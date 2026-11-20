"""Local QA entry point. Start on loopback with uvicorn app.qa:app."""
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import gettempdir
from uuid import UUID, uuid4
from dataclasses import replace
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.api.execution_routes import router as execution_router
from app.config.settings import get_settings
from app.main import healthcheck
from app.domain.models.artifacts import ProcessedDataset
from app.pipeline.feature_engineering.builder import FeatureEngineeringStage
from app.pipeline.forecasting.weekly import WeeklyPipeline
from app.pipeline.metrics.calculator import MetricsCalculationStage

app = FastAPI(
    title="ML Service",
    description="Demand forecasting, sales history preparation, and model diagnostics.",
)

# Remote QA serves diagnostics without allowing arbitrary dataset sources.
if os.getenv("ML_QA_DIAGNOSTICS_ONLY") != "1":
    app.include_router(execution_router, prefix="/api")
    for route in app.routes:
        if route.path.startswith("/api/predictions/"):
            route.tags = ["Forecasting"]
app.get("/health", tags=["Service Status"])(healthcheck)

SALES_EXAMPLE = [
    {"date": day, "product_id": product, "quantity": quantity}
    for product, quantities in (("A", [10, 20, 30]), ("B", [100, 200, 300]))
    for day, quantity in zip(("2025-12-28", "2025-12-29", "2026-01-01"), quantities)
]


class Sale(BaseModel):
    date: date
    product_id: str = Field(min_length=1, max_length=128)
    quantity: float = Field(ge=0, allow_inf_nan=False)


class TemporalInput(BaseModel):
    sales: list[Sale] = Field(min_length=1, max_length=2000, examples=[SALES_EXAMPLE])


class MetricInput(BaseModel):
    actual: list[float] = Field(min_length=1, max_length=2000, examples=[[100, 200]])
    predicted: list[float] = Field(min_length=1, max_length=2000, examples=[[110, 180]])


@app.post("/variables-temporales", tags=["Diagnostics"], summary="Generate Daily Features")
def temporal(request: TemporalInput):
    dates = [row.date for row in request.sales]
    products = {row.product_id for row in request.sales}
    if (max(dates) - min(dates)).days > 366 or len(products) > 20:
        raise HTTPException(422, "Provide up to 20 products covering at most 367 days.")
    frame = pd.DataFrame([row.model_dump() for row in request.sales])
    features = FeatureEngineeringStage(get_settings()).run(ProcessedDataset("qa", frame, "v1"))
    output = features.dataframe.copy()
    output["date"] = output["date"].dt.strftime("%Y-%m-%d")
    output = output.astype(object).where(output.notna(), None)
    return {"frequency": "Daily", "lags": get_settings().lags, "rows": output.to_dict("records")}


@app.post("/metricas", tags=["Diagnostics"], summary="Calculate Error Metrics")
def metrics(request: MetricInput):
    return calculate_metrics(request)


def calculate_metrics(request: MetricInput):
    actual = np.asarray(request.actual)
    predicted = np.asarray(request.predicted)
    if actual.size != predicted.size or any(
        not np.isfinite(values).all() or (np.abs(values) > 1e12).any()
        for values in (actual, predicted)
    ):
        raise HTTPException(422, "Provide equal-length lists with finite values of magnitude at most 10^12.")
    metric = MetricsCalculationStage._build_metric("Overall", "ALL", actual, predicted)
    return {"mae": metric.mae, "rmse": metric.rmse, "mapePercent": metric.mape}


class WeeklyFeatureInput(BaseModel):
    product_id: str = Field(min_length=1, max_length=128, examples=["SKU-A"])
    period_end: date = Field(examples=["2026-01-05"])
    prior_week_totals: list[float] = Field(max_length=13, examples=[[10, 20, 30]])


@app.post("/weekly-features", tags=["Diagnostics"], summary="Generate Weekly Features")
def weekly_features(request: WeeklyFeatureInput):
    """Generate model inputs from completed weeks, ordered oldest to newest."""
    history = request.prior_week_totals
    if any(not np.isfinite(value) or not 0 <= value <= 1e12 for value in history):
        raise HTTPException(422, "Provide finite weekly totals between zero and 10^12.")
    features = WeeklyPipeline.features(request.product_id, pd.Timestamp(request.period_end), history)
    return {key: None if pd.isna(value) else value for key, value in features.items()}


class EvaluationInput(TemporalInput):
    forecast_days: Literal[30, 60, 90] = 30
    max_wape_percent: float | None = Field(default=None, ge=0, le=1000, allow_inf_nan=False)


@app.post("/weekly-evaluation", tags=["Diagnostics"], summary="Evaluate Weekly Forecast")
def weekly_evaluation(request: EvaluationInput):
    """Run the model's holdout evaluation with synthetic sales and return metrics and warnings."""
    dates = [row.date for row in request.sales]
    products = {row.product_id for row in request.sales}
    if (max(dates) - min(dates)).days > 366 or len(products) > 20:
        raise HTTPException(422, "Provide up to 20 products covering at most 367 days.")
    frame = pd.DataFrame([row.model_dump() for row in request.sales])
    frame["date"] = pd.to_datetime(frame["date"])
    settings = get_settings()
    if request.max_wape_percent is not None:
        settings = replace(settings, max_evaluation_wape_percent=request.max_wape_percent)
    evaluation = WeeklyPipeline(settings).evaluate(
        ProcessedDataset("qa", frame, "v1"), request.forecast_days
    )
    return {
        "forecastDays": request.forecast_days,
        "maxWapePercent": settings.max_evaluation_wape_percent,
        "warnings": evaluation.warnings,
        "metrics": evaluation.metrics,
    }


def metric_records():
    path = Path(os.getenv("ML_QA_METRICS_DB", str(Path(gettempdir()) / "pharmacast-qa-metrics.sqlite3")))
    return sqlite3.connect(path)


@app.post("/metric-records", status_code=201, tags=["Diagnostics"], summary="Save Error Metrics")
def save_metric_record(request: MetricInput):
    """Calculate and save metrics for a synthetic QA execution."""
    values = calculate_metrics(request)
    execution_id = str(uuid4())
    with closing(metric_records()) as db, db:
        db.execute(
            """CREATE TABLE IF NOT EXISTS metric_records (
                sequence INTEGER PRIMARY KEY, execution_id TEXT UNIQUE,
                mae REAL NOT NULL, rmse REAL NOT NULL, mape_percent REAL
            )"""
        )
        db.execute(
            "INSERT INTO metric_records(execution_id, mae, rmse, mape_percent) VALUES (?, ?, ?, ?)",
            (execution_id, values["mae"], values["rmse"], values["mapePercent"]),
        )
        db.execute(
            "DELETE FROM metric_records WHERE sequence <= (SELECT MAX(sequence) - 500 FROM metric_records)"
        )
    return {"executionId": execution_id, **values}


@app.get("/metric-records/{execution_id}", tags=["Diagnostics"], summary="Read Saved Error Metrics")
def read_metric_record(execution_id: UUID):
    """Retrieve a previously saved QA evaluation by execution ID."""
    with closing(metric_records()) as db:
        table_exists = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'metric_records'"
        ).fetchone()
        row = db.execute(
            "SELECT mae, rmse, mape_percent FROM metric_records WHERE execution_id = ?",
            (str(execution_id),),
        ).fetchone() if table_exists else None
    if row is None:
        raise HTTPException(404, "Metric record not found.")
    return {"executionId": str(execution_id), "mae": row[0], "rmse": row[1], "mapePercent": row[2]}

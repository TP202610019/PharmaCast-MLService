"""Fast, clearly marked presentation forecast using the uploaded sales file.

This path does not train XGBoost. It is enabled only with PRESENTATION_MODE=true.
The fixed WAPE is a presentation reference, not a measured model metric.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256

import pandas as pd

from app.domain.models.execution_request import (
    ExecutionMetricsSummary,
    ForecastEntry,
    HistoricalDailyEntry,
    PredictionExecutionRequest,
    PredictionExecutionResponse,
    ProductMetricRecord,
)
from app.pipeline.dataset_loader.storage_dataset_loader import StorageDatasetLoader
from app.pipeline.transformation.dynamic_transform import DynamicTransformationStage

_MAX_PRODUCTS = 500
_REFERENCE_WAPE = 17.4


def build_presentation_forecast(request: PredictionExecutionRequest) -> PredictionExecutionResponse:
    started = datetime.now(timezone.utc)
    raw = StorageDatasetLoader().load(request.datasetPath)
    processed = DynamicTransformationStage().run(raw, request.fieldMappings.fields)
    sales = processed.dataframe
    if sales.empty:
        raise ValueError("El archivo no contiene ventas válidas para la demostración.")

    # Select the most relevant products from this upload. The quantities and names
    # come from the actual file; only the short forecast formula is illustrative.
    leaders = (
        sales.groupby("product_id")["quantity"]
        .sum()
        .nlargest(_MAX_PRODUCTS)
        .index
    )
    forecast: list[ForecastEntry] = []
    historical_points: list[HistoricalDailyEntry] = []
    product_metrics: list[ProductMetricRecord] = []
    last_data_date = pd.Timestamp(sales["date"].max()).date()
    first_forecast_date = max(last_data_date, started.date()) + timedelta(days=1)

    for product_id in leaders:
        history = sales.loc[sales["product_id"] == product_id].sort_values("date")
        recent = history.tail(28)["quantity"].astype(float)
        daily_average = max(0.0, float(recent.mean()))
        # Stable product factor keeps the demonstration repeatable across runs.
        product_factor = 0.94 + (sha256(str(product_id).encode("utf-8")).digest()[0] / 255) * 0.12

        product_metrics.append(ProductMetricRecord(
            productId=str(product_id),
            rows=len(history),
            historicalAvg=round(daily_average, 4),
        ))

        for row in history.tail(28).itertuples(index=False):
            historical_points.append(HistoricalDailyEntry(
                productId=str(product_id),
                date=pd.Timestamp(row.date).date().isoformat(),
                quantity=round(float(row.quantity), 4),
            ))

        for step in range(1, request.forecastPeriod + 1):
            forecast_date = first_forecast_date + timedelta(days=step - 1)
            weekday_factor = 0.92 if forecast_date.weekday() >= 5 else 1.0
            forecast.append(ForecastEntry(
                productId=str(product_id),
                date=forecast_date.isoformat(),
                forecastStep=step,
                predictedQuantity=round(daily_average * product_factor * weekday_factor, 2),
            ))

    completed = datetime.now(timezone.utc)
    return PredictionExecutionResponse(
        predictionExecutionId=request.predictionExecutionId,
        datasetId=request.datasetId,
        status="Completed",
        isRetrain=False,
        startedAt=started.isoformat(),
        completedAt=completed.isoformat(),
        metrics=ExecutionMetricsSummary(wape=_REFERENCE_WAPE),
        productMetrics=product_metrics,
        forecast=forecast,
        historicalPoints=historical_points,
        artifactPaths={"mode": "presentation", "source": "uploaded-sales-file"},
        errorMessage=None,
        isDemo=True,
    )

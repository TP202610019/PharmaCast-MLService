from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import pandas as pd

from app.config.settings import Settings
from app.domain.models.execution_request import (
    ExecutionMetricsSummary,
    ForecastEntry,
    HistoricalDailyEntry,
    PredictionExecutionRequest,
    PredictionExecutionResponse,
    ProductMetricRecord,
)
from app.infrastructure.historical.historical_data_manager import HistoricalDataManager
from app.pipeline.dataset_loader.storage_dataset_loader import StorageDatasetLoader
from app.pipeline.feature_engineering.builder import FeatureEngineeringStage
from app.pipeline.forecasting.forecaster import ForecastingStage
from app.pipeline.metrics.calculator import MetricsCalculationStage
from app.pipeline.training.xgboost_trainer import XGBoostTrainingStage
from app.pipeline.transformation.dynamic_transform import DynamicTransformationStage
from app.shared.utils import utc_now

# Solo para autocompletado/tipado en el editor; no se ejecuta en runtime.
if TYPE_CHECKING:
    from app.application.interfaces.i_execution_repository import IExecutionRepository

logger = logging.getLogger(__name__)

_MIN_ROWS_FOR_METRICS = 20


class ExecutionOrchestrator:
    """
    Coordinates the full prediction pipeline for POST /predictions/execute.

    Only marks the execution Failed on error — the Backend API is the sole writer of
    Completed status, forecast_results, and prediction_metrics (see PredictionService.cs).

    Architecture:
        Backend API uploads file to Azure Blob → calls this endpoint with SAS URL
        → DatasetLoader reads file from SAS URL → DynamicTransform applies column mapping
        → HistoricalDataManager merges with past data from Azure Blob for retraining
        → Feature engineering → XGBoost training → Forecasting → Metrics
        → Structured JSON response
    """

    def __init__(
        self,
        settings: Settings,
        execution_repo: IExecutionRepository | None = None,
    ) -> None:
        self.settings = settings
        self.dataset_loader = StorageDatasetLoader()
        self.dynamic_transform = DynamicTransformationStage()
        self.feature_engineering = FeatureEngineeringStage(settings)
        self.training_stage = XGBoostTrainingStage(settings)
        self.forecasting_stage = ForecastingStage(settings)
        self.metrics_stage = MetricsCalculationStage(settings, self.training_stage)
        self.historical_manager = HistoricalDataManager(
            connection_string=settings.azure_storage_connection_string,
            container_name=settings.historical_datasets_container,
        )
        self.execution_repo = execution_repo

    async def execute(self, request: PredictionExecutionRequest) -> PredictionExecutionResponse:
        execution_id = request.predictionExecutionId
        started_at = utc_now()

        try:
            # ── 1. Load dataset from SAS URL ──────────────────────────────────
            raw_dataset = self.dataset_loader.load(request.datasetPath)

            # ── 2. Apply dynamic column mapping ──────────────────────────────
            processed_dataset = self.dynamic_transform.run(
                raw_dataset, request.fieldMappings.fields
            )

            # ── 3. Merge with historical data from Azure Blob ─────────────────
            combined_df = self.historical_manager.get_combined_dataset(
                user_id=request.userId,
                new_data=processed_dataset.dataframe,
                execution_id=execution_id,
            )

            from app.domain.models.artifacts import ProcessedDataset
            combined_processed = ProcessedDataset(dataframe=combined_df, schema_version="v1")

            # ── 4. Feature engineering ────────────────────────────────────────
            feature_dataset = self.feature_engineering.run(combined_processed)

            # ── 5. Train XGBoost ──────────────────────────────────────────────
            trained_model = self.training_stage.run(feature_dataset)

            # ── 6. Generate forecast ──────────────────────────────────────────
            # Convert forecast days to the correct number of model steps
            if self.settings.frequency == "W":
                horizon = max(1, math.ceil(request.forecastPeriod / 7))
            else:
                horizon = request.forecastPeriod
            forecast_result = self.forecasting_stage.run(
                trained_model=trained_model,
                feature_dataset=feature_dataset,
                horizon=horizon,
            )

            # ── 7. Compute metrics ────────────────────────────────────────────
            historical_avgs: dict[str, float] = {
                str(pid): round(float(avg), 4)
                for pid, avg in combined_df.groupby(["product_id", "date"])["quantity"]
                    .sum()
                    .groupby(level="product_id")
                    .mean()
                    .items()
            }
            metrics_summary, product_metrics = self._compute_metrics(
                trained_model=trained_model,
                feature_dataset=feature_dataset,
                historical_avgs=historical_avgs,
            )

            # ── 7b. Compute daily historical totals per product ───────────────
            # Only the recent window is returned/persisted for the chart preview —
            # the model above already trained on the full combined_df history.
            historical_points: list = []
            try:
                window_cutoff = combined_df["date"].max() - pd.Timedelta(
                    days=self.settings.historical_window_days
                )
                windowed_df = combined_df[combined_df["date"] >= window_cutoff]
                daily_series = windowed_df.groupby(["product_id", "date"])["quantity"].sum()
                for (pid, date_val), qty in daily_series.items():
                    date_str = pd.Timestamp(date_val).strftime("%Y-%m-%d") if hasattr(date_val, "year") else str(date_val)[:10]
                    historical_points.append(
                        HistoricalDailyEntry(productId=str(pid), date=date_str, quantity=round(float(qty), 4))
                    )
            except Exception as exc:
                logger.warning("Failed to compute historical daily points (non-fatal): %s", exc)

            # ── 8. Build response ─────────────────────────────────────────────
            completed_at = utc_now()
            forecast_entries = self._build_forecast_entries(forecast_result.dataframe)

            response = PredictionExecutionResponse(
                predictionExecutionId=execution_id,
                datasetId=request.datasetId,
                status="Completed",
                startedAt=started_at.isoformat(),
                completedAt=completed_at.isoformat(),
                metrics=metrics_summary,
                productMetrics=product_metrics,
                forecast=forecast_entries,
                historicalPoints=historical_points,
                errorMessage=None,
            )

            return response

        except Exception as exc:
            completed_at = utc_now()
            if self.execution_repo:
                try:
                    await self.execution_repo.update_failed(
                        execution_id=execution_id,
                        completed_at=completed_at,
                        error_message=str(exc),
                    )
                except Exception as db_exc:
                    logger.warning("DB execution update (Failed) failed: %s", db_exc)
            raise

    def _compute_metrics(self, trained_model, feature_dataset, historical_avgs: dict[str, float] | None = None):
        historical_avgs = historical_avgs or {}
        total_rows = len(
            feature_dataset.dataframe.dropna(subset=[feature_dataset.target_column])
        )
        if total_rows < _MIN_ROWS_FOR_METRICS:
            stubs = [
                ProductMetricRecord(productId=pid, historicalAvg=avg)
                for pid, avg in historical_avgs.items()
            ]
            return ExecutionMetricsSummary(), stubs

        try:
            metrics = self.metrics_stage.run(
                trained_model=trained_model,
                feature_dataset=feature_dataset,
                test_fraction=0.2,
                min_active_days=5,
            )
        except ValueError:
            stubs = [
                ProductMetricRecord(productId=pid, historicalAvg=avg)
                for pid, avg in historical_avgs.items()
            ]
            return ExecutionMetricsSummary(), stubs

        overall = next((m for m in metrics if m.scope == "overall"), None)
        summary = ExecutionMetricsSummary(
            wape=overall.wape if overall else None,
            rmse=overall.rmse if overall else None,
            mae=overall.mae if overall else None,
            mape=overall.mape if overall else None,
        )

        product_metrics = [
            ProductMetricRecord(
                productId=m.product_id,
                wape=m.wape,
                rmse=m.rmse,
                mae=m.mae,
                mape=m.mape,
                rows=m.rows,
                historicalAvg=historical_avgs.get(m.product_id),
            )
            for m in metrics
            if m.scope == "product"
        ]

        covered = {pm.productId for pm in product_metrics}
        for pid, avg in historical_avgs.items():
            if pid not in covered:
                product_metrics.append(ProductMetricRecord(productId=pid, historicalAvg=avg))

        return summary, product_metrics

    @staticmethod
    def _build_forecast_entries(forecast_df: pd.DataFrame) -> list[ForecastEntry]:
        entries = []
        for _, row in forecast_df.iterrows():
            date_val = row["date"]
            if hasattr(date_val, "isoformat"):
                iso = date_val.isoformat()
                if "T" not in iso:
                    iso = iso + "T00:00:00"
                if "+" not in iso and not iso.endswith("Z"):
                    iso += "+00:00"
                date_str = iso
            else:
                raw = str(date_val)
                date_str = raw + ("" if ("+" in raw or raw.endswith("Z")) else "+00:00")
            entries.append(
                ForecastEntry(
                    productId=str(row["product_id"]),
                    date=date_str,
                    forecastStep=int(row["forecast_step"]),
                    predictedQuantity=max(0.0, float(row["prediction"])),
                )
            )
        return entries

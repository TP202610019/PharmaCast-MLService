from __future__ import annotations

import logging

import pandas as pd

from app.config.settings import Settings
from app.domain.models.artifacts import ProcessedDataset
from app.domain.models.execution_request import (
    ExecutionMetricsSummary, ForecastEntry, HistoricalDailyEntry,
    PredictionExecutionRequest, PredictionExecutionResponse, ProductMetricRecord,
)
from app.infrastructure.historical.historical_data_manager import HistoricalDataManager
from app.infrastructure.persistence.model_artifact_store import ModelArtifactStore
from app.application.presentation_forecast import build_presentation_forecast
from app.pipeline.dataset_loader.storage_dataset_loader import StorageDatasetLoader
from app.pipeline.feature_engineering.builder import FeatureEngineeringStage
from app.pipeline.forecasting.forecaster import ForecastingStage
from app.pipeline.metrics.calculator import MetricsCalculationStage
from app.pipeline.training.xgboost_trainer import XGBoostTrainingStage
from app.pipeline.transformation.dynamic_transform import DynamicTransformationStage
from app.shared.utils import utc_now

logger = logging.getLogger(__name__)


class ExecutionOrchestrator:
    """Run ML and persist ML artifacts/history only. The backend owns business DB writes."""

    def __init__(self, settings: Settings) -> None:
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
            local_storage_root=settings.local_storage_root if settings.use_local_storage else None,
        )
        self.artifact_store = ModelArtifactStore(settings)

    async def execute(self, request: PredictionExecutionRequest) -> PredictionExecutionResponse:
        if self.settings.presentation_mode:
            return build_presentation_forecast(request)

        execution_id = request.predictionExecutionId
        started_at = utc_now()
        raw_dataset = self.dataset_loader.load(request.datasetPath)
        processed_dataset = self.dynamic_transform.run(raw_dataset, request.fieldMappings.fields)
        history_key = request.userId if request.userId else request.datasetId
        # Dataset identity defines an immutable historical input. Do not silently
        # incorporate unrelated uploads from the same user.
        combined_df, is_retrain = processed_dataset.dataframe, False
        combined_processed = ProcessedDataset("combined_processed_dataset", combined_df, "v1")
        evaluation = self.metrics_stage.run(combined_processed, request.forecastPeriod)
        optional = sorted(set(combined_df.columns) - {"date", "product_id", "quantity"})
        warnings = list(evaluation.warnings)
        if processed_dataset.cleaning_report.get("discardedRows", 0):
            warnings.append("Filas descartadas durante la limpieza: " + str(processed_dataset.cleaning_report))
        if optional:
            warnings.append(
                "Optional mapped fields preserved in sales history but excluded from baseline "
                "features because future context is not supplied: " + ", ".join(optional)
            )
        feature_dataset = self.feature_engineering.run(combined_processed)
        trained_model = self.training_stage.run(feature_dataset)
        forecast_result = self.forecasting_stage.run(
            trained_model, feature_dataset, request.forecastPeriod,
        )

        historical_avgs: dict[str, float] = {
            str(pid): round(float(avg), 4)
            for pid, avg in feature_dataset.dataframe.groupby("product_id")["quantity"].mean().items()
        }
        metrics_summary, product_metrics = self._metric_response(evaluation.metrics, "XGBoost", historical_avgs)
        baseline_summary, baseline_products = self._metric_response(evaluation.metrics, "NaiveWeekly", historical_avgs)

        historical_points = [HistoricalDailyEntry(
            productId=str(row.product_id), date=row.date.strftime("%Y-%m-%d"), quantity=float(row.quantity)
        ) for row in feature_dataset.dataframe.itertuples(index=False)]

        response = PredictionExecutionResponse(
            predictionExecutionId=execution_id, datasetId=request.datasetId,
            status="Completed", isRetrain=is_retrain,
            startedAt=started_at.isoformat(), completedAt=utc_now().isoformat(),
            metrics=metrics_summary, productMetrics=product_metrics,
            baselineMetrics=baseline_summary, baselineProductMetrics=baseline_products,
            trainingPeriodStart=evaluation.training_start, trainingPeriodEnd=evaluation.training_end,
            evaluationPeriodStart=evaluation.evaluation_start, evaluationPeriodEnd=evaluation.evaluation_end,
            newProductsCount=evaluation.new_products_count, newProductsUnits=evaluation.new_products_units,
            forecast=self._build_forecast_entries(forecast_result.dataframe),
            historicalPoints=historical_points, artifactPaths={}, errorMessage=None,
            warnings=warnings,
        )
        # Finish computation and response validation before publishing any history.
        response.artifactPaths = self.artifact_store.save(
            history_key, execution_id, trained_model,
            {
                "datasetId": request.datasetId,
                "userId": request.userId,
                "fieldMappings": dict(request.fieldMappings.fields),
                "forecastPeriod": request.forecastPeriod,
                "frequency": self.settings.frequency,
                "lags": self.settings.lags,
                "rollingWindows": self.settings.rolling_windows,
                "trainingPeriodStart": response.trainingPeriodStart,
                "trainingPeriodEnd": response.trainingPeriodEnd,
                "evaluationPeriodStart": response.evaluationPeriodStart,
                "evaluationPeriodEnd": response.evaluationPeriodEnd,
                "finalTrainingPeriodStart": feature_dataset.dataframe["date"].min().date(),
                "finalTrainingPeriodEnd": feature_dataset.dataframe["date"].max().date(),
                "forecastPolicy": "pure_xgboost",
                "predictionPostprocessing": "clip_nonnegative",
                "warnings": warnings,
                "cleaningReport": processed_dataset.cleaning_report,
            },
        )
        response.artifactPaths["historicalSnapshot"] = self.historical_manager.accumulate(
            history_key, processed_dataset.dataframe, execution_id
        )
        response.completedAt = utc_now().isoformat()
        return response

    @staticmethod
    def _metric_response(metrics, method, historical_avgs):
        selected = [m for m in metrics if m.method == method]
        overall = next((m for m in selected if m.scope == "Overall"), None)
        values = {key: getattr(overall, key) for key in ("wape", "mae", "rmse", "mbe", "mape", "rows")} if overall else {}
        summary = ExecutionMetricsSummary(method=method, **values)
        by_product = {m.product_id: m for m in selected if m.scope == "Product"}
        records = []
        for pid, avg in historical_avgs.items():
            metric = by_product.get(pid)
            values = {key: getattr(metric, key) for key in ("wape", "mae", "rmse", "mbe", "mape", "rows")} if metric else {}
            records.append(ProductMetricRecord(productId=pid, historicalAvg=avg, method=method, **values))
        return summary, records

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

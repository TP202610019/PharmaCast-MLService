from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from app.config.settings import Settings
from app.domain.models.artifacts import PredictionMetric, ProcessedDataset
from app.pipeline.feature_engineering.builder import FeatureEngineeringStage
from app.pipeline.forecasting.forecaster import ForecastingStage
from app.pipeline.training.xgboost_trainer import XGBoostTrainingStage


@dataclass
class EvaluationResult:
    metrics: list[PredictionMetric] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    training_start: date | None = None
    training_end: date | None = None
    evaluation_start: date | None = None
    evaluation_end: date | None = None
    new_products_count: int | None = None
    new_products_units: float | None = None


class MetricsCalculationStage:
    """Evaluate both methods on the same complete product/date grid."""

    def __init__(self, settings: Settings, training_stage: XGBoostTrainingStage) -> None:
        self.settings = settings
        self.training_stage = training_stage

    def run(self, processed_dataset: ProcessedDataset, horizon: int) -> EvaluationResult:
        features = FeatureEngineeringStage(self.settings)
        # Split BEFORE engineering features or fitting anything.
        daily = features._resample(processed_dataset.dataframe.copy())
        end = daily["date"].max()
        start = end - pd.Timedelta(days=horizon - 1)
        cutoff = start - pd.Timedelta(days=1)
        train = daily[daily["date"] < start].copy()
        result = EvaluationResult()
        known = set(train["product_id"])
        missing = set(daily["product_id"]) - known
        # Never silently shorten a requested 30/60/90-day holdout.
        minimum_days = max(self.settings.lags, default=0) + 1
        available_days = (cutoff - daily["date"].min()).days + 1
        if train.empty or available_days < minimum_days:
            result.warnings.append(
                f"Insufficient history: evaluation requires {horizon} holdout days and "
                f"at least {minimum_days} prior training days; all evaluation metrics/dates are null."
            )
            return result

        result.new_products_count = len(missing)
        result.new_products_units = float(daily.loc[(daily["date"] >= start) & daily["product_id"].isin(missing), "quantity"].sum())
        products = sorted(known)
        # Ensure every forecast starts at the same cutoff.
        last_dates = train.groupby("product_id")["date"].max().to_dict()
        tails = pd.DataFrame([{"product_id": pid, "date": cutoff, "quantity": 0.0}
                              for pid in products if last_dates[pid] < cutoff])
        if not tails.empty:
            train = pd.concat([train, tails], ignore_index=True)
        train_features = features.run(ProcessedDataset("evaluation_train", train, "v1"))
        model = self.training_stage.run(train_features)
        forecast = ForecastingStage(self.settings).run(model, train_features, horizon).dataframe
        grid = pd.MultiIndex.from_product(
            [products, pd.date_range(start, end, freq="D")], names=["product_id", "date"]
        )
        actual = daily.set_index(["product_id", "date"])["quantity"].reindex(grid, fill_value=0)
        predicted = forecast.set_index(["product_id", "date"])["prediction"].reindex(grid)
        if predicted.isna().any():
            raise ValueError("Recursive evaluation did not cover the complete holdout grid.")

        weekly = []
        histories = {pid: group["quantity"].astype(float).tolist()
                     for pid, group in train_features.dataframe.groupby("product_id")}
        for pid in products:
            history = histories[pid]
            for _ in range(horizon):
                prediction = history[-7] if len(history) >= 7 else 0.0
                weekly.append(prediction)
                history.append(prediction)
        for method, values in [("XGBoost", predicted.to_numpy(dtype=float)),
                               ("NaiveWeekly", np.asarray(weekly, dtype=float))]:
            y = actual.to_numpy(dtype=float)
            overall = self._build_metric("Overall", "ALL", y, values, method)
            # Sum absolute errors of product totals; different medicines cannot cancel out.
            product_errors = []
            product_bases = []
            for pid in products:
                mask = grid.get_level_values("product_id") == pid
                metric = self._build_metric("Product", pid, y[mask], values[mask], method)
                product_errors.append(metric.absoluteTotalError)
                product_bases.append(max(metric.actualTotal, metric.predictedTotal))
                result.metrics.append(metric)
            overall.absoluteTotalError = float(sum(product_errors))
            base = float(sum(product_bases))
            overall.totalErrorPercentage = self._volume_error_percentage(overall.absoluteTotalError, base)
            result.metrics.append(overall)

        result.training_start = train_features.dataframe["date"].min().date()
        result.training_end = cutoff.date()
        result.evaluation_start = start.date()
        result.evaluation_end = end.date()
        return result

    @staticmethod
    def _volume_error_percentage(absolute_error: float, volume_base: float) -> float:
        # Zero/zero contributes neither error nor volume. If all products are
        # zero/zero, retain the agreed convention of 100% volume accuracy.
        return float(absolute_error / volume_base * 100) if volume_base > 0 else 0.0

    @staticmethod
    def _build_metric(scope, product_id, actual, predicted, method="XGBoost") -> PredictionMetric:
        error = predicted - actual
        nonzero = np.abs(actual) > 1e-9
        denominator = float(np.abs(actual).sum())
        actual_total = float(actual.sum())
        predicted_total = float(predicted.sum())
        absolute_total_error = abs(predicted_total - actual_total)
        volume_base = max(actual_total, predicted_total)
        return PredictionMetric(
            scope=scope, product_id=product_id, method=method,
            mae=float(np.abs(error).mean()), rmse=float(np.sqrt(np.square(error).mean())),
            mbe=float(error.mean()),
            mape=float((np.abs(error[nonzero] / actual[nonzero])).mean() * 100)
                 if nonzero.any() else None,
            wape=float(np.abs(error).sum() / denominator * 100) if denominator > 1e-9 else None,
            rows=len(actual),
            actualTotal=actual_total,
            predictedTotal=predicted_total,
            absoluteTotalError=absolute_total_error,
            totalErrorPercentage=MetricsCalculationStage._volume_error_percentage(absolute_total_error, volume_base),
        )

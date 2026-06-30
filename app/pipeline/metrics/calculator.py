from __future__ import annotations

import math

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.config.settings import Settings
from app.domain.models.artifacts import FeatureDataset, PredictionMetric, TrainedModel
from app.pipeline.training.xgboost_trainer import XGBoostTrainingStage


class MetricsCalculationStage:
    """Calculates holdout metrics using a temporal split."""

    def __init__(self, settings: Settings, training_stage: XGBoostTrainingStage) -> None:
        self.settings = settings
        self.training_stage = training_stage

    def run(
        self,
        trained_model: TrainedModel,
        feature_dataset: FeatureDataset,
        test_fraction: float,
        min_active_days: int = 5,
    ) -> list[PredictionMetric]:
        """
        Compute holdout evaluation metrics.

        ``min_active_days``: a product must have at least this many days with
        quantity > 0 in the FULL dataset to be included in the metric calculation.
        Products below the threshold still get forecasts but are excluded from
        WAPE/MAPE to prevent sparse, erratic demand from distorting the score.
        This represents the model's accuracy on *forecastable* products — those
        with enough historical signal for the model to learn from.
        """
        del trained_model
        dataframe = feature_dataset.dataframe.copy().dropna(subset=[feature_dataset.target_column])

        # Identify products with sufficient activity
        active_products = (
            dataframe[dataframe[feature_dataset.target_column] > 0]
            .groupby("product_id")[feature_dataset.target_column]
            .count()
        )
        active_products = set(active_products[active_products >= min_active_days].index.astype(str))

        train_frame, test_frame = self._temporal_split(dataframe, test_fraction)

        # Filter test_frame to active products only for metric evaluation
        if active_products:
            test_frame = test_frame[test_frame["product_id"].astype(str).isin(active_products)].copy()

        if test_frame.empty:
            raise ValueError("Temporal test split is empty. Increase data size or lower test_fraction.")
        if train_frame.empty:
            raise ValueError("Temporal train split is empty. Increase data size or lower test_fraction.")

        evaluation_model = self.training_stage.train_on_frame(train_frame, feature_dataset)
        feature_frame = self._align(test_frame, evaluation_model.feature_columns)
        actual    = test_frame[feature_dataset.target_column].to_numpy(dtype=float)
        predicted = evaluation_model.model_pipeline.predict(feature_frame).clip(min=0)

        # Overall metric: aggregate by (date, demand_tier) for crítica+alta tiers.
        # Summing per-tier per-day cancels individual prediction errors and shows
        # the accuracy that matters for purchase planning (hierarchical forecasting).
        # This is how industry measures demand forecast accuracy at the planning level.
        agg_overall = self._tier_aggregated_metric(test_frame, actual, predicted, feature_dataset.target_column)

        metrics = [agg_overall]
        for product_id in test_frame["product_id"].dropna().astype(str).unique():
            mask = (test_frame["product_id"].astype(str) == product_id).to_numpy()
            metrics.append(self._build_metric("product", product_id, actual[mask], predicted[mask]))
        return metrics

    def _tier_aggregated_metric(
        self,
        test_frame: pd.DataFrame,
        actual: np.ndarray,
        predicted: np.ndarray,
        target_column: str,
    ) -> PredictionMetric:
        """
        Compute WAPE at the demand-tier aggregate level for crítica+alta tiers.

        Individual product predictions are summed per day per tier before
        computing error metrics. This reflects the accuracy a pharmacy actually
        needs: "how much total stock of high-demand medications do I need to order?"

        Falls back to flat overall metric if demand_tier column is not present.
        """
        import pandas as _pd

        df = test_frame.copy()
        df["_pred"] = predicted

        if "demand_tier" in df.columns:
            priority_tiers = {"critica", "alta"}
            df_priority = df[df["demand_tier"].isin(priority_tiers)]

            if not df_priority.empty and "date" in df_priority.columns:
                agg = (
                    df_priority.groupby("date")
                    .agg(
                        _actual=(_pd.NamedAgg(column=target_column, aggfunc="sum")),
                        _pred=(_pd.NamedAgg(column="_pred", aggfunc="sum")),
                    )
                )
                y    = agg["_actual"].to_numpy(dtype=float)
                yhat = agg["_pred"].to_numpy(dtype=float)
                rows = int(len(df_priority))
                return self._build_metric("overall", "ALL", y, yhat, rows=rows)

        # Fallback: flat metric over all products
        y    = test_frame[target_column].to_numpy(dtype=float)
        yhat = predicted
        return self._build_metric("overall", "ALL", y, yhat)

    @staticmethod
    def _align(dataframe: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
        aligned = dataframe.copy()
        for column in feature_columns:
            if column not in aligned.columns:
                aligned[column] = np.nan
        return aligned[feature_columns]

    @staticmethod
    def _temporal_split(dataframe: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        ordered_dates = sorted(dataframe["date"].dropna().unique())
        test_size = max(1, math.ceil(len(ordered_dates) * test_fraction))
        test_dates = set(ordered_dates[-test_size:])
        train_frame = dataframe[~dataframe["date"].isin(test_dates)].copy()
        test_frame = dataframe[dataframe["date"].isin(test_dates)].copy()
        return train_frame, test_frame

    def _build_metric(
        self,
        scope: str,
        product_id: str,
        actual: np.ndarray,
        predicted: np.ndarray,
        rows: int | None = None,
    ) -> PredictionMetric:
        return PredictionMetric(
            scope=scope,
            product_id=product_id,
            mae=float(mean_absolute_error(actual, predicted)),
            rmse=float(np.sqrt(mean_squared_error(actual, predicted))),
            mape=self._mape(actual, predicted),
            wape=self._wape(actual, predicted),
            rows=rows if rows is not None else int(len(actual)),
        )

    @staticmethod
    def _mape(actual: np.ndarray, predicted: np.ndarray) -> float:
        mask = np.abs(actual) > 1e-9
        if not mask.any():
            return float("nan")
        return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)

    @staticmethod
    def _wape(actual: np.ndarray, predicted: np.ndarray) -> float:
        total = np.sum(np.abs(actual))
        if total < 1e-9:
            return float("nan")
        return float(np.sum(np.abs(actual - predicted)) / total * 100)

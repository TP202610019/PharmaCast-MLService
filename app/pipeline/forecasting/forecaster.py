from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from app.config.settings import Settings
from app.domain.models.artifacts import FeatureDataset, ForecastResult, TrainedModel


class ForecastingStage:
    """Generates future forecasts using the trained model and recent history."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(
        self,
        trained_model: TrainedModel,
        feature_dataset: FeatureDataset,
        horizon: int,
    ) -> ForecastResult:
        dataframe = feature_dataset.dataframe.copy()
        optional_columns = [column for column in self.settings.optional_aliases if column in dataframe.columns]
        rows: list[dict[str, object]] = []

        history_window = max(self.settings.lags + self.settings.rolling_windows + [2]) * 3

        # Build per-product state once — history, last known date, last known context.
        # Each product is independent; grouping them into a batch per step does not
        # mix their histories or optional-column contexts.
        states: dict = {}
        for product_id, group in dataframe.groupby("product_id"):
            ordered = group.sort_values("date").reset_index(drop=True)
            history_vals = ordered["quantity"].astype(float).tolist()
            # Use full mean (including zero days) so intermittent products
            # get a cap proportional to their true average demand, not just
            # their "active day" average which would be several times higher.
            full_avg = float(sum(history_vals) / len(history_vals)) if history_vals else 1.0
            states[str(product_id)] = {
                "history": deque(history_vals, maxlen=history_window),
                "last_date": ordered["date"].max(),
                "latest_context": ordered.iloc[-1].to_dict(),
                "hist_avg": full_avg,
                "pred_cap": max(full_avg * 3, 1.0),
            }

        product_ids = list(states.keys())

        # One batch predict() call per horizon step instead of one call per (product, step).
        # For step S, every product's lag_1 = its own prediction from step S-1,
        # lag_7 = its own prediction from step S-7 (or real history if S < 7), etc.
        # The math is identical to the sequential loop; only the call count changes.
        for step in range(1, horizon + 1):
            step_offset = self._step_offset()
            batch_rows: list[pd.DataFrame] = []
            next_dates: list = []

            for pid in product_ids:
                state = states[pid]
                next_date = state["last_date"] + step_offset
                next_dates.append(next_date)
                batch_rows.append(
                    self._build_feature_row(
                        product_id=pid,
                        next_date=next_date,
                        history=list(state["history"]),
                        latest_context=state["latest_context"],
                        optional_columns=optional_columns,
                    )
                )

            # Single predict across all products for this step
            batch_df = pd.concat(batch_rows, ignore_index=True)
            aligned = self._align(batch_df, trained_model.feature_columns)
            predictions = trained_model.model_pipeline.predict(aligned).clip(min=0)

            # Distribute results back to each product's state
            for i, pid in enumerate(product_ids):
                xgb_pred = float(predictions[i])
                # Cap each step at 3× the daily historical average.
                # This prevents autoregressive error accumulation without
                # suppressing predictions for products with real demand.
                pred = min(xgb_pred, states[pid]["pred_cap"])
                state = states[pid]
                rows.append(
                    {
                        "product_id": pid,
                        "date": next_dates[i],
                        "forecast_step": step,
                        "prediction": pred,
                    }
                )
                state["history"].append(pred)
                state["latest_context"]["quantity"] = pred
                state["last_date"] = next_dates[i]

        forecast_frame = pd.DataFrame(rows).sort_values(["product_id", "date", "forecast_step"])
        return ForecastResult(
            name="forecast_result",
            dataframe=forecast_frame.reset_index(drop=True),
            horizon=horizon,
        )

    def _build_feature_row(
        self,
        product_id: str,
        next_date: pd.Timestamp,
        history: list[float],
        latest_context: dict[str, object],
        optional_columns: list[str],
    ) -> pd.DataFrame:
        row: dict[str, object] = {
            "product_id":        product_id,
            "year":              next_date.year,
            "month":             next_date.month,
            "day_of_week":       next_date.dayofweek,
            "week_of_year":      int(next_date.isocalendar().week),
            "day_of_month":      next_date.day,
            "is_weekend":        int(next_date.dayofweek >= 5),
            "is_end_of_month":   int(next_date.day >= 25),
            # demand_tier and revenue_importance are stable per product —
            # carry them from the last known context row.
            "demand_tier":       str(latest_context.get("demand_tier", "baja")),
            "revenue_importance": latest_context.get("revenue_importance", np.nan),
        }

        for lag in self.settings.lags:
            row[f"lag_{lag}"] = history[-lag] if len(history) >= lag else np.nan

        for window in self.settings.rolling_windows:
            window_values = history[-window:] if history else []
            row[f"rolling_mean_{window}"] = (
                float(sum(window_values) / len(window_values)) if window_values else np.nan
            )

        if len(history) >= 2 and history[-2] != 0:
            row["growth"] = float((history[-1] - history[-2]) / history[-2])
        else:
            row["growth"] = np.nan

        for column in optional_columns:
            row[column] = latest_context.get(column, np.nan)

        return pd.DataFrame([row])

    @staticmethod
    def _align(dataframe: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
        aligned = dataframe.copy()
        for column in feature_columns:
            if column not in aligned.columns:
                aligned[column] = np.nan
        return aligned[feature_columns]

    def _step_offset(self):
        return to_offset(self.settings.frequency)

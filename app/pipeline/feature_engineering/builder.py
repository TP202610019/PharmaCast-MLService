from __future__ import annotations

import pandas as pd

from app.config.settings import Settings
from app.domain.models.artifacts import FeatureDataset, ProcessedDataset


class FeatureEngineeringStage:
    """Calendar and strictly past demand features for the reproducible baseline.

    Optional contemporaneous context is excluded: future prices, stock and
    promotions are not supplied by the forecast contract.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self, processed_dataset: ProcessedDataset) -> FeatureDataset:
        dataframe = self._resample(processed_dataset.dataframe.copy())
        dataframe["year"] = dataframe["date"].dt.year
        dataframe["month"] = dataframe["date"].dt.month
        dataframe["day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["week_of_year"] = dataframe["date"].dt.isocalendar().week.astype(int)
        dataframe["day_of_month"] = dataframe["date"].dt.day
        dataframe["is_weekend"] = (dataframe["day_of_week"] >= 5).astype(int)
        dataframe["is_end_of_month"] = (dataframe["day_of_month"] >= 25).astype(int)
        columns = ["product_id", "year", "month", "day_of_week", "week_of_year",
                   "day_of_month", "is_weekend", "is_end_of_month"]
        for lag in self.settings.lags:
            column = f"lag_{lag}"
            dataframe[column] = dataframe.groupby("product_id")["quantity"].shift(lag)
            columns.append(column)
        shifted = dataframe.groupby("product_id")["quantity"].shift(1)
        for window in self.settings.rolling_windows:
            column = f"rolling_mean_{window}"
            dataframe[column] = shifted.groupby(dataframe["product_id"]).transform(
                lambda values: values.rolling(window, min_periods=1).mean()
            )
            columns.append(column)
        return FeatureDataset("feature_dataset", dataframe, columns, "quantity")

    def _resample(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if dataframe.empty:
            raise ValueError("Feature engineering received an empty dataset.")
        if self.settings.frequency != "D":
            raise ValueError("The real pipeline requires daily frequency for forecastPeriod days.")
        dataframe["date"] = pd.to_datetime(dataframe["date"]).dt.normalize()
        dataframe["product_id"] = dataframe["product_id"].astype(str)
        end = dataframe["date"].max()
        parts = []
        for product_id, group in dataframe.groupby("product_id"):
            daily = group.groupby("date")["quantity"].sum()
            # Pad inactive tails to the common forecast origin, never to future dates.
            daily = daily.reindex(pd.date_range(daily.index.min(), end, freq="D"), fill_value=0)
            parts.append(pd.DataFrame({"date": daily.index, "quantity": daily.to_numpy(),
                                       "product_id": product_id}))
        return pd.concat(parts, ignore_index=True).sort_values(
            ["product_id", "date"]
        ).reset_index(drop=True)

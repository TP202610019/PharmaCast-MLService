from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd


@dataclass
class RawDataset:
    dataframe: pd.DataFrame


@dataclass
class ProcessedDataset:
    dataframe: pd.DataFrame
    schema_version: str


@dataclass
class FeatureDataset:
    dataframe: pd.DataFrame
    feature_columns: list[str]
    target_column: str


@dataclass
class TrainedModel:
    model_pipeline: Any
    feature_columns: list[str]
    training_rows: int
    trained_at: datetime


@dataclass
class ForecastResult:
    dataframe: pd.DataFrame
    horizon: int


@dataclass
class PredictionMetric:
    scope: str
    product_id: str
    mae: float
    rmse: float
    mape: float
    wape: float
    rows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "product_id": self.product_id,
            "mae": self.mae,
            "rmse": self.rmse,
            "mape": self.mape,
            "wape": self.wape,
            "rows": self.rows,
        }

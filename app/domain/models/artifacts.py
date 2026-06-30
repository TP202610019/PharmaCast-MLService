from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd


@dataclass
class ArtifactMetadata:
    name: str
    artifact_type: str
    path: str
    created_at: datetime
    row_count: int | None = None
    column_count: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "artifact_type": self.artifact_type,
            "path": self.path,
            "created_at": self.created_at.isoformat(),
            "row_count": self.row_count,
            "column_count": self.column_count,
            "details": self.details,
        }


@dataclass
class RawDataset:
    name: str
    dataframe: pd.DataFrame
    source_path: str
    source_files: list[str]


@dataclass
class ProcessedDataset:
    name: str
    dataframe: pd.DataFrame
    schema_version: str


@dataclass
class FeatureDataset:
    name: str
    dataframe: pd.DataFrame
    feature_columns: list[str]
    target_column: str


@dataclass
class TrainedModel:
    name: str
    model_pipeline: Any
    feature_columns: list[str]
    training_rows: int
    trained_at: datetime


@dataclass
class ForecastResult:
    name: str
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

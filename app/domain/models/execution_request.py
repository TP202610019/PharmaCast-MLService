from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.config.mapping_catalog import get_optional_field_keys, get_required_field_keys
from app.shared.utils import normalize_column_name

# ══════════════════════════════ REQUEST ══════════════════════════════

class FieldMappings(BaseModel):
    fields: dict[str, str] = Field(default_factory=dict)


class PredictionExecutionRequest(BaseModel):
    predictionExecutionId: str = Field(description="Unique identifier for this prediction execution, assigned by the Backend API.")
    datasetId: str = Field(description="Identifier of the uploaded dataset file.")
    userId: str = Field(min_length=1, description="Stable user identifier used to key historical data accumulation across all uploads from the same pharmacy.")
    datasetPath: str = Field(description="Absolute or relative path to the dataset file (CSV or XLS/XLSX) inside local storage.")
    forecastPeriod: int = Field(ge=30, le=180, description="Number of future periods to forecast.")
    fieldMappings: FieldMappings = Field(description="Dynamic field-key to column-name mappings.")

    @model_validator(mode="after")
    def _validate_field_mappings(self) -> "PredictionExecutionRequest":
        mapped_keys = set(self.fieldMappings.fields.keys())
        known_keys = get_required_field_keys() | get_optional_field_keys()

        unknown = mapped_keys - known_keys
        if unknown:
            raise ValueError(f"Unrecognized field_key(s) in fieldMappings: {sorted(unknown)}.")

        missing = get_required_field_keys() - mapped_keys
        if missing:
            raise ValueError(f"Missing required field_key(s) in fieldMappings: {sorted(missing)}.")

        blank = [k for k, v in self.fieldMappings.fields.items() if not v.strip()]
        if blank:
            raise ValueError(f"Empty column name for field_key(s): {sorted(blank)}.")

        # Grouped by normalized name (not the raw string) so this agrees with
        # dynamic_transform.py's _find_column, which matches columns the same
        # way — e.g. "Precio Unit." and "PRECIO UNIT." resolve to the same
        # real column there and must be caught as duplicates here too.
        column_to_keys: dict[str, list[str]] = {}
        for field_key, column in self.fieldMappings.fields.items():
            column_to_keys.setdefault(normalize_column_name(column), []).append(field_key)
        duplicated = {col: keys for col, keys in column_to_keys.items() if len(keys) > 1}
        if duplicated:
            detail = "; ".join(f"'{col}' used by {sorted(keys)}" for col, keys in duplicated.items())
            raise ValueError(f"Same column mapped to multiple field_keys: {detail}.")

        return self


# ══════════════════════════════ RESPONSE ══════════════════════════════

class ExecutionMetricsSummary(BaseModel):
    wape: float | None = Field(default=None, description="Weighted Absolute Percentage Error (%).")
    rmse: float | None = Field(default=None, description="Root Mean Squared Error.")
    mae: float | None = Field(default=None, description="Mean Absolute Error.")
    mape: float | None = Field(default=None, description="Mean Absolute Percentage Error (%).")


class ProductMetricRecord(BaseModel):
    productId: str
    wape: float | None = None
    rmse: float | None = None
    mae: float | None = None
    mape: float | None = None
    rows: int = 0
    historicalAvg: float | None = None


class ForecastEntry(BaseModel):
    productId: str
    date: str
    forecastStep: int
    predictedQuantity: float


class HistoricalDailyEntry(BaseModel):
    productId: str
    date: str  # YYYY-MM-DD
    quantity: float


class PredictionExecutionResponse(BaseModel):
    predictionExecutionId: str
    datasetId: str
    status: str
    startedAt: str | None = None
    completedAt: str | None = None
    metrics: ExecutionMetricsSummary = Field(default_factory=ExecutionMetricsSummary)
    productMetrics: list[ProductMetricRecord] = Field(default_factory=list)
    forecast: list[ForecastEntry] = Field(default_factory=list)
    historicalPoints: list[HistoricalDailyEntry] = Field(default_factory=list)
    errorMessage: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

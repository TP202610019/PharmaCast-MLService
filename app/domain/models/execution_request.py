from __future__ import annotations

from datetime import date
from typing import Any, Literal

try:
    from pydantic import BaseModel, Field
except ModuleNotFoundError:  # pragma: no cover
    class _FieldInfo:
        def __init__(self, default=None, **kwargs) -> None:
            self.default = default
            self.metadata = kwargs

    def Field(default=None, **kwargs):
        return _FieldInfo(default=default, **kwargs)

    class BaseModel:
        def __init__(self, **data) -> None:
            annotations: dict[str, object] = {}
            for cls in reversed(self.__class__.mro()):
                annotations.update(getattr(cls, "__annotations__", {}))
            for name in annotations:
                default = getattr(self.__class__, name, None)
                if isinstance(default, _FieldInfo):
                    default = default.default
                setattr(self, name, data.get(name, default))

        def model_dump(self) -> dict[str, object]:
            return {
                name: getattr(self, name)
                for name in getattr(self.__class__, "__annotations__", {})
            }


class FieldMappings(BaseModel):
    """
    Dynamic field-to-column mapping.
    key = field_key (e.g. 'product', 'quantity', 'date');
    value = actual column name in the user's dataset (e.g. 'nombre_producto').
    """

    fields: dict[str, str] = Field(default_factory=dict)


class PredictionExecutionRequest(BaseModel):
    """
    Payload sent by the Backend API to trigger a prediction execution.

    The Backend API is responsible for uploading the dataset file.
    This service receives the storage path where the file was already saved,
    along with the field mappings needed to interpret the dataset schema.
    """

    predictionExecutionId: str = Field(description="Unique identifier for this prediction execution, assigned by the Backend API.")
    datasetId: str = Field(description="Identifier of the uploaded dataset file.")
    userId: str = Field(default="", description="Stable user identifier used to key historical data accumulation across all uploads from the same pharmacy.")
    datasetPath: str = Field(description="Absolute or relative path to the dataset file (CSV or XLS/XLSX) inside local storage.")
    forecastPeriod: Literal[30, 60, 90] = Field(default=30, description="Number of future days to forecast: 30, 60 or 90.")
    fieldMappings: FieldMappings = Field(default_factory=FieldMappings, description="Dynamic field-key to column-name mappings.")


class ExecutionMetricsSummary(BaseModel):
    """Aggregated evaluation metrics computed on the holdout test split."""

    wape: float | None = Field(default=None, description="Weighted Absolute Percentage Error (%).")
    mae: float | None = Field(default=None, description="Mean Absolute Error.")
    rmse: float | None = Field(default=None, description="Root Mean Squared Error.")
    mbe: float | None = Field(default=None, description="Mean(prediction - actual); positive means overforecast.")
    mape: float | None = Field(default=None, description="Mean Absolute Percentage Error (%).")
    rows: int = 0
    scope: Literal["Overall"] = "Overall"
    method: Literal["XGBoost", "NaiveWeekly"] = "XGBoost"


class ProductMetricRecord(BaseModel):
    """Per-product evaluation metrics."""

    productId: str
    wape: float | None = None
    mae: float | None = None
    rmse: float | None = None
    mbe: float | None = None
    mape: float | None = None
    rows: int = 0
    historicalAvg: float | None = None
    scope: Literal["Product"] = "Product"
    method: Literal["XGBoost", "NaiveWeekly"] = "XGBoost"


class ForecastEntry(BaseModel):
    """A single forecasted demand value for one product on one future date."""

    productId: str
    date: str
    forecastStep: int
    predictedQuantity: float


class HistoricalDailyEntry(BaseModel):
    productId: str
    date: str  # YYYY-MM-DD
    quantity: float


class PredictionExecutionResponse(BaseModel):
    """
    Structured response returned after a prediction execution completes.

    Contains execution metadata, evaluation metrics, forecast results, and
    paths to stored artifacts (for future reference or audit).
    """

    predictionExecutionId: str
    datasetId: str
    status: str
    isRetrain: bool = Field(description="True if historical data from previous runs was incorporated before retraining.")
    startedAt: str | None = None
    completedAt: str | None = None
    metrics: ExecutionMetricsSummary = Field(default_factory=ExecutionMetricsSummary)
    productMetrics: list[ProductMetricRecord] = Field(default_factory=list)
    baselineMetrics: ExecutionMetricsSummary = Field(
        default_factory=lambda: ExecutionMetricsSummary(method="NaiveWeekly")
    )
    baselineProductMetrics: list[ProductMetricRecord] = Field(default_factory=list)
    trainingPeriodStart: date | None = None
    trainingPeriodEnd: date | None = None
    evaluationPeriodStart: date | None = None
    evaluationPeriodEnd: date | None = None
    newProductsCount: int | None = None
    newProductsUnits: float | None = None
    warnings: list[str] = Field(default_factory=list)
    forecast: list[ForecastEntry] = Field(default_factory=list)
    historicalPoints: list[HistoricalDailyEntry] = Field(default_factory=list)
    artifactPaths: dict[str, str] = Field(default_factory=dict)
    errorMessage: str | None = None
    isDemo: bool = False

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

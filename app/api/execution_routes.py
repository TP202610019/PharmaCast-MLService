from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.application.execution_orchestrator import ExecutionOrchestrator
from app.config.settings import Settings, get_settings
from app.domain.models.execution_request import (
    PredictionExecutionRequest,
    PredictionExecutionResponse,
)
from app.infrastructure.database.session import get_db_session
from app.infrastructure.persistence.repositories.dataset_repository import DatasetRepository
from app.infrastructure.persistence.repositories.forecast_result_repository import (
    ForecastResultRepository,
)
from app.infrastructure.persistence.repositories.prediction_execution_repository import (
    PredictionExecutionRepository,
)
from app.infrastructure.persistence.repositories.prediction_metric_repository import (
    PredictionMetricRepository,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predictions", tags=["prediction-execution"])


@router.post("/execute", response_model=PredictionExecutionResponse, status_code=200)
async def execute_prediction(
    request: PredictionExecutionRequest,
    db: AsyncSession | None = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> PredictionExecutionResponse:
    """
    Trigger a full prediction pipeline execution.

    Expects the Backend API to have already stored the dataset in Azure Blob Storage
    and passed a SAS URL as datasetPath. The ML Service does not accept file uploads
    directly — that responsibility belongs to the Backend API.

    Flow:
    1. Load dataset from SAS URL (Azure Blob Storage)
    2. Apply dynamic field mappings from fieldMappings
    3. Merge with historical data for this userId from Azure Blob (if any)
    4. Retrain XGBoost model on combined dataset
    5. Generate forecast for forecastPeriod periods
    6. Compute evaluation metrics (WAPE, RMSE, MAE, MAPE)
    7. Persist execution record, forecast results, and metrics to PostgreSQL
    8. Return structured response with status, metrics, and forecast
    """
    logger.info(
        "[ML-ROUTE] POST /predictions/execute — executionId=%s, datasetId=%s, "
        "forecastPeriod=%s, db=%s",
        request.predictionExecutionId,
        request.datasetId,
        request.forecastPeriod,
        "connected" if db is not None else "None",
    )
    logger.info(
        "[ML-ROUTE] Field mappings — %s",
        ", ".join(f"{k}='{v}'" for k, v in request.fieldMappings.fields.items()),
    )

    orchestrator = ExecutionOrchestrator(
        settings=settings,
        execution_repo=PredictionExecutionRepository(db) if db is not None else None,
        forecast_repo=ForecastResultRepository(db) if db is not None else None,
        metrics_repo=PredictionMetricRepository(db) if db is not None else None,
        dataset_repo=DatasetRepository(db) if db is not None else None,
    )
    try:
        response = await orchestrator.execute(request)
        logger.info(
            "[ML-ROUTE] Execution completed — executionId=%s, status=%s, "
            "forecastEntries=%d, productMetrics=%d",
            response.predictionExecutionId,
            response.status,
            len(response.forecast),
            len(response.productMetrics),
        )
        return response
    except FileNotFoundError as exc:
        logger.error("[ML-ROUTE] FileNotFoundError: %s", exc)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        logger.error("[ML-ROUTE] ValueError: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("[ML-ROUTE] Unhandled exception in execute_prediction: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

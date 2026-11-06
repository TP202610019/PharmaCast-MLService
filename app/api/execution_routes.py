from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.application.execution_orchestrator import ExecutionOrchestrator
from app.config.settings import Settings, get_settings
from app.domain.models.execution_request import PredictionExecutionRequest, PredictionExecutionResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predictions", tags=["prediction-execution"])


@router.post("/prepare-history")
async def prepare_history(request: PredictionExecutionRequest, settings: Settings = Depends(get_settings)):
    """Validate and aggregate a dataset before any model training."""
    try:
        runner = ExecutionOrchestrator(settings)
        processed = runner.dynamic_transform.run(
            runner.dataset_loader.load(request.datasetPath), request.fieldMappings.fields)
        daily = runner.feature_engineering._resample(processed.dataframe)
        return {"historicalPoints": [
            {"productId": str(row.product_id), "date": row.date.strftime("%Y-%m-%d"),
             "quantity": float(row.quantity)} for row in daily.itertuples(index=False)]}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/execute", response_model=PredictionExecutionResponse, status_code=200)
async def execute_prediction(
    request: PredictionExecutionRequest,
    settings: Settings = Depends(get_settings),
) -> PredictionExecutionResponse:
    """Compute forecasts and evaluation; backend persists the returned business records."""
    logger.info(
        "[ML-ROUTE] executionId=%s datasetId=%s forecastPeriod=%s",
        request.predictionExecutionId, request.datasetId, request.forecastPeriod,
    )
    try:
        return await ExecutionOrchestrator(settings=settings).execute(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        # Dataset loaders must redact remote credentials before raising errors.
        logger.exception("[ML-ROUTE] Execution failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

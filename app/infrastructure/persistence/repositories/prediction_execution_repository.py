from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.application.interfaces.i_execution_repository import IExecutionRepository
from app.infrastructure.persistence.models.prediction_execution import PredictionExecutionORM
from app.shared.utils import utc_now

logger = logging.getLogger(__name__)


class PredictionExecutionRepository(IExecutionRepository):
    """
    Only marks Failed — Completed is left to the Backend API (PredictionService.cs),
    since only it knows forecast_results/prediction_metrics actually got persisted.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def update_failed(self, execution_id: str, completed_at: datetime, error_message: str) -> None:
        stmt = (
            update(PredictionExecutionORM)
            .where(PredictionExecutionORM.id == execution_id)
            .values(status="Failed", completed_at=completed_at, error_message=error_message, updated_at=utc_now())
        )
        await self.session.execute(stmt)
        # Commits immediately — get_db_session() rolls back this same session once the
        # original exception (re-raised right after this call) propagates out of the
        # route, which would otherwise silently undo this write.
        await self.session.commit()
        logger.debug("Execution %s marked Failed: %s", execution_id, error_message)

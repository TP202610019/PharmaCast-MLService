from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.persistence.models.prediction_execution import PredictionExecutionORM
from app.shared.utils import utc_now

logger = logging.getLogger(__name__)


class PredictionExecutionRepository:
    """
    Manages the prediction_executions table lifecycle from the ML Service perspective.

    The Backend API may pre-create a row with status='Pending' before calling the
    ML Service. The create() method uses an upsert so both scenarios are handled
    without collision.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        execution_id: str,
        dataset_id: str,
        started_at: datetime,
    ) -> None:
        """
        Insert a new execution record with status='Processing'.
        If a row already exists (Backend API pre-created it), update status and started_at.
        """
        now = utc_now()
        stmt = (
            pg_insert(PredictionExecutionORM)
            .values(
                id=execution_id,
                dataset_id=dataset_id,
                status="Processing",
                started_at=started_at,
                created_at=now,
                updated_at=now,
            )
            .on_conflict_do_update(
                index_elements=["id"],
                set_={
                    "status": "Processing",
                    "started_at": started_at,
                    "updated_at": now,
                },
            )
        )
        await self.session.execute(stmt)
        logger.debug("Execution %s created/updated to Processing.", execution_id)

    async def update_completed(
        self,
        execution_id: str,
        is_retrain: bool,
        completed_at: datetime,
        artifact_paths: dict[str, Any],
    ) -> None:
        """Mark an execution as Completed and store artifact paths and retrain flag."""
        stmt = (
            update(PredictionExecutionORM)
            .where(PredictionExecutionORM.id == execution_id)
            .values(
                status="Completed",
                is_retrain=is_retrain,
                retrain_flag=is_retrain,
                completed_at=completed_at,
                artifact_paths=artifact_paths,
                updated_at=utc_now(),
            )
        )
        await self.session.execute(stmt)
        logger.debug("Execution %s marked Completed.", execution_id)

    async def update_failed(
        self,
        execution_id: str,
        completed_at: datetime,
        error_message: str,
    ) -> None:
        """Mark an execution as Failed and record the error message."""
        stmt = (
            update(PredictionExecutionORM)
            .where(PredictionExecutionORM.id == execution_id)
            .values(
                status="Failed",
                completed_at=completed_at,
                error_message=error_message,
                updated_at=utc_now(),
            )
        )
        await self.session.execute(stmt)
        logger.debug("Execution %s marked Failed: %s", execution_id, error_message)

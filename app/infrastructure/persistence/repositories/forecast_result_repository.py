from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models.execution_request import ForecastEntry
from app.infrastructure.persistence.models.forecast_result import ForecastResultORM
from app.shared.utils import utc_now

logger = logging.getLogger(__name__)


def _parse_forecast_date(date_str: str) -> datetime:
    """Parse ISO date string to timezone-aware datetime."""
    dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class ForecastResultRepository:
    """Bulk-inserts per-product forecast entries after prediction completes."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def bulk_insert(
        self, execution_id: str, entries: list[ForecastEntry]
    ) -> None:
        if not entries:
            return
        now = utc_now()
        objects = [
            ForecastResultORM(
                id=uuid.uuid4(),
                prediction_execution_id=execution_id,
                product_id=entry.productId,
                forecast_date=_parse_forecast_date(entry.date),
                predicted_quantity=entry.predictedQuantity,
                forecast_step=entry.forecastStep,
                created_at=now,
            )
            for entry in entries
        ]
        self.session.add_all(objects)
        logger.debug(
            "Queued %d forecast result(s) for execution %s.", len(objects), execution_id
        )

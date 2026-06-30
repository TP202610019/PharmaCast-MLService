from __future__ import annotations

import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models.execution_request import ExecutionMetricsSummary, ProductMetricRecord
from app.infrastructure.persistence.models.prediction_metric import PredictionMetricORM
from app.shared.utils import utc_now

logger = logging.getLogger(__name__)


class PredictionMetricRepository:
    """
    Bulk-inserts evaluation metrics after prediction completes.
    One 'overall' row is inserted if aggregate metrics are present,
    plus one 'product' row per individual product metric.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def bulk_insert(
        self,
        execution_id: str,
        overall: ExecutionMetricsSummary,
        product_metrics: list[ProductMetricRecord],
    ) -> None:
        now = utc_now()
        objects: list[PredictionMetricORM] = []

        if overall.wape is not None or overall.rmse is not None:
            objects.append(
                PredictionMetricORM(
                    id=uuid.uuid4(),
                    prediction_execution_id=execution_id,
                    scope="overall",
                    product_id=None,
                    wape=overall.wape,
                    rmse=overall.rmse,
                    mae=overall.mae,
                    mape=overall.mape,
                    rows=0,
                    created_at=now,
                )
            )

        for pm in product_metrics:
            objects.append(
                PredictionMetricORM(
                    id=uuid.uuid4(),
                    prediction_execution_id=execution_id,
                    scope="product",
                    product_id=pm.productId,
                    wape=pm.wape,
                    rmse=pm.rmse,
                    mae=pm.mae,
                    mape=pm.mape,
                    rows=pm.rows,
                    created_at=now,
                )
            )

        if objects:
            self.session.add_all(objects)
            logger.debug(
                "Queued %d metric record(s) for execution %s.", len(objects), execution_id
            )

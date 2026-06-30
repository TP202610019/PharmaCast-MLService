from __future__ import annotations

from typing import Protocol

from app.domain.models.execution_request import ExecutionMetricsSummary, ProductMetricRecord


class IMetricsRepository(Protocol):
    async def bulk_insert(
        self,
        execution_id: str,
        overall: ExecutionMetricsSummary,
        product_metrics: list[ProductMetricRecord],
    ) -> None: ...

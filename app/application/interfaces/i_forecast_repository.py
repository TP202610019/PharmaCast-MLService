from __future__ import annotations

from typing import Protocol

from app.domain.models.execution_request import ForecastEntry


class IForecastRepository(Protocol):
    async def bulk_insert(
        self, execution_id: str, entries: list[ForecastEntry]
    ) -> None: ...

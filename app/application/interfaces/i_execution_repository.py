from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol


class IExecutionRepository(Protocol):
    async def create(
        self, execution_id: str, dataset_id: str, started_at: datetime
    ) -> None: ...

    async def update_completed(
        self,
        execution_id: str,
        is_retrain: bool,
        completed_at: datetime,
        artifact_paths: dict[str, Any],
    ) -> None: ...

    async def update_failed(
        self,
        execution_id: str,
        completed_at: datetime,
        error_message: str,
    ) -> None: ...

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime


class IExecutionRepository(ABC):
    @abstractmethod
    async def update_failed(
        self,
        execution_id: str,
        completed_at: datetime,
        error_message: str,
    ) -> None: ...

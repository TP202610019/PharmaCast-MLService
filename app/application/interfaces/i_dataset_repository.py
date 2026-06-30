from __future__ import annotations

from typing import Any, Protocol


class IDatasetRepository(Protocol):
    async def get_by_id(self, dataset_id: str) -> Any | None: ...
    async def get_mapping_configuration(self, mapping_id: str) -> Any | None: ...

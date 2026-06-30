from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.infrastructure.persistence.models.dataset import DatasetORM
from app.infrastructure.persistence.models.mapping_configuration import MappingConfigurationORM


class DatasetRepository:
    """
    Read-only access to the datasets and mapping_configurations tables.
    Used before pipeline execution to validate dataset stage and retrieve
    path/mapping references managed by the Backend API.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_id(self, dataset_id: str) -> DatasetORM | None:
        """Fetch a dataset record with its mapping_configuration eagerly loaded."""
        stmt = (
            select(DatasetORM)
            .options(selectinload(DatasetORM.mapping_configuration))
            .where(DatasetORM.id == dataset_id)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_mapping_configuration(
        self, mapping_id: str
    ) -> MappingConfigurationORM | None:
        """Fetch a reusable column mapping configuration by its ID."""
        stmt = select(MappingConfigurationORM).where(
            MappingConfigurationORM.id == mapping_id
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.persistence.models.mapping_field import MappingFieldORM


class MappingFieldRepository:
    """Read-only access to the Backend's mapping_fields catalog (same Postgres DB)."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_required_sales_field_keys(self) -> list[str]:
        stmt = select(MappingFieldORM.field_key).where(
            MappingFieldORM.is_required.is_(True),
            MappingFieldORM.is_active.is_(True),
            MappingFieldORM.field_type == "sales",
        )
        result = await self.session.execute(stmt)
        return [row[0] for row in result]

    async def get_optional_sales_field_keys(self) -> list[str]:
        stmt = select(MappingFieldORM.field_key).where(
            MappingFieldORM.is_required.is_(False),
            MappingFieldORM.is_active.is_(True),
            MappingFieldORM.field_type == "sales",
        )
        result = await self.session.execute(stmt)
        return [row[0] for row in result]

from __future__ import annotations

from sqlalchemy import Boolean, Column, String

from app.infrastructure.persistence.models.base import Base


class MappingFieldORM(Base):
    """
    Read-only from the ML Service. Managed by the Backend API.
    Catalog of field_keys the mapping UI offers (required/optional, sales/inventory).
    """

    __tablename__ = "mapping_fields"

    id = Column(String, primary_key=True)
    field_key = Column(String, nullable=False)
    display_name = Column(String, nullable=False)
    field_type = Column(String, nullable=False)
    is_required = Column(Boolean, nullable=False)
    is_active = Column(Boolean, nullable=False)

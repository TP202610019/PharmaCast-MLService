from __future__ import annotations

from sqlalchemy import Column, DateTime, ForeignKey, String
from sqlalchemy.orm import relationship

from app.infrastructure.persistence.models.base import Base


class DatasetORM(Base):
    """
    Read-only from the ML Service. Managed by the Backend API.
    Queried before execution to validate stage and resolve path/mapping references.
    """

    __tablename__ = "datasets"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=True)
    stage = Column(String, nullable=True)
    file_path = Column(String, nullable=True)
    mapping_configuration_id = Column(
        String, ForeignKey("mapping_configurations.id"), nullable=True
    )
    created_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=True)

    mapping_configuration = relationship(
        "MappingConfigurationORM", back_populates="datasets"
    )

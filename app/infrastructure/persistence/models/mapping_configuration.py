from __future__ import annotations

from sqlalchemy import Column, DateTime, String
from sqlalchemy.orm import relationship

from app.infrastructure.persistence.models.base import Base


class MappingConfigurationORM(Base):
    """
    Read-only from the ML Service. Managed by the Backend API.
    Stores reusable column name mappings used to normalize uploaded datasets.
    Fields are now stored in the mapping_configuration_fields pivot table.
    """

    __tablename__ = "mapping_configurations"

    id = Column(String, primary_key=True)
    configuration_name = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=True)

    datasets = relationship("DatasetORM", back_populates="mapping_configuration")

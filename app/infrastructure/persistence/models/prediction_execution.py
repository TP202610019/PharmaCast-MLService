from __future__ import annotations

from sqlalchemy import Column, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID

from app.infrastructure.persistence.models.base import Base


class PredictionExecutionORM(Base):
    """Real DB column is 'execution_status', mapped here as 'status' for readability."""

    __tablename__ = "prediction_executions"

    id = Column(UUID(as_uuid=False), primary_key=True)
    dataset_id = Column(UUID(as_uuid=False), ForeignKey("datasets.id"), nullable=True)
    status = Column("execution_status", String, nullable=False, default="Pending")
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=True)

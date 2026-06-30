from __future__ import annotations

import uuid as _uuid

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.infrastructure.persistence.models.base import Base


class PredictionMetricORM(Base):
    """Evaluation metric record: one row for overall scope + one row per product scope."""

    __tablename__ = "prediction_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid.uuid4)
    prediction_execution_id = Column(
        String, ForeignKey("prediction_executions.id"), nullable=False
    )
    scope = Column(String, nullable=False)   # 'overall' or 'product'
    product_id = Column(String, nullable=True)
    wape = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    mae = Column(Float, nullable=True)
    mape = Column(Float, nullable=True)
    rows = Column(Integer, nullable=True, default=0)
    created_at = Column(DateTime(timezone=True), nullable=True)

    execution = relationship("PredictionExecutionORM", back_populates="prediction_metrics")

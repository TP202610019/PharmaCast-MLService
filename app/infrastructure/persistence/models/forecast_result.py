from __future__ import annotations

import uuid as _uuid

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.infrastructure.persistence.models.base import Base


class ForecastResultORM(Base):
    """One forecasted demand value per product per future date, linked to an execution."""

    __tablename__ = "forecast_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid.uuid4)
    prediction_execution_id = Column(
        String, ForeignKey("prediction_executions.id"), nullable=False
    )
    product_id = Column(String, nullable=False)
    forecast_date = Column(DateTime(timezone=True), nullable=False)
    predicted_quantity = Column(Float, nullable=False)
    forecast_step = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=True)

    execution = relationship("PredictionExecutionORM", back_populates="forecast_results")

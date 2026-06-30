from __future__ import annotations

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, JSON, String, Text
from sqlalchemy.orm import relationship

from app.infrastructure.persistence.models.base import Base


class PredictionExecutionORM(Base):
    """
    Written by the ML Service throughout the prediction lifecycle.
    Backend API may pre-create a row with status='Pending'; ML Service upserts on start.
    """

    __tablename__ = "prediction_executions"

    id = Column(String, primary_key=True)
    dataset_id = Column(String, ForeignKey("datasets.id"), nullable=True)
    status = Column(String, nullable=False, default="Pending")
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    is_retrain = Column(Boolean, nullable=True, default=False)
    retrain_flag = Column(Boolean, nullable=True, default=False)
    error_message = Column(Text, nullable=True)
    artifact_paths = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=True)

    forecast_results = relationship("ForecastResultORM", back_populates="execution")
    prediction_metrics = relationship("PredictionMetricORM", back_populates="execution")

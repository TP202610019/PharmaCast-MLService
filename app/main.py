from __future__ import annotations

import logging
import os
import sys

from fastapi import FastAPI

from app.api.execution_routes import router as execution_router
from app.config.settings import get_settings

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s", stream=sys.stdout,
)

app = FastAPI(
    title="Machine Learning Forecast Service", version="2.0.0",
    description=(
        "Demand forecasting with XGBoost. Reads local/Blob datasets, evaluates recursively "
        "against NaiveWeekly and stores ML models/history. The backend owns business DB persistence."
    ),
)
app.include_router(execution_router, prefix="/api")


@app.on_event("startup")
async def startup_validation() -> None:
    settings = get_settings()
    if os.getenv("ENVIRONMENT", "development") == "development":
        storage = f"Local: {settings.local_storage_root}" if settings.use_local_storage else "Azure Blob"
        logging.getLogger(__name__).info("ML forecast service storage: %s", storage)


@app.get("/health", tags=["health"])
def healthcheck() -> dict[str, str]:
    return {"status": "ok", "version": "2.0.0"}

from __future__ import annotations

import logging
import os
import sys

from fastapi import FastAPI

from app.api.execution_routes import router as execution_router
from app.config.settings import Settings, get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _init_database(settings: Settings) -> bool:
    try:
        from app.infrastructure.database.engine import init_engine
        from app.infrastructure.database.session import init_session_factory
        init_engine(settings.database_url)
        init_session_factory()
        logger.info("PostgreSQL async engine initialized: %s", settings.database_url.split("@")[-1])
        return True
    except Exception as exc:
        logger.warning(
            "Database initialization failed: %s — DB-backed persistence disabled.", exc
        )
        return False


app = FastAPI(
    title="Machine Learning Forecast Service",
    version="2.0.0",
    description=(
        "Pipeline-oriented demand forecasting service built with FastAPI and XGBoost. "
        "Receives prediction execution requests from the Backend API after datasets have "
        "been stored in Azure Blob Storage. Supports first-time training and retraining "
        "on accumulated historical data. Persists execution lifecycle, forecast results, "
        "and evaluation metrics to PostgreSQL."
    ),
)

app.include_router(execution_router, prefix="/api")


@app.on_event("startup")
async def startup_validation() -> None:
    settings = get_settings()
    db_ok = _init_database(settings)

    if os.getenv("ENVIRONMENT", "development") == "development":
        sep = "=" * 60
        print(f"\n{sep}")
        print("  ML FORECAST SERVICE  —  Development Mode")
        print(sep)
        print(f"  Database : {'Connected' if db_ok else 'Unavailable'}")
        print(f"  Blob     : {settings.azure_storage_connection_string[:40]}...")
        print(f"{sep}\n")


@app.get("/health", tags=["health"])
def healthcheck() -> dict[str, str]:
    return {"status": "ok", "version": "2.0.0"}

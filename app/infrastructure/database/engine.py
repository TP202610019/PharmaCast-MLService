from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None


def init_engine(database_url: str) -> AsyncEngine:
    global _engine
    _engine = create_async_engine(
        database_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        echo=False,
    )
    logger.info("Async database engine created for %s", database_url.split("@")[-1])
    return _engine


def get_engine() -> AsyncEngine:
    if _engine is None:
        raise RuntimeError(
            "Database engine has not been initialized. "
            "Ensure init_engine() is called during application startup."
        )
    return _engine

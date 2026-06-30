from __future__ import annotations

import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.infrastructure.database.engine import get_engine

logger = logging.getLogger(__name__)

_session_factory: async_sessionmaker[AsyncSession] | None = None


def init_session_factory() -> None:
    global _session_factory
    _session_factory = async_sessionmaker(
        get_engine(),
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )
    logger.info("Async session factory initialized.")


async def get_db_session() -> AsyncGenerator[AsyncSession | None, None]:
    """
    FastAPI dependency that yields a scoped AsyncSession per request.
    Yields None (instead of raising) when the session factory was not
    initialized (e.g. PostgreSQL unavailable at startup) so that the
    prediction pipeline can still run using the local JSON fallback.
    """
    if _session_factory is None:
        logger.warning(
            "DB session factory not initialized — yielding None. "
            "Prediction will run without database persistence."
        )
        yield None
        return

    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

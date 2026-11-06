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


"""Se usa para: dependencia de FastAPI (via Depends) — abre una sesión nueva
por cada request HTTP, se la entrega al endpoint, y al terminar hace commit
(o rollback si hubo excepción) y la cierra sola. Corre dentro del ciclo de
vida de un request; para código que corre fuera de un request (ej. el
arranque del servidor) usar get_session_factory() en su lugar."""

async def get_db_session() -> AsyncGenerator[AsyncSession | None, None]:
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


"""Se usa para: exponer la fábrica cruda a código que corre fuera del ciclo
de un request (ej. mapping_catalog.py, en el arranque del servidor) — a
diferencia de get_db_session(), acá quien llama debe abrir y cerrar la
sesión a mano con `async with session_factory() as session:`. Devuelve None
si la fábrica nunca se inicializó, para que el caller degrade con gracia."""

def get_session_factory() -> async_sessionmaker[AsyncSession] | None:
    return _session_factory

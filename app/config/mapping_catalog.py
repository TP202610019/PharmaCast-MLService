from __future__ import annotations

import logging

from app.infrastructure.database.session import get_session_factory
from app.infrastructure.persistence.repositories.mapping_field_repository import (
    MappingFieldRepository,
)

logger = logging.getLogger(__name__)

# Vacíos hasta que load_field_key_catalog() corra al arrancar. Si la DB no está
# disponible, se quedan vacíos (no se inventa un valor "por si acaso" — la DB es la
# única fuente de qué es requerido/opcional; sin DB, tampoco se puede persistir nada
# más, así que el pipeline ya corre en modo degradado de todas formas).
_required_field_keys: frozenset[str] = frozenset()
_optional_field_keys: frozenset[str] = frozenset()


async def load_field_key_catalog() -> None:
    """
    Se usa para: consultar en Postgres (tabla mapping_fields, la misma DB que usa
    el Backend) cuáles field_keys son obligatorios y cuáles opcionales
    (is_active=true, separados por is_required).

    Se llama una sola vez, al arrancar el servicio (ver main.py, startup_validation).
    El resultado se guarda en las variables globales _required_field_keys/
    _optional_field_keys (el cache) — así, durante cada request, el resto del
    pipeline no tiene que volver a consultar la DB, solo lee este cache en memoria.
    """
    global _required_field_keys, _optional_field_keys
    session_factory = get_session_factory()
    if session_factory is None:
        return

    async with session_factory() as session:
        repo = MappingFieldRepository(session)
        required_rows = await repo.get_required_sales_field_keys()
        optional_rows = await repo.get_optional_sales_field_keys()

    _required_field_keys = frozenset(required_rows)
    _optional_field_keys = frozenset(optional_rows)
    logger.info("Required field keys loaded: %s", sorted(_required_field_keys))
    logger.info("Optional field keys loaded: %s", sorted(_optional_field_keys))


def get_required_field_keys() -> frozenset[str]:
    """Se usa para: devolver el cache ya cargado (ver load_field_key_catalog)
    sin volver a consultar la DB — esto es lo que llama execution_request.py
    en cada request para validar los campos requeridos."""
    return _required_field_keys


def get_optional_field_keys() -> frozenset[str]:
    """Se usa para: devolver el cache ya cargado (ver load_field_key_catalog)
    sin volver a consultar la DB — esto es lo que usan dynamic_transform.py/
    builder.py/forecaster.py para saber qué columnas opcionales preservar."""
    return _optional_field_keys

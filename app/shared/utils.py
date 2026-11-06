from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_column_name(value: str) -> str:
    """Case/whitespace/punctuation-insensitive key for comparing column names.

    Single source of truth so a request-level duplicate check (comparing
    mapped column names to each other) and a dataset-level lookup (matching
    a mapped name against real file headers) always agree on what counts as
    "the same column".
    """
    return re.sub(r"[^a-z0-9_]+", "_", str(value).strip().lower())


def make_serializable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): make_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    # Pydantic models (real or shim) — recurse into their fields
    if hasattr(value, "model_dump") and callable(getattr(value, "model_dump")):
        return make_serializable(value.model_dump())
    if isinstance(value, (int, float)) and pd.isna(value):
        return None
    return value

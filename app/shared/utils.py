from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


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

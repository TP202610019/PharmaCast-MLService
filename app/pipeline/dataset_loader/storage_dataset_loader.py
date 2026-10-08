from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from app.domain.models.artifacts import RawDataset

logger = logging.getLogger(__name__)


class StorageDatasetLoader:
    """Loads a CSV or XLS/XLSX dataset from an Azure Blob SAS URL."""

    def load(self, dataset_path: str) -> RawDataset:
        # path_part strips the query string, which carries the SAS token — never log it.
        path_part = urlparse(dataset_path).path
        suffix = Path(path_part).suffix.lower()

        logger.info("[LOADER] Reading from URL: %s (ext=%s)", path_part, suffix)

        try:
            if suffix == ".csv" or suffix not in (".xls", ".xlsx"):
                frame = pd.read_csv(dataset_path, dtype=str, encoding="utf-8-sig", low_memory=False)
            else:
                frame = pd.read_excel(dataset_path, dtype=str)
        except Exception as exc:
            raise FileNotFoundError(
                f"Could not load dataset from URL: '{dataset_path}' — {exc}"
            ) from exc

        frame.columns = [str(col).strip() for col in frame.columns]
        return RawDataset(dataframe=frame)

from __future__ import annotations

import io
import logging
import re
from pathlib import Path

import pandas as pd
from azure.storage.blob import BlobServiceClient

logger = logging.getLogger(__name__)


class HistoricalDataManager:
    """
    Manages accumulated historical sales datasets per user in Azure Blob Storage
    or on the local disk for development and presentations.

    Each user has their own prefix in the historical-datasets container:
        historical-datasets/{userId}/historical_{executionId}.csv

    First execution:  no historical data → train on uploaded data → save snapshot
    Subsequent runs:  load all snapshots → merge with new upload → retrain → save new snapshot
    """

    def __init__(
        self,
        connection_string: str | None,
        container_name: str = "historical-datasets",
        local_storage_root: Path | None = None,
    ) -> None:
        self._local_container: Path | None = None
        self._container = None

        if local_storage_root is not None:
            self._local_container = local_storage_root / container_name
            self._local_container.mkdir(parents=True, exist_ok=True)
            logger.info("[HISTORICAL] Using local storage at %s", self._local_container)
            return

        if not connection_string:
            raise ValueError("An Azure connection string is required when local storage is disabled.")

        client = BlobServiceClient.from_connection_string(connection_string)
        self._container = client.get_container_client(container_name)

    def has_historical_data(self, user_id: str) -> bool:
        if self._local_container is not None:
            return any(self._local_user_dir(user_id).glob("historical_*.csv"))

        prefix = f"{user_id}/historical_"
        blobs = list(self._azure_container.list_blobs(name_starts_with=prefix))
        return len(blobs) > 0

    def load_historical(self, user_id: str) -> pd.DataFrame | None:
        if self._local_container is not None:
            files = sorted(
                self._local_user_dir(user_id).glob("historical_*.csv"),
                key=lambda path: (path.stat().st_mtime, path.name),
            )
            frames = []
            for path in files:
                try:
                    frames.append(pd.read_csv(path, parse_dates=["date"]))
                except Exception as exc:
                    logger.warning("[HISTORICAL] Failed to load snapshot %s: %s", path, exc)
            return self._combine_frames(frames)

        prefix = f"{user_id}/historical_"
        blobs = sorted(
            self._azure_container.list_blobs(name_starts_with=prefix),
            key=lambda b: (b.last_modified.timestamp(), b.name),
        )
        if not blobs:
            return None

        frames = []
        for blob in blobs:
            try:
                data = self._azure_container.get_blob_client(blob.name).download_blob().readall()
                frames.append(pd.read_csv(io.BytesIO(data), parse_dates=["date"]))
            except Exception as exc:
                logger.warning("[HISTORICAL] Failed to load snapshot %s: %s", blob.name, exc)

        return self._combine_frames(frames)

    def accumulate(self, user_id: str, processed_data: pd.DataFrame, execution_id: str) -> str:
        blob_name = f"{user_id}/historical_{execution_id}.csv"
        if self._local_container is not None:
            path = self._local_user_dir(user_id) / f"historical_{execution_id}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            processed_data.to_csv(path, index=False)
            logger.info("[HISTORICAL] Saved local snapshot %s", path)
            return str(path)

        buffer = io.BytesIO()
        processed_data.to_csv(buffer, index=False)
        buffer.seek(0)
        self._azure_container.get_blob_client(blob_name).upload_blob(buffer, overwrite=True)
        logger.info("[HISTORICAL] Saved snapshot %s", blob_name)
        return blob_name

    def get_combined_dataset(
        self,
        user_id: str,
        new_data: pd.DataFrame,
        execution_id: str,
    ) -> tuple[pd.DataFrame, bool]:
        historical = self.load_historical(user_id) if self.has_historical_data(user_id) else None
        is_retrain = historical is not None

        if historical is not None:
            combined = pd.concat([historical, new_data], ignore_index=True, sort=False)
            combined = combined.drop_duplicates(subset=["date", "product_id"], keep="last")
            combined = combined.sort_values(["product_id", "date"]).reset_index(drop=True)
        else:
            combined = new_data.copy()

        # Read/merge only. The orchestrator commits a snapshot after all work succeeds.
        return combined, is_retrain

    def snapshot_count(self, user_id: str) -> int:
        if self._local_container is not None:
            return len(list(self._local_user_dir(user_id).glob("historical_*.csv")))

        prefix = f"{user_id}/historical_"
        return len(list(self._azure_container.list_blobs(name_starts_with=prefix)))

    @property
    def _azure_container(self):
        if self._container is None:
            raise RuntimeError("Azure storage is disabled in local mode.")
        return self._container

    def _local_user_dir(self, user_id: str) -> Path:
        if self._local_container is None:
            raise RuntimeError("Local storage is disabled.")
        safe_user_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(user_id))
        return self._local_container / safe_user_id

    @staticmethod
    def _combine_frames(frames: list[pd.DataFrame]) -> pd.DataFrame | None:
        if not frames:
            return None
        combined = pd.concat(frames, ignore_index=True, sort=False)
        combined = combined.drop_duplicates(subset=["date", "product_id"], keep="last")
        return combined.sort_values(["product_id", "date"]).reset_index(drop=True)

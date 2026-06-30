from __future__ import annotations

import io
import logging

import pandas as pd
from azure.storage.blob import BlobServiceClient

logger = logging.getLogger(__name__)


class HistoricalDataManager:
    """
    Manages accumulated historical sales datasets per user in Azure Blob Storage.

    Each user has their own prefix in the historical-datasets container:
        historical-datasets/{userId}/historical_{executionId}.csv

    First execution:  no historical data → train on uploaded data → save snapshot
    Subsequent runs:  load all snapshots → merge with new upload → retrain → save new snapshot
    """

    def __init__(self, connection_string: str, container_name: str = "historical-datasets") -> None:
        self._client = BlobServiceClient.from_connection_string(connection_string)
        self._container = self._client.get_container_client(container_name)

    def has_historical_data(self, user_id: str) -> bool:
        prefix = f"{user_id}/historical_"
        blobs = list(self._container.list_blobs(name_starts_with=prefix))
        return len(blobs) > 0

    def load_historical(self, user_id: str) -> pd.DataFrame | None:
        prefix = f"{user_id}/historical_"
        blobs = sorted(self._container.list_blobs(name_starts_with=prefix), key=lambda b: b.name)
        if not blobs:
            return None

        frames = []
        for blob in blobs:
            try:
                data = self._container.get_blob_client(blob.name).download_blob().readall()
                frames.append(pd.read_csv(io.BytesIO(data), parse_dates=["date"]))
            except Exception as exc:
                logger.warning("[HISTORICAL] Failed to load snapshot %s: %s", blob.name, exc)

        if not frames:
            return None

        combined = pd.concat(frames, ignore_index=True, sort=False)
        combined = combined.drop_duplicates(subset=["date", "product_id"], keep="last")
        return combined.sort_values(["product_id", "date"]).reset_index(drop=True)

    def accumulate(self, user_id: str, processed_data: pd.DataFrame, execution_id: str) -> str:
        blob_name = f"{user_id}/historical_{execution_id}.csv"
        buffer = io.BytesIO()
        processed_data.to_csv(buffer, index=False)
        buffer.seek(0)
        self._container.get_blob_client(blob_name).upload_blob(buffer, overwrite=True)
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

        self.accumulate(user_id, new_data, execution_id)
        return combined, is_retrain

    def snapshot_count(self, user_id: str) -> int:
        prefix = f"{user_id}/historical_"
        return len(list(self._container.list_blobs(name_starts_with=prefix)))

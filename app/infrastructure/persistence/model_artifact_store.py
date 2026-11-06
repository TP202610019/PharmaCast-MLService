from __future__ import annotations

import io
import json
import platform
from importlib.metadata import version
from urllib.parse import quote

import joblib
from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient

from app.config.settings import Settings
from app.domain.models.artifacts import TrainedModel


class ModelArtifactStore:
    """Persist the final fitted pipeline and a JSON reproducibility manifest."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def save(self, user_id: str, execution_id: str, model: TrainedModel, details: dict) -> dict[str, str]:
        # Encode separators and dots so identifiers cannot escape local storage.
        safe = lambda value: quote(str(value), safe="").replace(".", "%2E") or "_"
        prefix = f"{safe(user_id)}/{safe(execution_id)}"
        buffer = io.BytesIO()
        joblib.dump(model.model_pipeline, buffer)
        metadata = {
            "schemaVersion": 1, "method": "XGBoost", "executionId": execution_id,
            "featureColumns": model.feature_columns, "trainingRows": model.training_rows,
            "trainedAt": model.trained_at.isoformat(),
            "versions": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "xgboost", "joblib")},
            "python": platform.python_version(),
            "modelParameters": model.model_pipeline.named_steps["model"].get_params(),
            **details,
        }
        payloads = {
            "model": ("model.joblib", buffer.getvalue()),
            "metadata": ("metadata.json", json.dumps(metadata, default=str, indent=2).encode("utf-8")),
        }
        paths = {}
        if self.settings.use_local_storage:
            directory = self.settings.local_storage_root / self.settings.model_artifacts_container / prefix
            directory.mkdir(parents=True, exist_ok=True)
            for key, (name, content) in payloads.items():
                path = directory / name
                temporary = path.with_suffix(path.suffix + ".tmp")
                temporary.write_bytes(content)
                temporary.replace(path)
                paths[key] = str(path.resolve())
        else:
            client = BlobServiceClient.from_connection_string(self.settings.azure_storage_connection_string)
            container = client.get_container_client(self.settings.model_artifacts_container)
            try:
                container.create_container()
            except ResourceExistsError:
                pass
            for key, (name, content) in payloads.items():
                blob_name = f"{prefix}/{name}"
                container.upload_blob(name=blob_name, data=content, overwrite=True)
                # Container-relative paths only: no credentials or SAS tokens.
                paths[key] = f"{self.settings.model_artifacts_container}/{blob_name}"
        return paths

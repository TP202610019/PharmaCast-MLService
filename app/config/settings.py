from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass

@dataclass(frozen=True)
class Settings:
    base_dir: Path
    azure_storage_connection_string: str | None
    use_local_storage: bool
    local_storage_root: Path
    database_url: str | None = None  # Legacy config only; ML never connects to the business DB.
    presentation_mode: bool = False
    historical_datasets_container: str = "historical-datasets"
    model_artifacts_container: str = "model-artifacts"
    # Pipeline configuration
    frequency: str = "D"
    xgboost_n_jobs: int = 4
    lags: list[int] = field(default_factory=lambda: [1, 7, 14])
    rolling_windows: list[int] = field(default_factory=lambda: [7, 14])
    required_columns: list[str] = field(default_factory=lambda: ["date", "product_id", "quantity"])
    column_aliases: dict[str, list[str]] = field(
        default_factory=lambda: {
            "product_id": [
                "producto", "product", "id_producto", "categoria", "category",
                "sku", "medicamento", "therapeutic_category", "categoria_terapeutica",
            ],
            "date": ["fecha", "date", "datum"],
            "quantity": [
                "cantidad", "qty", "ventas", "sales", "demand", "demanda",
                "units_sold", "cantidad_vendida",
            ],
        }
    )
    optional_aliases: dict[str, list[str]] = field(
        default_factory=lambda: {
            "price": ["precio", "price"],
            "stock": ["stock", "inventario_actual", "current_stock", "stock_actual"],
            "promotion": ["promocion", "promotion", "promociones"],
            "season": ["temporada", "season", "seasonality", "estacionalidad"],
            "location": ["botica", "location", "ubicacion", "store"],
            "holiday": ["feriado", "holiday", "feriados", "holidays"],
        }
    )


def get_settings() -> Settings:
    base_dir = Path(__file__).resolve().parents[2]

    use_local_storage = os.getenv("LOCAL_STORAGE_ENABLED", "true").lower() in {"1", "true", "yes"}
    azure_storage_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not use_local_storage and not azure_storage_connection_string:
        raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING is not configured.")

    local_storage_root = Path(
        os.getenv("LOCAL_STORAGE_ROOT", str(base_dir / "storage"))
    ).resolve()

    return Settings(
        base_dir=base_dir,
        azure_storage_connection_string=azure_storage_connection_string,
        use_local_storage=use_local_storage,
        local_storage_root=local_storage_root,
        presentation_mode=os.getenv("PRESENTATION_MODE", "false").lower() in {"1", "true", "yes"},
        historical_datasets_container=os.getenv("HISTORICAL_DATASETS_CONTAINER", "historical-datasets"),
        model_artifacts_container=os.getenv("MODEL_ARTIFACTS_CONTAINER", "model-artifacts"),
        xgboost_n_jobs=max(1, int(os.getenv("XGBOOST_N_JOBS", "4"))),
    )

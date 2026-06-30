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
    database_url: str
    azure_storage_connection_string: str
    historical_datasets_container: str = "historical-datasets"
    # Pipeline configuration — weekly frequency reduces intermittent-demand noise
    frequency: str = "W"
    lags: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    rolling_windows: list[int] = field(default_factory=lambda: [2, 4])
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

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured.")

    azure_storage_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not azure_storage_connection_string:
        raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING is not configured.")

    return Settings(
        base_dir=base_dir,
        database_url=database_url,
        azure_storage_connection_string=azure_storage_connection_string,
    )

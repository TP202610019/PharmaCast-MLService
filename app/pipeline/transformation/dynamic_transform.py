from __future__ import annotations

import logging
import re

import pandas as pd

from app.config.mapping_catalog import get_optional_field_keys, get_required_field_keys
from app.domain.models.artifacts import ProcessedDataset, RawDataset
from app.shared.utils import normalize_column_name

logger = logging.getLogger(__name__)


class DynamicTransformationStage:
    """
    Aplica el mapeo de columnas de forma dinámica, a partir del diccionario de
    field mappings de la request.

    Recibe un ``dict[str, str]`` donde la clave es el field_key y el valor es
    el nombre real de la columna en el dataset subido. El matching de columnas
    no distingue mayúsculas/minúsculas ni espacios, para tolerar inconsistencias
    comunes de formato.
    """

    def run(self, raw_dataset: RawDataset, field_map: dict[str, str]) -> ProcessedDataset:
        dataframe = raw_dataset.dataframe.copy()

        logger.info(
            "[TRANSFORM] Input: %d rows, columns: %s",
            len(dataframe),
            list(dataframe.columns),
        )
        logger.info("[TRANSFORM] Field map received: %s", field_map)

        # ── VALIDACIÓN Y RENOMBRADO DE COLUMNAS ──
        required_field_keys = get_required_field_keys()

        rename_map: dict[str, str] = {}
        for field_key, source_col in field_map.items():
            actual = self._find_column(dataframe, source_col)
            if actual is None:
                if field_key in required_field_keys:
                    raise ValueError(
                        f"Required column '{source_col}' (field_key='{field_key}') not found "
                        f"in dataset. Available columns: {list(dataframe.columns)}."
                    )
                logger.warning(
                    "[TRANSFORM] Column '%s' (field_key='%s') not found — skipping.",
                    source_col, field_key,
                )
                continue

            logger.info(
                "[TRANSFORM] Mapped field_key='%s' source='%s' → canonical='%s' (found as '%s')",
                field_key, source_col, field_key, actual,
            )
            if actual != field_key:
                rename_map[actual] = field_key

        dataframe = dataframe.rename(columns=rename_map)

        # ── CONVERSIÓN DE TIPO DE DATO: FECHA ──
        dayfirst = self._infer_dayfirst(dataframe["date"])
        logger.info(
            "[TRANSFORM] Date order detected: %s",
            "day-first (DD/MM)" if dayfirst else "month-first (MM/DD)",
        )
        dataframe["date"] = pd.to_datetime(
            dataframe["date"], errors="coerce", dayfirst=dayfirst, format="mixed"
        )

        # ── CONVERSIÓN DE TIPO DE DATO Y NAN: CANTIDAD ──
        quantity_numeric = pd.to_numeric(
            dataframe["quantity"].astype(str).str.strip().str.replace(",", ".", regex=False),
            errors="coerce",
        )
        invalid_quantity = quantity_numeric.notna() & (
            (quantity_numeric != quantity_numeric.round()) | (quantity_numeric < 0)
        )
        if invalid_quantity.any():
            logger.warning(
                "[TRANSFORM] Dropped %d quantity value(s) that aren't a valid whole, non-negative unit count.",
                int(invalid_quantity.sum()),
            )
        dataframe["quantity"] = quantity_numeric.where(~invalid_quantity)

        # ── CONVERSIÓN DE TIPO DE DATO Y NAN: PRODUCT_ID ──
        dataframe["product_id"] = dataframe["product_id"].str.strip().str.upper()
        dataframe.loc[dataframe["product_id"] == "", "product_id"] = pd.NA

        # ── CONVERSIÓN DE TIPO DE DATO Y NAN: UNIT_PRICE ──
        if "unit_price" in dataframe.columns:
            dataframe["unit_price"] = pd.to_numeric(
                dataframe["unit_price"].astype(str).str.strip().str.replace(",", ".", regex=False),
                errors="coerce",
            )

        before_filter = len(dataframe)

        # ── LIMPIEZA: DESCARTE DE FILAS SIN DATO OBLIGATORIO ──
        dataframe = dataframe.dropna(subset=["date", "product_id", "quantity"])

        logger.info(
            "[TRANSFORM] After filters: %d/%d rows kept (dropped %d)",
            len(dataframe), before_filter, before_filter - len(dataframe),
        )

        # ── LIMPIEZA: DEDUPLICACIÓN POR TRANSACCIÓN ──
        if "transaction_id" in dataframe.columns:
            has_id = dataframe["transaction_id"].notna() & (
                dataframe["transaction_id"].astype(str).str.strip() != ""
            )
            with_id, without_id = dataframe[has_id], dataframe[~has_id]
            before_dedup = len(with_id)
            with_id = with_id.drop_duplicates(subset=["transaction_id", "product_id"], keep="first")
            dropped = before_dedup - len(with_id)
            if dropped:
                logger.warning(
                    "[TRANSFORM] Dropped %d duplicate (transaction_id, product_id) row(s).",
                    dropped,
                )
            dataframe = pd.concat([with_id, without_id], ignore_index=True)
            dataframe = dataframe.drop(columns=["transaction_id"])

        # ── AGRUPACIÓN POR FECHA Y PRODUCTO ──
        optional_canonicals = [c for c in get_optional_field_keys() if c in dataframe.columns]
        agg_dict: dict[str, str] = {
            "quantity": "sum",
            **{c: ("mean" if c == "unit_price" else "last") for c in optional_canonicals},
        }
        dataframe = (
            dataframe.groupby(["date", "product_id"], as_index=False)
            .agg(agg_dict)
        )
        dataframe = dataframe.sort_values(["product_id", "date"]).reset_index(drop=True)

        logger.info(
            "[TRANSFORM] Output: %d rows, %d unique products: %s",
            len(dataframe),
            dataframe["product_id"].nunique(),
            list(dataframe["product_id"].unique()),
        )

        return ProcessedDataset(dataframe=dataframe, schema_version="v1")

    @staticmethod
    def _find_column(dataframe: pd.DataFrame, target: str) -> str | None:
        """Busca la columna sin distinguir mayúsculas/minúsculas ni espacios."""
        normalized_map = {
            normalize_column_name(col): col
            for col in dataframe.columns
        }
        return normalized_map.get(normalize_column_name(target))

    @staticmethod
    def _infer_dayfirst(raw_dates: pd.Series) -> bool:
        # ── DETECCIÓN DE FORMATO DD/MM VS MM/DD ──
        pattern = re.compile(r"^\s*(\d{1,4})[/\-.](\d{1,4})[/\-.](\d{1,4})")
        for value in raw_dates.dropna().astype(str):
            match = pattern.match(value)
            if not match:
                continue
            parts = [int(part) for part in match.groups()]
            candidates = [part for part in parts if part <= 31]
            if len(candidates) < 2:
                continue
            first, second = candidates[0], candidates[1]
            if first > 12:
                return True
            if second > 12:
                return False
        return True

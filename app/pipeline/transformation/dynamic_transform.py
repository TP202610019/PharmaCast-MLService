from __future__ import annotations

import logging
import re

import pandas as pd

from app.domain.models.artifacts import ProcessedDataset, RawDataset

logger = logging.getLogger(__name__)

# Maps field_key → canonical column name used by the rest of the pipeline.
# field_keys not listed here are silently ignored (optional enrichment columns).
FIELD_KEY_TO_CANONICAL: dict[str, str] = {
    # sales domain — required
    "date":      "date",
    "product":   "product_id",
    "quantity":  "quantity",
    # sales domain — optional enrichment
    "price":     "price",
    "category":  "category",
    "discount":  "promotion",   # DB field_key → ML canonical name
    "promotion": "promotion",   # direct alias
    "season":    "season",
    "location":  "location",
    "holiday":   "holiday",
    # inventory domain — optional; enables purchase-plan generation when present
    "inv_product": "product_id",
    "stock":       "stock",
    "min_stock":   "min_stock",
}

# Canonical columns that must exist after transformation for the pipeline to proceed.
REQUIRED_CANONICAL = {"date", "product_id", "quantity"}

# field_keys that belong exclusively to the inventory domain.
# Even if their canonical maps to a required column, missing them is not fatal
# because the ML endpoint only receives the sales dataset, not the inventory file.
INVENTORY_ONLY_FIELD_KEYS = {"inv_product", "stock", "min_stock"}

# Inventory canonical columns; their presence enables purchase-plan generation.
INVENTORY_CANONICAL = {"stock", "min_stock"}


class DynamicTransformationStage:
    """
    Applies column mapping dynamically from the request's field mappings dict.

    Accepts a ``dict[str, str]`` where key = field_key and value = actual
    column name in the uploaded dataset.  Column matching is case-insensitive
    and whitespace-normalised to handle common formatting inconsistencies.
    """

    def run(self, raw_dataset: RawDataset, field_map: dict[str, str]) -> ProcessedDataset:
        dataframe = raw_dataset.dataframe.copy()

        logger.info(
            "[TRANSFORM] Input: %d rows, columns: %s",
            len(dataframe),
            list(dataframe.columns),
        )
        logger.info("[TRANSFORM] Field map received: %s", field_map)

        rename_map: dict[str, str] = {}
        for field_key, source_col in field_map.items():
            canonical = FIELD_KEY_TO_CANONICAL.get(field_key)
            if canonical is None:
                # Optional field with no canonical pipeline name — skip silently.
                continue

            actual = self._find_column(dataframe, source_col)
            if actual is None:
                if canonical in REQUIRED_CANONICAL and field_key not in INVENTORY_ONLY_FIELD_KEYS:
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
                field_key, source_col, canonical, actual,
            )
            if actual != canonical:
                rename_map[actual] = canonical

        # Check all required canonical columns will be present after renaming
        projected = {rename_map.get(c, c) for c in dataframe.columns}
        missing_required = REQUIRED_CANONICAL - projected
        if missing_required:
            raise ValueError(
                f"Missing required canonical columns after mapping: {missing_required}. "
                f"Check that field_keys 'date', 'product', and 'quantity' are included in the mapping."
            )

        dataframe = dataframe.rename(columns=rename_map)

        dataframe["date"] = pd.to_datetime(dataframe["date"], errors="coerce")
        dataframe["quantity"] = pd.to_numeric(dataframe["quantity"], errors="coerce")
        dataframe["product_id"] = dataframe["product_id"].astype(str).str.strip().str.upper()

        before_filter = len(dataframe)
        dataframe = dataframe.dropna(subset=["date", "product_id", "quantity"])
        dataframe = dataframe[dataframe["product_id"] != ""]
        dataframe = dataframe[dataframe["quantity"] >= 0]
        logger.info(
            "[TRANSFORM] After filters: %d/%d rows kept (dropped %d)",
            len(dataframe), before_filter, before_filter - len(dataframe),
        )

        # Preserve optional columns through the groupby using "last" value per day.
        # Without this, price/stock/promotion are silently dropped before feature engineering.
        optional_canonicals = [
            c for c in ["price", "category", "promotion", "season", "location", "holiday"]
            if c in dataframe.columns
        ]
        agg_dict: dict[str, str] = {"quantity": "sum", **{c: "last" for c in optional_canonicals}}
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

        return ProcessedDataset(
            name="processed_dataset",
            dataframe=dataframe,
            schema_version="v1",
        )

    @staticmethod
    def _find_column(dataframe: pd.DataFrame, target: str) -> str | None:
        """Case-insensitive, whitespace-normalized column lookup."""
        normalized_map = {
            DynamicTransformationStage._normalize(col): col
            for col in dataframe.columns
        }
        return normalized_map.get(DynamicTransformationStage._normalize(target))

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"[^a-z0-9_]+", "_", str(value).strip().lower())

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from app.config.settings import Settings
from app.domain.models.artifacts import FeatureDataset, ProcessedDataset

logger = logging.getLogger(__name__)


class FeatureEngineeringStage:
    """Builds time-series features from the processed dataset."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self, processed_dataset: ProcessedDataset) -> FeatureDataset:
        dataframe = processed_dataset.dataframe.copy()

        logger.info(
            "[FEATURE-ENG] Input dataset shape: %d rows × %d cols, products: %s",
            len(dataframe),
            len(dataframe.columns),
            list(dataframe["product_id"].unique()) if "product_id" in dataframe.columns else "?",
        )

        if dataframe.empty:
            raise ValueError(
                "Feature engineering received an empty dataset. "
                "All rows were removed during transformation — check that the column mapping "
                "matches actual column names and that the dataset contains valid dates and non-negative quantities."
            )

        dataframe = dataframe.sort_values(["product_id", "date"]).reset_index(drop=True)
        dataframe = self._resample(dataframe)

        dataframe["year"]          = dataframe["date"].dt.year
        dataframe["month"]         = dataframe["date"].dt.month
        dataframe["day_of_week"]   = dataframe["date"].dt.dayofweek
        dataframe["week_of_year"]  = dataframe["date"].dt.isocalendar().week.astype(int)
        dataframe["day_of_month"]  = dataframe["date"].dt.day
        dataframe["is_weekend"]    = (dataframe["date"].dt.dayofweek >= 5).astype(int)
        # Last 6 days of each month: salary/payment cycles drive demand spikes in pharmacies
        dataframe["is_end_of_month"] = (dataframe["date"].dt.day >= 25).astype(int)

        size_series = dataframe.groupby("product_id")["quantity"].size()
        max_history = int(size_series.max()) if not size_series.empty else 0
        logger.info("[FEATURE-ENG] max_history=%d, products=%d", max_history, len(size_series))
        for lag in self.settings.lags:
            if max_history > lag:
                dataframe[f"lag_{lag}"] = dataframe.groupby("product_id")["quantity"].shift(lag)

        shifted = dataframe.groupby("product_id")["quantity"].shift(1)
        for window in self.settings.rolling_windows:
            if max_history > 1:
                dataframe[f"rolling_mean_{window}"] = shifted.groupby(dataframe["product_id"]).transform(
                    lambda values: values.rolling(window, min_periods=1).mean()
                )

        previous = dataframe.groupby("product_id")["quantity"].shift(1)
        dataframe["growth"] = (dataframe["quantity"] - previous) / previous.replace(0, pd.NA)

        # demand_tier: classify each product by its average daily demand.
        # XGBoost uses this to learn different demand patterns per tier without
        # needing an external category column — it's derived from the data itself.
        product_avg = (
            dataframe.groupby("product_id")["quantity"]
            .mean()
            .rename("_avg_qty")
        )
        dataframe = dataframe.join(product_avg, on="product_id")
        dataframe["demand_tier"] = pd.cut(
            dataframe["_avg_qty"],
            bins=[-np.inf, 0.08, 0.42, 1.67, np.inf],
            labels=["baja", "media", "alta", "critica"],
        ).astype(str)
        dataframe = dataframe.drop(columns=["_avg_qty"])

        # price feature engineering: auto-detect if price varies meaningfully.
        # If the median coefficient of variation across products > 2%, price is
        # considered dynamic and gets change/vs-avg features. Otherwise only
        # revenue_importance is kept and raw price is dropped (it adds no signal).
        if "price" in dataframe.columns:
            dataframe["price"] = pd.to_numeric(dataframe["price"], errors="coerce").fillna(0)
            dataframe["revenue_importance"] = (
                dataframe["price"] * dataframe.groupby("product_id")["quantity"].transform("mean")
            ).clip(lower=0)

            price_cv = (
                dataframe.groupby("product_id")["price"]
                .agg(lambda s: s.std() / s.mean() if s.mean() > 0 else 0)
            )
            price_is_dynamic = float(price_cv.median()) > 0.02
            logger.info("[FEATURE-ENG] price_cv median=%.4f → %s",
                        float(price_cv.median()), "dynamic" if price_is_dynamic else "fixed (dropping raw price)")

            if price_is_dynamic:
                price_avg = dataframe.groupby("product_id")["price"].transform("mean")
                dataframe["price_change"] = (
                    dataframe.groupby("product_id")["price"].pct_change()
                    .replace([np.inf, -np.inf], 0).fillna(0)
                )
                dataframe["price_vs_avg"] = (
                    (dataframe["price"] - price_avg) / price_avg.replace(0, pd.NA).fillna(1)
                ).replace([np.inf, -np.inf], 0).fillna(0)
            else:
                dataframe = dataframe.drop(columns=["price"])

        # stock feature engineering:
        # Guard 1 — snapshot detection: if stock barely changes per product it's a
        #   current-inventory snapshot, not a time-series → useless as a daily feature.
        # Guard 2 — stockout flag: days with stock=0 likely have suppressed demand
        #   (no stock → no sales), not true zero demand. Mark them separately so the
        #   model learns stockout ≠ low demand.
        # Guard 3 — active check: >5% of rows must have stock > 0.
        if "stock" in dataframe.columns:
            dataframe["stock"] = pd.to_numeric(dataframe["stock"], errors="coerce").fillna(0).clip(lower=0)

            stock_cv_per_product = dataframe.groupby("product_id")["stock"].agg(
                lambda s: s.std() / s.mean() if s.mean() > 0 else 0
            )
            stock_is_dynamic = float(stock_cv_per_product.median()) > 0.05
            stock_active_pct = (dataframe["stock"] > 0).mean()

            logger.info(
                "[FEATURE-ENG] stock cv_median=%.4f active=%.1f%% → %s",
                float(stock_cv_per_product.median()), stock_active_pct * 100,
                "dynamic+useful" if (stock_is_dynamic and stock_active_pct > 0.05)
                else "static snapshot — dropping" if not stock_is_dynamic
                else "mostly zero — dropping"
            )

            if stock_is_dynamic and stock_active_pct > 0.05:
                # Stockout flag: separates "no stock" from "no demand"
                dataframe["is_stockout"] = (dataframe["stock"] == 0).astype(int)
                rolling_cols = [c for c in dataframe.columns if c.startswith("rolling_mean_")]
                rolling_ref = dataframe[rolling_cols[0]] if rolling_cols else dataframe["quantity"]
                dataframe["stock_coverage_days"] = (
                    dataframe["stock"] / rolling_ref.replace(0, pd.NA).fillna(1)
                ).clip(lower=0, upper=365).fillna(0)
            else:
                dataframe = dataframe.drop(columns=["stock"])

        # promotion feature engineering:
        # Guard 1 — rate check: if <2% of rows have promotion=1, there's not enough
        #   signal for the model to learn from → drop.
        # Guard 2 — collinearity check: if promotion correlates >0.7 with is_weekend
        #   or is_end_of_month, the effect is already captured → drop to avoid redundancy.
        if "promotion" in dataframe.columns:
            dataframe["promotion"] = (pd.to_numeric(dataframe["promotion"], errors="coerce").fillna(0) > 0).astype(int)
            promo_rate = dataframe["promotion"].mean()

            promo_corr_weekend  = abs(dataframe["promotion"].corr(dataframe["is_weekend"]))      if "is_weekend"      in dataframe.columns else 0.0
            promo_corr_monthend = abs(dataframe["promotion"].corr(dataframe["is_end_of_month"])) if "is_end_of_month" in dataframe.columns else 0.0
            is_redundant = max(promo_corr_weekend, promo_corr_monthend) > 0.7

            logger.info(
                "[FEATURE-ENG] promotion rate=%.1f%% corr_weekend=%.2f corr_monthend=%.2f → %s",
                promo_rate * 100, promo_corr_weekend, promo_corr_monthend,
                "dropping (redundant with calendar)" if is_redundant
                else "useful" if promo_rate >= 0.02
                else "dropping (never promoted)"
            )

            if promo_rate >= 0.02 and not is_redundant:
                def days_since_promo(series: pd.Series) -> pd.Series:
                    result, count = [], 30
                    for val in series:
                        count = 0 if val == 1 else min(count + 1, 30)
                        result.append(count)
                    return pd.Series(result, index=series.index)
                dataframe["days_since_last_promo"] = (
                    dataframe.groupby("product_id")["promotion"].transform(days_since_promo)
                )
            else:
                dataframe = dataframe.drop(columns=["promotion"])

        # Guard: replace inf/nan and clip extreme numeric values before XGBoost
        dataframe = dataframe.replace([np.inf, -np.inf], np.nan)
        numeric_cols = dataframe.select_dtypes(include="number").columns.difference(["quantity"])
        dataframe[numeric_cols] = dataframe[numeric_cols].clip(-1e6, 1e6)

        feature_columns = self._feature_columns(dataframe)
        return FeatureDataset(
            name="feature_dataset",
            dataframe=dataframe,
            feature_columns=feature_columns,
            target_column="quantity",
        )

    def _resample(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        optional_columns = [column for column in self.settings.optional_aliases if column in dataframe.columns]
        parts: list[pd.DataFrame] = []

        for product_id, group in dataframe.groupby("product_id"):
            series = group.set_index("date")[["quantity"] + optional_columns]
            aggregations: dict[str, str] = {
                "quantity": "sum",
                **{column: "last" for column in optional_columns},
            }
            series = series.resample(self.settings.frequency).agg(aggregations)
            series["quantity"] = series["quantity"].fillna(0)
            series["product_id"] = product_id
            parts.append(series.reset_index())

        if not parts:
            return dataframe

        return pd.concat(parts, ignore_index=True).sort_values(["product_id", "date"]).reset_index(drop=True)

    def _feature_columns(self, dataframe: pd.DataFrame) -> list[str]:
        base = ["product_id", "year", "month", "day_of_week", "week_of_year",
                "day_of_month", "is_weekend", "is_end_of_month", "demand_tier"]
        lag_columns = [f"lag_{value}" for value in self.settings.lags if f"lag_{value}" in dataframe.columns]
        rolling_columns = [
            f"rolling_mean_{value}"
            for value in self.settings.rolling_windows
            if f"rolling_mean_{value}" in dataframe.columns
        ]
        growth        = ["growth"] if "growth" in dataframe.columns else []
        revenue       = ["revenue_importance"] if "revenue_importance" in dataframe.columns else []
        price_derived = [c for c in ["price_change", "price_vs_avg"] if c in dataframe.columns]
        stock_derived = [c for c in ["stock", "is_stockout", "stock_coverage_days"] if c in dataframe.columns]
        promo_derived = [c for c in ["promotion", "days_since_last_promo"] if c in dataframe.columns]
        optional      = [c for c in self.settings.optional_aliases
                         if c in dataframe.columns and c not in {"price", "stock", "promotion"}]
        return base + lag_columns + rolling_columns + growth + revenue + price_derived + stock_derived + promo_derived + optional

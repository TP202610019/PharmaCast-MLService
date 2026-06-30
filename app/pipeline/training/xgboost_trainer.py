from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from xgboost import XGBRegressor

from app.config.settings import Settings
from app.domain.models.artifacts import FeatureDataset, TrainedModel
from app.shared.utils import utc_now

logger = logging.getLogger(__name__)


@dataclass
class TrainingComponents:
    feature_columns: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]


class XGBoostTrainingStage:
    """Fits the XGBoost regression pipeline."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self, feature_dataset: FeatureDataset) -> TrainedModel:
        return self.train_on_frame(feature_dataset.dataframe.copy(), feature_dataset)

    def train_on_frame(self, dataframe: pd.DataFrame, feature_dataset: FeatureDataset) -> TrainedModel:
        dataframe = dataframe.dropna(subset=[feature_dataset.target_column]).copy()
        components = self._resolve_components(dataframe, feature_dataset.feature_columns)
        feature_frame = dataframe[components.feature_columns].copy()
        target = dataframe[feature_dataset.target_column]

        logger.info(
            "[XGBOOST-TRAIN] Training on %d rows, %d features — categorical: %s, numeric: %s",
            len(dataframe),
            len(components.feature_columns),
            components.categorical_columns,
            components.numeric_columns,
        )

        # Replace ALL non-finite values (inf, -inf, nan) with 0 for numeric columns.
        # XGBoost cannot handle inf and raises C++ assertion when values overflow float32.
        for col in components.numeric_columns:
            arr = feature_frame[col].to_numpy(dtype=np.float64)
            feature_frame[col] = np.where(np.isfinite(arr), arr, 0.0)

        pipeline = self._build_pipeline(
            categorical_columns=components.categorical_columns,
            numeric_columns=components.numeric_columns,
        )
        pipeline.fit(feature_frame, target)
        logger.info("[XGBOOST-TRAIN] Training complete.")

        return TrainedModel(
            name="xgboost_trained_model",
            model_pipeline=pipeline,
            feature_columns=components.feature_columns,
            training_rows=int(len(dataframe)),
            trained_at=utc_now(),
        )

    @staticmethod
    def _resolve_components(dataframe: pd.DataFrame, feature_columns: list[str]) -> TrainingComponents:
        categorical_columns = [
            column
            for column in feature_columns
            if pd.api.types.is_object_dtype(dataframe[column]) or pd.api.types.is_string_dtype(dataframe[column])
        ]
        numeric_columns = [column for column in feature_columns if column not in categorical_columns]
        return TrainingComponents(
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
            numeric_columns=numeric_columns,
        )

    def _build_pipeline(self, categorical_columns: list[str], numeric_columns: list[str]) -> Pipeline:
        transformers: list[tuple[str, object, list[str]]] = []
        if categorical_columns:
            transformers.append(
                (
                    "categorical",
                    Pipeline(
                        steps=[
                            ("clean", FunctionTransformer(self._clean_categories, validate=False)),
                            ("one_hot", self._make_ohe()),
                        ]
                    ),
                    categorical_columns,
                )
            )
        if numeric_columns:
            transformers.append(("numeric", "passthrough", numeric_columns))

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        model = XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
        return Pipeline([("preprocess", preprocessor), ("model", model)])

    @staticmethod
    def _clean_categories(values) -> pd.DataFrame:
        frame = pd.DataFrame(values).astype("object")
        return frame.where(frame.notna(), "missing").astype(str)

    @staticmethod
    def _make_ohe() -> OneHotEncoder:
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)

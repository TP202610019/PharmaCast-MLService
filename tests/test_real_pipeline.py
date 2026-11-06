"""Synthetic tests only: no database, HTTP datasets or real Blob service."""
from __future__ import annotations

import asyncio
import inspect
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pandas as pd

from app.application.execution_orchestrator import ExecutionOrchestrator
from app.config.settings import Settings, get_settings
from app.domain.models.artifacts import ProcessedDataset, RawDataset, TrainedModel
from app.domain.models.execution_request import PredictionExecutionRequest
from app.infrastructure.historical.historical_data_manager import HistoricalDataManager
from app.infrastructure.persistence.model_artifact_store import ModelArtifactStore
from app.pipeline.feature_engineering.builder import FeatureEngineeringStage
from app.pipeline.forecasting.forecaster import ForecastingStage
from app.pipeline.metrics.calculator import MetricsCalculationStage
from app.pipeline.training.xgboost_trainer import XGBoostTrainingStage
from app.shared.utils import utc_now


def settings(root):
    return Settings(base_dir=Path(root), azure_storage_connection_string=None,
                    use_local_storage=True, local_storage_root=Path(root))


def dataset(days=75, products=("A", "B"), quantity=10.0):
    dates = pd.date_range("2025-01-01", periods=days)
    return pd.DataFrame([{"date": day, "product_id": pid, "quantity": quantity}
                         for pid in products for day in dates])


class LagModel:
    def __init__(self, increment=0):
        self.increment = increment
        self.inputs = []

    def predict(self, frame):
        self.inputs.append(frame.copy())
        return frame["lag_1"].fillna(0).to_numpy(dtype=float) + self.increment


class FakeTraining:
    def __init__(self, increment=0):
        self.frames = []
        self.increment = increment

    def run(self, features):
        self.frames.append(features.dataframe.copy())
        return TrainedModel("synthetic", LagModel(self.increment), features.feature_columns,
                            len(features.dataframe), utc_now())


class RealPipelineTests(unittest.TestCase):
    def test_prepare_history_groups_sales_and_does_not_train(self):
        from app.api.execution_routes import prepare_history
        frame = pd.DataFrame({"date": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-03"]),
                              "product_id": ["A", "A", "A"], "quantity": [2., 3., 4.]})
        runner = Mock()
        runner.dynamic_transform.run.return_value = ProcessedDataset("test", frame, "v1")
        runner.feature_engineering = FeatureEngineeringStage(self.settings)
        request = PredictionExecutionRequest(predictionExecutionId="exec", datasetId="dataset",
            datasetPath="unused", forecastPeriod=30)
        with patch("app.api.execution_routes.ExecutionOrchestrator", return_value=runner):
            result = asyncio.run(prepare_history(request, self.settings))
        self.assertEqual([p["quantity"] for p in result["historicalPoints"]], [5., 0., 4.])
        runner.training_stage.run.assert_not_called()

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = settings(self.temp.name)

    def features(self, frame):
        return FeatureEngineeringStage(self.settings).run(ProcessedDataset("test", frame, "v1"))

    def test_features_exclude_target_leakage_and_unknown_future_context(self):
        original = dataset()
        original["price"] = np.arange(len(original)) + 1
        original["promotion"] = 1
        mutated = original.copy()
        mutated.loc[mutated["date"] >= "2025-02-01", "quantity"] = 10000
        first, second = self.features(original), self.features(mutated)
        forbidden = {"growth", "revenue_importance", "demand_tier", "price", "promotion"}
        self.assertFalse(forbidden.intersection(first.feature_columns))
        before = first.dataframe["date"] <= "2025-02-01"
        pd.testing.assert_frame_equal(
            first.dataframe.loc[before, first.feature_columns].reset_index(drop=True),
            second.dataframe.loc[before, second.feature_columns].reset_index(drop=True),
        )
        model = FakeTraining().run(first)
        predicted = ForecastingStage(self.settings).run(model, first, 1)
        next_date = predicted.dataframe.iloc[0]["date"]
        extended = pd.concat([original, pd.DataFrame([
            {"date": next_date, "product_id": pid, "quantity": 999} for pid in ("A", "B")
        ])], ignore_index=True)
        expected = self.features(extended).dataframe
        expected = expected.loc[expected["date"] == next_date, first.feature_columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(model.model_pipeline.inputs[0], expected, check_dtype=False)

    def test_pure_recursive_forecasts_exact_horizons_without_blend_or_cap(self):
        features = self.features(dataset(20))
        for horizon in (30, 60, 90):
            with self.subTest(horizon=horizon):
                model = FakeTraining(increment=1).run(features)
                result = ForecastingStage(self.settings).run(model, features, horizon).dataframe
                self.assertEqual(len(result), 2 * horizon)
                np.testing.assert_allclose(result.loc[result.product_id == "A", "prediction"],
                                           np.arange(11, 11 + horizon))
                self.assertEqual(result["date"].min(), pd.Timestamp("2025-01-21"))
                self.assertEqual(result["date"].max(), pd.Timestamp("2025-01-20") + pd.Timedelta(days=horizon))
                self.assertEqual(model.model_pipeline.inputs[1].iloc[0]["lag_1"], 11)

    def test_exact_holdout_recursive_and_all_product_flat_metrics(self):
        for horizon in (30, 60, 90):
            with self.subTest(horizon=horizon):
                frame = dataset(20 + horizon, ("A", "B", "ZERO"))
                frame.loc[frame.product_id == "ZERO", "quantity"] = 0
                frame.loc[(frame.date > "2025-01-20") & (frame.product_id == "A"), "quantity"] = 12
                frame.loc[(frame.date > "2025-01-20") & (frame.product_id == "B"), "quantity"] = 8
                training = FakeTraining()
                result = MetricsCalculationStage(self.settings, training).run(ProcessedDataset("test", frame, "v1"), horizon)
                self.assertEqual(result.training_end.isoformat(), "2025-01-20")
                self.assertEqual(result.evaluation_start.isoformat(), "2025-01-21")
                self.assertEqual((result.evaluation_end - result.evaluation_start).days + 1, horizon)
                self.assertEqual(training.frames[0]["date"].max(), pd.Timestamp("2025-01-20"))
                for method in ("XGBoost", "NaiveWeekly"):
                    overall = next(m for m in result.metrics if m.method == method and m.scope == "Overall")
                    self.assertEqual(overall.rows, 3 * horizon)
                    self.assertAlmostEqual(overall.mae, 4 / 3)
                    self.assertAlmostEqual(overall.wape, 20)
                    self.assertAlmostEqual(overall.mbe, 0)
                    products = [m for m in result.metrics if m.method == method and m.scope == "Product"]
                    self.assertEqual({m.product_id for m in products}, {"A", "B", "ZERO"})
                    zero = next(m for m in products if m.product_id == "ZERO")
                    self.assertIsNone(zero.wape)
                    self.assertIsNone(zero.mape)

    def test_evaluation_uses_predictions_not_actual_holdout_lags(self):
        frame = dataset(50, ("A",))
        frame.loc[frame.date > "2025-01-20", "quantity"] = 100
        result = MetricsCalculationStage(self.settings, FakeTraining(1)).run(ProcessedDataset("test", frame, "v1"), 30)
        xgb = next(m for m in result.metrics if m.method == "XGBoost" and m.scope == "Overall")
        self.assertAlmostEqual(xgb.mae, 74.5)  # predictions 11..40, never actual 100
        self.assertAlmostEqual(xgb.mbe, -74.5)

    def test_naive_weekly_repeats_last_seven_days_recursively(self):
        frame = dataset(50, ("A",))
        frame["quantity"] = np.tile(np.arange(1, 8), 8)[:50]
        frame.loc[frame.date > "2025-01-20", "quantity"] = 0
        result = MetricsCalculationStage(self.settings, FakeTraining()).run(ProcessedDataset("test", frame, "v1"), 30)
        weekly = next(m for m in result.metrics if m.method == "NaiveWeekly" and m.scope == "Overall")
        expected = np.resize(frame.iloc[13:20].quantity.to_numpy(), 30)
        self.assertAlmostEqual(weekly.mae, expected.mean())
        self.assertAlmostEqual(weekly.mbe, expected.mean())

    def test_insufficient_history_returns_null_metrics_and_dates(self):
        for days in (10, 30, 44):  # holdout 30 needs >=15 training dates
            training = FakeTraining()
            result = MetricsCalculationStage(self.settings, training).run(
                ProcessedDataset("test", dataset(days), "v1"), 30
            )
            self.assertFalse(result.metrics)
            self.assertFalse(training.frames)
            self.assertTrue(result.warnings)
            self.assertIsNone(result.training_start)
            self.assertIsNone(result.training_end)
            self.assertIsNone(result.evaluation_start)
            self.assertIsNone(result.evaluation_end)

    def test_fifteen_training_days_are_sufficient(self):
        result = MetricsCalculationStage(self.settings, FakeTraining()).run(
            ProcessedDataset("test", dataset(45), "v1"), 30)
        self.assertEqual(result.training_end.isoformat(), "2025-01-15")
        self.assertTrue(result.metrics)

    def test_holdout_only_products_are_excluded_from_evaluation(self):
        frame = dataset(50, ("A",))
        frame = pd.concat([frame, pd.DataFrame([{"date": pd.Timestamp("2025-02-15"),
                                                "product_id": "NEW", "quantity": 7}])])
        training = FakeTraining()
        result = MetricsCalculationStage(self.settings, training).run(ProcessedDataset("test", frame, "v1"), 30)
        self.assertEqual(result.new_products_count, 1)
        self.assertEqual(result.new_products_units, 7)
        self.assertNotIn("NEW", set(training.frames[0]["product_id"]))
        for method in ("XGBoost", "NaiveWeekly"):
            self.assertFalse(any(m.method == method and m.product_id == "NEW" for m in result.metrics))
            overall = next(m for m in result.metrics if m.method == method and m.scope == "Overall")
            self.assertEqual(overall.rows, 30)

    def test_snapshot_merge_is_read_only_and_local_blob_history_matches(self):
        local = HistoricalDataManager(None, local_storage_root=Path(self.temp.name))
        old = dataset(2, ("A",))
        correction = old.copy()
        correction["quantity"] = 42
        self.assertEqual(local.snapshot_count("user"), 0)
        local.get_combined_dataset("user", old, "z")
        self.assertEqual(local.snapshot_count("user"), 0)
        local.accumulate("user", old, "z")
        local.accumulate("user", correction, "a")
        files = list((Path(self.temp.name) / "historical-datasets" / "user").glob("*.csv"))
        for path in files:
            timestamp = 100 if path.stem == "historical_z" else 200
            os.utime(path, (timestamp, timestamp))
        blobs = [SimpleNamespace(name=f"user/{path.name}", last_modified=pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC"))
                 for path in files]
        payloads = {f"user/{path.name}": path.read_bytes() for path in files}
        remote = HistoricalDataManager.__new__(HistoricalDataManager)
        remote._local_container = None
        remote._container = Mock()
        remote._container.list_blobs.return_value = blobs
        remote._container.get_blob_client.side_effect = lambda name: SimpleNamespace(
            download_blob=lambda: SimpleNamespace(readall=lambda: payloads[name]))
        pd.testing.assert_frame_equal(local.load_historical("user"), remote.load_historical("user"))
        self.assertEqual(local.load_historical("user").quantity.tolist(), [42, 42])
        merged, retrain = local.get_combined_dataset("user", old, "next")
        self.assertTrue(retrain)
        self.assertEqual(local.snapshot_count("user"), 2)
        self.assertEqual(merged.quantity.tolist(), [10, 10])

    def orchestrator(self):
        orchestrator = ExecutionOrchestrator(self.settings)
        orchestrator.dataset_loader.load = Mock(return_value=RawDataset("test", dataset(50), "synthetic", []))
        orchestrator.training_stage = FakeTraining()
        orchestrator.metrics_stage = MetricsCalculationStage(self.settings, orchestrator.training_stage)
        orchestrator.artifact_store.save = Mock(return_value={"model": "model.joblib", "metadata": "metadata.json"})
        return orchestrator

    def request(self):
        return PredictionExecutionRequest(predictionExecutionId="execution", datasetId="dataset",
                                          userId="user", datasetPath="synthetic.csv", forecastPeriod=30)

    def test_failure_never_publishes_history(self):
        for stage in ("metrics", "forecast", "artifact"):
            with self.subTest(stage=stage):
                orchestrator = self.orchestrator()
                if stage == "metrics":
                    orchestrator.metrics_stage.run = Mock(side_effect=RuntimeError("synthetic failure"))
                elif stage == "forecast":
                    orchestrator.forecasting_stage.run = Mock(side_effect=RuntimeError("synthetic failure"))
                else:
                    orchestrator.artifact_store.save.side_effect = RuntimeError("synthetic failure")
                with self.assertRaises(RuntimeError):
                    asyncio.run(orchestrator.execute(self.request()))
                self.assertEqual(orchestrator.historical_manager.snapshot_count("user"), 0)

    def test_success_response_contract_and_snapshot_after_artifact(self):
        orchestrator = self.orchestrator()
        def save(*args):
            self.assertEqual(orchestrator.historical_manager.snapshot_count("user"), 0)
            return {"model": "model.joblib", "metadata": "metadata.json"}
        orchestrator.artifact_store.save.side_effect = save
        response = asyncio.run(orchestrator.execute(self.request()))
        self.assertEqual(orchestrator.historical_manager.snapshot_count("user"), 1)
        self.assertEqual(response.metrics.method, "XGBoost")
        self.assertEqual(response.baselineMetrics.method, "NaiveWeekly")
        self.assertEqual(response.metrics.scope, "Overall")
        self.assertEqual(response.productMetrics[0].scope, "Product")
        self.assertEqual(len(response.forecast), 60)
        self.assertEqual(set(response.artifactPaths), {"model", "metadata", "historicalSnapshot"})
        manifest = orchestrator.artifact_store.save.call_args.args[3]
        self.assertEqual(manifest["datasetId"], "dataset")
        self.assertEqual(manifest["userId"], "user")
        self.assertEqual(manifest["fieldMappings"], {})
        self.assertFalse(response.isDemo)
        json.loads(response.model_dump_json())  # includes valid nullable metrics/dates

    def test_optional_sales_context_is_preserved_in_snapshot_and_warned(self):
        orchestrator = self.orchestrator()
        raw = dataset(50)
        raw["price"] = 3.25
        orchestrator.dataset_loader.load.return_value.dataframe = raw
        response = asyncio.run(orchestrator.execute(self.request()))
        self.assertTrue(any("price" in warning and "excluded" in warning for warning in response.warnings))
        history = orchestrator.historical_manager.load_historical("user")
        self.assertIn("price", history)
        self.assertNotIn("price", orchestrator.artifact_store.save.call_args.args[2].feature_columns)

    def test_real_xgboost_artifact_round_trip_and_blob_parity(self):
        features = self.features(dataset(20))
        trainer = XGBoostTrainingStage(self.settings)
        model = trainer.run(features)
        paths = ModelArtifactStore(self.settings).save("user", "execution", model, {"forecastPolicy": "pure_xgboost"})
        restored = joblib.load(paths["model"])
        expected = ForecastingStage(self.settings).run(model, features, 3).dataframe
        restored_model = TrainedModel("restored", restored, model.feature_columns, model.training_rows, model.trained_at)
        actual = ForecastingStage(self.settings).run(restored_model, features, 3).dataframe
        pd.testing.assert_frame_equal(actual, expected)
        metadata = json.loads(Path(paths["metadata"]).read_text())
        self.assertEqual(metadata["featureColumns"], model.feature_columns)
        self.assertEqual(metadata["forecastPolicy"], "pure_xgboost")
        remote_settings = Settings(base_dir=Path(self.temp.name), use_local_storage=False,
                                   azure_storage_connection_string="synthetic", local_storage_root=Path(self.temp.name))
        client = Mock()
        with patch("app.infrastructure.persistence.model_artifact_store.BlobServiceClient") as factory:
            factory.from_connection_string.return_value = client
            remote_paths = ModelArtifactStore(remote_settings).save("user", "execution", model, {"forecastPolicy": "pure_xgboost"})
        uploads = {call.kwargs["name"]: call.kwargs["data"]
                   for call in client.get_container_client.return_value.upload_blob.call_args_list}
        self.assertEqual(json.loads(uploads["user/execution/metadata.json"]), metadata)
        np.testing.assert_allclose(joblib.load(io.BytesIO(uploads["user/execution/model.joblib"])).predict(
            features.dataframe[model.feature_columns]), restored.predict(features.dataframe[model.feature_columns]))
        self.assertEqual(remote_paths["model"], "model-artifacts/user/execution/model.joblib")

    def test_settings_route_and_startup_need_no_database(self):
        from app.api.execution_routes import execute_prediction
        from app.main import startup_validation
        self.assertNotIn("db", inspect.signature(execute_prediction).parameters)
        with patch.dict(os.environ, {"LOCAL_STORAGE_ENABLED": "true", "HISTORICAL_DATASETS_CONTAINER": "test-history"}, clear=True):
            config = get_settings()
            self.assertIsNone(config.database_url)
            self.assertEqual(config.historical_datasets_container, "test-history")
        with patch("app.main.get_settings", return_value=self.settings):
            asyncio.run(startup_validation())

    def test_forecast_period_defaults_to_thirty_and_only_accepts_supported_horizons(self):
        payload = {"predictionExecutionId": "execution", "datasetId": "dataset", "datasetPath": "synthetic.csv"}
        self.assertEqual(PredictionExecutionRequest(**payload).forecastPeriod, 30)
        for horizon in (30, 60, 90):
            self.assertEqual(PredictionExecutionRequest(**payload, forecastPeriod=horizon).forecastPeriod, horizon)
        for invalid in (0, 7, 31, 91):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                PredictionExecutionRequest(**payload, forecastPeriod=invalid)

    def test_historical_average_includes_zero_days_and_metric_order(self):
        orchestrator = self.orchestrator()
        frame = dataset(50)
        frame = frame[(frame.product_id == "B") | frame.date.isin([frame.date.min(), frame.date.max()])].copy()
        frame.loc[(frame.product_id == "A") & (frame.date == frame.date.max()), "quantity"] = 20
        orchestrator.dataset_loader.load.return_value.dataframe = frame
        response = asyncio.run(orchestrator.execute(self.request()))
        series = [p for p in response.historicalPoints if p.productId == "A"]
        self.assertEqual(len(series), 50)
        self.assertEqual(sum(p.quantity for p in series), 30)
        self.assertEqual(series[1].quantity, 0)
        for metrics in (response.productMetrics, response.baselineProductMetrics):
            self.assertEqual(next(m for m in metrics if m.productId == "A").historicalAvg, 0.6)
        keys = ("wape", "mae", "rmse", "mbe", "mape")
        for record in (response.metrics, response.baselineMetrics, response.productMetrics[0], response.baselineProductMetrics[0]):
            self.assertEqual([key for key in record.model_dump() if key in keys], list(keys))


    def test_cleaning_report_counts_invalid_rows_and_groups_daily_sales(self):
        from app.pipeline.transformation.dynamic_transform import DynamicTransformationStage
        frame = pd.DataFrame({
            "date": ["2025-01-01 10:00", "2025-01-01 10:00", "bad", "2025-01-01 10:00", "2025-01-01 10:00"],
            "product_id": ["a", "A", "A", None, "A"],
            "quantity": [2, 3, 5, 1, -1]})
        result = DynamicTransformationStage().run(RawDataset("test", frame, "unused", []), {})
        self.assertEqual(result.cleaning_report["discardedRows"], 3)
        self.assertEqual(result.cleaning_report["invalidDate"], 1)
        self.assertEqual(result.cleaning_report["invalidProduct"], 1)
        self.assertEqual(result.cleaning_report["negativeQuantity"], 1)
        self.assertEqual(result.dataframe.quantity.tolist(), [5])
        self.assertEqual(result.dataframe.date.iloc[0].hour, 0)

    def test_mixed_iso_and_spanish_dates_keep_june_in_june(self):
        from app.pipeline.transformation.dynamic_transform import DynamicTransformationStage
        frame = pd.DataFrame({"date": ["2025-01-01 00:00:00", "12/06/2025", "30/06/2025"],
                              "product_id": ["A"] * 3, "quantity": [1, 2, 3]})
        result = DynamicTransformationStage().run(RawDataset("test", frame, "unused", []), {})
        self.assertEqual(result.cleaning_report["discardedRows"], 0)
        self.assertEqual(result.dataframe.date.dt.strftime("%Y-%m-%d").tolist(),
                         ["2025-01-01", "2025-06-12", "2025-06-30"])

if __name__ == "__main__":
    unittest.main()

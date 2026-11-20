import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from app.config.settings import Settings
from app.domain.models.artifacts import ProcessedDataset
from app.pipeline.feature_engineering.builder import FeatureEngineeringStage
from app.pipeline.forecasting.weekly import WeeklyPipeline


class TemporalAcceptanceTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.stage = FeatureEngineeringStage(Settings(
            base_dir=root, azure_storage_connection_string=None,
            use_local_storage=True, local_storage_root=root,
        ))

    def features(self, data):
        return self.stage.run(ProcessedDataset("acceptance", data, "v1")).dataframe

    def test_cp183_lags_are_past_values_from_the_same_product(self):
        dates = pd.date_range("2025-12-29", periods=18)
        sales = pd.DataFrame([
            {"date": date, "product_id": product, "quantity": offset + 10 * (i + 1)}
            for product, offset in (("A", 0), ("B", 1000))
            for i, date in enumerate(dates)
        ]).iloc[::-1]
        cutoff = dates[15]
        before = self.features(sales)
        updated = sales.copy()
        updated.loc[updated.date >= cutoff, "quantity"] = 9999
        after = self.features(updated)
        for product, offset in (("A", 0), ("B", 1000)):
            current = before.loc[(before.product_id == product) & (before.date == cutoff)].iloc[0]
            revised = after.loc[(after.product_id == product) & (after.date == cutoff)].iloc[0]
            for lag in self.stage.settings.lags:
                expected = offset + 10 * (16 - lag)
                self.assertEqual(current[f"lag_{lag}"], expected)
                self.assertEqual(revised[f"lag_{lag}"], expected)
            weekly = WeeklyPipeline.features(product, cutoff, [offset + 10, offset + 20, offset + 30])
            self.assertEqual(weekly["lag_1"], offset + 30)
            self.assertEqual(weekly["lag_3"], offset + 10)
            self.assertTrue(pd.isna(weekly["lag_4"]))

    def test_cp184_calendar_across_month_and_iso_week_boundaries(self):
        dates = pd.date_range("2025-12-28", "2026-01-05")
        data = pd.DataFrame({"date": dates, "product_id": "A", "quantity": 1})
        result = self.features(data).set_index("date")
        for date, month, week, weekday, day in [
            ("2025-12-28", 12, 52, 6, 28),
            ("2025-12-29", 12, 1, 0, 29),
            ("2026-01-01", 1, 1, 3, 1),
            ("2026-01-05", 1, 2, 0, 5),
        ]:
            with self.subTest(date=date):
                row = result.loc[date]
                self.assertEqual(tuple(row[["month", "week_of_year", "day_of_week", "day_of_month"]]),
                                 (month, week, weekday, day))
                weekly = WeeklyPipeline.features("A", pd.Timestamp(date), [])
                self.assertEqual((weekly["month"], weekly["week_of_year"]), (month, week))

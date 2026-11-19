import unittest
import pandas as pd
import numpy as np
from app.config.settings import get_settings
from app.domain.models.artifacts import ProcessedDataset
from app.pipeline.forecasting.weekly import WeeklyPipeline


class WeeklyPipelineTests(unittest.TestCase):
    def setUp(self):
        self.pipeline=WeeklyPipeline(get_settings())
        dates=pd.date_range('2025-01-01',periods=120)
        self.data=pd.DataFrame([dict(product_id=pid,date=d,quantity=float((i%7==0)*amount))
            for pid,amount in [('a',7),('b',2)] for i,d in enumerate(dates)])

    def test_blocks_cover_horizon_without_overlap(self):
        for horizon in (30,60,90):
            result,_=self.pipeline.forecast(self.data,horizon)
            for _,group in result.dataframe.groupby('product_id'):
                lengths=(group.period_end-group.date).dt.days+1
                self.assertEqual(lengths.sum(),horizon)
                self.assertEqual(len(group),(horizon+6)//7)
                self.assertEqual(group.date.min(),self.data.date.max()+pd.Timedelta(days=1))
                self.assertTrue(np.isfinite(group.prediction).all())
                self.assertTrue(group.prediction.ge(0).all())
                self.assertEqual(lengths.iloc[-1],horizon%7 or 7)

    def test_evaluation_weekly_metrics_and_new_products(self):
        end=self.data.date.max()
        data=pd.concat([self.data,pd.DataFrame([dict(product_id='new',date=end,quantity=10.)])],ignore_index=True)
        ev=self.pipeline.evaluate(ProcessedDataset('test',data,'v1'),30)
        self.assertEqual(ev.new_products_count,1)
        self.assertEqual(ev.new_products_units,10)
        self.assertEqual(len(ev.metrics),6)
        for m in ev.metrics:
            self.assertEqual(m.rows,10 if m.scope=='Overall' else 5)
            self.assertGreaterEqual(m.totalErrorPercentage,0)
            self.assertLessEqual(m.totalErrorPercentage,100)
        baseline=next(m for m in ev.metrics if m.method=='NaiveWeekly' and m.scope=='Product' and m.product_id=='a')
        self.assertAlmostEqual(baseline.predictedTotal,30.)

    def test_uncertainty_calibration_and_missing_bounds(self):
        data = pd.concat([self.data.assign(product_id=f'p{i}')
                          for i in range(11)], ignore_index=True)
        evaluation = self.pipeline.evaluate(ProcessedDataset('test', data, 'v1'), 30)
        self.assertEqual(set(evaluation.weekly_error_quantiles), {1, 2, 3, 4, 5})
        forecast, _ = self.pipeline.forecast(data, 30)
        self.pipeline.add_uncertainty(forecast, data, evaluation.weekly_error_quantiles)
        rows = forecast.dataframe
        self.assertTrue(rows.lower_bound.between(0, rows.prediction).all())
        self.assertTrue(rows.upper_bound.ge(rows.prediction).all())
        self.pipeline.add_uncertainty(forecast, data, {})
        self.assertTrue(rows[['lower_bound', 'upper_bound']].isna().all().all())

if __name__=='__main__':unittest.main()

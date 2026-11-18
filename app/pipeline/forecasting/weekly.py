"""Weekly XGBoost forecasts, with the same temporal protocol in evaluation and serving."""
import math
import numpy as np
import pandas as pd
from app.domain.models.artifacts import FeatureDataset, ForecastResult
from app.pipeline.training.xgboost_trainer import XGBoostTrainingStage
from app.pipeline.metrics.calculator import EvaluationResult, MetricsCalculationStage


class WeeklyTrainer(XGBoostTrainingStage):
    def _build_pipeline(self, categorical_columns, numeric_columns):
        pipeline = super()._build_pipeline(categorical_columns, numeric_columns)
        pipeline.named_steps['model'].set_params(objective='reg:tweedie', max_depth=8, min_child_weight=3)
        return pipeline


class WeeklyPipeline:
    def __init__(self, settings):
        self.settings = settings

    @staticmethod
    def features(pid, date, history):
        row = dict(product_id=pid, month=date.month, week_of_year=int(date.isocalendar().week))
        for lag in range(1, 9):
            row[f'lag_{lag}'] = history[-lag] if len(history) >= lag else np.nan
        for window in (2, 4, 8, 13):
            row[f'rolling_mean_{window}'] = float(np.mean(history[-window:])) if history else np.nan
        return row

    def forecast(self, data, horizon, cutoff=None):
        cutoff = pd.Timestamp(cutoff if cutoff is not None else data.date.max())
        n = ((cutoff-data.date.min()).days+1)//7
        if n < 4:
            raise ValueError('El entrenamiento semanal requiere al menos cuatro semanas completas.')
        earliest = cutoff-pd.Timedelta(days=n*7-1)
        source = data.loc[data.date.between(earliest, cutoff)].copy()
        source['block'] = (cutoff-source.date).dt.days//7
        grouped = source.groupby(['product_id', 'block']).quantity.sum()
        frames, histories = [], {}
        for pid in sorted(source.product_id.unique()):
            first = int(source.loc[source.product_id.eq(pid), 'block'].max())
            values = [float(grouped.get((pid, b), 0)) for b in range(first, -1, -1)]
            histories[pid] = values
            for i, value in enumerate(values):
                row = self.features(pid, cutoff-pd.Timedelta(days=(len(values)-1-i)*7), values[:i])
                frames.append(dict(row, quantity=value))
        frame = pd.DataFrame(frames)
        columns = [c for c in frame if c != 'quantity']
        model = WeeklyTrainer(self.settings).run(FeatureDataset('weekly', frame, columns, 'quantity'))
        # Products only observed in the discarded initial partial week remain in output/evaluation.
        ids = sorted(data.product_id.unique())
        rows = []
        for step in range(math.ceil(horizon/7)):
            end = cutoff+pd.Timedelta(days=(step+1)*7)
            known = list(histories)
            x = pd.DataFrame([self.features(pid, end, histories[pid]) for pid in known])[columns]
            values = model.model_pipeline.predict(x).clip(min=0)
            predicted = dict(zip(known, values))
            days = min(7, horizon-step*7)
            for pid in ids:
                value = float(predicted.get(pid, 0))
                if not np.isfinite(value):
                    raise ValueError('El modelo semanal devolvió una cantidad no finita.')
                rows.append(dict(product_id=pid, date=cutoff+pd.Timedelta(days=step*7+1),
                    period_end=cutoff+pd.Timedelta(days=step*7+days), forecast_step=step+1,
                    prediction=value*days/7))
                if pid in histories:
                    histories[pid].append(value)
        return ForecastResult('weekly_forecast', pd.DataFrame(rows), horizon), model

    def evaluate(self, processed, horizon):
        data = processed.dataframe
        end = data.date.max()
        start = end-pd.Timedelta(days=horizon-1)
        cutoff = start-pd.Timedelta(days=1)
        train = data.loc[data.date<start].copy()
        result = EvaluationResult()
        if train.empty or (cutoff-train.date.min()).days+1 < 28:
            result.warnings.append('Evaluación semanal no disponible: se requieren cuatro semanas completas antes del periodo de evaluación.')
            return result
        ids = sorted(train.product_id.unique())
        new = data.loc[data.date.between(start,end)&~data.product_id.isin(ids)]
        result.new_products_count = int(new.product_id.nunique())
        result.new_products_units = float(new.quantity.sum())
        forecast, _ = self.forecast(train,horizon,cutoff)
        grid = pd.MultiIndex.from_product([ids,range(1,math.ceil(horizon/7)+1)],names=['product_id','forecast_step'])
        predicted = forecast.dataframe.set_index(['product_id','forecast_step']).prediction.reindex(grid).to_numpy()
        actual, naive = [], []
        holdout=data.loc[data.date.between(start,end)].copy()
        holdout['forecast_step']=((holdout.date-start).dt.days//7)+1
        grouped_actual=holdout.groupby(['product_id','forecast_step']).quantity.sum()
        last = train.loc[train.date>cutoff-pd.Timedelta(days=7)].groupby('product_id').quantity.sum()
        for pid in ids:
            for offset in range(0,horizon,7):
                days=min(7,horizon-offset)
                actual.append(float(grouped_actual.get((pid,offset//7+1),0)))
                naive.append(float(last.get(pid,0))*days/7)
        y=np.asarray(actual)
        for method, values in [('XGBoost',predicted),('NaiveWeekly',np.asarray(naive))]:
            overall=MetricsCalculationStage._build_metric('Overall','ALL',y,values,method)
            products=[]
            for pid in ids:
                mask=grid.get_level_values('product_id')==pid
                products.append(MetricsCalculationStage._build_metric('Product',pid,y[mask],values[mask],method))
            overall.absoluteTotalError=sum(m.absoluteTotalError for m in products)
            overall.totalErrorPercentage=MetricsCalculationStage._volume_error_percentage(overall.absoluteTotalError,sum(max(m.actualTotal,m.predictedTotal) for m in products))
            result.metrics.extend(products+[overall])
        result.training_start=train.date.min().date()
        result.training_end=cutoff.date()
        result.evaluation_start=start.date()
        result.evaluation_end=end.date()
        return result

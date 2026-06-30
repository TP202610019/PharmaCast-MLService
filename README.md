# XGBoost Demand Forecasting for Pharmacy Medication Sales

This project trains an experimental demand forecasting model for medication
sales in small pharmacies (MYPE). It supports thesis testing with small CSV
datasets and can scale to larger historical datasets later.

## What the pipeline does

- Loads one CSV file or automatically merges multiple CSV files from a folder.
- Detects date columns named `date`, `fecha` or `datum`.
- Normalizes wide datasets (`date` plus product/category columns) into long
  format (`date`, `product`, `sales`).
- Uses optional context columns only when they exist, such as `stock_actual`,
  `precio`, `promociones`, `tiempo_reposicion`, `ubicacion`, `feriados`,
  `temperatura` or `proveedor`.
- Creates calendar, lag and rolling-average features.
- Uses a temporal train/test split, not a random split.
- Trains and tunes an `XGBRegressor` because medication demand is a numeric
  quantity.
- Reports MAE, RMSE, MAPE and WMAPE.
- Saves every training execution under an independent run folder.
- Predicts future demand for the next 7 periods by default.
- Detects shortage risk and suggests purchase quantities when `stock_actual`
  exists.

## Install

```powershell
pip install -r requirements.txt
```

## Run with the current dataset

```powershell
python src\entrenamiento.py --input dataset\processed --horizon 7 --forecast-freq D
```

Each execution is stored in:

```text
outputs/runs/<run_id>/
```

The most recent run path is also written to:

```text
outputs/latest_run.txt
```

For weekly aggregate data, use:

```powershell
python src\entrenamiento.py --input dataset\processed --horizon 7 --forecast-freq W
```

To infer the historical frequency automatically:

```powershell
python src\entrenamiento.py --input dataset\processed --forecast-freq auto
```

## Useful thesis testing options

Train only selected categories:

```powershell
python src\entrenamiento.py --products M01AB,N02BE,R06
```

Limit the training range:

```powershell
python src\entrenamiento.py --train-start 2014-01-01 --train-end 2018-12-31
```

Set an explicit prediction window:

```powershell
python src\entrenamiento.py --prediction-start 2019-01-01 --prediction-end 2019-01-07
```

Use a readable run id:

```powershell
python src\entrenamiento.py --run-id prueba_tesis_01
```

Disable automatic tuning:

```powershell
python src\entrenamiento.py --no-tuning
```

## Outputs

- `outputs/runs/<run_id>/metrics/metrics.csv`
- `outputs/runs/<run_id>/metrics/baseline_metrics.csv`
- `outputs/runs/<run_id>/metrics/tuning_results.csv`
- `outputs/runs/<run_id>/metrics/run_summary.json`
- `outputs/runs/<run_id>/predictions/historical_test_predictions.csv`
- `outputs/runs/<run_id>/predictions/future_demand_forecast.csv`
- `outputs/runs/<run_id>/plots/actual_vs_predicted.png`
- `outputs/runs/<run_id>/models/xgboost_pipeline.joblib`

The future forecast includes:

- `product`
- `date`
- `predicted_demand`
- `shortage_risk` and `suggested_purchase_qty` when `stock_actual` exists

## Precision Notes

The pipeline tracks a target MAPE of 15% by default, but it does not force that
number artificially. If the temporal holdout does not reach the target,
`run_summary.json` marks `target_mape_met` as `false` and records the best
available result. This is important for thesis integrity because a time-series
model should not report an accuracy level unsupported by the available data.

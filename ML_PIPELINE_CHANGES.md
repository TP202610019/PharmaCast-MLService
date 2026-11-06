# Real ML pipeline and backend contract

This document describes the real FastAPI pipeline in `app/`. Older CLI and
presentation instructions in README/MODELO_DOCS are not the runtime contract.
Use `PRESENTATION_MODE=false` for this pipeline. The existing presentation branch
and historicalPoints construction are unchanged; presentation output is not measured
XGBoost accuracy.

## Runtime and persistence ownership

Install the runtime with `python -m pip install -r requirements-runtime.txt`.
The older requirements.txt also includes legacy DB and plotting dependencies.
No database URL, database session, repository dependency, startup database
initialization or migration is required by the real ML service.
The backend owns business database writes, execution status and failure recording.
Legacy persistence modules remain unused by the API.

Configuration:

| Variable | Meaning |
| --- | --- |
| PRESENTATION_MODE | Set false for real training; explicitly overrides presentation settings in existing .env files |
| LOCAL_STORAGE_ENABLED | true uses local storage; false uses Azure Blob |
| LOCAL_STORAGE_ROOT | Local datasets/artifacts root; defaults to storage under this project |
| AZURE_STORAGE_CONNECTION_STRING | Required only in Blob mode |
| HISTORICAL_DATASETS_CONTAINER | Historical snapshot container/directory; default historical-datasets |
| MODEL_ARTIFACTS_CONTAINER | Fitted model container/directory; default model-artifacts |

Daily frequency is required. forecastPeriod accepts exactly 30, 60 or 90 days
and defaults to 30. Unsupported horizons fail request validation.

## Features and forecasting

Features are product_id, calendar fields, quantity lags (1/7/14 by default)
and shifted rolling means (7/14 by default). growth, revenue_importance and
demand_tier are excluded from both fitting and recursive forecast rows.
Every lag and rolling mean uses only earlier quantities.

Optional sales columns retained by the existing mapping/transformation are
preserved in historical snapshots. They are excluded from this baseline because
the request does not supply their future values; warnings lists the retained
columns excluded from features. This also avoids deriving price/stock/promotion
features over the holdout or silently substituting unknown future context.
The existing mapping rules, including inventory-specific behavior, are unchanged.

Missing daily sales and inactive product tails are filled with zero through the
common final observed date. historicalAvg uses this resampled daily history,
including zero days, from each product's first observation through that common
date. historicalPoints continues to report the existing observed daily totals.

The final model is fitted on all available combined history. Predictions are
pure XGBoost, clipped at zero, and feed subsequent lags/rolling means recursively.
There is no historical-mean blend or arbitrary forecast cap.
Numeric missing-value handling is identical during fitting and prediction.

## Holdout and metric definitions

The holdout is the final forecastPeriod consecutive calendar days, with a single
shared cutoff for all products. Training ends the day before evaluation starts.
Feature construction/fitting for evaluation uses only the training prefix.
Evaluation calls the same ForecastingStage as final forecasting for the entire
horizon; actual holdout quantities never feed back into forecast features.

Evaluation requires at least max(lags) + 1 training calendar days, in addition
to the full holdout: defaults require at least 45/75/105 total days for a
30/60/90-day request. Otherwise evaluation metrics are null, row counts are zero,
all four evaluation/training period fields are null and warnings explains the
shortfall. The final forecast and saved model are still produced.
There is no shorter substitute holdout or training-set accuracy fallback.

Both methods use the complete product/date holdout grid, including sparse,
zero-sales and inactive products. Overall errors are calculated directly over
these individual observations; they are never grouped into tiers or daily totals
before computing error. Product metrics use the same observations per product.

Products first appearing in holdout are explicitly included using the observed
future catalog and a zero quantity seed at the training cutoff. A warning marks
this unseen-catalog evaluation assumption. The seed gives no actual holdout
quantity to training, but catalog membership is known from the complete dataset.

NaiveWeekly recursively repeats the training history's last seven daily values.
If a product has fewer than seven prior values, an unavailable weekly lag is
zero. It never reads actual quantities from the holdout.

Metrics appear in order wape, mae, rmse, mbe, mape:

- WAPE: 100 * sum(abs(prediction - actual)) / sum(abs(actual)).
- MAE: mean(abs(prediction - actual)).
- RMSE: sqrt(mean((prediction - actual)^2)).
- MBE: mean(prediction - actual); positive means overforecast.
- MAPE: mean absolute percentage error over nonzero actual observations only.

WAPE is null for zero total actual demand, and MAPE is null when there are no
nonzero actual observations. Undefined percentages serialize as JSON null,
never NaN. Other metrics still include zero-demand observations.

## Response fields coordinated with the backend

POST /api/predictions/execute retains existing forecast/execution fields and:

| Field | Contract |
| --- | --- |
| metrics | XGBoost Overall summary; retains previous metric names |
| productMetrics | XGBoost Product records; retains productId, rows, historicalAvg |
| baselineMetrics | Equivalent Overall summary for NaiveWeekly |
| baselineProductMetrics | Equivalent Product records for NaiveWeekly |
| trainingPeriodStart / trainingPeriodEnd | Inclusive training dates of the evaluation model; nullable ISO dates |
| evaluationPeriodStart / evaluationPeriodEnd | Inclusive holdout dates; nullable ISO dates |
| warnings | Array of strings for insufficient history, holdout-only catalog or excluded optional context |
| artifactPaths | model, metadata and historicalSnapshot references for backend persistence |

Every metric record includes method, scope, rows and nullable numeric metrics,
including mbe. Exact case-sensitive enum strings are:

- scope: Overall for summaries, Product for individual product records.
- method: XGBoost or NaiveWeekly.

The period fields describe the evaluation model's training prefix, not the final
all-history refit. The final model's actual date range is recorded separately as
finalTrainingPeriodStart/finalTrainingPeriodEnd in metadata.

The backend should persist the returned artifactPaths on its execution record
to retain the fitted model, manifest and input-snapshot relationship, alongside
both sets of metrics and their method/scope strings. ML performs no business DB
writes. No backend files or migrations were changed by this work.

## Model and historical artifacts

The final fitted sklearn/XGBoost pipeline is serialized to model.joblib and its
manifest to metadata.json under user/execution directories. Metadata includes
datasetId, userId, executionId, fieldMappings, forecastPeriod, frequency,
lags/rolling windows, exact feature columns, training rows/dates, model parameters,
package/Python versions, evaluation periods, warnings and
forecastPolicy=pure_xgboost with clip_nonnegative postprocessing.
No dataset URL, SAS token or connection string is recorded.

Local artifact paths are absolute paths. Blob artifact paths are
container-relative references without SAS credentials. The historicalSnapshot
value is the path returned by HistoricalDataManager (local absolute path or
Blob name relative to HISTORICAL_DATASETS_CONTAINER).
The model-artifacts Blob container is created if absent; the configured historical
container must already be provisioned, as in the existing history implementation.

History loading/merging performs no writes. Snapshots are committed only after
evaluation, final training, forecasting, response construction and model/manifest
storage succeed. A failed earlier stage cannot contaminate later retraining.
The current upload overrides overlapping historical product/date rows.
Historical snapshots use ascending storage modification time with filename as a
tie breaker, in both local and Blob modes; later snapshots override earlier ones.
The two stores therefore follow the same history ordering and merge rules.

There is no cross-storage transaction: artifact or snapshot write failures fail
the execution and can leave unreferenced model files, which the backend must not
treat as a Completed execution. A successful response contains all three paths.

## Verification

Run from Modelo_XGBOOST:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Synthetic tests cover causal features and forecast-row alignment; pure recursive
30/60/90-day forecasts; exact holdouts and training cutoffs; all-product flat
metrics and enum casing; signed MBE; zero denominators; NaiveWeekly recursion;
insufficient training history; unseen products; optional history/warnings;
deferred snapshot saving on success/failure; local/Blob history ordering;
real XGBoost joblib round trips and mocked Blob artifact parity; startup and
route operation without database dependencies; allowed/default request horizons;
daily historical averages and metric field order. The user's separate loader
tests cover remote/local CSV parity and SAS redaction.

No live database, migration, real Blob request or application server is used by
these tests. Blob integration is mocked; live cloud permissions are not verified.

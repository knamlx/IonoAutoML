# fof2_fast_ml_24h results summary

Source artifact directory: `artifacts/fof2_fast_ml_24h`. Full artifacts are intentionally kept outside git because they are large.

## Run status

- Stations in artifact folders: 41
- Stations with final metrics: 35
- Stations with partial metrics only: 1
- Stations without usable metrics: 5
- Metric rows loaded: 175995
- Test interval: `2025-01-01T00:00:00+00:00` to `2025-12-30T23:45:00+00:00`

## Models

| model            |   mae |   rmse |       r2 |   corr |   mape |
|:-----------------|------:|-------:|---------:|-------:|-------:|
| CatBoost         | 0.911 |  1.117 |    0.063 |  0.867 | 13.889 |
| RandomForest     | 0.912 |  1.129 |    0.019 |  0.855 | 13.609 |
| XGBoost          | 0.923 |  1.142 |    0.22  |  0.851 | 13.786 |
| ElasticNet       | 1.242 |  1.51  |   -0.787 |  0.804 | 19.002 |
| LinearRegression | 4.984 |  6.416 | -223.279 |  0.683 | 77.921 |

## Latitude zones

| latitude_zone   |   mae |   rmse |       r2 |   corr |   mape |
|:----------------|------:|-------:|---------:|-------:|-------:|
| S_mid           | 1.407 |  1.755 |   -7.186 |  0.85  | 23.539 |
| N_mid           | 1.583 |  2.006 |  -43.166 |  0.839 | 26.382 |
| N_high          | 2.026 |  2.467 | -117.079 |  0.448 | 41.492 |
| Low             | 2.267 |  2.868 |  -48.117 |  0.825 | 28.64  |

## Best model counts

| model            |   best_count |
|:-----------------|-------------:|
| CatBoost         |        11000 |
| RandomForest     |         9597 |
| XGBoost          |         6537 |
| ElasticNet       |         6107 |
| LinearRegression |         1958 |

## Notes

- The completed fast run used fixed model parameters, without Optuna/AutoML.
- The new configuration `fof2_fast_ml_24h_automl_shap` is prepared for Optuna tuning, saved models, and SHAP summaries in a separate output directory.
- Existing full artifacts in `artifacts/fof2_fast_ml_24h` should not be overwritten by the new run.
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


FUTURE_DERIVED_MARKERS = ("_target_", "_pred", "_savgol")
BASELINE_MODELS = {"persistence_state", "persistence_horizon_lag", "seasonal_24h_lag"}
ML_MODELS = {"LinearRegression", "ElasticNet", "RandomForest", "XGBoost", "CatBoost"}
OPTUNA_MODELS = {"ElasticNet", "RandomForest", "XGBoost", "CatBoost"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leak-aware ML baselines and Optuna models on feature CSVs.")
    parser.add_argument("--config", default="configs/experiments/baseline_v0.1.json")
    parser.add_argument("--station", default=None, help="Optional single station override.")
    parser.add_argument("--horizon", default=None, help="Optional single horizon override, for example 1h.")
    parser.add_argument("--no-automl", action="store_true", help="Disable Optuna even if enabled in the config.")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration and print planned run size without training.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_name(value: str) -> str:
    return value.replace(" ", "").replace("/", "_").replace("-", "_")


def horizon_to_hours(horizon: str) -> float:
    return pd.Timedelta(horizon) / pd.Timedelta(hours=1)


def as_utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def resolve_stations(config: dict[str, Any], station_override: str | None = None) -> list[str]:
    if station_override:
        return [station_override]
    configured = config.get("stations", [])
    if configured == "all":
        station_dir = Path(config["feature_dir"]) / "stations"
        return sorted(path.name.removesuffix("_features.csv") for path in station_dir.glob("*_features.csv"))
    return list(configured)


def iter_test_days(config: dict[str, Any]) -> list[pd.Timestamp]:
    start = as_utc_timestamp(config["test_start"])
    end = as_utc_timestamp(config["test_end"])
    if end <= start:
        raise ValueError("test_end must be after test_start.")
    return list(pd.date_range(start, end - pd.Timedelta(days=1), freq="1D"))


def split_walk_forward_window(
    frame: pd.DataFrame,
    *,
    current_day: pd.Timestamp,
    train_days: int,
    forecast_horizon: str,
    dataset_step: str,
    time_column: str = "time_utc",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.Timestamp]]:
    step = pd.Timedelta(dataset_step)
    forecast_delta = pd.Timedelta(forecast_horizon)
    if forecast_delta < step or forecast_delta % step != pd.Timedelta(0):
        raise ValueError(f"forecast_horizon {forecast_horizon!r} is not compatible with dataset_step {dataset_step!r}.")

    test_start = as_utc_timestamp(current_day)
    test_end = test_start + pd.Timedelta(days=1) - step
    train_end = test_start - step
    train_start = train_end - pd.Timedelta(days=train_days) + step
    safe_train_end = train_end - forecast_delta

    time = frame[time_column]
    train = frame[(time >= train_start) & (time <= safe_train_end)].copy()
    test = frame[(time >= test_start) & (time <= test_end)].copy()
    bounds = {
        "train_start": train_start,
        "train_end": train_end,
        "safe_train_end": safe_train_end,
        "test_start": test_start,
        "test_end": test_end,
    }
    return train, pd.DataFrame(columns=frame.columns), test, bounds


def split_fixed_year(
    frame: pd.DataFrame,
    *,
    forecast_horizon: str,
    dataset_step: str,
    config: dict[str, Any],
    time_column: str = "time_utc",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.Timestamp]]:
    step = pd.Timedelta(dataset_step)
    forecast_delta = pd.Timedelta(forecast_horizon)
    if forecast_delta < step or forecast_delta % step != pd.Timedelta(0):
        raise ValueError(f"forecast_horizon {forecast_horizon!r} is not compatible with dataset_step {dataset_step!r}.")

    train_start = as_utc_timestamp(config["train_start"])
    train_end = as_utc_timestamp(config["train_end"]) - step
    val_start = as_utc_timestamp(config["validation_start"])
    requested_val_end = as_utc_timestamp(config["validation_end"]) - step
    test_start = as_utc_timestamp(config["test_start"])
    test_end = as_utc_timestamp(config["test_end"]) - step

    if not (train_start <= val_start <= requested_val_end <= train_end < test_start <= test_end):
        raise ValueError("Expected validation inside train period and test after train period for fixed_year_split.")

    safe_train_end = val_start - forecast_delta - step
    val_end = min(requested_val_end, train_end - forecast_delta)
    time = frame[time_column]
    train_core = frame[(time >= train_start) & (time <= safe_train_end)].copy()
    validation = frame[(time >= val_start) & (time <= val_end)].copy()
    test = frame[(time >= test_start) & (time <= test_end)].copy()
    bounds = {
        "train_start": train_start,
        "train_end": train_end,
        "safe_train_end": safe_train_end,
        "validation_start": val_start,
        "validation_end": val_end,
        "requested_validation_end": requested_val_end,
        "test_start": test_start,
        "test_end": test_end,
    }
    return train_core, validation, test, bounds


def split_frames(
    frame: pd.DataFrame,
    *,
    config: dict[str, Any],
    horizon: str,
    current_day: pd.Timestamp | None = None,
    train_days: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.Timestamp]]:
    mode = config.get("evaluation_mode", "fixed_year_split")
    if mode == "fixed_year_split":
        return split_fixed_year(
            frame,
            forecast_horizon=horizon,
            dataset_step=config["dataset_step"],
            config=config,
        )
    if mode == "walk_forward_daily":
        if current_day is None or train_days is None:
            raise ValueError("walk_forward_daily requires current_day and train_days.")
        return split_walk_forward_window(
            frame,
            current_day=current_day,
            train_days=train_days,
            forecast_horizon=horizon,
            dataset_step=config["dataset_step"],
        )
    raise ValueError(f"Unsupported evaluation_mode: {mode!r}")


def split_iterations(config: dict[str, Any]) -> list[dict[str, Any]]:
    if config.get("evaluation_mode", "fixed_year_split") == "fixed_year_split":
        return [{"split_id": "fixed_2024_train_2025_test", "train_days": None, "test_day": None}]
    return [
        {"split_id": f"{day.date().isoformat()}_{train_days}d", "train_days": int(train_days), "test_day": day}
        for train_days in config.get("walk_forward_train_days", [])
        for day in iter_test_days(config)
    ]


def numeric_feature_pool(frame: pd.DataFrame, target: str) -> list[str]:
    excluded = {"hour", "month", "doy"}
    target_prefix = f"{target}_target_"
    columns = []
    for column in frame.select_dtypes(include=np.number).columns:
        if column in excluded:
            continue
        if column.startswith(target_prefix) or any(marker in column for marker in FUTURE_DERIVED_MARKERS):
            continue
        columns.append(column)
    return columns


def select_features(train: pd.DataFrame, feature_pool: list[str], min_coverage: float) -> list[str]:
    return [feature for feature in feature_pool if feature in train.columns and train[feature].notna().mean() >= min_coverage]


def metric_record(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    valid = pd.DataFrame({"actual": y_true, "predicted": y_pred}).dropna()
    if valid.empty:
        return {"n": 0, "mae": math.nan, "rmse": math.nan, "r2": math.nan, "corr": math.nan}
    error = valid["predicted"] - valid["actual"]
    mae = float(mean_absolute_error(valid["actual"], valid["predicted"]))
    rmse = float(np.sqrt(mean_squared_error(valid["actual"], valid["predicted"])))
    r2 = float(r2_score(valid["actual"], valid["predicted"])) if len(valid) > 1 else math.nan
    corr = float(valid["actual"].corr(valid["predicted"])) if len(valid) > 1 else math.nan
    bias = float(error.mean())
    return {"n": int(len(valid)), "mae": mae, "rmse": rmse, "r2": r2, "corr": corr, "bias": bias}


def make_baseline_predictions(test: pd.DataFrame, model_name: str, target: str, horizon: str) -> pd.Series:
    if model_name == "persistence_state":
        return pd.to_numeric(test.get(f"{target}_state"), errors="coerce")
    if model_name == "persistence_horizon_lag":
        return pd.to_numeric(test.get(f"{target}_lag_{safe_name(horizon)}"), errors="coerce")
    if model_name == "seasonal_24h_lag":
        return pd.to_numeric(test.get(f"{target}_lag_24h"), errors="coerce")
    raise ValueError(f"Unsupported baseline model: {model_name}")


def default_model_params(model_name: str, config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("model_params", {}).get(model_name, {}))


def make_estimator(model_name: str, config: dict[str, Any], params: dict[str, Any] | None = None) -> object:
    model_params = default_model_params(model_name, config)
    if params:
        model_params.update(params)
    if model_name == "LinearRegression":
        return LinearRegression(**model_params)
    if model_name == "ElasticNet":
        return ElasticNet(**model_params)
    if model_name == "RandomForest":
        return RandomForestRegressor(**model_params)
    if model_name == "XGBoost":
        return XGBRegressor(**model_params)
    if model_name == "CatBoost":
        return CatBoostRegressor(**model_params)
    raise ValueError(f"Unsupported ML model: {model_name}")


def impute_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    validation: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    imputer = SimpleImputer(strategy="median")
    train_imputed = pd.DataFrame(imputer.fit_transform(train[features]), columns=features, index=train.index)
    test_imputed = pd.DataFrame(imputer.transform(test[features]), columns=features, index=test.index)
    val_imputed = None
    if validation is not None:
        val_imputed = pd.DataFrame(imputer.transform(validation[features]), columns=features, index=validation.index)
    return train_imputed, val_imputed, test_imputed


def fit_predict_ml(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target_column: str,
    selected_features: list[str],
    model_name: str,
    config: dict[str, Any],
) -> tuple[pd.Series, int]:
    if not selected_features:
        return pd.Series(index=test.index, dtype=float), 0

    y_train = pd.to_numeric(train[target_column], errors="coerce")
    valid_train = y_train.notna()
    if valid_train.sum() < int(config["minimum_train_rows"]):
        return pd.Series(index=test.index, dtype=float), int(valid_train.sum())

    train_imputed, _, test_imputed = impute_frames(train.loc[valid_train], test, selected_features)
    estimator = make_estimator(model_name, config)
    estimator.fit(train_imputed, y_train.loc[valid_train])
    return pd.Series(estimator.predict(test_imputed), index=test.index), int(valid_train.sum())


def objective_direction(metric: str) -> str:
    return "maximize" if metric.upper() in {"R2", "CORR", "CC"} else "minimize"


def objective_value(metric: str, y_true: pd.Series, y_pred: pd.Series) -> float:
    metrics = metric_record(y_true, y_pred)
    key = {"RMSE": "rmse", "MAE": "mae", "R2": "r2", "CORR": "corr", "CC": "corr"}.get(metric.upper())
    if key is None:
        raise ValueError(f"Unsupported Optuna metric: {metric!r}.")
    value = metrics[key]
    if not np.isfinite(value):
        return np.inf if objective_direction(metric) == "minimize" else -np.inf
    return float(value)


def sample_optuna_params(model_name: str, trial: optuna.Trial) -> dict[str, Any]:
    if model_name == "ElasticNet":
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 1e2, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.01, 0.99),
            "max_iter": 10000,
            "random_state": 42,
        }
    if model_name == "RandomForest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "n_jobs": 1,
            "random_state": 42,
        }
    if model_name == "XGBoost":
        return {
            "objective": "reg:squarederror",
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 20.0),
            "n_jobs": 1,
            "verbosity": 0,
            "random_state": 42,
        }
    if model_name == "CatBoost":
        return {
            "loss_function": "RMSE",
            "iterations": trial.suggest_int("iterations", 300, 1200, step=100),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "random_seed": 42,
            "thread_count": 1,
            "verbose": False,
        }
    raise ValueError(f"Unsupported Optuna model: {model_name}")


def run_optuna_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target_column: str,
    selected_features: list[str],
    model_name: str,
    config: dict[str, Any],
    station: str,
    horizon: str,
    output_dir: Path,
) -> tuple[pd.Series, dict[str, Any], list[dict[str, Any]], int]:
    automl = config.get("automl", {})
    metric = automl.get("metric", "RMSE")
    random_state = int(automl.get("random_state", 42))
    direction = objective_direction(metric)

    y_train = pd.to_numeric(train[target_column], errors="coerce")
    y_val = pd.to_numeric(validation[target_column], errors="coerce")
    y_test = pd.to_numeric(test[target_column], errors="coerce")
    valid_train = y_train.notna()
    valid_val = y_val.notna()
    valid_test = y_test.notna()
    if valid_train.sum() < int(config["minimum_train_rows"]) or valid_val.sum() < int(config["minimum_eval_rows"]):
        return pd.Series(index=test.index, dtype=float), {}, [], int(valid_train.sum())

    x_train, x_val, x_test = impute_frames(train.loc[valid_train], test, selected_features, validation.loc[valid_val])
    assert x_val is not None
    y_train_fit = y_train.loc[valid_train]
    y_val_eval = y_val.loc[valid_val]

    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner = optuna.pruners.MedianPruner() if automl.get("method") == "optuna_tpe_pruning" else optuna.pruners.NopPruner()
    storage = automl.get("storage")
    study_name = f"{station}_{horizon}_{model_name}_{target_column}"
    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        params = sample_optuna_params(model_name, trial)
        model = make_estimator(model_name, config, params=params)
        model.fit(x_train, y_train_fit)
        pred = pd.Series(model.predict(x_val), index=y_val_eval.index)
        value = objective_value(metric, y_val_eval, pred)
        trial.report(value, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return value

    target_trials = int(automl.get("n_trials", 25))
    remaining_trials = max(0, target_trials - len(study.trials))
    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials)
    best_params = dict(study.best_params)
    best_value = float(study.best_value)

    final_train = train
    if automl.get("train_final_on_train_validation", True):
        final_train = pd.concat([train, validation], axis=0)
    final_y = pd.to_numeric(final_train[target_column], errors="coerce")
    valid_final = final_y.notna()
    train_label_rows = int(valid_final.sum())
    final_x, _, final_test_x = impute_frames(final_train.loc[valid_final], test, selected_features)
    final_model = make_estimator(model_name, config, params=best_params)
    final_model.fit(final_x, final_y.loc[valid_final])
    y_pred = pd.Series(final_model.predict(final_test_x), index=test.index)

    if automl.get("save_models", False):
        model_dir = output_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, model_dir / f"{station}_{safe_name(horizon)}_{model_name}.joblib")

    trials = []
    for trial in study.trials:
        trials.append(
            {
                "station": station,
                "horizon": horizon,
                "model": model_name,
                "trial_number": trial.number,
                "value": trial.value,
                "state": str(trial.state),
                "params": json.dumps(trial.params, sort_keys=True),
            }
        )
    info = {"best_params": json.dumps(best_params, sort_keys=True), "best_val_score": best_value}
    if valid_test.sum() < int(config["minimum_eval_rows"]):
        return pd.Series(index=test.index, dtype=float), info, trials, train_label_rows
    return y_pred, info, trials, train_label_rows


def append_metric_and_predictions(
    *,
    metric_rows: list[dict[str, Any]],
    prediction_frames: list[pd.DataFrame],
    station: str,
    horizon: str,
    split_id: str,
    train_days: int | None,
    model_name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    bounds: dict[str, pd.Timestamp],
    selected_features: list[str],
    target_column: str,
    y_pred: pd.Series,
    extra: dict[str, Any] | None = None,
) -> None:
    y_test = pd.to_numeric(test[target_column], errors="coerce")
    metrics = metric_record(y_test, y_pred)
    if metrics["n"] <= 0:
        return
    extra = extra or {}
    metric_rows.append(
        {
            "station": station,
            "horizon": horizon,
            "horizon_hours": horizon_to_hours(horizon),
            "split_id": split_id,
            "train_days": train_days,
            "model": model_name,
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "feature_count": int(len(selected_features)),
            **metrics,
            **extra,
            **{key: value.isoformat() for key, value in bounds.items()},
        }
    )
    prediction_frames.append(
        pd.DataFrame(
            {
                "station": station,
                "time_utc": test["time_utc"],
                "target_time_utc": test["time_utc"] + pd.Timedelta(horizon),
                "horizon": horizon,
                "split_id": split_id,
                "train_days": train_days,
                "model": model_name,
                "actual": y_test,
                "predicted": y_pred,
            }
        )
    )


def run_station(
    config: dict[str, Any],
    station: str,
    horizons: list[str],
    output_dir: Path,
    run_automl: bool,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[dict[str, Any]]]:
    path = Path(config["feature_dir"]) / "stations" / f"{station}_features.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    frame["time_utc"] = pd.to_datetime(frame["time_utc"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["time_utc"]).sort_values("time_utc").reset_index(drop=True)
    feature_pool = numeric_feature_pool(frame, config["target"])

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    trial_rows: list[dict[str, Any]] = []
    for horizon in horizons:
        target_column = f"{config['target']}_target_{safe_name(horizon)}"
        if target_column not in frame.columns:
            raise KeyError(f"Missing target column {target_column!r} in {path}.")
        for iteration in split_iterations(config):
            train, validation, test, bounds = split_frames(
                frame,
                config=config,
                horizon=horizon,
                current_day=iteration["test_day"],
                train_days=iteration["train_days"],
            )
            selected_features = select_features(train, feature_pool, float(config["minimum_feature_coverage"]))
            y_test = pd.to_numeric(test[target_column], errors="coerce")
            if (
                len(train) < int(config["minimum_train_rows"])
                or y_test.notna().sum() < int(config["minimum_eval_rows"])
                or not selected_features
            ):
                continue

            for model_name in config["models"]:
                if model_name in BASELINE_MODELS:
                    y_pred = make_baseline_predictions(test, model_name, config["target"], horizon)
                    extra = {"train_label_rows": int(pd.to_numeric(train[target_column], errors="coerce").notna().sum())}
                elif model_name in ML_MODELS:
                    y_pred, train_label_rows = fit_predict_ml(
                        train,
                        test,
                        target_column=target_column,
                        selected_features=selected_features,
                        model_name=model_name,
                        config=config,
                    )
                    extra = {"train_label_rows": train_label_rows}
                else:
                    raise ValueError(f"Unsupported model in config: {model_name}")
                append_metric_and_predictions(
                    metric_rows=metric_rows,
                    prediction_frames=prediction_frames,
                    station=station,
                    horizon=horizon,
                    split_id=iteration["split_id"],
                    train_days=iteration["train_days"],
                    model_name=model_name,
                    train=train,
                    validation=validation,
                    test=test,
                    bounds=bounds,
                    selected_features=selected_features,
                    target_column=target_column,
                    y_pred=y_pred,
                    extra=extra,
                )

            automl = config.get("automl", {})
            if run_automl and automl.get("enabled", False):
                for model_name in automl.get("models", []):
                    if model_name not in OPTUNA_MODELS:
                        raise ValueError(f"Unsupported Optuna model in config: {model_name}")
                    y_pred, info, trials, train_label_rows = run_optuna_model(
                        train,
                        validation,
                        test,
                        target_column=target_column,
                        selected_features=selected_features,
                        model_name=model_name,
                        config=config,
                        station=station,
                        horizon=horizon,
                        output_dir=output_dir,
                    )
                    trial_rows.extend(trials)
                    append_metric_and_predictions(
                        metric_rows=metric_rows,
                        prediction_frames=prediction_frames,
                        station=station,
                        horizon=horizon,
                        split_id=iteration["split_id"],
                        train_days=iteration["train_days"],
                        model_name=f"Optuna_{model_name}",
                        train=train,
                        validation=validation,
                        test=test,
                        bounds=bounds,
                        selected_features=selected_features,
                        target_column=target_column,
                        y_pred=y_pred,
                        extra={"train_label_rows": train_label_rows, **info},
                    )
    return metric_rows, prediction_frames, trial_rows


def write_outputs(
    output_dir: Path,
    config: dict[str, Any],
    stations: list[str],
    horizons: list[str],
    metrics: list[dict[str, Any]],
    trials: list[dict[str, Any]],
    prediction_rows: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(metrics)
    trials_df = pd.DataFrame(trials)

    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    trials_df.to_csv(output_dir / "optuna_trials.csv", index=False)
    if not metrics_df.empty:
        summary = (
            metrics_df.groupby(["station", "horizon", "model"], dropna=False)[["n", "mae", "rmse", "r2", "corr", "bias"]]
            .mean(numeric_only=True)
            .reset_index()
        )
    else:
        summary = pd.DataFrame()
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)

    if config.get("automl", {}).get("export_trials_parquet", False) and not trials_df.empty:
        trials_df.to_parquet(output_dir / "optuna_trials.parquet", index=False)
    if not metrics_df.empty:
        metrics_df.to_parquet(output_dir / "metrics.parquet", index=False)

    run_manifest = {
        "config": config,
        "stations": stations,
        "horizons": horizons,
        "metrics_rows": int(len(metrics_df)),
        "prediction_rows": int(prediction_rows),
        "trial_rows": int(len(trials_df)),
        "prediction_layout": "artifacts/<experiment>/predictions/<station>_predictions.{csv,parquet}",
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def dry_run(config: dict[str, Any], stations: list[str], horizons: list[str], run_automl: bool) -> None:
    iterations = split_iterations(config)
    regular_models = len(config.get("models", []))
    optuna_models = len(config.get("automl", {}).get("models", [])) if run_automl and config.get("automl", {}).get("enabled") else 0
    total_fit_groups = len(stations) * len(horizons) * len(iterations)
    total_regular_model_fits = total_fit_groups * regular_models
    total_optuna_trials = total_fit_groups * optuna_models * int(config.get("automl", {}).get("n_trials", 0))
    print("DRY RUN")
    print(f"evaluation_mode: {config.get('evaluation_mode')}")
    print(f"stations: {len(stations)}")
    print(f"horizons: {horizons}")
    print(f"split_iterations: {len(iterations)}")
    print(f"regular_models: {regular_models} -> estimated fits {total_regular_model_fits}")
    print(f"optuna_models: {optuna_models} -> estimated trials {total_optuna_trials}")
    print(f"output_dir: {config['output_dir']}")
    if config.get("evaluation_mode") == "fixed_year_split" and stations and horizons:
        sample_path = Path(config["feature_dir"]) / "stations" / f"{stations[0]}_features.csv"
        sample = pd.read_csv(sample_path, usecols=["time_utc", f"{config['target']}_target_{safe_name(horizons[-1])}"])
        sample["time_utc"] = pd.to_datetime(sample["time_utc"], utc=True, errors="coerce")
        train, validation, test, bounds = split_fixed_year(
            sample,
            forecast_horizon=horizons[-1],
            dataset_step=config["dataset_step"],
            config=config,
        )
        print("sample_split_for_largest_horizon:")
        for key, value in bounds.items():
            print(f"  {key}: {value}")
        print(f"  rows: train={len(train)} validation={len(validation)} test={len(test)}")


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    stations = resolve_stations(config, args.station)
    horizons = [args.horizon] if args.horizon else list(config["horizons"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    run_automl = not args.no_automl
    if args.dry_run:
        dry_run(config, stations, horizons, run_automl)
        return

    all_metrics: list[dict[str, Any]] = []
    all_trials: list[dict[str, Any]] = []
    prediction_rows = 0
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for station in stations:
        metrics, predictions, trials = run_station(config, station, horizons, output_dir, run_automl)
        all_metrics.extend(metrics)
        all_trials.extend(trials)
        station_predictions = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
        prediction_rows += int(len(station_predictions))
        if not station_predictions.empty:
            station_predictions.to_csv(prediction_dir / f"{station}_predictions.csv", index=False)
            if config.get("automl", {}).get("export_predictions_parquet", True):
                station_predictions.to_parquet(prediction_dir / f"{station}_predictions.parquet", index=False)
        print(f"{station}: metrics={len(metrics)} prediction_blocks={len(predictions)} trials={len(trials)}")

    write_outputs(output_dir, config, stations, horizons, all_metrics, all_trials, prediction_rows)
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()

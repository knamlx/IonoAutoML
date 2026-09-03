import pandas as pd

from filter_time_series import filter_column, filter_station_file


def test_filter_column_replaces_local_outlier() -> None:
    values = pd.Series([10.0, 10.1, 10.0, 50.0, 10.2, 10.1, 10.0])
    config = {
        "methods": {
            "hampel": {"enabled": True, "window": 5, "n_sigma": 3.0},
            "rolling_median": {"enabled": False},
        }
    }

    filtered, changed = filter_column(values, config)

    assert changed == 1
    assert filtered.iloc[3] < 11.0


def test_filter_station_file_keeps_raw_column(tmp_path) -> None:
    source = tmp_path / "AA000_time_grid.csv"
    target = tmp_path / "out" / "stations" / "AA000_time_grid.csv"
    pd.DataFrame(
        {
            "time_utc": pd.date_range("2024-01-01", periods=5, freq="15min", tz="UTC"),
            "station": ["AA000"] * 5,
            "foF2": [5.0, 5.1, 30.0, 5.2, 5.0],
            "gfz_Kp": [1.0, 1.0, 9.0, 1.0, 1.0],
        }
    ).to_csv(source, index=False)
    config = {
        "columns": ["foF2"],
        "methods": {
            "hampel": {"enabled": True, "window": 5, "n_sigma": 3.0},
            "rolling_median": {"enabled": False},
        },
        "keep_original_columns": True,
        "replace_columns": True,
        "filtered_suffix": "_filtered",
    }

    summary = filter_station_file(source, target, config)
    result = pd.read_csv(target)

    assert summary["filtered_columns"] == ["foF2"]
    assert "foF2_raw" in result.columns
    assert "gfz_Kp_raw" not in result.columns

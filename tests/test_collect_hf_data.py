import argparse
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import collect_hf_data as collector


def assert_payload_shape(test_case, payload, dataset, count):
    test_case.assertEqual(payload["schema_version"], "1.0")
    test_case.assertEqual(payload["dataset"], dataset)
    test_case.assertEqual(payload["record_count"], count)
    test_case.assertIsInstance(payload["records"], list)
    test_case.assertEqual(len(payload["records"]), count)


class JsonOutputTests(unittest.TestCase):
    def test_make_records_payload_has_uniform_shape(self):
        payload = collector.make_records_payload("sample", [{"time_utc": "2026-01-01T00:00:00Z"}])
        assert_payload_shape(self, payload, "sample", 1)

    def test_read_json_bytes_accepts_swpc_trailing_nul_bytes(self):
        payload = collector.read_json_bytes(b'[["time_tag","speed"],["2026-01-01 00:00:00.000","400"]]\x00\x00')
        self.assertEqual(payload[0], ["time_tag", "speed"])


class ParserTests(unittest.TestCase):
    def test_collect_noaa_normalizes_table_json(self):
        config = {
            "noaa": {
                "enabled": True,
                "endpoints": [{"name": "solar_wind_plasma_2_hour", "url": "https://example.test/plasma.json"}],
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            paths = collector.Paths(
                root=Path(tmp),
                raw_noaa=Path(tmp) / "raw" / "noaa",
                raw_gfz=Path(tmp) / "raw" / "gfz",
                raw_omni=Path(tmp) / "raw" / "omni",
                raw_giro=Path(tmp) / "raw" / "giro",
                processed=Path(tmp) / "processed",
                metadata=Path(tmp) / "metadata",
                logs=Path(tmp) / "logs",
            )
            for path in (paths.raw_noaa, paths.raw_gfz, paths.raw_omni, paths.raw_giro, paths.processed, paths.metadata, paths.logs):
                path.mkdir(parents=True, exist_ok=True)
            raw = b'[["time_tag","density","speed"],["2026-01-01 00:00:00.000","5.1","410.2"]]'
            with patch.object(collector, "fetch_url", return_value=raw):
                rows = collector.collect_noaa(config, paths, timeout=1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source"], "NOAA_SWPC")
        self.assertEqual(rows[0]["product"], "solar_wind_plasma_2_hour")
        self.assertEqual(rows[0]["time_utc"], "2026-01-01T00:00:00Z")
        self.assertEqual(rows[0]["speed"], "410.2")

    def test_parse_giro_table_normalizes_success_and_error_rows(self):
        text = "\n".join(
            [
                "# Time CS foF2 MUFD hmF2 TEC fmin",
                "# Time CS foF2 QD MUFD QD hmF2 QD TEC QD fmin QD",
                "2012-07-02T21:01:00.000Z 100 5.15 // 15.58 // 311.5 // 3.9 // 1.7 //",
                "ERROR: No data found for requested period",
            ]
        )
        rows = collector.parse_giro_table(text, station="MO155", raw_file="sample.txt")
        self.assertEqual(rows[0]["source"], "GIRO_DIDBase")
        self.assertEqual(rows[0]["station"], "MO155")
        self.assertEqual(rows[0]["time_utc"], "2012-07-02T21:01:00Z")
        self.assertEqual(rows[0]["foF2"], "5.15")
        self.assertEqual(rows[0]["foF2_QD"], "//")
        self.assertEqual(rows[1]["error"], "ERROR: No data found for requested period")

    def test_parse_giro_station_codes_deduplicates_form_options(self):
        html = '<select name="location"><option value="MO155">MOSCOW</option><option value="MO155">MOSCOW</option><option value="BC840">BOULDER</option></select>'
        self.assertEqual(collector.parse_giro_station_codes(html), ["MO155", "BC840"])

    def test_parse_giro_station_options_keeps_station_labels(self):
        html = '<option value="MO155">Moscow Parus-A</option><option value="BC840">Boulder</option>'
        self.assertEqual(
            collector.parse_giro_station_options(html),
            [{"station": "MO155", "name": "Moscow Parus-A"}, {"station": "BC840", "name": "Boulder"}],
        )

    def test_iter_time_windows_splits_long_collection_periods(self):
        start = collector.parse_dt("2024-01-01T00:00:00Z")
        end = collector.parse_dt("2024-03-01T00:00:00Z")
        windows = collector.iter_time_windows(start, end, 31)
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0][0], start)
        self.assertEqual(windows[-1][1], end)

    def test_iter_time_windows_never_crosses_calendar_year_boundary(self):
        start = collector.parse_dt("2024-01-01T00:00:00Z")
        end = collector.parse_dt("2026-01-01T00:00:00Z")
        windows = collector.iter_time_windows(start, end, 999)
        self.assertEqual(
            windows,
            [
                (collector.parse_dt("2024-01-01T00:00:00Z"), collector.parse_dt("2025-01-01T00:00:00Z")),
                (collector.parse_dt("2025-01-01T00:00:00Z"), collector.parse_dt("2026-01-01T00:00:00Z")),
            ],
        )

    def test_normalize_giro_headers_keeps_quality_columns(self):
        headers = collector.normalize_giro_headers(["Time", "CS", "foF2", "QD", "TEC", "QD"])
        self.assertEqual(headers, ["Time", "CS", "foF2", "foF2_QD", "TEC", "TEC_QD"])

    def test_build_analytical_dataset_uses_giro_mufd_column(self):
        rows = [
            {
                "station": "MO155",
                "time_utc": "2012-07-02T21:01:00Z",
                "foF2": "5.15",
                "MUF(D)": "15.583",
            }
        ]
        dataset = collector.build_analytical_dataset(rows, [], [], tolerance_minutes=90)
        self.assertEqual(dataset[0]["MUFD_3000_MHz"], "15.583")
        self.assertEqual(dataset[0]["muf_3000_proxy_MHz"], 15.583)

    def test_parse_gfz_index_payload_normalizes_records(self):
        payload = {
            "Kp": [0.667],
            "datetime": ["2024-01-01T00:00:00Z"],
            "status": ["def"],
            "meta": {"source": "GFZ Potsdam"},
        }
        rows = collector.parse_gfz_index_payload(payload, "Kp", "kp.json")
        self.assertEqual(
            rows[0],
            {
                "source": "GFZ",
                "product": "kp_gfz_json_api",
                "index": "Kp",
                "time_utc": "2024-01-01T00:00:00Z",
                "value": 0.667,
                "status": "def",
                "raw_file": "kp.json",
            },
        )

    def test_build_analytical_dataset_adds_nearest_gfz_index_features(self):
        giro_rows = [{"station": "MO155", "time_utc": "2024-01-01T01:00:00Z", "foF2": "5.0"}]
        gfz_rows = [
            {"index": "Kp", "time_utc": "2024-01-01T00:00:00Z", "value": 0.667, "status": "def"},
            {"index": "Ap", "time_utc": "2024-01-01T00:00:00Z", "value": 7, "status": "def"},
        ]
        dataset = collector.build_analytical_dataset(giro_rows, [], gfz_rows, tolerance_minutes=90)
        self.assertEqual(dataset[0]["Kp"], 0.667)
        self.assertEqual(dataset[0]["Kp_time_utc"], "2024-01-01T00:00:00Z")
        self.assertEqual(dataset[0]["Ap"], 7)

    def test_parse_omni_hourly_listing_normalizes_rows(self):
        text = """
        <pre>YEAR DOY HR    1     2
        2024   1  0  -2.8  -11
        2024   1  1 9999.99 99999
        </pre>
        """
        rows = collector.parse_omni_hourly_listing(text, ["bz_gsm_nT", "dst_nT"], "omni.html")
        self.assertEqual(rows[0]["time_utc"], "2024-01-01T00:00:00Z")
        self.assertEqual(rows[0]["bz_gsm_nT"], -2.8)
        self.assertEqual(rows[0]["dst_nT"], -11.0)
        self.assertEqual(rows[1]["bz_gsm_nT"], "")


class EndToEndJsonTests(unittest.TestCase):
    def test_main_writes_uniform_json_outputs(self):
        noaa_raw = {
            "https://example.test/kp.json": json.dumps(
                [{"time_tag": "2026-01-01T00:00:00", "estimated_kp": 2.0}]
            ).encode("utf-8"),
            "https://kp.gfz.de/app/json/?start=2026-01-01T00%3A00%3A00Z&end=2026-01-01T01%3A00%3A00Z&index=Kp": json.dumps(
                {"Kp": [2.0], "datetime": ["2026-01-01T00:00:00Z"], "status": ["def"]}
            ).encode("utf-8"),
            "https://giro.uml.edu/didbase/scaled.php": "\n".join(
                [
                    "# Time CS foF2 MUFD hmF2 TEC fmin",
                    "2026-01-01T00:00:00.000Z 100 5.0 15.0 300.0 10.0 1.5",
                ]
            ).encode("latin-1"),
        }

        def fake_fetch(url, *, data, timeout, attempts=3):
            return noaa_raw[url]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.toml"
            output_dir = root / "out"
            config_path.write_text(
                "\n".join(
                    [
                        "[run]",
                        'output_dir = "out"',
                        "timeout_seconds = 1",
                        "[giro]",
                        "enabled = true",
                        'stations = ["MO155"]',
                        'characteristics = "all"',
                        "muf_distance_km = 3000",
                        "chunk_days = 31",
                        "[noaa]",
                        "enabled = true",
                        'endpoints = [{ name = "planetary_k_index_1m", url = "https://example.test/kp.json" }]',
                        "[gfz]",
                        "enabled = true",
                        'api_url = "https://kp.gfz.de/app/json/"',
                        'indices = ["Kp"]',
                        "[omni]",
                        "enabled = false",
                        "[preprocess]",
                        "join_tolerance_minutes = 90",
                    ]
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                config=str(config_path),
                start="2026-01-01T00:00:00Z",
                end="2026-01-01T01:00:00Z",
                output_dir=str(output_dir),
                sources="all",
            )
            with (
                patch.object(collector, "parse_args", return_value=args),
                patch.object(collector, "fetch_url", side_effect=fake_fetch),
                redirect_stdout(StringIO()),
            ):
                self.assertEqual(collector.main(), 0)

            expected = {
                "noaa_observations.json": "noaa_observations",
                "geophysical_indices.json": "geophysical_indices",
                "giro_scaled.json": "giro_scaled",
                "analytical_hf_dataset.json": "analytical_hf_dataset",
            }
            for file_name, dataset in expected.items():
                payload = json.loads((output_dir / "processed" / file_name).read_text(encoding="utf-8"))
                assert_payload_shape(self, payload, dataset, 1)

            manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["outputs"]["json"]), 5)
            self.assertEqual(len(manifest["outputs"]["csv"]), 5)


if __name__ == "__main__":
    unittest.main()

"""Artificial MATLAB fixtures test parser behavior, not NASA schema validity."""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.io import savemat

from ev_battery.config import ProjectConfig, ProjectPaths
from ev_battery.data_loader import DatasetError, discover_mat_files, read_mat_file
from ev_battery.inspection import describe, inspect_file, main


class InspectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="artificial-mat-parser-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def make_fixture(self, filename: str = "artificial_fixture.mat") -> Path:
        """Deliberately unrelated variable/file names and mixed struct data fields."""
        cycles = np.empty((1, 3), dtype=[("type", "O"), ("data", "O")])
        cycles[0, 0] = ("charge", {"Voltage": np.array([[3.1, 3.2]])})
        cycles[0, 1] = ("discharge", {"Capacity": np.array([[1.9]]), "Time": np.array([[0, 1, 2]])})
        cycles[0, 2] = ("discharge", {"Capacity": np.array([[1.8]]), "Time": np.array([[0, 1, 2, 3]]), "LateField": np.array([[1]])})
        path = self.root / filename
        savemat(path, {"arbitrary_root": {"cycle": cycles}, "matrix": np.zeros((2, 4))})
        return path

    def test_recursive_discovery_filters_archive_metadata(self) -> None:
        nested = self.root / "nested"
        nested.mkdir()
        (nested / "example.MAT").touch()
        (nested / "._example.mat").touch()
        meta = self.root / "__MACOSX"
        meta.mkdir()
        (meta / "example.mat").touch()
        self.assertEqual(discover_mat_files(self.root), [nested / "example.MAT"])
        with self.assertRaisesRegex(DatasetError, "directory"):
            discover_mat_files(self.root / "absent")

    def test_structure_preserves_dimensions_and_counts_all_cycle_types(self) -> None:
        path = self.make_fixture()
        result = inspect_file(path, sample_items=1)
        self.assertEqual(result["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        matrix = next(v for v in result["variables"] if v["name"] == "matrix")
        self.assertEqual(matrix["shape"], [2, 4])
        self.assertEqual(result["structure"]["matrix"]["dtype"], "float64")
        container = result["cycle_discovery"]["containers"][0]
        self.assertEqual(container["path"], "arbitrary_root.cycle")
        self.assertEqual(container["container_shape"], [1, 3])
        self.assertEqual(container["observed_entry_count"], 3)
        discharge = container["by_observed_type"]["discharge"]
        self.assertEqual(discharge["count"], 2)
        self.assertEqual(discharge["data_fields"]["Time"]["distinct_shape_count"], 2)
        self.assertEqual(discharge["data_fields"]["LateField"]["present_entries"], 1)
        json.dumps(result, allow_nan=False)

    def test_singleton_cycle_and_cell_wrappers(self) -> None:
        cell = np.empty((1, 1), dtype=object)
        cell[0, 0] = {"cycles": {"data": {"Time": np.array([[0]])}}}
        path = self.root / "singleton.mat"
        savemat(path, {"wrapper": cell})
        result = inspect_file(path)
        container = result["cycle_discovery"]["containers"][0]
        self.assertEqual(container["observed_entry_count"], 1)
        self.assertEqual(container["by_observed_type"]["<missing-or-nontext-type>"]["count"], 1)

    def test_cycle_search_checks_later_struct_entries(self) -> None:
        wrappers = np.empty((1, 2), dtype=object)
        wrappers[0, 0] = {"unrelated": np.ones((1, 1))}
        wrappers[0, 1] = {"cycle": {"type": "impedance", "data": {"complex_signal": np.array([[1+2j]])}}}
        path = self.root / "later.mat"
        savemat(path, {"wrappers": wrappers})
        result = inspect_file(path, sample_items=1)
        found = result["cycle_discovery"]["containers"]
        self.assertEqual(len(found), 1)
        self.assertIn("impedance", found[0]["by_observed_type"])

    def test_empty_cycles_and_unknown_structure(self) -> None:
        path = self.root / "empty.mat"
        savemat(path, {"root": {"cycle": np.array([])}})
        self.assertEqual(inspect_file(path)["cycle_discovery"]["containers"][0]["observed_entry_count"], 0)
        plain = self.root / "plain.mat"
        savemat(plain, {"readings": np.ones((1, 4))})
        self.assertTrue(inspect_file(plain)["warnings"])

    def test_truncation_is_explicit(self) -> None:
        path = self.make_fixture()
        _, content = read_mat_file(path)
        summary = describe(content["arbitrary_root"], max_depth=1, sample_items=1)
        self.assertEqual(summary["sampled_entries"][0]["truncated"], "max_depth reached")

    def test_v73_and_corrupt_files_have_actionable_errors(self) -> None:
        v73 = self.root / "v73.mat"
        v73.write_bytes(b"MATLAB 7.3 MAT-file" + b" " * 128)
        with self.assertRaisesRegex(DatasetError, "v7.3/HDF5"):
            read_mat_file(v73)
        corrupt = self.root / "corrupt.mat"
        corrupt.write_bytes(b"not a MAT file")
        with self.assertRaisesRegex(DatasetError, "Cannot read MATLAB"):
            read_mat_file(corrupt)

    def test_missing_data_cli_returns_truthful_report(self) -> None:
        output = self.root / "report.json"
        with self.assertLogs(level="ERROR"):
            code = main(["--raw-dir", str(self.root), "--output", str(output)])
        report = json.loads(output.read_text())
        self.assertEqual(code, 2)
        self.assertEqual(report["status"], "no_data")
        self.assertEqual(report["file_count"], 0)
        self.assertNotIn("metrics", report)

    def test_cli_success_and_no_overwrite(self) -> None:
        self.make_fixture()
        output = self.root / "report.json"
        self.assertEqual(main(["--raw-dir", str(self.root), "--output", str(output)]), 0)
        original = output.read_bytes()
        with self.assertLogs(level="ERROR"):
            self.assertEqual(main(["--raw-dir", str(self.root), "--output", str(output)]), 1)
        self.assertEqual(output.read_bytes(), original)
        self.assertEqual(json.loads(original)["inspected_count"], 1)

    def test_corrupt_file_does_not_hide_successful_file(self) -> None:
        self.make_fixture()
        (self.root / "bad.mat").write_text("corrupt")
        stream = io.StringIO()
        with redirect_stdout(stream), self.assertLogs(level="ERROR"):
            code = main(["--raw-dir", str(self.root)])
        report = json.loads(stream.getvalue())
        self.assertEqual(code, 1)
        self.assertEqual(report["file_count"], 2)
        self.assertEqual(report["inspected_count"], 1)

    def test_invalid_display_limits_are_rejected(self) -> None:
        for args in (["--max-depth", "0"], ["--sample-items", "-1"]):
            with self.assertRaises(SystemExit) as caught:
                main(args)
            self.assertEqual(caught.exception.code, 2)


class ConfigurationTests(unittest.TestCase):
    def test_paths_and_serializable_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            config = ProjectConfig(paths=ProjectPaths(Path(folder)))
            self.assertEqual(config.paths.raw, Path(folder).resolve() / "data" / "raw")
            self.assertFalse(config.paths.raw.exists())
            self.assertAlmostEqual(config.eol_capacity_ah, 1.4)
            json.dumps(config.to_dict(), allow_nan=False)

    def test_invalid_capacity_threshold_and_seed(self) -> None:
        for kwargs in (
            {"nominal_capacity_ah": 0}, {"nominal_capacity_ah": float("nan")},
            {"nominal_capacity_ah": float("inf")}, {"eol_soh_fraction": 70},
            {"eol_soh_fraction": 1}, {"eol_soh_fraction": float("nan")},
            {"random_seed": -1}, {"random_seed": 1.5}, {"random_seed": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ProjectConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()

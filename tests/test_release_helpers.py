from __future__ import annotations

import importlib.util
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from utils.path_resolution import resolve_cv2024_or_kvasir_path  # noqa: E402


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PathResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cv = Path("/cv2024")
        self.kv = Path("/kvasir/labelled_images")

    def test_cv2024_backslash_path(self) -> None:
        out = resolve_cv2024_or_kvasir_path(
            "training\\Normal\\KVASIR\\frame.jpg",
            cv2024_root=self.cv,
            kvasir_data_root=self.kv,
        )
        self.assertEqual(out, self.cv / "training/Normal/KVASIR/frame.jpg")

    def test_kvasir_split_1_path(self) -> None:
        out = resolve_cv2024_or_kvasir_path(
            "kvasir_capsule_split_1/ulcer/img001.jpg",
            cv2024_root=self.cv,
            kvasir_data_root=self.kv,
        )
        self.assertEqual(out, self.kv / "ulcer/img001.jpg")

    def test_historical_labelled_images_prefixes(self) -> None:
        for rel in (
            "kvasir_capsule/labelled_images/normal_clean_mucosa/a.jpg",
            "labelled_images/normal_clean_mucosa/a.jpg",
        ):
            with self.subTest(rel=rel):
                out = resolve_cv2024_or_kvasir_path(
                    rel,
                    cv2024_root=self.cv,
                    kvasir_data_root=self.kv,
                )
                self.assertEqual(out, self.kv / "normal_clean_mucosa/a.jpg")

    def test_absolute_path_passthrough(self) -> None:
        out = resolve_cv2024_or_kvasir_path(
            "/already/absolute.jpg",
            cv2024_root=self.cv,
            kvasir_data_root=self.kv,
        )
        self.assertEqual(out, Path("/already/absolute.jpg"))


class AggregateLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.aggregate = load_module(
            SRC / "counterfactual/08_aggregate_r4.py",
            "aggregate_r4_under_test",
        )

    def test_missing_json_returns_none(self) -> None:
        self.assertIsNone(self.aggregate.load("/definitely/not/present.json"))

    def test_invalid_json_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{not-json")
            with self.assertRaises(ValueError):
                self.aggregate.load(str(path))

    def test_valid_json_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ok.json"
            path.write_text(json.dumps({"runs": []}))
            self.assertEqual(self.aggregate.load(str(path)), {"runs": []})


class BountyArtifactTests(unittest.TestCase):
    def test_public_reports_do_not_reference_local_artifact_paths(self) -> None:
        report_paths = sorted((ROOT / "bounty-artifacts").glob("*/REPORT.md"))
        self.assertGreater(len(report_paths), 0)

        local_path = re.compile(r"(?<![\w/])(?:/home/|~/|[A-Za-z]:[\\/])")
        offenders: list[str] = []
        for path in report_paths:
            for lineno, line in enumerate(path.read_text().splitlines(), start=1):
                if local_path.search(line):
                    offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {line}")

        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()

"""Dataset validation utilities for handwashing training pipeline."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Tuple, List, Dict, Optional

import config as cfg

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "datasets" / "raw"
PROCESSED_DIR = REPO_ROOT / "datasets" / "processed"

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")

LABEL_TOKENS = {
    "step1": 1,
    "step2": 2,
    "step3": 3,
    "step4": 4,
    "step5": 5,
    "step6": 6,
    "other": 0,
}


@dataclass
class ValidationReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def extend(self, other: "ValidationReport", prefix: str) -> None:
        for err in other.errors:
            self.errors.append(f"{prefix}: {err}")
        for warn in other.warnings:
            self.warnings.append(f"{prefix}: {warn}")


def _is_archive(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in ARCHIVE_SUFFIXES)


def _iter_files(root: Path, predicate, max_hits: int) -> List[Path]:
    hits: List[Path] = []
    if not root.exists():
        return hits
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if predicate(path):
            hits.append(path)
            if len(hits) >= max_hits:
                break
    return hits


def _collect_archives(root: Path, max_hits: int = 10) -> List[Path]:
    return _iter_files(root, _is_archive, max_hits)


def infer_label_from_path(path: Path, num_classes: int) -> Optional[int]:
    for part in reversed(path.parts):
        if part.isdigit():
            class_id = int(part)
            if 0 <= class_id < num_classes:
                return class_id
    text = str(path).lower()
    for token, idx in LABEL_TOKENS.items():
        if token in text:
            return idx
    return None


def _csv_sample_rows(csv_path: Path, max_rows: int) -> Tuple[List[Dict[str, str]], List[str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for i, row in enumerate(reader):
            rows.append(row)
            if i + 1 >= max_rows:
                break
    return rows, fieldnames


def _resolve_path(path_str: str, dataset_dir: Path) -> Path:
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate
    if (dataset_dir / candidate).exists():
        return (dataset_dir / candidate).resolve()
    return (REPO_ROOT / candidate).resolve()


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def validate_raw_kaggle(raw_dir: Path, max_samples: int, strict_archives: bool) -> ValidationReport:
    report = ValidationReport()
    if not raw_dir.exists():
        report.add_error("raw dir missing")
        return report

    kaggle_root = raw_dir / "kaggle-dataset-6classes"
    if not kaggle_root.exists():
        report.add_error("kaggle-dataset-6classes folder missing")
    videos = _iter_files(kaggle_root, lambda p: p.suffix.lower() in VIDEO_EXTS, max_samples)
    if not videos:
        report.add_error("no video files found under kaggle-dataset-6classes")
    for video in videos:
        class_id = infer_label_from_path(video, cfg.NUM_CLASSES)
        if class_id is None:
            report.add_error(f"could not infer label from {video}")

    archives = _collect_archives(raw_dir)
    if archives:
        message = f"archive files still present: {', '.join(p.name for p in archives[:5])}"
        if strict_archives:
            report.add_error(message)
        else:
            report.add_warning(message)

    return report


def _find_pskus_split_csv(raw_dir: Path) -> Optional[Path]:
    csv_path = raw_dir / "statistics-with-locations.csv"
    if csv_path.exists():
        return csv_path
    fallback = REPO_ROOT / "code" / "edgewash" / "dataset-pskus" / "statistics-with-locations.csv"
    if fallback.exists():
        return fallback
    return None


def validate_raw_pskus(raw_dir: Path, max_samples: int, strict_archives: bool) -> ValidationReport:
    report = ValidationReport()
    if not raw_dir.exists():
        report.add_error("raw dir missing")
        return report

    dataset_dirs = [p for p in raw_dir.glob("DataSet*") if p.is_dir()]
    if not dataset_dirs:
        report.add_error("no DataSet* folders found")

    videos = _iter_files(raw_dir, lambda p: p.suffix.lower() in VIDEO_EXTS, max_samples)
    if not videos:
        report.add_error("no video files found")

    annotations = _iter_files(raw_dir, lambda p: p.suffix.lower() == ".json" and "Annotations" in p.parts, 1)
    if not annotations:
        report.add_error("no annotation JSON files found in Annotations/")

    split_csv = _find_pskus_split_csv(raw_dir)
    if split_csv is None:
        report.add_error("statistics-with-locations.csv not found (needed for split)")

    archives = _collect_archives(raw_dir)
    if archives:
        message = f"archive files still present: {', '.join(p.name for p in archives[:5])}"
        if strict_archives:
            report.add_error(message)
        else:
            report.add_warning(message)

    return report


def validate_raw_metc(raw_dir: Path, max_samples: int, strict_archives: bool) -> ValidationReport:
    report = ValidationReport()
    if not raw_dir.exists():
        report.add_error("raw dir missing")
        return report

    interface_dirs = [p for p in raw_dir.glob("Interface_number_*") if p.is_dir()]
    if not interface_dirs:
        report.add_error("no Interface_number_* folders found")

    videos = _iter_files(raw_dir, lambda p: p.suffix.lower() in VIDEO_EXTS, max_samples)
    if not videos:
        report.add_error("no video files found")

    annotations = _iter_files(raw_dir, lambda p: p.suffix.lower() == ".json" and "Annotations" in p.parts, 1)
    if not annotations:
        report.add_error("no annotation JSON files found in Annotations/")

    archives = _collect_archives(raw_dir)
    if archives:
        message = f"archive files still present: {', '.join(p.name for p in archives[:5])}"
        if strict_archives:
            report.add_error(message)
        else:
            report.add_warning(message)

    return report


def validate_raw_synthetic(raw_dir: Path, max_samples: int, strict_archives: bool) -> ValidationReport:
    report = ValidationReport()
    if not raw_dir.exists():
        report.add_error("raw dir missing")
        return report

    pngs = _iter_files(
        raw_dir,
        lambda p: p.suffix.lower() == ".png" and "gesture" in "".join(p.parts).lower() and "rgb" in "".join(p.parts).lower(),
        max_samples,
    )
    if not pngs:
        report.add_error("no RGB gesture PNG files found")

    archives = _collect_archives(raw_dir)
    if archives:
        message = f"archive files still present: {', '.join(p.name for p in archives[:5])}"
        if strict_archives:
            report.add_error(message)
        else:
            report.add_warning(message)

    return report


def validate_processed_dataset(name: str, dataset_dir: Path, max_rows: int) -> ValidationReport:
    report = ValidationReport()

    if not dataset_dir.exists():
        report.add_error("processed dataset dir missing")
        return report

    frames_dir = dataset_dir / "frames"
    if not frames_dir.exists():
        report.add_error("frames/ directory missing")

    for split in ("train", "val", "test"):
        csv_path = dataset_dir / f"{split}.csv"
        if not csv_path.exists():
            report.add_error(f"missing {split}.csv")
            continue
        rows, fieldnames = _csv_sample_rows(csv_path, max_rows=max_rows)
        required = {"frame_path", "class_id", "video_id", "frame_idx"}
        missing = required - set(fieldnames)
        if missing:
            report.add_error(f"{split}.csv missing columns: {sorted(missing)}")
            continue
        if not rows:
            report.add_error(f"{split}.csv has no rows")
            continue

        class_ids = set()
        for row in rows:
            frame_path = _resolve_path(row["frame_path"], dataset_dir)
            if not frame_path.exists():
                report.add_error(f"missing frame file: {frame_path}")
                continue
            if frame_path.suffix.lower() not in IMAGE_EXTS:
                report.add_warning(f"unexpected frame extension: {frame_path.name}")
            if not _is_under(frame_path, dataset_dir):
                report.add_warning(f"frame path outside dataset dir: {frame_path}")
            try:
                class_id = int(row["class_id"])
            except (ValueError, TypeError):
                report.add_error(f"invalid class_id in {split}.csv: {row['class_id']}")
                continue
            if not (0 <= class_id < cfg.NUM_CLASSES):
                report.add_error(f"class_id out of range in {split}.csv: {class_id}")
            class_ids.add(class_id)
            try:
                frame_idx = int(row["frame_idx"])
                if frame_idx < 0:
                    report.add_error(f"negative frame_idx in {split}.csv: {frame_idx}")
            except (ValueError, TypeError):
                report.add_error(f"invalid frame_idx in {split}.csv: {row['frame_idx']}")
            if not row.get("video_id"):
                report.add_error(f"missing video_id in {split}.csv")

        if len(class_ids) <= 1:
            report.add_warning(f"{split}.csv contains only {len(class_ids)} class(es) in sample")

    return report


def validate_all_datasets(
    datasets: Optional[List[str]] = None,
    validate_raw: bool = True,
    validate_processed: bool = True,
    max_samples: int = 5,
    max_rows: int = 25,
    strict_archives: bool = True,
) -> ValidationReport:
    report = ValidationReport()
    dataset_names = datasets or list(cfg.DATASETS.keys())

    for name in dataset_names:
        raw_dir = RAW_DIR / name
        if validate_raw:
            if name == "kaggle":
                sub = validate_raw_kaggle(raw_dir, max_samples, strict_archives)
            elif name == "pskus":
                sub = validate_raw_pskus(raw_dir, max_samples, strict_archives)
            elif name == "metc":
                sub = validate_raw_metc(raw_dir, max_samples, strict_archives)
            elif name == "synthetic_blender_rozakar":
                sub = validate_raw_synthetic(raw_dir, max_samples, strict_archives)
            else:
                sub = ValidationReport(errors=["unknown dataset name"])
            report.extend(sub, f"raw/{name}")

        if validate_processed:
            processed_dir = PROCESSED_DIR / name
            sub = validate_processed_dataset(name, processed_dir, max_rows)
            report.extend(sub, f"processed/{name}")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate raw/processed datasets")
    parser.add_argument("--datasets", nargs="*", default=None, help="Datasets to validate")
    parser.add_argument("--raw", action="store_true", help="Validate raw datasets")
    parser.add_argument("--processed", action="store_true", help="Validate processed datasets")
    parser.add_argument("--max-samples", type=int, default=5, help="Max raw files to sample")
    parser.add_argument("--max-rows", type=int, default=25, help="Max CSV rows to sample")
    parser.add_argument("--allow-archives", action="store_true", help="Do not error on leftover archives")
    args = parser.parse_args()

    validate_raw = args.raw or not args.processed
    validate_processed = args.processed or not args.raw
    report = validate_all_datasets(
        datasets=args.datasets,
        validate_raw=validate_raw,
        validate_processed=validate_processed,
        max_samples=args.max_samples,
        max_rows=args.max_rows,
        strict_archives=not args.allow_archives,
    )

    if report.errors:
        print("Validation errors:")
        for err in report.errors:
            print("-", err)
    if report.warnings:
        print("Validation warnings:")
        for warn in report.warnings:
            print("-", warn)

    if report.errors:
        return 1
    print("Validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

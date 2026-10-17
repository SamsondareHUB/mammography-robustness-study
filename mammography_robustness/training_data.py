"""Validate image-level training records and patient-level split assignments."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


REQUIRED_COLUMNS = frozenset({"patient_id", "case_id", "breast_side", "view", "label", "split"})
SPLITS = ("train", "val", "test")
VIEWS = frozenset({"CC", "MLO"})
SIDES = frozenset({"L", "R"})


@dataclass(frozen=True)
class ImageRecord:
    patient_id: str
    case_id: str
    breast_side: str
    view: str
    label: int
    split: str


@dataclass(frozen=True)
class DataSummary:
    image_counts: dict[str, int]
    patient_counts: dict[str, int]
    breast_counts: dict[str, int]
    incomplete_breasts: int
    conflicting_label_breasts: int


def read_manifest(path: str | Path) -> list[ImageRecord]:
    """Read the six-column low-energy image manifest from dataset preparation."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            raise ValueError(f"Manifest must contain: {', '.join(sorted(REQUIRED_COLUMNS))}")
        records = []
        used_case_ids: set[str] = set()
        for lineno, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"Extra columns in manifest row {lineno}")
            values = {key: row[key].strip() if isinstance(row[key], str) else "" for key in REQUIRED_COLUMNS}
            if not all(values.values()):
                raise ValueError(f"Missing field in manifest row {lineno}")
            if values["label"] not in {"0", "1"}:
                raise ValueError(f"Invalid binary label on manifest row {lineno}")
            if values["split"] not in SPLITS:
                raise ValueError(f"Invalid split on manifest row {lineno}")
            if values["breast_side"] not in SIDES or values["view"] not in VIEWS:
                raise ValueError(f"Invalid side or view on manifest row {lineno}")
            case_id = values["case_id"]
            if case_id in used_case_ids:
                raise ValueError(f"Duplicate case_id in manifest: {case_id}")
            if "/" in case_id or "\\" in case_id or case_id in {".", ".."}:
                raise ValueError(f"Invalid case_id: {case_id}")
            used_case_ids.add(case_id)
            records.append(ImageRecord(
                values["patient_id"], case_id, values["breast_side"],
                values["view"], int(values["label"]), values["split"],
            ))
    if not records:
        raise ValueError("Manifest contains no image records")
    return records


def read_patient_splits(path: str | Path) -> dict[str, set[str]]:
    """Load the patient assignments from the companion JSON split file."""
    with Path(path).open(encoding="utf-8") as handle:
        contents = json.load(handle)
    if not isinstance(contents, dict) or set(contents) != set(SPLITS):
        raise ValueError("Split JSON must contain train, val and test arrays")
    seen: set[str] = set()
    result: dict[str, set[str]] = {}
    for split in SPLITS:
        patients = contents[split]
        if not isinstance(patients, list):
            raise ValueError(f"{split} must be an array of patient IDs")
        ids: set[str] = set()
        for patient in patients:
            if isinstance(patient, bool) or not isinstance(patient, (str, int)):
                raise ValueError(f"Invalid patient ID in {split}")
            value = str(patient)
            if not value.strip() or value != value.strip() or value in ids:
                raise ValueError(f"Duplicate or invalid patient ID in {split}")
            ids.add(value)
        if seen & ids:
            raise ValueError(f"Patient appears in multiple JSON splits: {sorted(seen & ids)[:3]}")
        seen.update(ids)
        result[split] = ids
    return result


def validate_records(records: list[ImageRecord], splits: dict[str, set[str]]) -> DataSummary:
    """Check split agreement and breast-level annotations before model training."""
    if not records:
        raise ValueError("No image records to validate")
    patient_assignments: dict[str, str] = {}
    breasts: dict[tuple[str, str], list[ImageRecord]] = defaultdict(list)
    for row in records:
        earlier = patient_assignments.setdefault(row.patient_id, row.split)
        if earlier != row.split:
            raise ValueError(f"Patient {row.patient_id} spans multiple manifest splits")
        breasts[(row.patient_id, row.breast_side)].append(row)

    json_assignments = {patient: split for split, ids in splits.items() for patient in ids}
    missing = set(patient_assignments) ^ set(json_assignments)
    if missing:
        raise ValueError(f"Manifest and JSON list different patients ({len(missing)} unmatched)")
    mismatches = [patient for patient, split in patient_assignments.items()
                  if json_assignments[patient] != split]
    if mismatches:
        raise ValueError(
            f"Manifest and JSON disagree on {len(mismatches)} patient assignments "
            f"(examples: {', '.join(sorted(mismatches)[:5])})"
        )

    incomplete = 0
    label_conflicts = 0
    for (patient, side), breast_rows in breasts.items():
        views = [row.view for row in breast_rows]
        if len(views) != len(set(views)):
            raise ValueError(f"Duplicate view for patient {patient}, breast {side}")
        if len({row.label for row in breast_rows}) != 1:
            label_conflicts += 1
        if set(views) != VIEWS:
            incomplete += 1

    return DataSummary(
        image_counts={s: sum(row.split == s for row in records) for s in SPLITS},
        patient_counts={s: sum(split == s for split in patient_assignments.values()) for s in SPLITS},
        breast_counts={s: sum(rows[0].split == s for rows in breasts.values()) for s in SPLITS},
        incomplete_breasts=incomplete,
        conflicting_label_breasts=label_conflicts,
    )


def index_images(root: str | Path, records: list[ImageRecord]) -> dict[str, Path]:
    """Match case IDs to image filename stems, rejecting missing or ambiguous files."""
    directory = Path(root)
    if not directory.is_dir():
        raise ValueError(f"Image directory does not exist: {directory}")
    available: dict[str, Path] = {}
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            if path.stem in available:
                raise ValueError(f"Multiple image files match {path.stem}")
            available[path.stem] = path
    missing = sorted({row.case_id for row in records} - available.keys())
    if missing:
        raise ValueError(f"Missing {len(missing)} image files (examples: {missing[:5]})")
    return {record.case_id: available[record.case_id] for record in records}

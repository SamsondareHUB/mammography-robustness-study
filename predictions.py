"""Validate paired mammography predictions and write evaluation-ready outputs."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Iterable


IMAGE_COLUMNS = (
    "case_id", "patient_id", "breast_side", "view", "label", "score",
    "model", "degradation", "severity", "seed",
)
BREAST_COLUMNS = (
    "patient_id", "breast_side", "label", "score", "model",
    "degradation", "severity", "seed",
)
SEVERITIES = {"low", "medium", "high"}
SEVERITY_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}


@dataclass(frozen=True)
class ImagePrediction:
    """Prediction for one low-energy image.

    ``image_id`` is a stable internal image identifier, deliberately excluded
    from the agreed ten-column CSV. ``case_id`` is the source case identifier;
    ``patient_id`` and ``breast_side`` identify the evaluation unit.
    """

    case_id: str
    patient_id: str
    breast_side: str
    image_id: str
    view: str
    label: int
    score: float
    model: str
    degradation: str
    severity: str
    seed: int


@dataclass(frozen=True)
class BreastPrediction:
    """Maximum malignancy score from the CC and MLO views of one breast."""

    patient_id: str
    breast_side: str
    label: int
    score: float
    model: str
    degradation: str
    severity: str
    seed: int


@dataclass(frozen=True)
class SplitImage:
    """One manifest image and its assigned partition (train/validation/test)."""

    patient_id: str
    breast_side: str
    image_id: str
    split: str


def _nonempty(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string")


def _validate_common(row: ImagePrediction | BreastPrediction) -> None:
    for name in ("patient_id", "model", "degradation", "severity"):
        _nonempty(getattr(row, name), name)
    if row.breast_side not in {"L", "R"}:
        raise ValueError("breast_side must be L or R")
    if type(row.label) is not int or row.label not in (0, 1):
        raise ValueError("label must be integer 0 or 1")
    if isinstance(row.score, bool) or not isinstance(row.score, Real):
        raise ValueError("score must be a real number between 0 and 1")
    if not math.isfinite(row.score) or not 0 <= row.score <= 1:
        raise ValueError("score must be finite and between 0 and 1")
    if isinstance(row.seed, bool) or not isinstance(row.seed, Integral) or row.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if row.degradation == "clean":
        if row.severity != "none":
            raise ValueError("clean predictions require severity='none'")
    elif row.severity not in SEVERITIES:
        raise ValueError("degraded predictions require low/medium/high severity")


def _condition(row: ImagePrediction | BreastPrediction) -> tuple[str, str]:
    return row.degradation, row.severity


def _check_coverage(groups: dict, expected_conditions: Iterable[tuple[str, str]] | None) -> None:
    """Every breast/model/seed must have matching conditions and a clean baseline."""
    per_run: dict[tuple[str, int], dict[tuple[str, str], set[tuple[str, str]]]] = defaultdict(lambda: defaultdict(set))
    for patient, side, model, degradation, severity, seed in groups:
        per_run[(model, seed)][(patient, side)].add((degradation, severity))
    expected = set(expected_conditions) if expected_conditions is not None else None
    if expected is not None:
        expected.add(("clean", "none"))
    reference = None
    for run, breasts in per_run.items():
        signature = {breast: frozenset(conds) for breast, conds in breasts.items()}
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError(f"Mismatched breast or condition coverage for {run}")
        if any(("clean", "none") not in conditions for conditions in breasts.values()):
            raise ValueError(f"Missing clean baseline for {run}")
        if expected is not None:
            for breast, conditions in breasts.items():
                if conditions != expected:
                    raise ValueError(f"Unexpected or missing conditions for {breast} in {run}")
        else:
            first = next(iter(signature.values()))
            if any(conditions != first for conditions in signature.values()):
                raise ValueError(f"Missing conditions within model/seed {run}")


def _image_sort(row: ImagePrediction) -> tuple:
    return (row.model, int(row.seed), row.patient_id, row.breast_side,
            row.degradation != "clean", row.degradation,
            SEVERITY_ORDER[row.severity], row.view, row.image_id)


def _breast_sort(row: BreastPrediction) -> tuple:
    return (row.model, int(row.seed), row.patient_id, row.breast_side,
            row.degradation != "clean", row.degradation, SEVERITY_ORDER[row.severity])


def _validated_images(
    records: Iterable[ImagePrediction], *,
    expected_conditions: Iterable[tuple[str, str]] | None = None,
) -> tuple[list[ImagePrediction], dict[tuple[str, str, str, str, str, int], list[ImagePrediction]]]:
    rows = list(records)
    if not rows:
        raise ValueError("No image predictions supplied")
    groups: dict[tuple[str, str, str, str, str, int], list[ImagePrediction]] = defaultdict(list)
    image_owner: dict[str, tuple[str, str, str, str]] = {}
    breast_labels: dict[tuple[str, str], int] = {}
    case_owner: dict[str, tuple[str, str]] = {}
    image_by_key: set[tuple[str, str, str, str, int]] = set()

    for row in rows:
        if not isinstance(row, ImagePrediction):
            raise TypeError("Expected ImagePrediction")
        _validate_common(row)
        for name in ("case_id", "image_id"):
            _nonempty(getattr(row, name), name)
        if row.view not in {"CC", "MLO"}:
            raise ValueError("view must be CC or MLO")
        breast = (row.patient_id, row.breast_side)
        if breast in breast_labels and breast_labels[breast] != row.label:
            raise ValueError(f"Conflicting breast labels for {breast}")
        breast_labels[breast] = row.label
        owner = (row.case_id, row.patient_id, row.breast_side, row.view)
        if row.image_id in image_owner and image_owner[row.image_id] != owner:
            raise ValueError(f"Inconsistent image_id ownership: {row.image_id}")
        image_owner[row.image_id] = owner
        # One source case may legitimately contain images of both breasts.
        case_key = (row.case_id, row.patient_id)
        if row.case_id in case_owner and case_owner[row.case_id][1] != row.patient_id:
            raise ValueError(f"case_id spans different patients: {row.case_id}")
        case_owner[row.case_id] = case_key
        key = (row.image_id, row.model, row.degradation, row.severity, int(row.seed))
        if key in image_by_key:
            raise ValueError(f"Duplicate image prediction: {key}")
        image_by_key.add(key)
        groups[(row.patient_id, row.breast_side, row.model,
                row.degradation, row.severity, int(row.seed))].append(row)

    views_by_group: dict[tuple[str, str, str, str, str, int], dict[str, str]] = {}
    for key, views in groups.items():
        mapping = {r.view: r.image_id for r in views}
        if len(views) != 2 or set(mapping) != {"CC", "MLO"}:
            raise ValueError(f"Expected one CC and one MLO image for {key}")
        views_by_group[key] = mapping

    reference_image_sets: dict[tuple[str, str], dict[str, str]] = {}
    for (patient, side, model, degr, sev, seed), mapping in views_by_group.items():
        baseline = (patient, side, model, "clean", "none", seed)
        if baseline not in views_by_group:
            raise ValueError(f"Missing clean baseline for {baseline}")
        if mapping != views_by_group[baseline]:
            raise ValueError(f"Clean and degraded image views differ for {patient}/{side}")
        breast = (patient, side)
        if breast in reference_image_sets and reference_image_sets[breast] != mapping:
            raise ValueError(f"Cross-model image views differ for {breast}")
        reference_image_sets[breast] = mapping

    _check_coverage(groups, expected_conditions)
    return sorted(rows, key=_image_sort), groups


def aggregate_breast_predictions(
    records: Iterable[ImagePrediction], *,
    expected_conditions: Iterable[tuple[str, str]] | None = None,
) -> list[BreastPrediction]:
    """Aggregate CC/MLO using the protocol's fixed maximum-score rule."""
    _, groups = _validated_images(records, expected_conditions=expected_conditions)
    out = []
    for (patient, side, model, degradation, severity, seed), rows in groups.items():
        out.append(BreastPrediction(
            patient, side, rows[0].label, max(float(r.score) for r in rows),
            model, degradation, severity, seed,
        ))
    return sorted(out, key=_breast_sort)


def _write_csv(rows: list, columns: tuple[str, ...], path: str | Path) -> None:
    # All validation is completed before the output is opened.
    target = Path(path)
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: float(row.score) if col == "score" else getattr(row, col)
                             for col in columns})


def write_image_csv(
    records: Iterable[ImagePrediction], path: str | Path, *,
    expected_conditions: Iterable[tuple[str, str]] | None = None,
) -> None:
    """Write the agreed ten-column image-level table."""
    rows, _ = _validated_images(records, expected_conditions=expected_conditions)
    _write_csv(rows, IMAGE_COLUMNS, path)


def write_breast_csv(
    records: Iterable[BreastPrediction], path: str | Path, *,
    expected_conditions: Iterable[tuple[str, str]] | None = None,
) -> None:
    """Write the agreed eight-column breast-level table."""
    rows = list(records)
    if not rows:
        raise ValueError("No breast predictions supplied")
    groups = {}
    breast_labels = {}
    for row in rows:
        if not isinstance(row, BreastPrediction):
            raise TypeError("Expected BreastPrediction")
        _validate_common(row)
        breast = (row.patient_id, row.breast_side)
        if breast in breast_labels and breast_labels[breast] != row.label:
            raise ValueError(f"Conflicting breast labels for {breast}")
        breast_labels[breast] = row.label
        key = (row.patient_id, row.breast_side, row.model,
               row.degradation, row.severity, int(row.seed))
        if key in groups:
            raise ValueError(f"Duplicate breast prediction: {key}")
        groups[key] = [row]
    _check_coverage(groups, expected_conditions)
    _write_csv(sorted(rows, key=_breast_sort), BREAST_COLUMNS, path)


def summarize_splits(records: Iterable[SplitImage]) -> dict[str, dict[str, int]]:
    """Count distinct patients, breasts and images; reject patient leakage."""
    rows = list(records)
    if not rows:
        raise ValueError("No split manifest images supplied")
    patients: dict[str, str] = {}
    images: dict[str, tuple[str, str, str]] = {}
    counts: dict[str, dict[str, set]] = defaultdict(lambda: {
        "patients": set(), "breasts": set(), "images": set()
    })
    for row in rows:
        if not isinstance(row, SplitImage):
            raise TypeError("Expected SplitImage")
        for name in ("patient_id", "image_id"):
            _nonempty(getattr(row, name), name)
        if row.breast_side not in {"L", "R"}:
            raise ValueError("breast_side must be L or R")
        if row.split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation or test")
        if row.patient_id in patients and patients[row.patient_id] != row.split:
            raise ValueError(f"Patient appears in multiple splits: {row.patient_id}")
        patients[row.patient_id] = row.split
        owner = (row.patient_id, row.breast_side, row.split)
        if row.image_id in images and images[row.image_id] != owner:
            raise ValueError(f"Image appears under multiple assignments: {row.image_id}")
        images[row.image_id] = owner
        counts[row.split]["patients"].add(row.patient_id)
        counts[row.split]["breasts"].add((row.patient_id, row.breast_side))
        counts[row.split]["images"].add(row.image_id)
    return {
        split: {key: len(values) for key, values in group.items()}
        for split, group in sorted(counts.items())
    }


def write_split_summary_json(records: Iterable[SplitImage], path: str | Path) -> None:
    """Write counts only after a patient-disjointness check succeeds."""
    counts = summarize_splits(records)
    Path(path).write_text(
        json.dumps({"patient_disjoint": True, "counts": counts}, indent=2) + "\n",
        encoding="utf-8",
    )


def export_prediction_tables(
    records: Iterable[ImagePrediction],
    image_path: str | Path,
    breast_path: str | Path, *,
    expected_conditions: Iterable[tuple[str, str]] | None = None,
) -> None:
    """Validate once, then export both tables from the same image predictions."""
    if Path(image_path).resolve() == Path(breast_path).resolve():
        raise ValueError("Image and breast outputs must have different paths")
    rows = list(records)
    expected = tuple(expected_conditions) if expected_conditions is not None else None
    aggregated = aggregate_breast_predictions(rows, expected_conditions=expected)
    image_target, breast_target = Path(image_path), Path(breast_path)
    for target in (image_target, breast_target):
        if not target.parent.is_dir():
            raise FileNotFoundError(f"Output directory does not exist: {target.parent}")

    # Stage both complete tables before replacing either destination. The two
    # final replacements cannot be made atomic as a pair across filesystems.
    staged: list[Path] = []
    try:
        for target in (image_target, breast_target):
            fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
            os.close(fd)
            staged.append(Path(name))
        write_image_csv(rows, staged[0], expected_conditions=expected)
        write_breast_csv(aggregated, staged[1], expected_conditions=expected)
        os.replace(staged[0], image_target)
        os.replace(staged[1], breast_target)
    finally:
        for temporary in staged:
            temporary.unlink(missing_ok=True)

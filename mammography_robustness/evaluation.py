"""Compute AUC, sensitivity, specificity and F1 from the agreed prediction CSVs.

Run ``python -m mammography_robustness.evaluation --help`` for the command line.
See docs/EVALUATION.md for the protocol and output columns.
"""

from __future__ import annotations

import argparse
import math
import zlib
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from mammography_robustness.predictions import (
    BREAST_COLUMNS, IMAGE_COLUMNS, SEVERITIES, SEVERITY_ORDER,
)


METRICS = ("auc", "sensitivity", "specificity", "f1")
UNIT_COLUMNS = {
    "breast": ("patient_id", "breast_side"),
    "image": ("patient_id", "breast_side", "view"),
}
CLEAN = ("clean", "none")
THRESHOLD_METHODS = ("youden",)

RUN_COLUMNS = (
    "level", "model", "seed", "degradation", "severity",
    "threshold", "threshold_source",
    "n_patients", "n_units", "n_positive", "n_negative", "tp", "fp", "tn", "fn",
    *[f"{prefix}{metric}{suffix}"
      for metric in METRICS
      for prefix in ("", "delta_")
      for suffix in ("", "_ci_low", "_ci_high")],
    "n_bootstrap", "confidence",
)
SUMMARY_COLUMNS = (
    "level", "model", "degradation", "severity", "n_seeds", "threshold_source",
    *[f"{prefix}{metric}_{stat}"
      for metric in METRICS
      for prefix in ("", "delta_")
      for stat in ("mean", "sd")],
)

Thresholds = Mapping[tuple[str, int], tuple[float, str]]


# --------------------------------------------------------------------------
# Loading and validation


def _fail(mask: pd.Series, message: str) -> None:
    if mask.any():
        # +2: one for the header line, one for 1-based line numbers.
        raise ValueError(f"{message} (CSV line {int(np.flatnonzero(mask.to_numpy())[0]) + 2})")


def validate_predictions(frame: pd.DataFrame, level: str) -> pd.DataFrame:
    """Check an agreed-format table and return it with typed label/score/seed.

    Every model/seed run must contain a clean baseline, and every condition in
    that run must cover exactly the same evaluation units as the baseline so
    degraded results can be compared with clean results on identical cases.
    """
    if level not in UNIT_COLUMNS:
        raise ValueError(f"level must be one of {sorted(UNIT_COLUMNS)}")
    expected = IMAGE_COLUMNS if level == "image" else BREAST_COLUMNS
    if tuple(frame.columns) != expected:
        raise ValueError(f"{level}-level columns must be exactly {','.join(expected)}")
    if frame.empty:
        raise ValueError("No predictions supplied")
    df = frame.astype(str).reset_index(drop=True)

    for name in set(expected) - {"label", "score", "seed"}:
        values = df[name]
        _fail((values == "") | (values != values.str.strip()), f"{name} must be a non-empty trimmed string")
    _fail(~df["breast_side"].isin(["L", "R"]), "breast_side must be L or R")
    if level == "image":
        _fail(~df["view"].isin(["CC", "MLO"]), "view must be CC or MLO")
    _fail(~df["label"].isin(["0", "1"]), "label must be 0 or 1")
    _fail(~df["seed"].str.fullmatch(r"\d+"), "seed must be a nonnegative integer")
    scores = pd.to_numeric(df["score"], errors="coerce").astype(float)
    _fail(~np.isfinite(scores) | (scores < 0) | (scores > 1), "score must be finite and between 0 and 1")
    clean = df["degradation"] == "clean"
    _fail(clean & (df["severity"] != "none"), "clean predictions require severity='none'")
    _fail(~clean & ~df["severity"].isin(SEVERITIES), "degraded predictions require low/medium/high severity")

    df["label"] = df["label"].astype(int)
    df["score"] = scores
    df["seed"] = df["seed"].astype(int)

    units = list(UNIT_COLUMNS[level])
    _fail(df.duplicated(units + ["model", "seed", "degradation", "severity"]), "Duplicate prediction")
    _fail(df.groupby(units)["label"].transform("nunique") > 1, "Conflicting labels for one evaluation unit")

    for (model, seed), run in df.groupby(["model", "seed"], sort=True):
        coverage = {
            condition: frozenset(group[units].itertuples(index=False, name=None))
            for condition, group in run.groupby(["degradation", "severity"])
        }
        if CLEAN not in coverage:
            raise ValueError(f"Missing clean baseline for model={model} seed={seed}")
        for condition, covered in coverage.items():
            if covered != coverage[CLEAN]:
                raise ValueError(
                    f"{condition} covers different {level}s than the clean baseline "
                    f"for model={model} seed={seed}")
    return df


def load_predictions(path: str | Path) -> tuple[pd.DataFrame, str]:
    """Read an image- or breast-level CSV; the level is detected from its header."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    header = tuple(frame.columns)
    if header == IMAGE_COLUMNS:
        level = "image"
    elif header == BREAST_COLUMNS:
        level = "breast"
    else:
        raise ValueError(f"{path}: header matches neither agreed prediction format")
    return validate_predictions(frame, level), level


# --------------------------------------------------------------------------
# Metrics


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ordered)) + 1]
    ends = np.r_[starts[1:], values.size]
    ranks = np.empty(values.size)
    ranks[order] = np.repeat((starts + ends + 1) / 2, ends - starts)
    return ranks


def roc_auc(labels, scores) -> float:
    """Area under the ROC curve (Mann-Whitney form, ties count one half).

    Returns NaN when only one class is present.
    """
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=float)
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return math.nan
    return float((_average_ranks(s)[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else math.nan


def confusion_counts(labels, scores, threshold: float) -> dict[str, int]:
    """A score at or above ``threshold`` is a positive (malignant) call."""
    y = np.asarray(labels, dtype=bool)
    predicted = np.asarray(scores, dtype=float) >= threshold
    return {
        "tp": int((predicted & y).sum()), "fp": int((predicted & ~y).sum()),
        "tn": int((~predicted & ~y).sum()), "fn": int((~predicted & y).sum()),
    }


def binary_metrics(labels, scores, threshold: float) -> dict[str, float]:
    """AUC (threshold-free) plus sensitivity, specificity and F1 at ``threshold``.

    Undefined values (for example sensitivity with no positive cases) are NaN.
    """
    c = confusion_counts(labels, scores, threshold)
    return {
        "auc": roc_auc(labels, scores),
        "sensitivity": _ratio(c["tp"], c["tp"] + c["fn"]),
        "specificity": _ratio(c["tn"], c["tn"] + c["fp"]),
        "f1": _ratio(2 * c["tp"], 2 * c["tp"] + c["fp"] + c["fn"]),
    }


# --------------------------------------------------------------------------
# Thresholds


def youden_threshold(labels, scores) -> float:
    """Observed score maximising sensitivity + specificity - 1.

    Ties go to the highest threshold, matching the usual
    ``thresholds[argmax(tpr - fpr)]`` idiom on scikit-learn's ROC curve.
    """
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=float)
    positives, negatives = np.sort(s[y]), np.sort(s[~y])
    if positives.size == 0 or negatives.size == 0:
        raise ValueError("Youden threshold needs both positive and negative cases")
    candidates = np.unique(s)
    sensitivity = 1 - np.searchsorted(positives, candidates, side="left") / positives.size
    specificity = np.searchsorted(negatives, candidates, side="left") / negatives.size
    youden = sensitivity + specificity - 1
    return float(candidates[np.flatnonzero(youden == youden.max())[-1]])


def fixed_thresholds(predictions: pd.DataFrame, value: float) -> dict[tuple[str, int], tuple[float, str]]:
    """Use the same threshold for every model/seed run."""
    if not math.isfinite(value):
        raise ValueError("threshold must be finite")
    runs = predictions[["model", "seed"]].drop_duplicates().itertuples(index=False, name=None)
    return {run: (float(value), "fixed") for run in runs}


def validation_thresholds(validation: pd.DataFrame, method: str = "youden") -> dict[tuple[str, int], tuple[float, str]]:
    """Choose one threshold per model/seed from its clean validation predictions.

    The threshold is then frozen and applied to the clean and every degraded
    test condition, so degradations are not given their own tuned threshold.
    """
    if method not in THRESHOLD_METHODS:
        raise ValueError(f"method must be one of {THRESHOLD_METHODS}")
    clean = validation[(validation["degradation"] == "clean") & (validation["severity"] == "none")]
    return {
        (model, int(seed)): (youden_threshold(run["label"], run["score"]), f"{method}_validation")
        for (model, seed), run in clean.groupby(["model", "seed"], sort=True)
    }


# --------------------------------------------------------------------------
# Evaluation


def _condition_order(condition: tuple[str, str]) -> tuple:
    degradation, severity = condition
    return degradation != "clean", degradation, SEVERITY_ORDER[severity]


def _percentile_ci(samples: np.ndarray, confidence: float) -> tuple[np.ndarray, np.ndarray]:
    """Percentile interval per column, ignoring NaN replicates."""
    alpha = (1 - confidence) / 2
    low = np.full(samples.shape[1:], math.nan)
    high = np.full(samples.shape[1:], math.nan)
    valid = ~np.all(np.isnan(samples), axis=0)
    if valid.any():
        low[valid] = np.nanquantile(samples[:, valid], alpha, axis=0)
        high[valid] = np.nanquantile(samples[:, valid], 1 - alpha, axis=0)
    return low, high


def _run_rng(bootstrap_seed: int, model: str, seed: int) -> np.random.Generator:
    # Independent of which other runs are in the file, so adding a model does
    # not change the intervals already reported for another model.
    return np.random.default_rng([bootstrap_seed, int(seed), zlib.crc32(model.encode("utf-8"))])


def _evaluate_run(run: pd.DataFrame, level: str, model: str, seed: int, threshold: float,
                  source: str, n_bootstrap: int, confidence: float, bootstrap_seed: int) -> list[dict]:
    units = list(UNIT_COLUMNS[level])
    conditions = sorted(run.groupby(["degradation", "severity"]).groups, key=_condition_order)
    by_condition = {cond: g.set_index(units).sort_index()
                    for cond, g in run.groupby(["degradation", "severity"])}
    reference = by_condition[CLEAN]
    labels = reference["label"].to_numpy()
    scores = np.vstack([by_condition[c]["score"].reindex(reference.index).to_numpy() for c in conditions])
    clean_index = conditions.index(CLEAN)

    patients = reference.index.get_level_values("patient_id").to_numpy()
    patient_ids, patient_codes = np.unique(patients, return_inverse=True)
    rows_of_patient = [np.flatnonzero(patient_codes == p) for p in range(patient_ids.size)]

    def metric_matrix(index: np.ndarray) -> np.ndarray:
        return np.array([[binary_metrics(labels[index], scores[c, index], threshold)[m] for m in METRICS]
                         for c in range(len(conditions))])

    point = metric_matrix(np.arange(labels.size))
    point_delta = point - point[clean_index]

    ci = np.full((2, 2, len(conditions), len(METRICS)), math.nan)  # [metric|delta][low|high]
    if n_bootstrap > 0:
        # Resample patients, not breasts or images, and reuse each resample for
        # every condition so the delta intervals are paired.
        rng = _run_rng(bootstrap_seed, model, seed)
        samples = np.empty((n_bootstrap, len(conditions), len(METRICS)))
        for b in range(n_bootstrap):
            chosen = rng.integers(0, patient_ids.size, patient_ids.size)
            samples[b] = metric_matrix(np.concatenate([rows_of_patient[p] for p in chosen]))
        ci[0] = _percentile_ci(samples, confidence)
        ci[1] = _percentile_ci(samples - samples[:, [clean_index]], confidence)

    records = []
    for c, (degradation, severity) in enumerate(conditions):
        counts = confusion_counts(labels, scores[c], threshold)
        record = {
            "level": level, "model": model, "seed": seed,
            "degradation": degradation, "severity": severity,
            "threshold": threshold, "threshold_source": source,
            "n_patients": int(patient_ids.size), "n_units": int(labels.size),
            "n_positive": int(labels.sum()), "n_negative": int(labels.size - labels.sum()),
            **counts, "n_bootstrap": n_bootstrap, "confidence": confidence,
        }
        for m, metric in enumerate(METRICS):
            record[metric] = point[c, m]
            record[f"delta_{metric}"] = point_delta[c, m]
            for kind, prefix in enumerate(("", "delta_")):
                record[f"{prefix}{metric}_ci_low"] = ci[kind, 0, c, m]
                record[f"{prefix}{metric}_ci_high"] = ci[kind, 1, c, m]
        records.append(record)
    return records


def evaluate_predictions(predictions: pd.DataFrame, level: str, thresholds: Thresholds, *,
                         n_bootstrap: int = 2000, confidence: float = 0.95,
                         bootstrap_seed: int = 0) -> pd.DataFrame:
    """One row per model, seed and condition, with deltas from that run's clean baseline.

    ``predictions`` must already be validated (see ``load_predictions``).
    """
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be nonnegative")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    records = []
    for (model, seed), run in predictions.groupby(["model", "seed"], sort=True):
        if (model, seed) not in thresholds:
            raise ValueError(f"No threshold for model={model} seed={seed}")
        threshold, source = thresholds[(model, seed)]
        records += _evaluate_run(run, level, model, int(seed), float(threshold), source,
                                 n_bootstrap, confidence, bootstrap_seed)
    return pd.DataFrame.from_records(records, columns=RUN_COLUMNS)


def summarize_over_seeds(per_run: pd.DataFrame) -> pd.DataFrame:
    """Mean and sample SD (NaN for a single seed) of each metric across training seeds."""
    keys = ["level", "model", "degradation", "severity"]
    records = []
    for (level, model, degradation, severity), group in per_run.groupby(keys, sort=False):
        record = {
            "level": level, "model": model, "degradation": degradation, "severity": severity,
            "n_seeds": int(group["seed"].nunique()),
            "threshold_source": ";".join(sorted(group["threshold_source"].unique())),
        }
        for metric in METRICS:
            for column in (metric, f"delta_{metric}"):
                values = group[column].astype(float)
                record[f"{column}_mean"] = values.mean()
                record[f"{column}_sd"] = values.std(ddof=1) if len(values) > 1 else math.nan
        records.append(record)
    return pd.DataFrame.from_records(records, columns=SUMMARY_COLUMNS)


# --------------------------------------------------------------------------
# Command line


def _parse_threshold(text: str) -> float | str:
    if text in THRESHOLD_METHODS:
        return text
    try:
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number or one of {THRESHOLD_METHODS}") from None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m mammography_robustness.evaluation",
        description="Compute AUC, sensitivity, specificity and F1 from an agreed prediction CSV.")
    parser.add_argument("predictions", help="test-set image- or breast-level prediction CSV")
    parser.add_argument("--threshold", type=_parse_threshold, default=0.5,
                        help="fixed threshold (default 0.5) or 'youden' to select it on --validation")
    parser.add_argument("--validation", help="validation-set CSV at the same level, required for 'youden'")
    parser.add_argument("--bootstrap", type=int, default=2000,
                        help="patient-level bootstrap resamples per run; 0 disables CIs (default 2000)")
    parser.add_argument("--confidence", type=float, default=0.95, help="CI level (default 0.95)")
    parser.add_argument("--bootstrap-seed", type=int, default=0, help="random seed for resampling (default 0)")
    parser.add_argument("--out-dir", default="outputs/metrics", help="output directory (default outputs/metrics)")
    args = parser.parse_args(argv)

    predictions, level = load_predictions(args.predictions)
    if isinstance(args.threshold, str):
        if not args.validation:
            parser.error(f"--threshold {args.threshold} requires --validation")
        validation, validation_level = load_predictions(args.validation)
        if validation_level != level:
            parser.error(f"validation CSV is {validation_level}-level but predictions are {level}-level")
        thresholds = validation_thresholds(validation, args.threshold)
    else:
        if args.validation:
            parser.error("--validation is only used with a threshold method such as 'youden'")
        thresholds = fixed_thresholds(predictions, args.threshold)

    per_run = evaluate_predictions(predictions, level, thresholds, n_bootstrap=args.bootstrap,
                                   confidence=args.confidence, bootstrap_seed=args.bootstrap_seed)
    summary = summarize_over_seeds(per_run)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(out_dir / f"{level}_metrics_by_run.csv", index=False, lineterminator="\n")
    summary.to_csv(out_dir / f"{level}_metrics_summary.csv", index=False, lineterminator="\n")
    print(f"Wrote {len(per_run)} run rows and {len(summary)} summary rows to {out_dir}")


if __name__ == "__main__":
    main()

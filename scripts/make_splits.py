"""Build patient-level train/val/test splits for CDD-CESM low-energy images.

Usage:
    python scripts/make_splits.py --annotations path/to/Radiology-manual-annotations.xlsx
    python scripts/make_splits.py --annotations ... --image-dir "path/to/Low energy images of CDD-CESM"

Writes to data/:
    cdd_cesm_low_energy_splits.csv  patient_id, case_id, breast_side, view, label, split
    split_summary.csv               patients / breasts / images per split
    splits.json                     {"train": [patient_ids], "val": [...], "test": [...]}
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
VAL_FRAC, TEST_FRAC = 0.15, 0.15  # 70/15/15 at patient level
EXPECTED_IMAGES, EXPECTED_PATIENTS = 1003, 326
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data"


def load_low_energy(annotations: Path) -> pd.DataFrame:
    df = pd.read_excel(annotations, sheet_name="all")
    df = df[df["Type"].str.strip() == "DM"]  # DM = low-energy, CESM = subtracted
    img = pd.DataFrame({
        "patient_id": df["Patient_ID"].astype(int),
        "case_id": df["Image_name"].str.strip(),
        "breast_side": df["Side"].str.strip(),
        "view": df["View"].str.strip(),
        # malignant = 1, normal or benign = 0
        "label": (df["Pathology Classification/ Follow up"].str.strip() == "Malignant").astype(int),
    }).reset_index(drop=True)
    assert len(img) == EXPECTED_IMAGES, f"expected {EXPECTED_IMAGES} images, got {len(img)}"
    assert img.patient_id.nunique() == EXPECTED_PATIENTS, "unexpected patient count"
    assert img.case_id.is_unique, "case_id must be unique"
    assert img.breast_side.isin(["L", "R"]).all() and img.view.isin(["CC", "MLO"]).all()
    return img


def split_patients(img: pd.DataFrame) -> dict:
    # Patient-level status: 1 if any image of the patient is malignant
    pat = img.groupby("patient_id")["label"].max().rename("pat_label").reset_index()
    trainval, test = train_test_split(
        pat, test_size=TEST_FRAC, stratify=pat.pat_label, random_state=SEED)
    train, val = train_test_split(
        trainval, test_size=VAL_FRAC / (1 - TEST_FRAC),
        stratify=trainval.pat_label, random_state=SEED)
    splits = {"train": set(train.patient_id), "val": set(val.patient_id), "test": set(test.patient_id)}
    assert not (splits["train"] & splits["val"]), "leakage: train/val"
    assert not (splits["train"] & splits["test"]), "leakage: train/test"
    assert not (splits["val"] & splits["test"]), "leakage: val/test"
    assert set().union(*splits.values()) == set(pat.patient_id), "patients missing from splits"
    return splits


def summarise(img: pd.DataFrame) -> pd.DataFrame:
    pat = img.groupby(["split", "patient_id"])["label"].max().reset_index()
    breast = img.groupby(["split", "patient_id", "breast_side"])["label"].max().reset_index()
    rows = []
    for s in ["train", "val", "test"]:
        i, p, b = img[img.split == s], pat[pat.split == s], breast[breast.split == s]
        rows.append({"split": s, "patients": len(p), "patients_cancer": int(p.label.sum()),
                     "breasts": len(b), "breasts_cancer": int(b.label.sum()),
                     "images": len(i), "images_cancer": int(i.label.sum())})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--annotations", required=True, type=Path, help="Radiology-manual-annotations.xlsx")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--image-dir", type=Path, default=None,
                    help="optional: low-energy image folder, checks every case_id has a file")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    img = load_low_energy(args.annotations)
    splits = split_patients(img)
    split_of = {p: s for s, ids in splits.items() for p in ids}
    img["split"] = img.patient_id.map(split_of)
    img = img.sort_values(["patient_id", "breast_side", "view", "case_id"]).reset_index(drop=True)

    if args.image_dir is not None:
        stems = {p.stem for p in args.image_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
        missing = sorted(set(img.case_id) - stems)
        assert not missing, f"{len(missing)} case_ids have no image file, e.g. {missing[:5]}"
        print(f"[CHECK] all {len(img)} case_ids found in {args.image_dir}")

    img.to_csv(args.out_dir / "cdd_cesm_low_energy_splits.csv", index=False)
    summ = summarise(img)
    summ.to_csv(args.out_dir / "split_summary.csv", index=False)
    with open(args.out_dir / "splits.json", "w") as f:
        json.dump({k: sorted(int(p) for p in v) for k, v in splits.items()}, f, indent=2)

    print(summ.to_string(index=False))
    print("[CHECK] zero patient overlap between train, val and test")


if __name__ == "__main__":
    main()
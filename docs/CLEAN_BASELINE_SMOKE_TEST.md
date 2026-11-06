# Clean-baseline real-data smoke test

Use the first trained CDD-CESM checkpoint to verify the clean inference exports before running all six model/seed checkpoints. Run from the repository root in a Bash terminal (including GitHub Codespaces). This procedure checks export coverage and consistency, not predictive performance.

## Inputs and run commands

Use the committed `data/cdd_cesm_low_energy_splits.csv` and `data/splits.json`, the corresponding permitted low-energy images, a verified annotation CSV with original `normal`/`benign`/`malignant` labels for **all 1,003 images**, and a checkpoint trained with those exact split files. The annotation CSV must have the columns described in [`CLEAN_BASELINE_INFERENCE.md`](CLEAN_BASELINE_INFERENCE.md). Set the paths to private, accessible locations **outside the public repository**; do not upload restricted data to Codespaces without permission.

```bash
export ANNOTATIONS="/private/path/verified_original_classes.csv"
export IMAGES="/private/path/low_energy_images"
export CHECKPOINT="/private/path/mobilenet_v3_small_seed42_best.pt"
export RUN1="/private/path/clean_smoke_first"
export RUN2="/private/path/clean_smoke_repeat"

python -m mammography_robustness.clean_inference \
  --annotations "$ANNOTATIONS" --images "$IMAGES" \
  --checkpoint "$CHECKPOINT" --output-dir "$RUN1" --device cpu

python -m mammography_robustness.clean_inference \
  --annotations "$ANNOTATIONS" --images "$IMAGES" \
  --checkpoint "$CHECKPOINT" --output-dir "$RUN2" --device cpu
```

Use two **new, distinct output directories**; the exporter refuses to overwrite its CSVs. Install the training/inference dependencies in the active Python environment first. Checkpoint loading verifies model/seed configuration and SHA-256 hashes of the manifest and split JSON. For a run record, capture the commands and paths (without publishing restricted paths), checkpoint identity and seed, Python/PyTorch versions, and input hashes:

```bash
python -c 'import sys, torch; print(sys.version); print("torch", torch.__version__)'
sha256sum data/cdd_cesm_low_energy_splits.csv data/splits.json "$CHECKPOINT"
```

## Expected CSV coverage

The manifest has 159 validation images, but `P281_L_DM_MLO2` is excluded from evaluation under the fixed acquisition rule. The **158 selected validation images** represent **88 breasts**. All **158 test images**, representing **89 breasts**, remain included. Patient 99 retains both MLO acquisitions in the training source records; neither belongs in validation/test exports.

| CSV file | First checkpoint | Six checkpoints |
| --- | ---: | ---: |
| `val_image_predictions.csv` | 158 | 948 |
| `val_image_metadata.csv` | 158 | 948 |
| `val_breast_predictions.csv` | 88 | 528 |
| `val_breast_metadata.csv` | 88 | 528 |
| `test_image_predictions.csv` | 158 | 948 |
| `test_image_metadata.csv` | 158 | 948 |
| `test_breast_predictions.csv` | 89 | 534 |
| `test_breast_metadata.csv` | 89 | 534 |

Counts are data rows, excluding headers. The six-checkpoint figures assume two architectures with three distinct seeds each.

## Automated verification

After both commands finish, run the following block in the **same Bash terminal**. It validates all eight CSVs against the selected manifest records and the verified source annotations, including original-class preservation, binary-label agreement, single/paired-view status, breast-level severity, and maximum-view probability aggregation. It also checks the reviewed Patient 16 source annotations (Patient 16 belongs to training, not these exports), prints SHA-256 hashes, and requires identical hashes between the two CPU runs. Any mismatch stops with an error.

```bash
python - "$RUN1" "$RUN2" "$ANNOTATIONS" <<'PY'
import csv
import hashlib
import math
import sys
from collections import defaultdict
from pathlib import Path

first, repeat, annotation_path = map(Path, sys.argv[1:4])
manifest_path = Path('data/cdd_cesm_low_energy_splits.csv')
classes = {'normal': 0, 'benign': 0, 'malignant': 1}
severity = {'normal': 0, 'benign': 1, 'malignant': 2}
counts = {
    'val_image_predictions.csv': 158, 'val_image_metadata.csv': 158,
    'val_breast_predictions.csv': 88, 'val_breast_metadata.csv': 88,
    'test_image_predictions.csv': 158, 'test_image_metadata.csv': 158,
    'test_breast_predictions.csv': 89, 'test_breast_metadata.csv': 89,
}


def check(condition, message):
    if not condition:
        raise SystemExit(f'FAILED: {message}')


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        check(reader.fieldnames is not None, f'missing header: {path}')
        rows = list(reader)
        check(all(None not in r for r in rows), f'malformed rows: {path}')
        return rows


def indexed(rows, key, description):
    result = {key(row): row for row in rows}
    check(len(result) == len(rows), f'duplicate {description}')
    return result


manifest_rows = read_csv(manifest_path)
manifest = indexed(manifest_rows, lambda r: r['case_id'], 'manifest case_id')
check(len(manifest) == 1003, 'unexpected manifest size')
annotations = indexed(read_csv(annotation_path), lambda r: r['case_id'], 'annotation case_id')
check(set(annotations) == set(manifest), 'annotation coverage differs from manifest')
for case_id, source in manifest.items():
    record = annotations[case_id]
    check(all(record.get(k) == source[k] for k in ('patient_id', 'breast_side', 'view', 'label', 'split')),
          f'annotation identity mismatch: {case_id}')
    label = record.get('original_class', '').strip().lower()
    check(label in classes and classes[label] == int(source['label']), f'annotation class mismatch: {case_id}')
check(annotations['P16_R_DM_CC']['original_class'].strip().lower() == 'normal' and
      annotations['P16_R_DM_MLO']['original_class'].strip().lower() == 'malignant',
      'Patient 16 source CC/MLO classes differ from reviewed annotations')

expected = {}
for split, source_count, image_count, breast_count in (('val', 159, 158, 88), ('test', 158, 158, 89)):
    source = [r for r in manifest_rows if r['split'] == split]
    selected = [r for r in source if r['case_id'] != 'P281_L_DM_MLO2']
    breasts = defaultdict(list)
    for row in selected:
        breasts[(row['patient_id'], row['breast_side'])].append(row)
    check((len(source), len(selected), len(breasts)) == (source_count, image_count, breast_count),
          f'unexpected {split} manifest coverage')
    expected[split] = (indexed(selected, lambda r: r['case_id'], f'{split} image'), breasts)


def verify(run_dir):
    names = {path.name for path in run_dir.glob('*.csv')}
    check(names == set(counts), f'CSV filenames differ: {run_dir}')
    files = {name: read_csv(run_dir / name) for name in counts}
    for name, count in counts.items():
        check(len(files[name]) == count, f'{name}: expected {count} rows, got {len(files[name])}')
    pairs = {(r.get('model'), r.get('seed')) for rows in files.values() for r in rows}
    check(len(pairs) == 1 and all(all(p) for p in pairs), f'multiple/missing model-seed values: {run_dir}')
    for split in ('val', 'test'):
        images, breasts = expected[split]
        predictions = indexed(files[f'{split}_image_predictions.csv'], lambda r: r['case_id'], 'image prediction')
        metadata = indexed(files[f'{split}_image_metadata.csv'], lambda r: r['case_id'], 'image metadata')
        check(set(predictions) == set(metadata) == set(images), f'{split} selected image coverage mismatch')
        for case_id, source in images.items():
            pred, meta = predictions[case_id], metadata[case_id]
            anno = annotations[case_id]
            for row in (pred, meta):
                check(all(row.get(k) == source[k] for k in ('patient_id', 'breast_side', 'view', 'label')),
                      f'{split} image identity/label mismatch: {case_id}')
            check(meta.get('image_id') == case_id and
                  meta.get('original_class') == anno['original_class'].strip().lower() and
                  pred.get('score') == meta.get('score'),
                  f'{split} image metadata mismatch: {case_id}')
        breast_predictions = indexed(files[f'{split}_breast_predictions.csv'],
                                     lambda r: (r['patient_id'], r['breast_side']), 'breast prediction')
        breast_metadata = indexed(files[f'{split}_breast_metadata.csv'],
                                  lambda r: (r['patient_id'], r['breast_side']), 'breast metadata')
        check(set(breast_predictions) == set(breast_metadata) == set(breasts),
              f'{split} breast coverage mismatch')
        for breast, views in breasts.items():
            pred, meta = breast_predictions[breast], breast_metadata[breast]
            expected_label = max(int(r['label']) for r in views)
            expected_class = max((annotations[r['case_id']]['original_class'].strip().lower() for r in views),
                                 key=severity.__getitem__)
            expected_score = max(float(predictions[r['case_id']]['score']) for r in views)
            check(pred['label'] == meta['label'] == str(expected_label) and
                  meta.get('original_class') == expected_class and
                  meta.get('view_status') == ('paired' if len(views) == 2 else 'single') and
                  meta.get('view_label_conflict') == str(int(len({r['label'] for r in views}) > 1)),
                  f'{split} breast metadata mismatch: {breast}')
            score = float(pred['score'])
            check(math.isfinite(score) and abs(score - expected_score) <= 1e-12,
                  f'{split} max-view score mismatch: {breast}')
    hashes = {name: hashlib.sha256((run_dir / name).read_bytes()).hexdigest() for name in sorted(counts)}
    for name, digest in hashes.items():
        print(f'{run_dir.name}/{name}: rows={counts[name]}, sha256={digest}')
    return hashes

check(first.resolve() != repeat.resolve(), 'repeat run must use a separate directory')
check(verify(first) == verify(repeat), 'CSV checksums differ between runs')
print('PASS: source classes, selected views, eight CSVs, breast aggregation, and repeat-run hashes')
PY
```

Matching output hashes demonstrate byte-for-byte equality for the two checked runs. They do not establish accuracy or reproducibility across different hardware and software versions. Save the eight CSV row counts, hashes, environment details, and any failures in a private run record; share only information permitted by the data-use terms. Real model performance should be reported only after the full evaluation protocol is run.

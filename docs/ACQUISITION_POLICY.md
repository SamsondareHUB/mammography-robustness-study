# Repeated MLO acquisitions in CDD-CESM

On November 2, 2026, the project lead confirmed that the unsuffixed MLO images should be used for evaluation for Patients 99 and 281. This selection is fixed in advance and does not depend on model predictions.

## Images and splits

| Patient | Split | MLO selected for evaluation | Additional MLO retained in source |
| --- | --- | --- | --- |
| 99 | Training | `P99_L_DM_MLO` | `P99_L_DM_MLO_2` |
| 281 | Validation | `P281_L_DM_MLO` | `P281_L_DM_MLO2` |

Sallew inspected these image pairs and found differences in dimensions, file contents, and positioning. The images are distinct and may be retakes, but their acquisition history has not been verified. Both records and their original annotations remain in the source manifest.

## Implementation

- The manifest validator accepts only these two repeated MLO pairs, with their expected IDs, splits, and labels. Other repeated breast-view combinations remain invalid.
- Both Patient 99 MLO acquisitions may be used for training. Patient 281's images remain in validation and must not be used for training.
- The additional Patient 281 MLO is excluded from validation-loss calculation and validation prediction exports. The unsuffixed MLO is used instead.
- Apply `select_evaluation_images()` to the relevant source records before generating evaluation predictions. Use the same image selection across models, seeds, and conditions.
- Neither patient belongs to the test split, so no test images are excluded by this rule.
- The existing rules for single-view breasts and Patient 16's view-label discrepancy remain unchanged.

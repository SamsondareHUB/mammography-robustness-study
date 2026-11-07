"""Pure helpers for breast-level checkpoint selection and reproducible training."""

from __future__ import annotations

from collections import defaultdict


def breast_auc(patients, sides, labels, scores) -> float:
    """Compute breast-level ROC AUC with midranks for tied probabilities."""
    groups = defaultdict(list)
    for patient, side, label, score in zip(patients, sides, labels, scores, strict=True):
        if label not in (0, 1) or not 0 <= float(score) <= 1:
            raise ValueError("invalid binary label or probability")
        groups[(str(patient), side)].append((int(label), float(score)))
    items = sorted((max(x[0] for x in values), max(x[1] for x in values))
                   for values in groups.values())
    n_positive = sum(label == 1 for label, _ in items)
    n_negative = len(items) - n_positive
    if not n_positive or not n_negative:
        raise ValueError("breast-level validation AUC requires both classes")
    ranked = sorted(items, key=lambda x: x[1])
    rank_sum_positive = 0.0
    i = 0
    while i < len(ranked):
        j = i + 1
        while j < len(ranked) and ranked[j][1] == ranked[i][1]:
            j += 1
        average_rank = ((i + 1) + j) / 2
        rank_sum_positive += average_rank * sum(label == 1 for label, _ in ranked[i:j])
        i = j
    return (rank_sum_positive - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)


def should_replace_best(auc: float, loss: float, epoch: int, best) -> bool:
    """Tie-break exact equal AUC by lower loss and then earliest epoch."""
    if best is None:
        return True
    best_auc, best_loss, best_epoch = best
    return auc > best_auc or (auc == best_auc and
        (loss < best_loss or (loss == best_loss and epoch < best_epoch)))


def inverse_frequency_weights(labels):
    """Balanced two-class weights computed strictly from the training records."""
    labels = list(labels)
    c0 = labels.count(0)
    c1 = labels.count(1)
    if c0 == 0 or c1 == 0 or c0 + c1 != len(labels):
        raise ValueError("training split must have binary labels from both classes")
    return (len(labels) / (2 * c0), len(labels) / (2 * c1))

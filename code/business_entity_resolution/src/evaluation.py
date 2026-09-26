"""Evaluation module computing exact macro-averaged F_0.5 score.

F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
Averaged across all Source-1 entities including singletons.
"""

from typing import Dict, List, Set, Union
import pandas as pd


def compute_f05_score(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
) -> float:
    """Compute the macro-averaged F_0.5 score across all Source-1 entities.

    Parameters
    ----------
    ground_truth : dict of {s1_id: set_of_true_matched_ids}
        Ground truth match sets. Empty set denotes a singleton.
    predictions : dict of {s1_id: set_of_pred_matched_ids}
        Predicted match sets. Empty set denotes predicted singleton.

    Returns
    -------
    float
        Macro-averaged F_0.5 score across all S1 entities (0.0 to 1.0).
    """
    total_score = 0.0
    num_entities = len(ground_truth)

    if num_entities == 0:
        return 0.0

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())

        # Singleton ground truth handling
        if len(true_set) == 0:
            if len(pred_set) == 0:
                total_score += 1.0  # Correctly identified singleton
            else:
                total_score += 0.0  # False merge on singleton
            continue

        # Non-singleton ground truth but predicted empty
        if len(pred_set) == 0:
            total_score += 0.0
            continue

        # Both non-empty: calculate TP, Precision, Recall
        tp = len(true_set & pred_set)
        if tp == 0:
            total_score += 0.0
            continue

        precision = tp / len(pred_set)
        recall = tp / len(true_set)

        denom = (0.25 * precision) + recall
        if denom == 0:
            score = 0.0
        else:
            score = (1.25 * precision * recall) / denom

        total_score += score

    return total_score / num_entities

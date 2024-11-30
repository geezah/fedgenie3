from typing import Callable

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


def auprc(y_true: NDArray, y_scores: NDArray) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)


def auroc(y_true: NDArray, y_scores: NDArray) -> float:
    return roc_auc_score(y_true, y_scores)


# TODO: Write two separate functions for auroc and auprc p-values where the metric is passed as an argument
def permutation_test(
    metric: Callable[[NDArray, NDArray], float],
    y_true: NDArray,
    y_scores: NDArray,
    num_permutations: int = 1000,
) -> float:
    """
    Calculate the p-value for the observed AUC scores being greater than random AUC scores using permutation testing.

    Args:
        metric (Callable[[np.ndarray, np.ndarray], float]): Metric function to calculate AUC with.
        y_true (np.ndarray): Ground truth labels.
        y_scores (np.ndarray): Predicted scores.
        num_permutations (int): Number of permutations to perform.

    Returns:
        float: P-value for the observed AUC scores.
    """
    # Input validation
    if not isinstance(y_true, np.ndarray) or not isinstance(y_scores, np.ndarray):
        raise TypeError(
            f"y_true and y_scores must be numpy arrays. Got {type(y_true)} and {type(y_scores)} respectively instead."
        )

    if y_true.shape != y_scores.shape:
        raise ValueError("y_true and y_scores must have the same shape")

    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only binary values (0 or 1)")

    if num_permutations < 1:
        raise ValueError("num_permutations must be positive")

    # Calculate observed AUC metric
    observed_score = metric(y_true, y_scores)

    # Generate null distribution through permutation
    permuted_scores = np.zeros(num_permutations)
    rng = np.random.default_rng()  # Use newer random number generator

    for i in range(num_permutations):
        permutations = rng.permutation(y_scores)
        permuted_scores[i] = metric(y_true, permutations)

    # Calculate p-value based on the alternative hypothesis
    p_value = np.sum(permuted_scores >= observed_score) / num_permutations

    assert 0 < p_value <= 1, "P-value must be between 0 and 1"
    return p_value


def auroc_permutation_test(
    y_true: NDArray, y_scores: NDArray, num_permutations: int = 1000
) -> float:
    return permutation_test(auroc, y_true, y_scores, num_permutations)


def auprc_permutation_test(
    y_true: NDArray, y_scores: NDArray, num_permutations: int = 1000
) -> float:
    return permutation_test(auprc, y_true, y_scores, num_permutations)


def combined_log_p_value(auroc_p: float, auprc_p: float) -> float:
    return -0.5 * np.log10(auroc_p * auprc_p)


if __name__ == "__main__":
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 0])
    y_scores = np.array([0.1, 0.9, 0.8, 0.3, 0.7, 0.2, 0.6, 0.4, 0.5, 0.1])
    auroc_score = auroc(y_true, y_scores)
    auroc_p_value = permutation_test(auroc, y_true, y_scores)
    auprc_score = auprc(y_true, y_scores)
    auprc_p_value = permutation_test(auprc, y_true, y_scores)
    overall_score = combined_log_p_value(auroc_p_value, auprc_p_value)
    results = {
        "auroc": auroc_score,
        "auroc_p": auroc_p_value,
        "auprc": auprc_score,
        "aupr_p": auprc_p_value,
        "overall_score": overall_score,
    }
    print(results)

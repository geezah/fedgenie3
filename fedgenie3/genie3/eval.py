from typing import Tuple

import pandas as pd
from numpy.typing import NDArray

from fedgenie3.genie3.metrics import (
    auprc,
    auprc_permutation_test,
    auroc,
    auroc_permutation_test,
    combined_log_p_value,
)


def _prepare_evaluation(
    predictions: pd.DataFrame, gt: pd.DataFrame
) -> Tuple[NDArray, NDArray]:
    """
    Prepare the predictions and ground truth for evaluation.

    Args:
        predictions (pd.DataFrame): Predictions from the model
        gt (pd.DataFrame): Ground truth data

    Returns:
        Tuple[NDArray, NDArray]: Tuple containing importance scores and ground truths as NumPy arrays
    """
    merged = predictions.merge(
        gt, on=["transcription_factor", "target_gene"], how="outer"
    )
    merged = merged.fillna(0)
    y_scores = merged["importance"].values
    y_true = merged["label"].values
    return y_scores, y_true


def evaluate(
    predictions: pd.DataFrame, gt: pd.DataFrame
) -> Tuple[float, float, float, float]:
    """
    Evaluate the predictions against the ground truth data.


    Args:
        predictions (pd.DataFrame): Predictions from the model
        gt (pd.DataFrame): Ground truth data

    Returns:
        Tuple[float, float, float, float]: Tuple containing AUPR, AUPR p-value, AUROC, and AUROC p-value
    """
    y_scores, y_true = _prepare_evaluation(predictions, gt)
    auroc_score = auroc(y_true, y_scores)
    auprc_score = auprc(y_true, y_scores)
    auroc_p_value = auroc_permutation_test(auroc, y_true, y_scores)
    auprc_p_value = auprc_permutation_test(auprc, y_true, y_scores)
    overall_score = combined_log_p_value(auroc_p_value, auprc_p_value)

    return {
        "auroc": auroc_score,
        "aupr": auprc_score,
        "auroc_p_value": auroc_p_value,
        "auprc_p_value": auprc_p_value,
        "overall_score": overall_score,
    }

from typing import Tuple

import pandas as pd
from numpy.typing import NDArray
from fedgenie3.genie3.metrics import (
    auprc,
    auc_p_value,
    auroc,
)
import numpy as np


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
    aupr_score = auprc(y_true, y_scores)
    # auroc_p = auc_p_value(auroc, y_true, y_scores)
    # aupr_p = auc_p_value(auprc, y_true, y_scores)

    # overall_score = -0.5 * np.log10(aupr_p * auroc_p)
    return {
        "auroc": auroc_score,
        "aupr": aupr_score,
        # "auroc_p_value": auroc_p,
        # "aupr_p_value": aupr_p,
        # "overall_score": overall_score,
    }

from typing import Dict, Tuple

import pandas as pd
from numpy.typing import NDArray

from genie3.metrics import compute_auroc


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


def evaluate_ranking(
    predictions: pd.DataFrame, gt: pd.DataFrame
) -> Dict[str, float]:
    """
    Evaluate the predictions against the ground truth data.


    Args:
        predictions (pd.DataFrame): Predictions from the model
        gt (pd.DataFrame): Ground truth data

    Returns:
        Dict[str, float]: Dict containing AUROC score
    """
    y_scores, y_true = _prepare_evaluation(predictions, gt)
    auroc = compute_auroc(y_true, y_scores)
    return {"auroc": auroc}

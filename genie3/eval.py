from typing import Tuple

import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import auc, precision_recall_curve, roc_curve


class Results(BaseModel):
    """
    Container model to store and verify evaluation results of GENIE3.

    Attributes:
        auroc (float): Area under the ROC curve. Must be between 0 and 1.
        auprc (float): Area under the precision-recall curve. Must be between 0 and 1.
        fpr (NDArray): False positive rates.
        tpr (NDArray): True positive rates.
        recall (NDArray): Recall scores.
        precision (NDArray): Precision scores.
        pos_frac (float): Fraction of positive examples in the dataset. Must be between 0 and 1.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    auroc: float = Field(
        ..., description="Area under the ROC curve", ge=0, le=1
    )
    auprc: float = Field(
        ..., description="Area under the precision-recall curve", ge=0, le=1
    )
    fpr: NDArray = Field(..., description="False positive rates")
    tpr: NDArray = Field(..., description="True positive rates")
    recall: NDArray = Field(..., description="Recall scores")
    precision: NDArray = Field(..., description="Precision scores")
    pos_frac: float = Field(
        ...,
        description="Fraction of positive examples in the dataset",
        ge=0,
        le=1,
    )


def prepare_evaluation(
    predicted_network: pd.DataFrame, true_network: pd.DataFrame
) -> Tuple[NDArray, NDArray]:
    """
    Prepare the predicted and ground truth network for evaluation.

    Args:
        predicted_network (pd.DataFrame): Predicted network
        true_network (pd.DataFrame): Ground truth network

    Returns:
        Tuple[NDArray, NDArray]: Tuple containing importance scores and ground truths as NumPy arrays
    """
    merged = predicted_network.merge(
        true_network, on=["transcription_factor", "target_gene"], how="outer"
    )
    merged = merged.fillna(0)
    y_preds = merged["importance"].values
    y_true = merged["label"].values
    return y_preds, y_true


def run_evaluation(y_preds: NDArray, y_true: NDArray) -> Results:
    """
    Evaluate the predictions against the ground truth data.


    Args:
        y_preds (NDArray): Predicted importance scores
        y_true (NDArray): Ground truth labels
    Returns:
        Tuple[float, float]: AUROC and AUPRC scores
    """
    pos_frac: float = y_true.sum() / len(y_true)
    fpr, tpr, _ = roc_curve(y_true, y_preds)
    precision, recall, _ = precision_recall_curve(y_true, y_preds)
    auroc = auc(fpr, tpr)
    auprc = auc(recall, precision)
    return Results(
        auroc=auroc,
        auprc=auprc,
        fpr=fpr,
        tpr=tpr,
        recall=recall,
        precision=precision,
        pos_frac=pos_frac,
    )

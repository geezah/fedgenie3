import matplotlib.pyplot as plt
import seaborn as sns
from numpy.typing import NDArray
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def plot_roc_curve(
    fpr: NDArray, tpr: NDArray, roc_auc: float, regressor_name: str = ""
) -> plt.Figure:
    """
    Plots the Receiver Operating Characteristic (ROC) curve.

    Args:
        fpr (NDArray): Array of false positive rates.
        tpr (NDArray): Array of true positive rates.
        roc_auc (float): Area Under the ROC Curve (AUC) score.
        regressor_name (str, optional): Name of the regressor to be displayed in the plot label. Defaults to "".

    Returns:
        plt.Figure: The matplotlib figure object containing the ROC curve plot.
    """
    regressor_name = regressor_name + " " if regressor_name else ""
    fig, ax = plt.subplots()
    sns.lineplot(
        x=fpr,
        y=tpr,
        label=f"{regressor_name}AUC = {roc_auc:.2f}",
        linewidth=2,
        ax=ax,
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess")

    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.legend(loc="upper right")
    ax.grid(True)
    return fig


def plot_precision_recall_curve(
    recall: NDArray,
    precision: NDArray,
    pos_frac: float,
    auprc: float,
    regressor_name: str = "",
) -> plt.Figure:
    """
    Plots a precision-recall curve using the provided recall and precision values.

    Args:
        recall (NDArray): Array of recall values.
        precision (NDArray): Array of precision values.
        pos_frac (float): Fraction of positive samples.
        auprc (float): Area under the precision-recall curve.
        regressor_name (str, optional): Name of the regressor to be displayed in the plot label. Defaults to "".

    Returns:
        plt.Figure: The matplotlib figure object containing the precision-recall curve.
    """
    regressor_name = regressor_name + " " if regressor_name else ""
    fig, ax = plt.subplots()
    sns.lineplot(
        x=recall,
        y=precision,
        label=f"{regressor_name}AUC = {auprc:.4f}, %P: {pos_frac}",
        linewidth=2,
        ax=ax,
    )
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall Gain")
    ax.set_ylabel("Precision Gain")
    ax.legend(loc="upper right")
    ax.grid(True)
    return fig


# Toy example using sklearn's make_classification
if __name__ == "__main__":
    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train a classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Get prediction probabilities
    y_scores = clf.predict_proba(X_test)[:, 1]

    # Plot ROC and Precision-Recall curves

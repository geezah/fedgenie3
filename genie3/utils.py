import json
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from genie3.config import GENIE3Config


def save_results_inference_only(
    config: GENIE3Config,
    predicted_network: pd.DataFrame,
    output_dir: Path = Path("results"),
) -> None:
    """
    Save the results of the inference phase only.

    Args:
        config (GENIE3Config): The configuration object for the GENIE3 model.
        predicted_network (pd.DataFrame): The predicted network as a pandas DataFrame.
        output_dir (Path, optional): The directory where the results will be saved. Defaults to "results".

    Returns:
        None
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # Dump the model configuration
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        # Convert the GENIE3Config object to a dictionary with paths casted to strings and dump it as JSON
        config = json.loads(config.model_dump_json())
        json.dump(config, f, indent=4)
    # Dump the predicted network
    predicted_network.to_csv(output_dir / "predicted_network.csv", index=False)


def save_results_all(
    config: GENIE3Config,
    auroc: float,
    auprc: float,
    predicted_network: pd.DataFrame,
    reference_network: pd.DataFrame,
    roc_curve_plot: Figure,
    precision_recall_curve_plot: Figure,
    output_dir: Path,
) -> None:
    """
    Save all results including metrics, predicted and reference networks, and plots.

    Args:
        config (GENIE3Config): The configuration object for the GENIE3 model.
        auroc (float): The Area Under the Receiver Operating Characteristic curve score.
        auprc (float): The Area Under the Precision-Recall curve score.
        predicted_network (pd.DataFrame): The predicted network as a pandas DataFrame.
        reference_network (pd.DataFrame): The reference network as a pandas DataFrame.
        roc_curve_plot (Figure): The ROC curve plot as a matplotlib Figure.
        precision_recall_curve_plot (Figure): The precision-recall curve plot as a matplotlib Figure.
        output_dir (Path): The directory where the results will be saved.

    Returns:
        None
    """
    save_results_inference_only(config, predicted_network, output_dir)
    # Dump the metrics
    pd.DataFrame(
        {
            "metric": ["AUROC", "AUPRC"],
            "score": [auroc, auprc],
        },
    ).to_csv(output_dir / "metrics.csv", index=False)
    # Dump the predicted and reference networks
    reference_network.to_csv(output_dir / "reference_network.csv", index=False)
    # Dump the plots
    roc_curve_plot.savefig(output_dir / "roc_curve.png")
    precision_recall_curve_plot.savefig(
        output_dir / "precision_recall_curve.png"
    )

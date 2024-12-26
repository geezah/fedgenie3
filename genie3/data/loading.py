from pathlib import Path

import pandas as pd


def load_gene_expression_data(
    gene_expression_path: Path,
) -> pd.DataFrame:
    """
    Load gene expression data from a file.

    Args:
        gene_expression_path (Path): Path to the gene expression data file.

    Returns:
        pd.DataFrame: Gene expression data.
    """
    return pd.read_csv(gene_expression_path, sep="\t", header=0)


def load_transcription_factor_data(
    transcription_factor_path: Path,
) -> pd.Series:
    """
    Load transcription factor data from a file.

    Args:
        transcription_factor_path (Path): Path to the transcription factor data file.

    Returns:
        pd.Series: Transcription factor data.
    """
    transcription_factors: pd.Series = pd.read_csv(
        transcription_factor_path, sep="\t", header=0
    ).squeeze()
    return transcription_factors


def load_reference_network_data(reference_network_path: Path) -> pd.DataFrame:
    """
    Load reference network data from a given file path.
    This function reads a tab-separated values (TSV) file containing reference network data
    and returns it as a pandas DataFrame. The file is expected to have the following columns:
    "transcription_factor", "target_gene", and "label".
    Args:
        reference_network_path (Path): The file path to the reference network data.
    Returns:
        pd.DataFrame: A DataFrame containing the reference network data.
    """
    df: pd.DataFrame = pd.read_csv(reference_network_path, sep="\t", header=0)
    return df

from pathlib import Path

import pandas as pd


def load_gene_expression_data(
    gene_expression_path: Path,
) -> pd.DataFrame:
    return pd.read_csv(gene_expression_path, sep="\t")


def load_transcription_factor_data(
    transcription_factor_path: Path,
) -> pd.Series:
    transcription_factors: pd.Series = pd.read_csv(
        transcription_factor_path, sep="\t", header=0
    ).squeeze()
    return transcription_factors


def load_reference_network_data(reference_network_path: Path) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(reference_network_path, sep="\t", header=0)
    assert df.columns.to_list() == [
        "transcription_factor",
        "target_gene",
        "label",
    ]
    return df

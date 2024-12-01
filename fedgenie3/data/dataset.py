from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class GRNDataset:
    gene_expressions: pd.DataFrame
    transcription_factors: pd.Series
    reference_network: pd.DataFrame


def load_gene_expression_data(
    gene_expression_path: Path,
) -> pd.Series:
    df = pd.read_csv(gene_expression_path, sep="\t")
    return df


def load_transcription_factor_data(
    transcription_factor_path: Path,
) -> pd.DataFrame:
    series = pd.read_csv(transcription_factor_path, sep="\t").squeeze()
    assert isinstance(series, pd.Series)
    series.name = "transcription_factor"
    return series


def load_reference_network_data(reference_network_path: Path) -> pd.DataFrame:
    df = pd.read_csv(reference_network_path, sep="\t")
    assert list(df.columns) == [
        "transcription_factor",
        "target_gene",
        "label",
    ]
    return df


def construct_grn_dataset(
    gene_expression_path: Path,
    transcription_factor_path: Path,
    reference_network_path: Path,
) -> GRNDataset:
    gene_expressions = load_gene_expression_data(gene_expression_path)
    transcription_factors = load_transcription_factor_data(
        transcription_factor_path
    )
    reference_network = load_reference_network_data(reference_network_path)
    return GRNDataset(
        gene_expressions=gene_expressions,
        transcription_factors=transcription_factors,
        reference_network=reference_network,
    )


def load_dream_five(root: Path, net_id: int) -> GRNDataset:
    net_id_to_net_name = {
        1: "in-silico",
        3: "e-coli",
        4: "s-cerevisiae",
    }
    network_name = net_id_to_net_name[net_id]
    GENE_EXPRESSION_PATH = (
        root / f"net{net_id}_{network_name}" / "gene_expression_data.tsv"
    )
    REFERENCE_NETWORK_PATH = (
        root / f"net{net_id}_{network_name}" / "reference_network_data.tsv"
    )
    TRANSCRIPTION_FACTOR_PATH = (
        root / f"net{net_id}_{network_name}" / "transcription_factors.tsv"
    )

    grn_dataset = construct_grn_dataset(
        gene_expression_path=GENE_EXPRESSION_PATH,
        transcription_factor_path=TRANSCRIPTION_FACTOR_PATH,
        reference_network_path=REFERENCE_NETWORK_PATH,
    )
    return grn_dataset


if __name__ == "__main__":
    grn_dataset = load_dream_five(Path("local_data/processed/dream_five"), 1)
    print(grn_dataset.gene_expressions.head())

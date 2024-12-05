from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd


@dataclass
class GRNMetadata:
    """
    GRNMetadata is a class that holds metadata information for a Gene Regulatory Network (GRN).

    Attributes:
        transcription_factor_indices (pd.Series): A `pd.Series`-object containing indices of transcription factors.
        gene_names_to_indices (Dict[str, int]): A dictionary mapping gene names to their corresponding indices.
        gene_indices_to_names (List[int]): A list mapping gene indices to their corresponding names.
    """

    transcription_factor_indices: pd.Series
    gene_names_to_indices: Dict[str, int]
    indices_to_gene_names: Dict[int, str]


def construct_grn_metadata(
    transcription_factors_path: Path, indices_to_names_path: Path
) -> GRNMetadata:
    transcription_factors = load_transcription_factor_data(
        transcription_factors_path
    )
    indices_to_names = load_indices_to_names(indices_to_names_path)
    names_to_indices = {v: k for k, v in indices_to_names.items()}
    transcription_factor_indices = transcription_factors.map(names_to_indices)
    return GRNMetadata(
        transcription_factor_indices, indices_to_names, names_to_indices
    )


def load_transcription_factor_data(
    transcription_factor_path: Path,
) -> pd.Series:
    series = pd.read_csv(
        transcription_factor_path, sep="\t", header=0
    ).squeeze()
    assert series.name == "transcription_factor"
    return series


def load_indices_to_names(indices_to_gene_names_path: Path) -> Dict[int, str]:
    series = pd.read_csv(
        indices_to_gene_names_path, sep="\t", header=0, index_col=0
    ).squeeze()
    assert series.name == "gene_name"
    return series.to_dict()


@dataclass
class GRNDataset:
    """
    GRNDataset class to represent gene regulatory network datasets.

    Attributes:
        gene_expressions (NDArray): A numpy array containing gene expression data.
        reference_network (NDArray): A numpy array representing the reference network.
        metadata (GRNMetadata): An instance of GRNMetadata containing metadata information about the dataset.
    """

    gene_expressions: pd.DataFrame
    reference_network: pd.DataFrame
    metadata: GRNMetadata


def load_gene_expression_data(
    gene_expression_path: Path,
) -> pd.DataFrame:
    return pd.read_csv(gene_expression_path, sep="\t")


def load_reference_network_data(reference_network_path: Path) -> pd.DataFrame:
    df = pd.read_csv(reference_network_path, sep="\t", header=0)
    assert df.columns.to_list() == [
        "transcription_factor",
        "target_gene",
        "label",
    ]
    return df


def construct_grn_dataset(
    gene_expression_path: Path,
    transcription_factor_path: Path,
    reference_network_path: Path,
    indices_to_names_path: Path,
) -> GRNDataset:
    gene_expressions: pd.DataFrame = load_gene_expression_data(
        gene_expression_path
    )
    reference_network: pd.DataFrame = load_reference_network_data(
        reference_network_path
    )

    grn_metadata = construct_grn_metadata(
        transcription_factor_path, indices_to_names_path
    )
    return GRNDataset(
        gene_expressions=gene_expressions,
        reference_network=reference_network,
        metadata=grn_metadata,
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
    INDICES_TO_NAMES_PATH = (
        root / f"net{net_id}_{network_name}" / "indices_to_names.tsv"
    )

    grn_dataset = construct_grn_dataset(
        gene_expression_path=GENE_EXPRESSION_PATH,
        transcription_factor_path=TRANSCRIPTION_FACTOR_PATH,
        reference_network_path=REFERENCE_NETWORK_PATH,
        indices_to_names_path=INDICES_TO_NAMES_PATH,
    )
    return grn_dataset


if __name__ == "__main__":
    grn_dataset = load_dream_five(Path("local_data/processed/dream_five"), 1)
    print(grn_dataset.gene_expressions)
    print(grn_dataset.reference_network)
    print(grn_dataset.metadata.transcription_factor_indices)

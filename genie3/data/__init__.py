from pathlib import Path
from typing import Optional

from pandas import DataFrame, Series

from .dataset import GRNDataset
from .loading import (
    load_gene_expression_data,
    load_reference_network_data,
    load_transcription_factor_data,
)


def init_grn_dataset(
    gene_expressions_path: Path,
    transcription_factor_path: Optional[Path],
    reference_network_path: Optional[Path],
) -> GRNDataset:
    gene_expressions: DataFrame = load_gene_expression_data(
        gene_expressions_path
    )
    transcription_factor_names = None
    if transcription_factor_path is not None:
        transcription_factor_names: Series = load_transcription_factor_data(
            transcription_factor_path
        )

    reference_network = None
    if reference_network_path is not None:
        reference_network: DataFrame = load_reference_network_data(
            reference_network_path
        )
    return GRNDataset(
        gene_expressions=gene_expressions,
        reference_network=reference_network,
        transcription_factor_names=transcription_factor_names,
    )


__all__ = [
    "GRNDataset",
    "init_grn_dataset",
    "load_gene_expression_data",
    "load_transcription_factor_data",
    "load_reference_network_data",
]

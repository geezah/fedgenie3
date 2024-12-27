from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm.auto import tqdm

from genie3.config import RegressorConfig
from genie3.data import GRNDataset
from genie3.data.utils import map_data

from .regressor import (
    RegressorFactory,
)


def run(
    dataset: GRNDataset, regressor_config: RegressorConfig
) -> pd.DataFrame:
    importance_scores = calculate_importances(
        dataset.gene_expressions.values,
        dataset._transcription_factor_indices,
        regressor_config.name,
        regressor_config.init_params,
        **regressor_config.fit_params,
    )
    predicted_network = rank_genes_by_importance(dataset, importance_scores)
    return predicted_network


def partition_data(
    gene_expressions: NDArray,
    transcription_factor_indices: List[int],
    target_gene: int,
) -> Tuple[NDArray, NDArray, List[int]]:
    # Remove target gene from regulator list and gene expression matrix
    input_genes = [i for i in transcription_factor_indices if i != target_gene]
    X = gene_expressions[:, input_genes]
    y = gene_expressions[:, target_gene]
    return X, y, input_genes


def calculate_importances(
    gene_expressions: NDArray,
    transcription_factor_indices: List[int],
    regressor_type: str,
    regressor_init_params: Dict[str, Any],
    **fit_params: Dict[str, Any],
) -> NDArray[np.float32]:
    # Get the number of genes and transcription factors
    num_genes = gene_expressions.shape[1]
    num_transcription_factors = len(transcription_factor_indices)

    # Initialize importance matrix
    importance_matrix = np.zeros(
        (num_genes, num_transcription_factors), dtype=np.float32
    )

    progress_bar = tqdm(
        range(num_genes),
        total=num_genes,
        desc="Computing importances",
        unit="gene",
        miniters=num_genes // 100,
    )
    for target_gene in progress_bar:
        regressor = RegressorFactory[regressor_type](**regressor_init_params)
        X, y, input_genes = partition_data(
            gene_expressions,
            transcription_factor_indices,
            target_gene,
        )
        regressor.fit(X, y, **fit_params)
        importance_matrix[target_gene, input_genes] = (
            regressor.feature_importances
        )
    return importance_matrix


def rank_genes_by_importance(
    dataset: GRNDataset,
    importance_matrix: NDArray[np.float32],
) -> pd.DataFrame:
    predicted_network = []
    num_genes, num_transcription_factors = (
        importance_matrix.shape[0],
        importance_matrix.shape[1],
    )
    # Create a DataFrame of combinations of target genes and regulators
    for i in range(num_genes):
        for j in range(num_transcription_factors):
            regulator_target_importance_tuples = (
                dataset._transcription_factor_indices[j],
                i,
                importance_matrix[i, j],
            )
            predicted_network.append(regulator_target_importance_tuples)

    predicted_network = pd.DataFrame(
        predicted_network,
        columns=["transcription_factor", "target_gene", "importance"],
    )
    predicted_network.sort_values(
        by="importance", ascending=False, inplace=True
    )
    predicted_network.reset_index(drop=True, inplace=True)
    predicted_network = map_data(
        predicted_network,
        dataset._gene_names,
        subset=["transcription_factor", "target_gene"],
    )
    return predicted_network

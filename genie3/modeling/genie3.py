from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm.auto import tqdm

from genie3.data import GRNDataset
from genie3.data.processing import map_gene_indices_to_names
from genie3.modeling.regressor import (
    initialize_regressor,
)
from genie3.config import RegressorConfig


def run(
    dataset: GRNDataset, regressor_config: RegressorConfig
) -> pd.DataFrame:
    transcription_factor_indices = list(
        range(len(dataset.transcription_factor_names))
    )
    importance_scores = calculate_importances(
        dataset.gene_expressions.values,
        transcription_factor_indices,
        regressor_config.name,
        regressor_config.init_params,
        **regressor_config.fit_params,
    )
    predicted_network = rank_genes_by_importance(dataset, importance_scores)
    return predicted_network


def partition_data(
    gene_expressions: NDArray,
    transcription_factors: List[int],
    target_gene: int,
) -> Tuple[NDArray, NDArray, List[int]]:
    # Remove target gene from regulator list and gene expression matrix
    input_genes = [i for i in transcription_factors if i != target_gene]
    X = gene_expressions[:, input_genes]
    y = gene_expressions[:, target_gene]
    return X, y, input_genes


def calculate_importances(
    gene_expressions: NDArray,
    transcription_factors: List[int],
    regressor_type: str,
    regressor_init_params: Dict[str, Any],
    **fit_params: Dict[str, Any],
) -> NDArray[np.float32]:
    # Get the number of genes and transcription factors
    num_genes = gene_expressions.shape[1]
    num_candidate_regulators = len(transcription_factors)

    # Initialize importance matrix
    importance_matrix = np.zeros(
        (num_genes, num_candidate_regulators), dtype=np.float32
    )

    progress_bar = tqdm(
        range(num_genes),
        total=num_genes,
        desc="Computing importances",
        unit="gene",
    )
    for target_gene in progress_bar:
        regressor = initialize_regressor(regressor_type, regressor_init_params)
        X, y, input_genes = partition_data(
            gene_expressions,
            transcription_factors,
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
    num_genes, num_regulators = (
        importance_matrix.shape[0],
        importance_matrix.shape[1],
    )
    importance_series = importance_matrix.reshape(num_genes * num_regulators)
    # Create a DataFrame of combinations of target genes and regulators
    target_genes = np.repeat(np.arange(num_genes), num_regulators)
    regulators = np.tile(np.arange(num_regulators), num_genes)
    predicted_network = pd.DataFrame(
        {
            "transcription_factor": regulators,
            "target_gene": target_genes,
            "importance": importance_series,
        }
    )
    predicted_network.sort_values(
        by="importance", ascending=False, inplace=True
    )
    predicted_network.reset_index(drop=True, inplace=True)
    predicted_network = map_gene_indices_to_names(
        predicted_network,
        dataset._gene_names,
    )
    return predicted_network

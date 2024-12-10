from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm.auto import tqdm

from genie3.interfaces import (
    compute_importance_scores,
    initialize_regressor,
)


class GENIE3:
    def __init__(
        self,
        regressor_name: str = "LGBM",
        regressor_init_params: Dict[str, Any] = {},
    ):
        self.regressor_type = regressor_name
        self.regressor_init_params = regressor_init_params

    @staticmethod
    def _partition_data(
        gene_expressions: NDArray,
        target_gene_idx: int,
        candidate_regulator_indices: List[int],
    ) -> Tuple[NDArray, NDArray]:
        # Remove target gene from regulator list and gene expression matrix
        input_gene_indices = [
            i for i in candidate_regulator_indices if i != target_gene_idx
        ]
        X = gene_expressions[:, input_gene_indices]
        y = gene_expressions[:, target_gene_idx]
        return X, y, input_gene_indices

    def calculate_importances(
        self,
        gene_expressions: NDArray[np.float32],
        candidate_regulator_indices: Optional[List[int]],
        dev_run: bool = False,
    ) -> NDArray[np.float32]:
        num_genes = gene_expressions.shape[1]

        if candidate_regulator_indices is None:
            candidate_regulator_indices = list(range(num_genes))
        num_candidate_regulators = len(candidate_regulator_indices)

        importance_matrix = np.zeros(
            (num_genes, num_candidate_regulators), dtype=np.float32
        )
        progress_bar = tqdm(
            range(num_genes),
            total=num_genes,
            desc="Computing importances",
            unit="gene",
        )
        for target_gene_index in progress_bar:
            regressor = initialize_regressor(
                self.regressor_type, self.regressor_init_params
            )
            X, y, input_gene_indices = GENIE3._partition_data(
                gene_expressions,
                target_gene_index,
                candidate_regulator_indices,
            )
            feature_importances = compute_importance_scores(X, y, regressor)
            importance_matrix[target_gene_index, input_gene_indices] = (
                feature_importances
            )
            if dev_run:
                break
        importance_matrix = importance_matrix / np.sum(
            importance_matrix, axis=1, keepdims=True
        )
        return importance_matrix

    @staticmethod
    def _generate_gene_ranking(
        importance_matrix: NDArray[np.float32],
        candidate_regulator_indices: Optional[List[int]],
    ) -> List[Tuple[int, int, float]]:
        gene_rankings = []
        num_genes, num_regulators = (
            importance_matrix.shape[0],
            importance_matrix.shape[1],
        )
        if candidate_regulator_indices is None:
            candidate_regulator_indices = list(range(num_regulators))
        for i in range(num_genes):
            for j in range(num_regulators):
                regulator_target_importance_tuples = (
                    candidate_regulator_indices[j],
                    i,
                    importance_matrix[i, j],
                )
                gene_rankings.append(regulator_target_importance_tuples)
        return gene_rankings

    @staticmethod
    def _convert_to_dataframe(
        gene_rankings: List[Tuple[int, int, float]],
    ) -> pd.DataFrame:
        gene_rankings = pd.DataFrame(
            gene_rankings,
            columns=["transcription_factor", "target_gene", "importance"],
        )
        gene_rankings = gene_rankings.astype(
            {
                "transcription_factor": np.uint16,
                "target_gene": np.uint16,
                "importance": np.float64,
            }
        )
        gene_rankings.sort_values(
            by="importance", ascending=False, inplace=True
        )
        gene_rankings.reset_index(drop=True, inplace=True)
        return gene_rankings

    @staticmethod
    def rank_genes_by_importance(
        importance_matrix: NDArray[np.float32],
        candidate_regulator_indices: Optional[List[int]],
    ) -> pd.DataFrame:
        gene_ranking = GENIE3._generate_gene_ranking(
            importance_matrix,
            candidate_regulator_indices,
        )
        gene_ranking = GENIE3._convert_to_dataframe(gene_ranking)
        return gene_ranking

    def __call__(
        self,
        gene_expressions: NDArray,
        candidate_regulator_indices: Optional[List[int]],
        dev_run: bool = False,
    ) -> pd.DataFrame:
        importance_matrix = self.calculate_importances(
            gene_expressions, candidate_regulator_indices, dev_run
        )
        gene_ranking = self.rank_genes_by_importance(
            importance_matrix, candidate_regulator_indices
        )
        return gene_ranking

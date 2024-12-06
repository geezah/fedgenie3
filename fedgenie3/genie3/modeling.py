from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm.auto import tqdm

from fedgenie3.genie3.interfaces import (
    compute_importance_scores,
    initialize_regressor,
)


class GENIE3:
    def __init__(
        self,
        regressor_type: str = "RF",
        regressor_init_params: Dict[str, Any] = {},
    ):
        self.regressor_type = regressor_type
        self.regressor_init_params = regressor_init_params

    @staticmethod
    def _partition_data(
        gene_expressions: NDArray,
        target_gene_idx: int,
        indices_of_candidate_regulators: List[int],
    ) -> Tuple[NDArray, NDArray]:
        # Remove target gene from regulator list and gene expression matrix
        input_gene_indices = [
            i for i in indices_of_candidate_regulators if i != target_gene_idx
        ]
        X = gene_expressions[:, input_gene_indices]
        y = gene_expressions[:, target_gene_idx]
        return X, y, input_gene_indices

    def calculate_importances(
        self,
        gene_expressions: NDArray[np.float32],
        indices_of_candidate_regulators: List[int],
        dev_run: bool = False,
    ) -> NDArray[np.float32]:
        num_genes = gene_expressions.shape[1]
        num_candidate_regulators = len(indices_of_candidate_regulators)
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
                indices_of_candidate_regulators,
            )
            feature_importances = compute_importance_scores(X, y, regressor)
            importance_matrix[target_gene_index, input_gene_indices] = (
                feature_importances
            )
            if dev_run:
                break
        return importance_matrix

    @staticmethod
    def _generate_gene_ranking(
        importance_matrix: NDArray[np.float32],
        indices_of_candidate_regulators: List[int],
    ) -> List[Tuple[int, int, float]]:
        gene_rankings = []
        num_genes, num_regulators = (
            importance_matrix.shape[0],
            importance_matrix.shape[1],
        )
        for i in range(num_genes):
            for j in range(num_regulators):
                regulator_target_importance_tuples = (
                    indices_of_candidate_regulators[j],
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
        indices_of_candidate_regulators: List[int],
    ) -> pd.DataFrame:
        gene_rankings = GENIE3._generate_gene_ranking(
            importance_matrix,
            indices_of_candidate_regulators,
        )
        gene_rankings = GENIE3._convert_to_dataframe(gene_rankings)
        return gene_rankings

    @staticmethod
    def map_indices_to_gene_names(
        rankings: pd.DataFrame,
        indices_to_gene_names: Dict[int, str],
    ) -> pd.DataFrame:
        rankings["transcription_factor"] = rankings[
            "transcription_factor"
        ].apply(lambda i: indices_to_gene_names[i])
        rankings["target_gene"] = rankings["target_gene"].apply(
            lambda i: indices_to_gene_names[i]
        )
        return rankings

    def run(
        self,
        gene_expressions: NDArray[np.float32],
        indices_of_candidate_regulators: List[int],
    ) -> pd.DataFrame:
        importance_matrix = self.calculate_importances(
            gene_expressions, indices_of_candidate_regulators
        )
        gene_rankings = GENIE3.rank_genes_by_importance(
            importance_matrix, indices_of_candidate_regulators
        )
        return gene_rankings


if __name__ == "__main__":
    pass

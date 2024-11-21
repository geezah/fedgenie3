from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.ensemble._forest import ForestRegressor
from tqdm.auto import tqdm


class GENIE3:
    def __init__(
        self,
        tree_method: str = "RF",
        tree_init_kwargs: Dict[str, Any] = {},
        top_k_regulator_candidates: Optional[int] = None,
    ):
        self.tree_method = tree_method
        self.tree_init_kwargs = tree_init_kwargs
        self.top_k_regulator_candidates = top_k_regulator_candidates

    def _init_model(self) -> ForestRegressor:
        if self.tree_method == "RF":
            return RandomForestRegressor(**self.tree_init_kwargs)
        elif self.tree_method == "ET":
            return ExtraTreesRegressor(**self.tree_init_kwargs)
        else:
            raise ValueError(
                "Invalid tree method. Choose between 'RF' and 'ET'"
            )

    @staticmethod
    def _partition_data(
        gene_expressions: NDArray,
        target_gene_idx: int,
        indices_of_candidate_regulators: List[int],
    ) -> Tuple[NDArray, NDArray]:
        """
        Partition gene expression matrix into target gene and candidate regulators.

        Args:
            gene_expressions (NDArray): Gene expression matrix.
            target_gene_idx (int): Index of target gene.
            indices_of_candidate_regulators (List[int]): List of indices of candidate regulators.

        Returns:
            Tuple[NDArray, NDArray]: Input and target matrix.
        """
        # Remove target gene from regulator list and gene expression matrix
        input_gene_indices = [
            i for i in indices_of_candidate_regulators if i != target_gene_idx
        ]
        X = gene_expressions[:, input_gene_indices]
        y = gene_expressions[:, target_gene_idx]
        return X, y, input_gene_indices

    @staticmethod
    def _check_before_compute_importances(
        gene_expressions: NDArray[np.float32],
        indices_of_candidate_regulators: List[int],
    ) -> None:
        # Type checks
        assert isinstance(
            gene_expressions, np.ndarray
        ), "Input must be a numpy array"
        assert isinstance(
            indices_of_candidate_regulators, List
        ), "Regulators must be a list"
        assert all(
            isinstance(i, int) for i in indices_of_candidate_regulators
        ), "Regulator indices must be integers"

        # Shape checks
        assert len(gene_expressions.shape) == 2, "Input must be a 2D array"
        assert (
            len(indices_of_candidate_regulators) > 0
        ), "Regulator candidate list must not be empty"
        assert (
            gene_expressions.shape[1] > 1
        ), "Gene expression matrix must have more than one gene"
        assert (
            gene_expressions.shape[0] > 1
        ), "Gene expression matrix must have more than one sample"

        # Value checks
        assert len(set(indices_of_candidate_regulators)) == len(
            indices_of_candidate_regulators
        ), "Regulator candidate list must not contain duplicates"
        assert all(
            0 <= i < gene_expressions.shape[1]
            for i in indices_of_candidate_regulators
        ), "Regulator candidate indices must be within the range of gene expressions"

    @staticmethod
    def check_after_compute_importances(
        importance_matrix: NDArray,
    ) -> None:
        assert np.all(
            importance_matrix >= 0
        ), "Importances must be non-negative"
        assert np.allclose(
            importance_matrix.sum(axis=1), 1
        ), "Sum of importances assigned to regulator genes must be 1 for each target gene"

    def compute_importances(
        self,
        gene_expressions: NDArray[np.float32],
        indices_of_candidate_regulators: List[int],
        dev_run: bool = False,
    ) -> NDArray[np.float32]:
        GENIE3._check_before_compute_importances(
            gene_expressions, indices_of_candidate_regulators
        )
        num_genes = gene_expressions.shape[1]
        num_regulators = len(indices_of_candidate_regulators)
        importance_matrix = np.zeros(
            (num_genes, num_regulators), dtype=np.float32
        )
        progress_bar = tqdm(
            range(num_genes),
            total=num_genes,
            desc="Computing importances",
            unit="gene",
        )
        for target_gene_index in progress_bar:
            forest_regressor = self._init_model()
            X, y, input_gene_indices = GENIE3._partition_data(
                gene_expressions,
                target_gene_index,
                indices_of_candidate_regulators,
            )
            forest_regressor.fit(X, y)
            # TODO: Permutation Importance instead of impurity-based? Challenge: Limited sample size
            importance_matrix[target_gene_index, input_gene_indices] = (
                forest_regressor.feature_importances_
            )
            if dev_run:
                break
        if not dev_run:
            GENIE3.check_after_compute_importances(importance_matrix)

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

    def run(
        self,
        gene_expressions: NDArray[np.float32],
        indices_of_regulator_candidates: List[int],
    ) -> pd.DataFrame:
        importance_matrix = self.compute_importances(
            gene_expressions, indices_of_regulator_candidates
        )
        gene_rankings = GENIE3.rank_genes_by_importance(importance_matrix)
        return gene_rankings


if __name__ == "__main__":
    pass

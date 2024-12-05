import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from fedgenie3.data.dataset import (
    construct_grn_metadata,
    load_reference_network_data,
)
from fedgenie3.genie3.eval import evaluate as evaluate_ranking
from fedgenie3.genie3.modeling import GENIE3


class GENIE3Strategy(FedAvg):
    def __init__(
        self,
        reference_network_path: Path,
        transcription_factors_path: Path,
        indices_to_names_path: Path,
    ):
        super().__init__()
        self.metadata = construct_grn_metadata(
            transcription_factors_path, indices_to_names_path
        )
        self.reference_network = load_reference_network_data(
            reference_network_path
        )

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        client_instructions = super().configure_fit(
            server_round, parameters, client_manager
        )
        for _, fit_ins in client_instructions:
            fit_ins.config["transcription_factor_indices"] = json.dumps(
                self.metadata.transcription_factor_indices.to_list()
            )
        return client_instructions

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        print(f"Aggregating results for round {server_round}...")
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        importance_matrices = []
        sample_sizes = []
        for _, fit_res in results:
            importance_matrices.append(
                parameters_to_ndarrays(fit_res.parameters)[0]
            )
            sample_sizes.append(fit_res.num_examples)
        # Compute aggregated importance matrix
        aggregated_importance_matrix = self._aggregate_importance_matrices(
            importance_matrices, sample_sizes
        )
        self.global_importance_matrix = aggregated_importance_matrix
        return [], {}

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        print(f"Evaluating on server-side round {server_round}...")
        if server_round == 0:
            return 0.0, {}
        gene_ranking_with_indices = GENIE3.rank_genes_by_importance(
            self.global_importance_matrix,
            self.metadata.transcription_factor_indices,
        )
        gene_ranking_with_names = GENIE3.map_indices_to_gene_names(
            gene_ranking_with_indices, self.metadata.gene_names_to_indices
        )
        evaluation_results = evaluate_ranking(
            gene_ranking_with_names, self.reference_network
        )
        return 0.0, evaluation_results

    def _aggregate_importance_matrices(
        self, importance_matrices: List[np.ndarray], sample_sizes: List[int]
    ) -> np.ndarray:
        # Convert sample sizes to weights
        total_samples = sum(sample_sizes)
        sample_weights = np.array(sample_sizes) / total_samples

        # Stack importance matrices for easier computation
        stacked_importances = np.stack(importance_matrices)

        # Compute mean and variance across clients
        average_importances = np.average(
            stacked_importances, axis=0, weights=sample_weights
        )
        return average_importances

    # TODO: Implement this method when implementing inverse variance weighting
    @staticmethod
    def _normalize_importances(importance_matrix: np.ndarray) -> np.ndarray:
        pass

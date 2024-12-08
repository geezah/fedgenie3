import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from numpy.typing import NDArray

from fedgenie3.data.dataset import (
    construct_grn_metadata,
    load_reference_network_data,
)
from fedgenie3.genie3.eval import evaluate_ranking
from fedgenie3.genie3.modeling import GENIE3


class GENIE3Strategy(Strategy):
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

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize the (global) model parameters."""
        return None

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        transcription_factor_indices: List[int] = json.dumps(
            self.metadata.transcription_factor_indices.to_list()
        )
        config.update(
            {"transcription_factor_indices": transcription_factor_indices}
        )
        fit_ins = FitIns(parameters, config)
        clients: List[ClientProxy] = client_manager.all().values()
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        print(f"Aggregating results for round {server_round}...")
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
        return importance_matrices, {}

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        clients = list(client_manager.all().values())
        eval_instructions = []
        for idx in range(len(clients)):
            importances: Parameters = ndarrays_to_parameters([parameters[idx]])
            instruction = EvaluateIns(importances, {})
            eval_instructions.append((clients[idx], instruction))
        return eval_instructions

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], dict[str, Scalar]]:
        print(f"Aggregating evaluation results for round {server_round}...")
        num_clients = len(results)
        cumulated_auroc = 0.0
        for _, evaluate_res in results:
            metrics = evaluate_res.metrics
            cumulated_auroc += metrics["auroc"]
        average_auroc = cumulated_auroc / num_clients
        return 0.0, {"auroc": average_auroc}

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
        self, importance_matrices: List[NDArray], sample_sizes: List[int]
    ) -> NDArray:
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

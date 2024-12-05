import json
from pathlib import Path

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, NDArrays, Scalar

from fedgenie3.data.dataset import GRNDataset
from fedgenie3.data.simulation import simulate_dream_five
from fedgenie3.genie3.modeling import GENIE3


class GENIE3Client(NumPyClient):
    def __init__(self, dataset: GRNDataset):
        self.dataset = dataset
        self.model = GENIE3(
            tree_method="RF", tree_init_kwargs={"n_estimators": 50}
        )

    def fit(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        transcription_factor_indices = json.loads(
            config["transcription_factor_indices"]
        )
        importance_matrix = self.model.calculate_importances(
            self.dataset.gene_expressions.values,
            transcription_factor_indices,
        )
        return (
            [importance_matrix],
            len(self.dataset.gene_expressions.values),
            {},
        )


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""
    dataset = GRNDataset(
        gene_expression_path=context.node_config["gene_expression_path"],
        transcription_factor_path=context.node_config[
            "transcription_factor_path"
        ],
        reference_network_path=context.node_config["reference_network_path"],
    )
    # Return Client instance
    return GENIE3Client(dataset).to_client()


def client_fn_simulation(context: Context):
    """Construct a Client that will be run in a ClientApp."""
    # TODO: Find a way to not hardcode these values
    root = Path("local_data/processed/dream_five")
    network_id = 1
    random_seed = 42
    simulation_type = "even"

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    dataset_partitions = simulate_dream_five(
        root, network_id, simulation_type, num_partitions, random_seed
    )

    # Return Client instance
    return GENIE3Client(dataset_partitions[partition_id]).to_client()


client_app = ClientApp(client_fn_simulation)

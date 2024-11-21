from pathlib import Path

import flwr as fl
from flwr.common import Context
from flwr.server.strategy import FedAvg

from fedgenie3.client import GENIE3Client
from fedgenie3.genie3.config import GENIE3Config


def client_fn(ctx: Context) -> fl.client.Client:
    # Adjust the data path for each client
    data_path = Path("data/processed/ss/DREAM4_InSilico_Size10.tsv")
    config = GENIE3Config(
        gene_expression_path=data_path,
        gene_names=None,
        regulators=None,
        tree_method="RF",
        max_features="sqrt",
        n_estimators=30,
        random_state=42,
        top_k_regulators=1,
    )
    return GENIE3Client(config).to_client()


if __name__ == "__main__":
    backend_config = {"num_cpus": 1, "num_gpus": 0.0}
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=2,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=FedAvg(
            fraction_fit=1,
            fraction_evaluate=0,
            min_fit_clients=2,
            min_evaluate_clients=0,
            min_available_clients=2,
        ),
        ray_init_args={"num_cpus": backend_config["num_cpus"]},
    )

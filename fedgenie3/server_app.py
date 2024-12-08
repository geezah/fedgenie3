from pathlib import Path
from typing import Any, Dict

from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from .strategy import GENIE3Strategy


def get_strategy_config(root: Path, network_id: int) -> Dict[str, Any]:
    net_id_to_net_name = {
        1: "in-silico",
        3: "e-coli",
        4: "s-cerevisiae",
    }
    network_name = net_id_to_net_name[network_id]
    reference_network_path = (
        root / f"net{network_id}_{network_name}" / "reference_network_data.tsv"
    )
    transcription_factor_path = (
        root / f"net{network_id}_{network_name}" / "transcription_factors.tsv"
    )
    indices_to_names_path = (
        root / f"net{network_id}_{network_name}" / "indices_to_names.tsv"
    )

    node_config = {
        "reference_network_path": reference_network_path,
        "transcription_factors_path": transcription_factor_path,
        "indices_to_names_path": indices_to_names_path,
    }
    return node_config


def server_fn_simulation(context: Context):
    server_config = ServerConfig(num_rounds=1)
    root = Path("local_data/processed/dream_five").resolve()
    network_id = 1
    strategy_config = get_strategy_config(root, network_id)
    strategy = GENIE3Strategy(
        strategy_config["reference_network_path"],
        strategy_config["transcription_factors_path"],
        strategy_config["indices_to_names_path"],
    )
    return ServerAppComponents(strategy=strategy, config=server_config)


def server_fn(context: Context):
    server_config = ServerConfig(num_rounds=context.run_config["num_rounds"])
    strategy = GENIE3Strategy(
        context.run_config["reference_network_path"],
        context.run_config["transcription_factors_path"],
        context.run_config["indices_to_names_path"],
    )
    return ServerAppComponents(strategy=strategy, config=server_config)


server_app = ServerApp(server_fn=server_fn_simulation)

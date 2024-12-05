from flwr.simulation import run_simulation
from fedgenie3.client_app import client_app
from fedgenie3.server_app import server_app
from typer import Typer
import os

app = Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(num_clients: int):
    num_cpus = os.cpu_count()
    num_cpus_per_client = num_cpus // num_clients
    backend_config = {
        "client_resources": {"num_cpus": num_cpus_per_client, "num_gpus": 0.0}
    }
    # Run simulation
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=num_clients,
        backend_config=backend_config,
    )


if __name__ == "__main__":
    app()

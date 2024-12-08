from flwr.simulation import run_simulation
from typer import Typer

from fedgenie3.client_app import client_app
from fedgenie3.server_app import server_app

app = Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(num_clients: int):
    backend_config = {
        "client_resources": {"num_cpus": 1, "num_gpus": 0.0}
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

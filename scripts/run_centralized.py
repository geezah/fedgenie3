from pathlib import Path

from typer import Typer

from fedgenie3.data.dataset import load_dream_five
from fedgenie3.genie3.configs import get_regressor_init_params
from fedgenie3.genie3.run import run

app = Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    root: Path = Path("local_data/processed/dream_five"),
    network_id: int = 1,
    dev_run: bool = False,
):
    print(
        f"Running (non-federated) GRN inference for network {network_id} in {root}"
    )
    grn_dataset = load_dream_five(root, network_id)
    regressor_type = "LGBM"
    regressor_init_params = get_regressor_init_params(regressor_type)
    run(
        grn_dataset,
        regressor_type=regressor_type,
        regressor_init_params=regressor_init_params,
        dev_run=dev_run,
    )


if __name__ == "__main__":
    app()

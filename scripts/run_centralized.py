from typer import Typer
from fedgenie3.genie3.run import run
from pathlib import Path

app = Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    root: Path = Path("data/processed/dream_five"),
    network_id: int = 1,
):
    run(root, network_id)


if __name__ == "__main__":
    app()

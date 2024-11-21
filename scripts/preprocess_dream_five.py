from typer import Typer
from fedgenie3.data.preprocessing.dream_five import preprocess_dream_five
from pathlib import Path

app = Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    raw_data_root: Path = Path("local_data/raw/syn2787209/Gene Network Inference"),
    processed_data_root: Path = Path("local_data/processed/dream_five"),
):
    preprocess_dream_five(raw_data_root, processed_data_root)


if __name__ == "__main__":
    app()

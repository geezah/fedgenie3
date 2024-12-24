from pathlib import Path

from typer import Typer
from yaml import safe_load

from genie3.data import init_grn_dataset
from genie3.schema import ComposedConfig
from genie3.modeling import GENIE3

app = Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    cfg_path: Path,
):
    with open(cfg_path, "r") as f:
        cfg = safe_load(f)
    cfg = ComposedConfig.model_validate(cfg)
    grn_dataset = init_grn_dataset(
        cfg.data.gene_expressions_path,
        cfg.data.transcription_factors_path,
        cfg.data.reference_network_path,
    )
    genie3 = GENIE3(
        dataset=grn_dataset,
        regressor_config=cfg.regressor,
    )
    genie3.fit()
    results = genie3.evaluate()
    print(results)


if __name__ == "__main__":
    app()

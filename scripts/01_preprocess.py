from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from typer import Typer

app = Typer()


def check_if_steady_state_file(file: Path):
    if file.stem.endswith(("dualknockouts_indexes", "timeseries")):
        return False
    return True


def yield_steady_state_files(root: Path):
    for file in root.rglob("**/*.tsv"):
        if check_if_steady_state_file(file):
            yield file


def yield_time_series_files(root: Path):
    for file in root.rglob("**/*_timeseries.tsv"):
        yield file


class Mode(Enum):
    STEADY_STATE = "ss"
    TIME_SERIES = "ts"


def yield_files_by_mode(root: Path, mode: Mode):
    match mode:
        case Mode.STEADY_STATE:
            yield from yield_steady_state_files(root)
        case Mode.TIME_SERIES:
            yield from yield_time_series_files(root)
        case _:
            raise ValueError(f"Unknown mode: {mode}")


@app.command()
def main(root: Path, output: Path, mode: Mode):
    logger.info(
        f"Root Directory: {root}, Output Parent Directory of CSV: {output}, Mode: {mode}"
    )
    combined_df = pd.DataFrame()
    generator = yield_files_by_mode(root, mode)
    for file in generator:
        logger.info(f"Processing {file.name}...")
        df = pd.read_csv(file, delimiter="\t", dtype=np.float32)
        # Ensure all columns are float and get the highest precision
        combined_df = pd.concat([combined_df, df], ignore_index=True, axis=0)
        logger.info(
            f"Concatenated {df.shape[0]} rows from {file.name} to the combined dataframe"
        )
    if combined_df.empty:
        logger.warning("No files found to concatenate!")
        return
    logger.info(f"Writing the combined dataframe to {output}")
    output = output / mode.value
    output.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(
        output / f"{root.stem}.tsv", index=False, sep="\t", 
    )
    logger.info("Done!")


if __name__ == "__main__":
    app()

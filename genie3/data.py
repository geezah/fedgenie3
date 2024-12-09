from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd


# TODO: pydantic.BaseModel instead of dataclasses.dataclass
@dataclass
class GRNDataset:
    gene_expressions: pd.DataFrame
    transcription_factors: Optional[pd.Series]
    reference_network: Optional[pd.DataFrame]


def load_transcription_factor_data(
    transcription_factor_path: Path,
) -> pd.Series:
    series = pd.read_csv(
        transcription_factor_path, sep="\t", header=0
    ).squeeze()
    assert series.name == "transcription_factor"
    return series


def load_gene_expression_data(
    gene_expression_path: Path,
) -> pd.DataFrame:
    return pd.read_csv(gene_expression_path, sep="\t")


def load_reference_network_data(reference_network_path: Path) -> pd.DataFrame:
    df = pd.read_csv(reference_network_path, sep="\t", header=0)
    assert df.columns.to_list() == [
        "transcription_factor",
        "target_gene",
        "label",
    ]
    return df


def get_names_to_indices_mapping(dataframe: pd.DataFrame) -> Dict[str, int]:
    return {name: index for index, name in enumerate(dataframe.columns)}


def get_indices_to_names_mapping(dataframe: pd.DataFrame) -> Dict[int, str]:
    return {index: name for index, name in enumerate(dataframe.columns)}


def map_data(
    data: Union[pd.Series, pd.DataFrame],
    mapping: Dict,
    subset: Optional[List[str]] = None,
) -> Union[pd.Series, pd.DataFrame]:
    # If data is a Series, apply the function to each element
    if isinstance(data, pd.Series):
        return data.map(mapping)

    # If data is a DataFrame, apply the function to each element for each column in the subset.
    # If subset is None, apply the function to each element in all columns.
    elif isinstance(data, pd.DataFrame):
        if subset is not None:
            data[subset] = data[subset].map(lambda x: mapping[x])
            return data
        else:
            return data.map(lambda x: mapping[x])

    else:
        raise ValueError(
            f"`data` must be a pd.Series or pd.DataFrame. Got {type(data)}."
        )

def get_transcription_factor_indices(
    dataset: GRNDataset,
) -> Optional[List[int]]:
    names_to_indices = get_names_to_indices_mapping(dataset.gene_expressions)
    transcription_factor_indices = None
    if dataset.transcription_factors is not None:
        transcription_factor_indices = map_data(
            dataset.transcription_factors, names_to_indices
        ).tolist()
    return transcription_factor_indices


def construct_grn_dataset(
    gene_expressions_path: Path,
    transcription_factor_path: Optional[Path],
    reference_network_path: Optional[Path],
) -> GRNDataset:
    gene_expressions: pd.DataFrame = load_gene_expression_data(
        gene_expressions_path
    )
    transcription_factors = None
    if transcription_factor_path is not None:
        transcription_factors: pd.Series = load_transcription_factor_data(
            transcription_factor_path
        )

    reference_network = None
    if reference_network_path is not None:
        reference_network: pd.DataFrame = load_reference_network_data(
            reference_network_path
        )
    return GRNDataset(
        gene_expressions=gene_expressions,
        reference_network=reference_network,
        transcription_factors=transcription_factors,
    )


if __name__ == "__main__":
    pass

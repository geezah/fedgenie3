from typing import Dict, List, Optional, Union

import pandas as pd


def get_names_to_indices_mapping(gene_names: List[str]) -> Dict[str, int]:
    return {name: index for index, name in enumerate(gene_names)}


def map_data(
    data: Union[pd.Series, pd.DataFrame],
    mapping: Union[List, Dict],
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


def map_gene_indices_to_names(
    gene_ranking_with_indices: pd.DataFrame, gene_names: List[str]
) -> pd.DataFrame:
    gene_ranking_with_names = map_data(
        gene_ranking_with_indices,
        gene_names,
        subset=["transcription_factor", "target_gene"],
    )
    return gene_ranking_with_names

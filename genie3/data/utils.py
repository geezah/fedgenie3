from typing import Dict, List, Optional, Union

import pandas as pd


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

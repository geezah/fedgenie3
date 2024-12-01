from typing import List, Optional, Tuple

import pandas as pd
from numpy.typing import NDArray


def preprocess(
    gene_expression_data: pd.DataFrame,
    transcription_factor_data: Optional[pd.Series] = None,
) -> Tuple[NDArray, List[int]]:
    inputs = gene_expression_data.values
    if transcription_factor_data is None:
        # Get indices of all columns, since no TF data is provided
        transcription_factor_indices = list(range(inputs.shape[1]))
    # Get indices of TF genes from the columns in the expression data
    else:
        transcription_factor_indices = _get_indices_of_tfs_in_expression_data(
            gene_expression_data, transcription_factor_data
        )
    return inputs, transcription_factor_indices


def _get_indices_of_tfs_in_expression_data(
    gene_expression_data: pd.DataFrame,
    transcription_factor_data: pd.Series,
) -> List[int]:
    return [
        gene_expression_data.columns.get_loc(tf)
        for tf in transcription_factor_data
    ]


def postprocess(
    gene_ranking_df: pd.DataFrame,
    gene_expressions: pd.DataFrame,
) -> pd.DataFrame:
    assert gene_ranking_df.columns.to_list() == [
        "transcription_factor",
        "target_gene",
        "importance",
    ]
    # Map indices to gene names
    gene_ranking_df["transcription_factor"] = gene_ranking_df[
        "transcription_factor"
    ].apply(lambda i: gene_expressions.columns[i])
    gene_ranking_df["target_gene"] = gene_ranking_df["target_gene"].apply(
        lambda i: gene_expressions.columns[i]
    )
    return gene_ranking_df

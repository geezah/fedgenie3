import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from fedgenie3.genie3.config import GENIE3Config


def _load_data(config: GENIE3Config):
    gene_expressions = pd.read_csv(config.gene_expression_path, delimiter="\t")
    return gene_expressions.values


def _init_model(config: GENIE3Config):
    if config.tree_method == "RF":
        return RandomForestRegressor(
            n_estimators=config.n_estimators,
            max_features=config.max_features,
            random_state=config.random_state,
            verbose=1,
        )
    elif config.tree_method == "ET":
        return ExtraTreesRegressor(
            n_estimators=config.n_estimators,
            max_features=config.max_features,
            random_state=config.random_state,
            verbose=1,
        )
    else:
        raise ValueError(f"Unknown tree_method: {config.tree_method}")


def _prepare_inputs(
    gene_expressions: NDArray, target_gene_idx: int, config: GENIE3Config
):
    # Remove target gene from regulator list and gene expression matrix
    regulator_indices = [
        i for i, _ in enumerate(config.regulators) if i != target_gene_idx
    ]
    X = gene_expressions[:, regulator_indices]
    y = gene_expressions[:, target_gene_idx]
    return X, y, regulator_indices


def _get_ranked_list(
    importance_matrix: NDArray[np.float32], config: GENIE3Config
) -> pd.DataFrame:
    assert (
        importance_matrix.shape[0] == importance_matrix.shape[1]
    ), f"Importance matrix must be square. Got shape: {importance_matrix.shape}"

    gene_rankings = []
    for i in range(importance_matrix.shape[0]):
        gene_importances = importance_matrix[i]
        sorted_indices = np.argsort(gene_importances)[::-1]
        top_k_gene_regulations = [
            (config.gene_names[j], config.gene_names[i], gene_importances[j])
            for j in sorted_indices[: config.top_k_regulators]
        ]
        gene_rankings.extend(top_k_gene_regulations)
    gene_rankings = pd.DataFrame(
        gene_rankings, columns=["regulator_gene", "target_gene", "importance"]
    )
    gene_rankings = gene_rankings.sort_values(
        by="importance", ascending=False
    ).reset_index(drop=True)
    return gene_rankings


def _compute_importance_matrix(
    gene_expressions: NDArray[np.float32], config: GENIE3Config
) -> NDArray[np.float32]:
    num_genes = gene_expressions.shape[1]
    importance_matrix = np.zeros((num_genes, num_genes))

    for i in range(num_genes):
        model = _init_model(config)
        X, y, feature_candidates = _prepare_inputs(gene_expressions, i, config)
        model.fit(X, y)
        importance_matrix[i, feature_candidates] = model.feature_importances_
    return importance_matrix


def genie3(gene_expressions: NDArray[np.float32], config: GENIE3Config) -> pd.DataFrame:
    assert len(gene_expressions.shape) == 2, "Input must be a 2D array"
    assert gene_expressions.shape[1] == len(config.gene_names), (
        f"Number of columns in gene_expressions ({gene_expressions.shape[1]}) "
        f"must match the number of genes in gene_names ({len(config.gene_names)})"
    )
    importance_matrix = _compute_importance_matrix(gene_expressions, config)
    gene_ranking = _get_ranked_list(importance_matrix, config)
    return gene_ranking


def main(config: GENIE3Config):
    gene_expressions = _load_data(config)
    gene_ranking = genie3(gene_expressions, config)
    print("GENIE3 Done!")
    gene_ranking.to_csv(
        f"{config.gene_expression_path.stem}_GENIE3_{config.tree_method}_rankings.tsv",
        sep="\t",
        index=False,
    )


if __name__ == "__main__":
    config = GENIE3Config(
        gene_expression_path="data/processed/ss/DREAM4_InSilico_Size10.tsv",
        gene_names=None,
        regulators=None,
        tree_method="RF",
        max_features="sqrt",
        n_estimators=30,
        random_state=42,
        top_k_regulators=1,
    )
    # import json
    # config_json = config.model_dump(mode='json')
    # json.dump(config_json, open("config.json", "w"))
    main(config)

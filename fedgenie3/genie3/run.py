from typing import Any, Dict

from fedgenie3.data.dataset import GRNDataset
from fedgenie3.genie3.configs import get_regressor_init_params
from fedgenie3.genie3.eval import evaluate_ranking
from fedgenie3.genie3.modeling import GENIE3


def run(dataset: GRNDataset, **kwargs: Dict[str, Any]):
    regressor_type = kwargs.get("regressor_type", "LGBM")
    regressor_init_params = kwargs.get(
        "regressor_init_params", get_regressor_init_params(regressor_type)
    )
    genie3 = GENIE3(
        regressor_type=regressor_type,
        regressor_init_params=regressor_init_params,
    )
    importance_matrix = genie3.calculate_importances(
        dataset.gene_expressions.values,
        dataset.metadata.transcription_factor_indices.to_list(),
        dev_run=kwargs.get("dev_run", False),
    )
    gene_ranking_with_indices = GENIE3.rank_genes_by_importance(
        importance_matrix, dataset.metadata.transcription_factor_indices
    )
    gene_ranking_with_names = GENIE3.map_indices_to_gene_names(
        gene_ranking_with_indices, dataset.metadata.gene_names_to_indices
    )
    results = evaluate_ranking(
        gene_ranking_with_names, dataset.reference_network
    )
    print(results)
    return results

from pathlib import Path

from fedgenie3.data.dataset import load_dream_five
from fedgenie3.genie3.eval import evaluate_ranking
from fedgenie3.genie3.modeling import GENIE3
from fedgenie3.genie3.configs import get_regressor_init_params


def run(root: Path, network_id: int, dev_run: bool = False):
    print(
        f"Running (non-federated) GRN inference for network {network_id} in {root}"
    )
    grn_dataset = load_dream_five(root, network_id)
    regressor_type = "LGBM"
    regressor_init_params = get_regressor_init_params(regressor_type)
    genie3 = GENIE3(
        regressor_type=regressor_type,
        regressor_init_params=regressor_init_params,
    )
    importance_matrix = genie3.calculate_importances(
        grn_dataset.gene_expressions.values,
        grn_dataset.metadata.transcription_factor_indices.to_list(),
    )
    gene_ranking_with_indices = GENIE3.rank_genes_by_importance(
        importance_matrix, grn_dataset.metadata.transcription_factor_indices
    )
    gene_ranking_with_names = GENIE3.map_indices_to_gene_names(
        gene_ranking_with_indices, grn_dataset.metadata.gene_names_to_indices
    )
    results = evaluate_ranking(
        gene_ranking_with_names, grn_dataset.reference_network
    )
    print(results)
    return results

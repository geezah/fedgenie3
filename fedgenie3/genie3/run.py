from pathlib import Path

from fedgenie3.data.dataset import load_dream_five
from fedgenie3.genie3.eval import evaluate
from fedgenie3.genie3.modeling import GENIE3


def run(root: Path, network_id: int, dev_run: bool = False):
    print(
        f"Running (non-federated) GRN inference for network {network_id} in {root}"
    )
    grn_dataset = load_dream_five(root, network_id)

    tree_method = "RF"
    tree_init_kwargs = {
        "n_estimators": 100,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": -1,
    }
    genie3 = GENIE3(tree_method=tree_method, tree_init_kwargs=tree_init_kwargs)
    importance_matrix = genie3.calculate_importances(
        grn_dataset.gene_expressions.values,
        grn_dataset.metadata.transcription_factor_indices.to_list(),
        dev_run=dev_run,
    )
    gene_ranking_with_indices = GENIE3.rank_genes_by_importance(
        importance_matrix, grn_dataset.metadata.transcription_factor_indices
    )
    gene_ranking_with_names = GENIE3.map_indices_to_gene_names(
        gene_ranking_with_indices, grn_dataset.metadata.gene_names_to_indices
    )
    results = evaluate(gene_ranking_with_names, grn_dataset.reference_network)
    print(results)

from pathlib import Path
from fedgenie3.data.dataset import GRNDataset
from fedgenie3.data.processor import GRNProcessor
from fedgenie3.genie3.modeling import GENIE3
from fedgenie3.genie3.eval import evaluate

def run(root : Path, network_id : int):
    print(f"Running (non-federated) GRN inference for network {network_id} in {root}")
    net_id_to_net_name = {
        1: "in-silico",
        3: "e-coli",
        4: "s-cerevisiae",
    }
    network_name = net_id_to_net_name[network_id]
    GENE_EXPRESSION_PATH = (
        root
        / f"net{network_id}_{network_name}"
        / f"net{network_id}_{network_name}_expression_data.tsv"
    )
    REFERENCE_NETWORK_PATH = (
        root
        / f"net{network_id}_{network_name}"
        / f"net{network_id}_{network_name}_reference_network_data.tsv"
    )
    TRANSCRIPTION_FACTOR_PATH = (
        root
        / f"net{network_id}_{network_name}"
        / f"net{network_id}_{network_name}_transcription_factors.tsv"
    )

    grn_dataset = GRNDataset(
        gene_expression_path=GENE_EXPRESSION_PATH,
        reference_network_path=REFERENCE_NETWORK_PATH,
        transcription_factor_path=TRANSCRIPTION_FACTOR_PATH,
    ) 

    inputs, transcription_factor_indices = GRNProcessor.preprocess(grn_dataset.gene_expression_data, grn_dataset.transcription_factor_data)

    tree_method = "RF"
    tree_init_kwargs = {
        "n_estimators":  100,
        "max_features": 'sqrt',
        "random_state": 42,
        "n_jobs": -1,
    }
    genie3 = GENIE3(tree_method=tree_method, tree_init_kwargs=tree_init_kwargs)

    importance_matrix = genie3.compute_importances(inputs, transcription_factor_indices, dev_run=False)
    gene_ranking_with_indices = genie3.rank_genes_by_importance(importance_matrix, transcription_factor_indices)
    gene_ranking_with_names = GRNProcessor.postprocess(gene_ranking_with_indices, grn_dataset.gene_expression_data)
    
    results = evaluate(gene_ranking_with_names, grn_dataset.reference_network_data)
    print(results)
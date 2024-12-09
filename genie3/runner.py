from genie3.data import (
    GRNDataset,
    get_indices_to_names_mapping,
    get_transcription_factor_indices,
    map_data,
)
from genie3.eval import evaluate_ranking
from genie3.modeling import GENIE3
from genie3.schema import RegressorConfig


class GENIE3Runner:
    def __init__(self, dataset: GRNDataset, regressor_config: RegressorConfig):
        self.dataset = dataset
        self.regressor_config = regressor_config

    def _preprocess_data(self):
        transcription_factor_indices = get_transcription_factor_indices(
            self.dataset
        )
        return transcription_factor_indices

    def _postprocess_data(self, gene_ranking_with_indices):
        indices_to_names = get_indices_to_names_mapping(
            self.dataset.gene_expressions
        )
        gene_ranking_with_names = map_data(
            gene_ranking_with_indices,
            indices_to_names,
            subset=["transcription_factor", "target_gene"],
        )
        return gene_ranking_with_names

    def __call__(self, dev_run=False):
        transcription_factor_indices = self._preprocess_data()
        genie3 = GENIE3(
            regressor_name=self.regressor_config.name,
            regressor_init_params=self.regressor_config.init_params,
        )
        gene_ranking_with_indices = genie3(
            self.dataset.gene_expressions.values,
            transcription_factor_indices,
            dev_run,
        )
        gene_ranking_with_names = self._postprocess_data(
            gene_ranking_with_indices
        )
        results = None
        if self.dataset.reference_network is not None:
            results = evaluate_ranking(
                gene_ranking_with_names, self.dataset.reference_network
            )
        print(gene_ranking_with_names.head())
        print(results)
        return gene_ranking_with_names, results

from typing import Dict, List, Tuple

import numpy as np
from flwr.client import NumPyClient

from fedgenie3.genie3.config import GENIE3Config
from fedgenie3.genie3.modeling import _init_model, _load_data, _prepare_inputs


class GENIE3Client(NumPyClient):
    def __init__(self, config: GENIE3Config):
        self.config = config
        self.gene_expressions = _load_data(config)
        self.gene_names = config.gene_names

    def get_parameters(
        self, config: Dict[str, float] | None = None
    ) -> List[np.ndarray]:
        # Return an empty list or appropriate parameters
        return []

    def _compute_importance_matrix(self) -> np.ndarray:
        num_genes = self.gene_expressions.shape[1]
        importance_matrix = np.zeros((num_genes, num_genes))

        for i in range(num_genes):
            model = _init_model(self.config)
            X, y, feature_candidates = _prepare_inputs(
                self.gene_expressions, i, self.config
            )
            model.fit(X, y)
            importance_matrix[i, feature_candidates] = model.feature_importances_
        return importance_matrix

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, float] | None = None,
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Compute the local importance matrix
        importance_matrix = self._compute_importance_matrix()
        # Return the importance matrix as a list of numpy arrays
        return [importance_matrix], self.gene_expressions.shape[0], {}

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, float] | None = None,
    ) -> Tuple[float, int, Dict]:
        # Evaluation is not required for this task
        return 0.0, 0, {}

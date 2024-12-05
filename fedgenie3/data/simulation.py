from copy import deepcopy
from pathlib import Path
from typing import Callable, List, Literal, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from fedgenie3.data.dataset import GRNDataset, construct_grn_dataset


def _split_evenly(
    dataset: GRNDataset,
    n_partitions: int,
    random_state: int = 42,
) -> List[GRNDataset]:
    """Split gene expression data into partitions of equal size."""

    np.random.seed(random_state)
    gene_expression_inputs = dataset.gene_expressions.values
    np.random.shuffle(gene_expression_inputs)
    gene_expression_partitions = np.array_split(
        gene_expression_inputs, n_partitions
    )
    dataset_partitions: List[GRNDataset] = []
    for i in range(n_partitions):
        dataset_partition = deepcopy(dataset)
        dataset_partition.gene_expressions = pd.DataFrame(
            gene_expression_partitions[i],
            columns=dataset.gene_expressions.columns,
        )
        dataset_partition.reference_network = dataset.reference_network
        dataset_partition.metadata = dataset.metadata
        dataset_partitions.append(dataset_partition)

    return dataset_partitions


def _split_tf_centric(
    dataset: GRNDataset,
    n_partitions: int,
    random_state: int = 42,
) -> List[GRNDataset]:
    """Simulate TF-centric partitions of gene expression data."""

    np.random.seed(random_state)
    gene_expression_inputs = dataset.gene_expressions.values
    transcription_factor_indices = (
        dataset.metadata.transcription_factor_indices
    )

    if n_partitions > gene_expression_inputs.shape[0]:
        raise ValueError(
            "Number of partitions cannot exceed number of samples"
        )

    # Extract TF expression data for clustering
    tf_expression: npt.NDArray[np.float64] = gene_expression_inputs[
        :, transcription_factor_indices
    ]

    # Normalize TF expression
    scaler: StandardScaler = StandardScaler()
    tf_expression_normalized: npt.NDArray[np.float64] = scaler.fit_transform(
        tf_expression
    )

    # Cluster samples based on TF expression patterns
    kmeans: KMeans = KMeans(n_clusters=n_partitions, random_state=42)
    clusters: npt.NDArray[np.int64] = kmeans.fit_predict(
        tf_expression_normalized
    )

    # Create partitions with all genes
    dataset_partitions: List[GRNDataset] = []
    for i in range(n_partitions):
        indices_partition: npt.NDArray[np.int64] = np.where(clusters == i)[0]
        # Keep all genes for samples in this partition
        gene_expression_partition: npt.NDArray[np.float64] = (
            gene_expression_inputs[indices_partition, :]
        )
        dataset_partition = GRNDataset(
            gene_expressions=pd.DataFrame(
                gene_expression_partition,
                columns=dataset.gene_expressions.columns,
            ),
            reference_network=dataset.reference_network,
            metadata=dataset.metadata,
        )
        dataset_partitions.append(dataset_partition)

    return dataset_partitions


def _get_simulation_func(
    simulation_type: Literal["even", "tf_centric"],
) -> Callable[[GRNDataset, int, Optional[int]], List[GRNDataset]]:
    if simulation_type == "even":
        return _split_evenly
    elif simulation_type == "tf_centric":
        return _split_tf_centric
    else:
        raise ValueError("Invalid simulation type")


def simulate_dream_five(
    root: Path,
    network_id: int,
    simulation_type: Literal["even", "tf_centric"] = "even",
    n_partitions: int = 2,
    random_seed: int = 42,
):
    root = Path("local_data/processed/dream_five")
    net_id_to_net_name = {
        1: "in-silico",
        3: "e-coli",
        4: "s-cerevisiae",
    }
    network_name = net_id_to_net_name[network_id]
    gene_expression_path = (
        root / f"net{network_id}_{network_name}" / "gene_expression_data.tsv"
    )
    reference_network_path = (
        root / f"net{network_id}_{network_name}" / "reference_network_data.tsv"
    )
    transcription_factor_path = (
        root / f"net{network_id}_{network_name}" / "transcription_factors.tsv"
    )
    indices_to_names_path = (
        root / f"net{network_id}_{network_name}" / "indices_to_names.tsv"
    )
    dataset = construct_grn_dataset(
        gene_expression_path=gene_expression_path,
        transcription_factor_path=transcription_factor_path,
        reference_network_path=reference_network_path,
        indices_to_names_path=indices_to_names_path,
    )
    simulation_func = _get_simulation_func(simulation_type)
    dataset_partitions: List[GRNDataset] = simulation_func(
        dataset=dataset,
        n_partitions=n_partitions,
        random_state=random_seed,
    )
    return dataset_partitions


if __name__ == "__main__":
    root = Path("local_data/processed/dream_five")
    network_id = 1

    simulation_type = "even"
    num_partitions = 2
    random_seed = 42

    dataset_partitions = simulate_dream_five(
        root, network_id, simulation_type, num_partitions, random_seed
    )
    for i, dataset in enumerate(dataset_partitions):
        print(f"Expressions {i}: {dataset.gene_expressions.head()}")
        print(f"Reference Network {i}: {dataset.reference_network.head()}")
        print(f"Metadata {i}: {dataset.metadata}")

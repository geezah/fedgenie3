"""
This module contains unit tests for the GENIE3Config class from the fedgenie3.genie3.config module.
Disclaimer: This module was AI-generated and then manually checked and modified for correctness.

Fixtures:
    temp_tsv_file: Creates a temporary .tsv file with specified header and data.
    default_gene_expression_file: Provides a default gene expression data file.
    default_config: Provides a default GENIE3Config instance.

Tests:
    test_file_does_not_exist: Ensures ValueError is raised if the gene expression data file does not exist.
    test_file_not_tsv: Ensures ValueError is raised if the gene expression data file is not a .tsv file.
    test_file_no_header: Ensures ValueError is raised if the gene expression data file has no header.
    test_case_1_1: Tests case where gene_names and regulators are None.
    test_case_1_2_valid: Tests case where gene_names is None and valid regulators are provided.
    test_case_1_2_invalid: Tests case where gene_names is None and invalid regulators are provided.
    test_case_2_1: Tests case where gene_names are provided and regulators are None.
    test_case_2_2_valid: Tests case where valid gene_names and regulators are provided.
    test_case_2_2_invalid: Tests case where gene_names and invalid regulators are provided.
    test_top_k_regulators_default: Ensures top_k_regulators defaults to the length of regulators.
    test_top_k_regulators_specified: Ensures top_k_regulators is set when specified.
    test_max_features_float_valid: Tests valid float value for max_features.
    test_max_features_float_invalid: Ensures ValueError is raised for invalid float value for max_features.
    test_max_features_literal_valid: Tests valid literal values for max_features.
    test_random_state_valid: Tests valid random_state.
    test_n_estimators_valid: Tests valid n_estimators.
    test_tree_method_valid: Tests valid tree_method values.
    test_tree_method_invalid: Ensures ValueError is raised for invalid tree_method value.
    test_gene_expression_data_is_directory: Ensures ValueError is raised if gene_expression_data is a directory.
    test_gene_names_override_file_header: Tests that provided gene_names override file header.
    test_large_number_of_genes: Tests with a large number of genes.
    test_missing_gene_expression_data: Ensures ValidationError is raised if gene_expression_data is not provided.
    test_invalid_random_state: Ensures ValueError is raised for invalid random_state (negative integer).
"""
from typing import List
import pytest
import tempfile
from pathlib import Path
from fedgenie3.genie3.config import GENIE3Config  

# Fixture to create a temporary .tsv file with specified header and data
@pytest.fixture
def temp_tsv_file():
    def _create_temp_tsv_file(header : List[str] = None, data : List = None , suffix=".tsv"):
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, mode='w', delete=False)
        tmp_path = Path(tmp.name)
        if header:
            tmp.write("\t".join(header) + "\n")
        if data:
            for row in data:
                tmp.write("\t".join(map(str, row)) + "\n")
        tmp.flush()
        tmp.close()
        return tmp_path
    yield _create_temp_tsv_file
    # Cleanup code can be added here if necessary

# Fixture for a default gene expression data file
@pytest.fixture
def default_gene_expression_file(temp_tsv_file):
    header = ["gene1", "gene2", "gene3"]
    data = [[1, 2, 3]]
    return temp_tsv_file(header=header, data=data)

# Fixture for a default GENIE3Config instance
@pytest.fixture
def default_config(default_gene_expression_file):
    return GENIE3Config(gene_expression_path=default_gene_expression_file)

def test_file_does_not_exist():
    with pytest.raises(ValueError, match="Gene expression data file does not exist"):
        GENIE3Config(gene_expression_path=Path("/non/existent/file.tsv"))

def test_file_not_tsv(temp_tsv_file):
    tmp_path = temp_tsv_file(suffix=".csv")
    with pytest.raises(ValueError, match="Gene expression data file is not a .tsv file"):
        GENIE3Config(gene_expression_path=tmp_path)

def test_file_no_header(temp_tsv_file):
    tmp_path = temp_tsv_file()
    with pytest.raises(ValueError):
        GENIE3Config(gene_expression_path=tmp_path)

def test_case_1_1(temp_tsv_file):
    # Case 1.1: gene_names is None, regulators is None
    header = ["gene1", "gene2", "gene3"]
    data = [[1, 2, 3]]
    tmp_path = temp_tsv_file(header=header, data=data)
    config = GENIE3Config(gene_expression_path=tmp_path)
    assert config.gene_names == header
    assert config.regulators == header

def test_case_1_2_valid(temp_tsv_file):
    # Case 1.2: gene_names is None, regulators provided and valid
    header = ["gene1", "gene2", "gene3"]
    data = [[4, 5, 6]]
    tmp_path = temp_tsv_file(header=header, data=data)
    config = GENIE3Config(
        gene_expression_path=tmp_path,
        regulators=["gene1", "gene3"]
    )
    assert config.gene_names == header
    assert config.regulators == ["gene1", "gene3"]

def test_case_1_2_invalid(temp_tsv_file):
    # Case 1.2: gene_names is None, regulators provided and invalid
    header = ["geneA", "geneB", "geneC"]
    data = [[7, 8, 9]]
    tmp_path = temp_tsv_file(header=header, data=data)
    with pytest.raises(ValueError, match="Regulators .* are not in the list of gene names"):
        GENIE3Config(
            gene_expression_path=tmp_path,
            regulators=["geneX", "geneY"]
        )

def test_case_2_1(temp_tsv_file):
    # Case 2.1: gene_names provided, regulators is None
    header = ["geneX", "geneY", "geneZ"]
    data = [[10, 11, 12]]
    tmp_path = temp_tsv_file(header=header, data=data)
    gene_names = ["gene1", "gene2", "gene3"]
    config = GENIE3Config(
        gene_expression_path=tmp_path,
        gene_names=gene_names
    )
    assert config.gene_names == gene_names
    assert config.regulators == gene_names

def test_case_2_2_valid(temp_tsv_file):
    # Case 2.2: gene_names and regulators provided and valid
    header = ["geneA", "geneB", "geneC"]
    data = [[13, 14, 15]]
    tmp_path = temp_tsv_file(header=header, data=data)
    gene_names = ["gene1", "gene2", "gene3"]
    regulators = ["gene1", "gene2"]
    config = GENIE3Config(
        gene_expression_path=tmp_path,
        gene_names=gene_names,
        regulators=regulators
    )
    assert config.gene_names == gene_names
    assert config.regulators == regulators

def test_case_2_2_invalid(temp_tsv_file):
    # Case 2.2: gene_names and regulators provided but regulators invalid
    header = ["geneM", "geneN", "geneO"]
    data = [[16, 17, 18]]
    tmp_path = temp_tsv_file(header=header, data=data)
    gene_names = ["gene1", "gene2", "gene3"]
    regulators = ["geneX", "geneY"]
    with pytest.raises(ValueError, match="Regulators .* are not in the list of gene names"):
        GENIE3Config(
            gene_expression_path=tmp_path,
            gene_names=gene_names,
            regulators=regulators
        )

def test_top_k_regulators_default(temp_tsv_file):
    # Test that top_k_regulators defaults to len(regulators)
    header = ["gene1", "gene2"]
    data = [[19, 20]]
    tmp_path = temp_tsv_file(header=header, data=data)
    config = GENIE3Config(gene_expression_path=tmp_path)
    assert config.top_k_regulators == 2

def test_top_k_regulators_specified(temp_tsv_file):
    # Test that top_k_regulators is set when specified
    header = ["gene1", "gene2", "gene3"]
    data = [[21, 22, 23]]
    tmp_path = temp_tsv_file(header=header, data=data)
    config = GENIE3Config(
        gene_expression_path=tmp_path,
        top_k_regulators=1
    )
    assert config.top_k_regulators == 1

def test_max_features_float_valid(default_gene_expression_file):
    # Test valid float value for max_features
    config = GENIE3Config(
        gene_expression_path=default_gene_expression_file,
        max_features=0.5
    )
    assert config.max_features == 0.5

def test_max_features_float_invalid(default_gene_expression_file):
    # Test invalid float value for max_features
    with pytest.raises(ValueError):
        GENIE3Config(
            gene_expression_path=default_gene_expression_file,
            max_features=1.5  # Invalid since it's greater than 1.0
        )

def test_max_features_literal_valid(default_gene_expression_file):
    # Test valid literal values for max_features
    for value in ["sqrt", "log2"]:
        config = GENIE3Config(
            gene_expression_path=default_gene_expression_file,
            max_features=value
        )
        assert config.max_features == value

def test_random_state_valid(default_gene_expression_file):
    # Test valid random_state
    config = GENIE3Config(
        gene_expression_path=default_gene_expression_file,
        random_state=123
    )
    assert config.random_state == 123

def test_n_estimators_valid(default_gene_expression_file):
    # Test valid n_estimators
    config = GENIE3Config(
        gene_expression_path=default_gene_expression_file,
        n_estimators=500
    )
    assert config.n_estimators == 500

def test_tree_method_valid(default_gene_expression_file):
    # Test valid tree_method values
    for method in ["RF", "ET"]:
        config = GENIE3Config(
            gene_expression_path=default_gene_expression_file,
            tree_method=method
        )
        assert config.tree_method == method

def test_tree_method_invalid(default_gene_expression_file):
    # Test invalid tree_method value
    with pytest.raises(ValueError):
        GENIE3Config(
            gene_expression_path=default_gene_expression_file,
            tree_method="InvalidMethod"
        )

def test_gene_expression_data_is_directory():
    # Test when gene_expression_data is a directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        with pytest.raises(ValueError, match="Gene expression data file does not exist"):
            GENIE3Config(gene_expression_path=tmp_path)

def test_gene_names_override_file_header(temp_tsv_file):
    # Test that provided gene_names override file header
    header = ["file_gene1", "file_gene2"]
    data = [[34, 35]]
    tmp_path = temp_tsv_file(header=header, data=data)
    gene_names = ["gene1", "gene2"]
    config = GENIE3Config(
        gene_expression_path=tmp_path,
        gene_names=gene_names
    )
    assert config.gene_names == gene_names

def test_large_number_of_genes(temp_tsv_file):
    # Test with a large number of genes
    gene_list = [f"gene{i}" for i in range(1000)]
    data = [range(1000)]
    tmp_path = temp_tsv_file(header=gene_list, data=data)
    config = GENIE3Config(gene_expression_path=tmp_path)
    assert len(config.gene_names) == 1000
    assert len(config.regulators) == 1000

def test_missing_gene_expression_data():
    from pydantic import ValidationError
    # Test when gene_expression_data is not provided
    with pytest.raises(ValidationError):
        GENIE3Config()

def test_invalid_random_state(default_gene_expression_file):
    # Test invalid random_state (negative integer)
    with pytest.raises(ValueError):
        GENIE3Config(
            gene_expression_path=default_gene_expression_file,
            random_state=-1
        )
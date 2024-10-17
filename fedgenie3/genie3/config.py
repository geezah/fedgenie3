from pathlib import Path
from typing import Annotated, List, Literal, Union

from pydantic import BaseModel, Field, PositiveInt, field_validator, model_validator
import csv


class GENIE3Config(BaseModel):
    gene_expression_path: Path = Field(
        ...,
        description="Path to the expression data file",
        examples=["/path/to/gene_expression_data.tsv"],
    )
    gene_names: List[str] | None = Field(
        None,
        description="List of gene names",
        examples=["gene1", "gene2", "gene3"],
    )
    regulators: List[str] | None = Field(
        None,
        description="List of regulator genes",
        examples=[["gene1", "gene2"]],
    )
    tree_method: Literal["RF", "ET"] = Field(
        "RF",
        description="Tree method to use",
        examples=["RF", "ET"],
    )
    n_estimators: PositiveInt = Field(1000)
    max_features: Union[
        Literal["sqrt", "log2"],
        Annotated[float, Field(ge=0.0, le=1.0)],
    ] = Field(
        1.0,
        description=(
            "Maximum number of features to consider at each split. Can be a float "
            "between 0 and 1 or 'sqrt' or 'log2'"
        ),
        examples=[1.0, 0.5, "sqrt", "log2"],
    )
    random_state: PositiveInt = Field(
        42,
        description="Seed for the random number generator. Default is 42.",
        examples=[42],
    )
    top_k_regulators: PositiveInt | None = Field(
        None,
        description="Number of top regulators to return for each gene. If None, all regulators are returned.",
        examples=[1, 5, None],
    )

    @field_validator("gene_expression_path")
    @classmethod
    def validate_gene_expression_path(cls, v):
        if not Path(v).is_file():
            raise ValueError(
                f"Gene expression data file does not exist. Given: {v}. Please provide a valid path."
            )
        if v.suffix.lower() != ".tsv":
            raise ValueError(
                f"Gene expression data file is not a .tsv file. Given: {v}. Please provide a .tsv file."
            )
        return v

    @model_validator(mode="after")
    def check_gene_names_and_regulators(self):
        """
        Validates and sets the gene names and regulators for the instance.
        This method performs the following checks and actions:
        1. If `self.gene_names` is None, it attempts to read the gene names from the
           `self.gene_expression_data` file. The file is expected to be a tab-delimited
           CSV (TSV) with a header row containing the gene names.
        2. If `self.regulators` is None, it sets `self.regulators` to be the same as
           `self.gene_names`.
        3. If `self.regulators` is not None, it checks if all regulators are present
           in the list of gene names. If any regulators are missing, it raises a
           `ValueError`.

        Raises:
            ValueError: If the gene expression data file has no header row, if the file
                cannot be read, or if any regulators are not in the list of gene names.

        Returns:
            self: The instance with validated and possibly updated `gene_names` and `regulators`.
        """

        if self.gene_names is None:
            try:
                with open(self.gene_expression_path, "r", newline="") as f:
                    reader = csv.reader(f, delimiter="\t")
                    header = next(reader)
                    if not header:
                        raise ValueError(
                            "The gene expression data file has no header row."
                        )
                    self.gene_names = header
            except Exception as e:
                raise ValueError(
                    f"Failed to read gene names from gene_expression_data file: {e}"
                )
        if self.regulators is None:
            self.regulators = self.gene_names
        else:
            missing_regulators = set(self.regulators) - set(self.gene_names)
            if missing_regulators:
                raise ValueError(
                    f"Regulators {missing_regulators} are not in the list of gene names."
                )
        return self

    @model_validator(mode="after")
    @classmethod
    def set_top_k_regulators_if_none(cls, model):
        if model.top_k_regulators is None:
            model.top_k_regulators = len(model.regulators)
        return model

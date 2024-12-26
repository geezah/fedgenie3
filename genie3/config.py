from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from .modeling.regressor import RegressorName


class DataConfig(BaseModel):
    """
    DataConfig is a configuration model for specifying paths to various data files used in the application.
    Attributes:
        gene_expressions_path (Path): Path to the gene expression data.
        transcription_factors_path (Optional[Path]): Path to the transcription factor data. Defaults to None.
        reference_network_path (Optional[Path]): Path to the reference network data. Defaults to None.
    """

    gene_expressions_path: Path = Field(
        ..., description="Path to the gene expression data"
    )
    transcription_factors_path: Optional[Path] = Field(
        None, description="Path to the transcription factor data"
    )
    reference_network_path: Optional[Path] = Field(
        None, description="Path to the reference network data"
    )


class RegressorConfig(BaseModel):
    """
    RegressorConfig is a configuration class for specifying the parameters of a supported regressor.
    Attributes:
        name (RegressorName): Type of regressor to use. One of the values defined in RegressorName.
        init_params (Dict[str, Any]): Parameters to initialize the regressor with. Must comply with the regressor's API.
        fit_params (Dict[str, Any]): Parameters to fit the regressor with. Must comply with the regressor's API.
    """

    name: RegressorName = Field(
        "ET",
        description=f"Type of regressor to use. One of: {RegressorName.__args__}",
    )
    init_params: Dict[str, Any] = Field(
        {"n_estimators": 100, "max_depth": 3},
        description="Parameters to initialize the regressor with. Must comply with the regressor's API.",
    )
    fit_params: Dict[str, Any] = Field(
        {},
        description="Parameters to fit the regressor with. Must comply with the regressor's API.",
    )


class GENIE3Config(BaseModel):
    """
    Configuration class for the GENIE3 pipeline.
    Attributes:
        data (DataConfig): Configuration for the data.
        regressor (RegressorConfig): Configuration for the regressor.
    """
    data: DataConfig
    regressor: RegressorConfig


if __name__ == "__main__":
    from pprint import pprint

    from yaml import safe_load

    CFG_PATH = Path("configs/lightgbm.yaml")
    with open(CFG_PATH, "r") as f:
        cfg = safe_load(f)
    cfg = GENIE3Config.model_validate(cfg)
    print(cfg.data.gene_expressions_path)
    print(cfg.data.transcription_factors_path)
    print(cfg.data.reference_network_path)
    print(cfg.regressor.name)
    pprint(cfg.regressor.init_params)

from .extratrees import DefaultExtraTreesConfiguration, ExtraTreesRegressor
from .lightgbm import DefaultLightGBMConfiguration, LGBMRegressor
from .randomforest import (
    DefaultRandomForestConfiguration,
    RandomForestRegressor,
)
from .xgboost import DefaultXGBoostConfiguration, XGBRegressor

RegressorFactory = {
    "RandomForestRegressor": RandomForestRegressor,
    "ExtraTreesRegressor": ExtraTreesRegressor,
    "LGBMRegressor": LGBMRegressor,
    "XGBRegressor": XGBRegressor,
}
ConfigurationFactory = {
    "RandomForestRegressor": DefaultRandomForestConfiguration,
    "ExtraTreesRegressor": DefaultExtraTreesConfiguration,
    "LGBMRegressor": DefaultLightGBMConfiguration,
    "XGBRegressor": DefaultXGBoostConfiguration,
}

__all__ = ["RegressorFactory", "ConfigurationFactory"]

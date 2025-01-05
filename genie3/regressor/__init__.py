from .extratrees import DefaultExtraTreesConfiguration, ExtraTreesRegressor
from .gradientboosting import (
    DefaultGradientBoostingConfiguration,
    GradientBoostingRegressor,
)
from .lightgbm import DefaultLightGBMConfiguration, LGBMRegressor
from .randomforest import (
    DefaultRandomForestConfiguration,
    RandomForestRegressor,
)

RegressorFactory = {
    "RandomForestRegressor": RandomForestRegressor,
    "ExtraTreesRegressor": ExtraTreesRegressor,
    "LGBMRegressor": LGBMRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
}
ConfigurationFactory = {
    "RandomForestRegressor": DefaultRandomForestConfiguration,
    "ExtraTreesRegressor": DefaultExtraTreesConfiguration,
    "GradientBoostingRegressor": DefaultGradientBoostingConfiguration,
    "LGBMRegressor": DefaultLightGBMConfiguration,
}

__all__ = ["RegressorFactory", "ConfigurationFactory"]

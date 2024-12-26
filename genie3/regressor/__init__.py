from .extratrees import ExtraTreesRegressor
from .lightgbm import LGBMRegressor
from .randomforest import RandomForestRegressor

RegressorFactory = {
    "RandomForestRegressor": RandomForestRegressor,
    "ExtraTreesRegressor": ExtraTreesRegressor,
    "LGBMRegressor": LGBMRegressor,
}

__all__ = [
    "RegressorFactory",
]

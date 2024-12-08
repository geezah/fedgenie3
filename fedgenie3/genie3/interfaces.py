from typing import Any, Dict

import lightgbm as lgb
from numpy.typing import NDArray
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
)
from sklearn.ensemble._forest import ForestRegressor
from sklearn.model_selection import train_test_split


def initialize_regressor(
    regressor_type: str, regressor_init_params: Dict[str, Any]
):
    if regressor_type == "RF":
        return RandomForestRegressor(**regressor_init_params)
    elif regressor_type == "ET":
        return ExtraTreesRegressor(**regressor_init_params)
    elif regressor_type == "LGBM":
        return lgb.LGBMRegressor(**regressor_init_params)
    else:
        raise ValueError(
            "Invalid tree method. Choose between: ['RF', 'ET', 'GBDT', 'LGBM']"
        )


def compute_importance_scores(
    X: NDArray, y: NDArray, regressor: Any, **fit_params: Dict[str, Any]
) -> NDArray:
    if isinstance(regressor, lgb.LGBMRegressor):
        return _compute_importance_scores_lgbm(X, y, regressor, **fit_params)
    elif isinstance(regressor, ForestRegressor):
        return _compute_importance_scores_scikit(X, y, regressor, **fit_params)
    else:
        raise ValueError("Invalid regressor type")


def _compute_importance_scores_lgbm(
    X: NDArray,
    y: NDArray,
    regressor: lgb.LGBMRegressor,
    **kwargs: Dict[str, Any],
) -> NDArray:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    regressor = regressor.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    return regressor.feature_importances_


def _compute_importance_scores_scikit(
    X: NDArray,
    y: NDArray,
    regressor: ForestRegressor,
    **fit_kwargs: Dict[str, Any],
) -> NDArray:
    regressor.fit(X, y, **fit_kwargs)
    return regressor.feature_importances_

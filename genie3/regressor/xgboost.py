from xgboost import XGBRegressor as _XGBRegressor
from typing import Any, Dict

from lightgbm import LGBMRegressor as _LGBMRegressor
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from .protocol import RegressorProtocol

DefaultXGBoostConfiguration = {
    "init_params": {
        "n_estimators": 100,
        "learning_rate": 0.01,
        "max_depth": 3,
        "early_stopping_rounds": 10,
        "random_state": 42,
        "importance_type": "gain",
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbosity": 0,
        "n_jobs" : 8
    },
    "fit_params": {
        "verbose": 0,
    },
}


class XGBRegressor(RegressorProtocol):
    def __init__(self, **init_params: Dict[str, Any]):
        self.regressor: _LGBMRegressor = _XGBRegressor(
            **init_params
            if init_params
            else DefaultXGBoostConfiguration["init_params"]
        )

    def fit(self, X: NDArray, y: NDArray, **fit_params: Dict[str, Any]) -> Any:
        # In the case that early_stopping_rounds is provided, we need to split the data into training and validation sets.
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=42
        )
        regressor = self.regressor.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            **fit_params
            if fit_params
            else DefaultXGBoostConfiguration["fit_params"],
        )
        importances = regressor.feature_importances_
        normalized_importances = importances / importances.sum()
        self.feature_importances = normalized_importances

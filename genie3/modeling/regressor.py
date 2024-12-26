from typing import Any, Dict, Literal, Protocol

from lightgbm import LGBMRegressor as _LGBMRegressor
from numpy.typing import NDArray
from sklearn.ensemble import (
    ExtraTreesRegressor as _ExtraTreesRegressor,
)
from sklearn.ensemble import (
    RandomForestRegressor as _RandomForestRegressor,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor as _XGBRegressor

RegressorName = Literal["RF", "ET", "LGBM", "XGB"]


class RegressorProtocol(Protocol):
    def fit(
        self, X: NDArray, y: NDArray, **fit_kwargs: Dict[str, Any]
    ) -> Any: ...

    @property
    def feature_importances(self) -> NDArray:
        if not hasattr(self, "_feature_importances"):
            raise ValueError(
                "Model has not been fitted yet. Therefore, no feature importances available."
            )
        return self._feature_importances

    @feature_importances.setter
    def feature_importances(self, value: NDArray) -> None:
        self._feature_importances = value


class RandomForestRegressor(RegressorProtocol):
    def __init__(self, **init_params: Dict[str, Any]):
        self.regressor: _RandomForestRegressor = _RandomForestRegressor(
            **init_params
        )

    def fit(self, X: NDArray, y: NDArray, **fit_params: Dict[str, Any]) -> Any:
        self.regressor.fit(X, y, **fit_params)
        self.feature_importances = self.regressor.feature_importances_


class ExtraTreesRegressor(RegressorProtocol):
    def __init__(self, **init_params: Dict[str, Any]):
        self.regressor: _ExtraTreesRegressor = _ExtraTreesRegressor(
            **init_params
        )

    def fit(self, X: NDArray, y: NDArray, **fit_params: Dict[str, Any]) -> Any:
        self.regressor.fit(X, y, **fit_params)
        self.feature_importances = self.regressor.feature_importances_


class LGBMRegressor(RegressorProtocol):
    def __init__(self, **init_params: Dict[str, Any]):
        self.regressor = _LGBMRegressor(**init_params)

    def fit(self, X: NDArray, y: NDArray, **fit_params: Dict[str, Any]) -> Any:
        # In the case that early_stopping_rounds is provided, we need to split the data into training and validation sets.
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=42
        )
        regressor = self.regressor.fit(
            X_train, y_train, eval_set=[(X_val, y_val)], **fit_params
        )
        importances = regressor.feature_importances_
        normalized_importances = importances / importances.sum()
        self.feature_importances = normalized_importances


class XGBRegressor(RegressorProtocol):
    def __init__(self, **init_params: Dict[str, Any]):
        self.regressor = _XGBRegressor(**init_params)

    def fit(self, X: NDArray, y: NDArray, **fit_params: Dict[str, Any]) -> Any:
        test_size = fit_params.get("test_size", 0.1)
        random_state = fit_params.get("random_state", 42)

        # In the case that early_stopping_rounds is provided, we need to split the data into training and validation sets.
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        regressor = self.regressor.fit(
            X_train, y_train, eval_set=[(X_val, y_val)], **fit_params
        )
        importances = regressor.feature_importances_
        normalized_importances = importances / importances.sum()
        self.feature_importances = normalized_importances


def initialize_regressor(
    name: RegressorName, init_params: Dict[str, Any]
) -> RegressorProtocol:
    if name == "RF":
        return RandomForestRegressor(**init_params)
    elif name == "ET":
        return ExtraTreesRegressor(**init_params)
    elif name == "LGBM":
        return LGBMRegressor(**init_params)
    elif name == "XGB":
        return XGBRegressor(**init_params)
    else:
        raise ValueError(
            f"Invalid tree method. Choose between: {RegressorName.__args__}"
        )


if __name__ == "__main__":
    from sklearn.datasets import make_regression

    # Example usage
    X, y = make_regression(n_samples=100, n_features=2, noise=0.1)

    rf_regressor = RandomForestRegressor()
    rf_regressor.fit(X, y)
    print(rf_regressor.feature_importances)

    et_regressor = ExtraTreesRegressor()
    et_regressor.fit(X, y)
    print(et_regressor.feature_importances)

    lgbm_regressor = LGBMRegressor(verbose=-1)
    lgbm_regressor.fit(
        X,
        y,
    )
    print(lgbm_regressor.feature_importances)

    xgb_regressor = XGBRegressor()
    xgb_regressor.fit(
        X,
        y,
    )
    print(xgb_regressor.feature_importances)
    try:
        failed_regressor = initialize_regressor("DUMMY", {})
    except ValueError as e:
        print(e)

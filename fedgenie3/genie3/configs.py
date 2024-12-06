def get_regressor_init_params(regressor_type: str):
    if regressor_type == "LGBM":
        return LGBM_INIT_PARAMS
    elif regressor_type == "RF":
        return RF_INIT_PARAMS
    elif regressor_type == "ET":
        return ET_INIT_PARAMS


LGBM_INIT_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.01,
    "max_depth": 3,
    "min_samples_leaf": 1,
    "n_iter_no_change": 25,
    "random_state": 42,
    "importance_type": "gain",
    "extra_trees": True,
    "early_stopping_min_delta": 1e-4,
    "n_jobs": 8,
    "verbosity": -1,
}
RF_INIT_PARAMS = {
    "n_estimators": 1000,
    "random_state": 42,
    "n_jobs": 8,
}
ET_INIT_PARAMS = {
    "n_estimators": 1000,
    "random_state": 42,
    "n_jobs": 8,
}

"""Train-only preprocessing and small, documented regression search spaces."""
from typing import Any
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

MODEL_NAMES = ['mean_baseline', 'ridge', 'random_forest', 'xgboost', 'svr', 'lstm']
SEARCH_SPACES = {
    'random_forest': [{'max_depth': d, 'min_samples_leaf': l} for d in [8, None] for l in [2, 5]],
    'xgboost': [{'max_depth': d, 'learning_rate': lr} for d in [2, 4] for lr in [.05, .1]],
    'svr': [{'C': c, 'epsilon': e} for c in [1., 10.] for e in [.05, .15]],
}
DEFAULTS = {'random_forest': {'max_depth': 8, 'min_samples_leaf': 2},
            'xgboost': {'max_depth': 2, 'learning_rate': .05},
            'svr': {'C': 10., 'epsilon': .05}}


def make_model(name: str, seed: int = 42, params: dict[str, Any] | None = None) -> TransformedTargetRegressor:
    """A fresh estimator per fit; target scaling also learns from training only."""
    params = params or DEFAULTS.get(name, {})
    if name == 'mean_baseline': estimator = DummyRegressor(strategy='mean')
    elif name == 'ridge': estimator = Ridge(alpha=1.)
    elif name == 'random_forest':
        estimator = RandomForestRegressor(n_estimators=64, random_state=seed, n_jobs=1, **params)
    elif name == 'xgboost':
        from xgboost import XGBRegressor
        estimator = XGBRegressor(n_estimators=80, objective='reg:squarederror', tree_method='hist',
                                 random_state=seed, n_jobs=1, verbosity=0, **params)
    elif name == 'svr': estimator = SVR(kernel='rbf', **params)
    else: raise ValueError(f'Unknown tabular model: {name}')
    pipeline = Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
                         ('scaler', StandardScaler()), ('model', estimator)])
    return TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())

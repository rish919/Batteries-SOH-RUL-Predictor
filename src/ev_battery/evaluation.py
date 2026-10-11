"""Group-only tuning and metrics with explicit undefined R²."""
from typing import Any
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from ev_battery.models import make_model, SEARCH_SPACES, DEFAULTS


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    a = np.asarray(actual, dtype=float); p = np.asarray(predicted, dtype=float)
    if a.shape != p.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError('Metrics require equal nonempty arrays with finite actuals and predictions')
    residual = a-p; denom = float(np.sum((a-a.mean())**2))
    r2 = 1-float(np.sum(residual**2))/denom if len(a)>1 and denom>0 else None
    return {'mae': float(np.mean(np.abs(residual))), 'rmse': float(np.sqrt(np.mean(residual**2))),
            'r2': r2, 'r2_status': 'defined' if r2 is not None else 'undefined: fewer than 2 rows or constant target', 'n_rows': len(a)}


def tune_model(name: str, X: pd.DataFrame, y: pd.Series, groups: pd.Series, seed: int) -> tuple[dict, dict]:
    """Choose by equal-battery MAE on 3 inner group folds; never receive outer test data."""
    ids = sorted(groups.unique().tolist())
    if name not in SEARCH_SPACES:
        return {}, {'method': 'fixed', 'reason': 'prespecified baseline', 'training_groups': ids}
    if len(ids)<4:
        return DEFAULTS[name], {'method': 'fixed', 'reason': 'fewer than four training groups for stable inner tuning', 'training_groups': ids}
    splits = list(GroupKFold(n_splits=3).split(X, y, groups))
    trials = []
    for params in SEARCH_SPACES[name]:
        losses = []; details = []
        for tr, va in splits:
            estimator = make_model(name, seed, params)
            estimator.fit(X.iloc[tr], y.iloc[tr]); pred = estimator.predict(X.iloc[va])
            for group in sorted(groups.iloc[va].unique()):
                mask = groups.iloc[va].to_numpy()==group
                losses.append(float(np.abs(y.iloc[va].to_numpy()[mask]-pred[mask]).mean()))
            details.append({'train_groups': sorted(groups.iloc[tr].unique().tolist()),
                            'validation_groups': sorted(groups.iloc[va].unique().tolist())})
        trials.append({'parameters': params, 'macro_battery_mae': float(np.mean(losses)), 'splits': details})
    best = min(trials, key=lambda t:t['macro_battery_mae'])
    return best['parameters'], {'method': 'nested_3fold_group_search', 'training_groups': ids, 'trials': trials,
                               'selected_parameters': best['parameters']}


def aggregate_metrics(oof: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, model), p in oof.groupby(['target', 'model'], sort=True):
        f = folds[(folds.target==target)&(folds.model==model)&(folds.status=='evaluated')]
        m = regression_metrics(p.actual.to_numpy(), p.predicted.to_numpy())
        rows.append({'target':target,'model':model,'units':'percentage_points' if target=='soh' else 'cycles',
                     'n_groups':p.battery_id.nunique(),'n_predictions':len(p),
                     'macro_mae':f.mae.mean(),'macro_rmse':f.rmse.mean(),'macro_r2':f.r2.mean(),
                     'defined_r2_folds':int(f.r2.notna().sum()), 'fold_mae_std':f.mae.std(ddof=1),
                     'pooled_mae':m['mae'],'pooled_rmse':m['rmse'],'pooled_r2':m['r2']})
    return pd.DataFrame(rows)

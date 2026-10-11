"""Held-out permutation effects and optional SHAP of final demonstration models."""
import numpy as np
import pandas as pd


def permutation_rows(predict, X: np.ndarray, y: np.ndarray, features: list[str], seed: int,
                     target: str, model: str, fold: str, repeats: int = 3) -> list[dict]:
    """MAE increase in target units. For sequences permute whole feature histories."""
    baseline=float(np.abs(y-predict(X)).mean()); rng=np.random.default_rng(seed); rows=[]
    for j,name in enumerate(features):
        values=[]
        for _ in range(repeats):
            shuffled=X.copy(); permutation=rng.permutation(len(X))
            if X.ndim==3: shuffled[:,:,j]=X[permutation,:,j]
            else: shuffled[:,j]=X[permutation,j]
            values.append(float(np.abs(y-predict(shuffled)).mean())-baseline)
        rows.append({'target':target,'model':model,'fold_id':fold,'feature':name,
                     'method':'heldout_permutation_mae_increase','importance':float(np.mean(values)),
                     'repeat_std':float(np.std(values)), 'n_rows':len(y),'units':'percentage_points' if target=='soh' else 'cycles'})
    return rows


def shap_final(estimator, X: pd.DataFrame, target: str, name: str) -> tuple[list[dict],dict]:
    """Final-fit descriptive TreeSHAP only; never used for feature/model selection."""
    if name not in {'random_forest','xgboost'}:
        return [], {'status':'not_supported','reason':'TreeExplainer restricted to compatible tree models; permutation importance provided'}
    try:
        import shap
        pipe=estimator.regressor_; transformed=pipe[:-1].transform(X.iloc[:128])
        explainer=shap.TreeExplainer(pipe.named_steps['model'])
        values=np.asarray(explainer.shap_values(transformed))*float(estimator.transformer_.scale_[0])
        if values.shape!=transformed.shape or not np.isfinite(values).all(): raise ValueError('Invalid SHAP shape/values')
        rows=[{'target':target,'model':name,'fold_id':'final_demo','feature':feature,
               'method':'final_demo_mean_abs_shap','importance':float(np.abs(values[:,j]).mean()),
               'repeat_std':None,'n_rows':len(transformed),'units':'percentage_points' if target=='soh' else 'cycles'} for j,feature in enumerate(X.columns)]
        return rows, {'status':'available','scope':'first <=128 final-training rows; descriptive only, not CV evidence','explainer':'TreeExplainer'}
    except Exception as exc:
        return [], {'status':'unavailable','reason':f'{type(exc).__name__}: {exc}'}

"""Nested leave-one-battery-out experiments and distinct final inference artifacts."""
import argparse
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any
import joblib
import numpy as np
import pandas as pd
from ev_battery.config import ProjectConfig
from ev_battery.feature_engineering import SOH_FEATURES, RUL_FEATURES, validate_features
from ev_battery.models import MODEL_NAMES, make_model
from ev_battery.evaluation import regression_metrics, tune_model, aggregate_metrics
from ev_battery.explainability import permutation_rows, shap_final
from ev_battery.sequences import SETTINGS, SequenceModel, choose_epochs, sequence_indices

LOGGER=logging.getLogger(__name__)
LABELS={'soh':'soh_pct','rul':'rul_cycles'}


def clean_json(value: Any) -> Any:
    if isinstance(value,dict): return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean_json(v) for v in value]
    if isinstance(value,np.integer): return int(value)
    if isinstance(value,(float,np.floating)): return float(value) if np.isfinite(value) else None
    return value


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(clean_json(data),indent=2,sort_keys=True,allow_nan=False)+'\n')


def load_target(directory: Path, target: str) -> tuple[pd.DataFrame,list[str],dict]:
    path=directory/f'{target}_dataset.csv'
    if not path.is_file(): raise FileNotFoundError(f'Missing prepared data: {path}. Run python scripts/prepare_data.py first.')
    frame=pd.read_csv(path); features=SOH_FEATURES.copy() if target=='soh' else RUL_FEATURES.copy()
    validate_features(features,target); label=LABELS[target]
    required=set(features+[label,'battery_id']) | ({'eol_censored'} if target=='rul' else set())
    if not required.issubset(frame): raise ValueError(f'{path}: missing columns {sorted(required-set(frame))}')
    if frame.battery_id.isna().any() or frame.cycle_index.isna().any(): raise ValueError(f'{path}: missing group/cycle IDs')
    if frame.duplicated(['battery_id','cycle_index']).any(): raise ValueError(f'{path}: duplicate battery/cycle keys')
    if (frame.cycle_index<1).any() or (frame.cycle_index%1!=0).any(): raise ValueError('Invalid cycle index')
    for col in features+[label]: frame[col]=pd.to_numeric(frame[col],errors='raise')
    frame[features]=frame[features].replace([np.inf,-np.inf],np.nan)
    eligible=np.isfinite(frame[label]); reasons=[]
    if target=='rul':
        censored=frame.eol_censored.astype(str).str.lower()
        if not censored.isin(['true','false']).all(): raise ValueError('Unrecognized censor flag')
        eligible &= censored.eq('false') & (frame[label]>=0)
    for bid,g in frame.groupby('battery_id',sort=True):
        valid=g[eligible.loc[g.index]]
        reason=None
        if not len(valid): reason='no finite uncensored labels'
        elif len(valid)<2: reason='fewer than two labelled records; initial-threshold-only RUL is not a prognostic trajectory'
        elif target=='rul' and valid[label].max()<=0: reason='no observed positive-RUL interval'
        if reason:
            eligible.loc[g.index]=False; reasons.append({'battery_id':bid,'reason':reason,'rows':len(g)})
    selected=frame.loc[eligible].sort_values(['battery_id','cycle_index']).reset_index(drop=True)
    if selected.battery_id.nunique()<3: raise ValueError(f'{target}: need >=3 eligible batteries for LOBO; found {selected.battery_id.nunique()}')
    return selected,features,{'input_path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                             'input_rows':len(frame),'eligible_rows':len(selected),'eligible_groups':sorted(selected.battery_id.unique().tolist()),
                             'excluded_rows':int((~eligible).sum()),'excluded_groups':reasons}


def optional_availability() -> dict[str,dict]:
    result={}
    for name,module in [('xgboost','xgboost'),('lstm','torch')]:
        try: __import__(module); result[name]={'available':True}
        except Exception as exc: result[name]={'available':False,'reason':f'{type(exc).__name__}: {exc}'}
    return result


def audit_saved(results: Path) -> dict:
    """Recompute every fold and aggregate metric from the persisted OOF CSV."""
    oof=pd.read_csv(results/'oof_predictions.csv'); folds=pd.read_csv(results/'fold_metrics.csv')
    keys=['target','model','battery_id','cycle_index']
    if oof.duplicated(keys).any(): raise AssertionError('Duplicate OOF keys')
    for row in folds[folds.status=='evaluated'].itertuples():
        p=oof[(oof.target==row.target)&(oof.model==row.model)&(oof.fold_id==row.fold_id)]
        assert p.battery_id.nunique()==1 and p.battery_id.iloc[0]==row.test_battery
        m=regression_metrics(p.actual.to_numpy(),p.predicted.to_numpy())
        assert m['n_rows']==row.n_rows
        for key in ['mae','rmse','r2']:
            actual=getattr(row,key)
            assert (pd.isna(actual) and m[key] is None) or np.isclose(actual,m[key],rtol=1e-9,atol=1e-9), (row,key)
    aggregate=aggregate_metrics(oof,folds)
    for target in LABELS:
        saved=pd.read_csv(results/f'{target}_metrics.csv').sort_values('model').reset_index(drop=True)
        recomputed=aggregate[aggregate.target==target].sort_values('model').reset_index(drop=True)
        pd.testing.assert_frame_equal(saved,recomputed,check_dtype=False,rtol=1e-9,atol=1e-9)
    return {'status':'passed','oof_rows':len(oof),'evaluated_folds':int((folds.status=='evaluated').sum()),'check':'all fold and aggregate metrics recomputed from saved held-out predictions'}


def run_experiments(data_dir: Path, results: Path, models_dir: Path, seed: int = 42) -> dict:
    # Validate both inputs before creating model/results artifacts.
    datasets={t:load_target(data_dir,t) for t in LABELS}
    results.mkdir(parents=True,exist_ok=True); models_dir.mkdir(parents=True,exist_ok=True)
    optional=optional_availability()
    comparison={'protocol':'nested leave-one-battery-out; equal-battery inner MAE', 'seed':seed,
                'optional_dependencies':optional,'datasets':{t:v[2] for t,v in datasets.items()},
                'lstm_settings':SETTINGS,'tuning':{},'final_fits':{},'shap':{},'failures':[],
                'interpretation':{'primary_metrics':'macro average over held-out batteries; pooled metrics weight long trajectories more',
                                  'selection':'No winning model is selected using outer test outcomes. Final models are refit separately for every model.',
                                  'coverage':'LSTM uses contiguous length-5 windows, so early/gapped cycles lack predictions. Compare models on common support separately.',
                                  'limitations':'34 SOH / target-specific eligible groups are independent units; fold variation is not a confidence interval. RUL omits right-censored and initial-threshold-only trajectories and is conditional on observed EOL.',
                                  'importance':'held-out permutation is descriptive only; never used to select features or hyperparameters; correlated predictors and temporal dependence limit interpretation.'}}
    comparison['versions']={'python':platform.python_version()}
    for name in ['numpy','scipy','pandas','scikit-learn','joblib','xgboost','torch','shap']:
        try: comparison['versions'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: comparison['versions'][name]=None
    (results/'experiment_environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    source_paths=sorted(Path(__file__).parent.glob('*.py'))
    comparison['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    oof=[]; folds=[]; importances=[]
    for target,(frame,features,_) in datasets.items():
        label=LABELS[target]; groups=sorted(frame.battery_id.unique())
        for name in MODEL_NAMES:
            key=f'{target}/{name}'; comparison['tuning'][key]=[]
            if name in optional and not optional[name]['available']:
                reason=optional[name]['reason']; comparison['failures'].append({'target':target,'model':name,'reason':reason})
                for bid in groups: folds.append({'target':target,'model':name,'fold_id':f'{target}:test={bid}','test_battery':bid,'status':'skipped','reason':reason})
                continue
            for bid in groups:
                fold=f'{target}:test={bid}'
                train=frame[frame.battery_id!=bid].reset_index(drop=True); test=frame[frame.battery_id==bid].reset_index(drop=True)
                record={'target':target,'model':name,'fold_id':fold,'test_battery':bid,'n_train_groups':train.battery_id.nunique()}
                LOGGER.info('%s %s train_groups=%d test_rows=%d',key,bid,train.battery_id.nunique(),len(test))
                try:
                    if name=='lstm':
                        windows,ends=sequence_indices(test)
                        train_windows,_=sequence_indices(train)
                        if not len(ends) or len(train_windows)<SETTINGS['min_windows']:
                            folds.append({**record,'status':'skipped','reason':'insufficient contiguous test/training windows'}); continue
                        epochs,tuning=choose_epochs(train,features,label,seed)
                        estimator=SequenceModel(seed).fit(train,features,label,epochs)
                        evaluated=test.iloc[ends]; raw=test[features].to_numpy()[windows]
                        predict=estimator.predict_windows; pred=predict(raw)
                    else:
                        params,tuning=tune_model(name,train[features],train[label],train.battery_id,seed)
                        estimator=make_model(name,seed,params).fit(train[features],train[label])
                        evaluated=test; raw=test[features].to_numpy()
                        predict=lambda x,e=estimator,f=features:e.predict(pd.DataFrame(x,columns=f))
                        pred=predict(raw)
                    tuning.update({'fold_id':fold,'outer_train_groups':sorted(train.battery_id.unique().tolist()),'test_group':bid})
                    comparison['tuning'][key].append(tuning)
                    metrics=regression_metrics(evaluated[label].to_numpy(),pred)
                    folds.append({**record,'status':'evaluated','reason':'','units':'percentage_points' if target=='soh' else 'cycles',**metrics})
                    for (_,row),p in zip(evaluated.iterrows(),pred):
                        oof.append({'target':target,'model':name,'fold_id':fold,'battery_id':row.battery_id,'cycle_index':int(row.cycle_index),'actual':float(row[label]),'predicted':float(p)})
                    # Deterministic held-out subsample; all model predictions above remain saved.
                    sample=np.linspace(0,len(evaluated)-1,min(64,len(evaluated)),dtype=int)
                    try:
                        importances.extend(permutation_rows(predict,raw[sample],evaluated[label].to_numpy()[sample],features,seed,target,name,fold))
                    except Exception as exc:
                        comparison['failures'].append({'target':target,'model':name,'fold_id':fold,'stage':'permutation','reason':str(exc)})
                except Exception as exc:
                    if name not in {'lstm','xgboost'}: raise
                    LOGGER.exception('Optional model failed: %s %s',key,bid)
                    folds.append({**record,'status':'failed','reason':f'{type(exc).__name__}: {exc}'})
            # No final in-sample scores enter the results. The final model gets separate metadata.
            try:
                stem=f'{target}_{name}'
                metadata={'target':target,'model':name,'features':features,'seed':seed,'label':label,
                          'units':'percentage_points' if target=='soh' else 'cycles','training_groups':groups,
                          'purpose':'final full-eligible-data fit for optional inference; not CV evaluation',
                          'observation_time':'end of completed discharge','feature_dtypes':{c:str(frame[c].dtype) for c in features},
                          'missing_value_policy':'training-median imputation; all-missing columns filled with zero',
                          'versions':comparison['versions'],'input_sha256':comparison['datasets'][target]['sha256']}
                if name=='lstm':
                    import torch
                    epochs,tuning=choose_epochs(frame,features,label,seed)
                    estimator=SequenceModel(seed).fit(frame,features,label,epochs)
                    torch.save(estimator.model.state_dict(),models_dir/f'{stem}.pt')
                    joblib.dump({'preprocessor':estimator.preprocessor,'target_scaler':estimator.target_scaler},models_dir/f'{stem}_preprocessing.joblib')
                    metadata.update({'sequence_settings':SETTINGS,'epochs':epochs})
                    comparison['shap'][key]={'status':'not_supported','reason':'TreeExplainer not compatible with LSTM; held-out history permutation provided'}
                else:
                    params,tuning=tune_model(name,frame[features],frame[label],frame.battery_id,seed)
                    estimator=make_model(name,seed,params).fit(frame[features],frame[label])
                    joblib.dump(estimator,models_dir/f'{stem}.joblib')
                    rows,status=shap_final(estimator,frame[features],target,name)
                    importances.extend(rows); comparison['shap'][key]=status
                metadata['tuning']=tuning; write_json(models_dir/f'{stem}_schema.json',metadata)
                comparison['final_fits'][key]={'status':'saved','metadata':str(models_dir/f'{stem}_schema.json')}
            except Exception as exc:
                comparison['final_fits'][key]={'status':'failed','reason':f'{type(exc).__name__}: {exc}'}
    predictions=pd.DataFrame(oof).sort_values(['target','model','battery_id','cycle_index'])
    fold_frame=pd.DataFrame(folds).sort_values(['target','model','test_battery'])
    summary=aggregate_metrics(predictions,fold_frame)
    for filename,df in [('oof_predictions.csv',predictions),('fold_metrics.csv',fold_frame),('feature_importance.csv',pd.DataFrame(importances))]:
        df.to_csv(results/filename,index=False,float_format='%.17g')
    for target in LABELS: summary[summary.target==target].to_csv(results/f'{target}_metrics.csv',index=False,float_format='%.17g')
    # Direct comparisons against LSTM must use the same endpoint support.
    common_rows=[]
    for target,p in predictions.groupby('target'):
        names=sorted(p.model.unique()); support=None
        for name in names:
            keys=set(map(tuple,p[p.model==name][['battery_id','cycle_index']].to_numpy()))
            support=keys if support is None else support & keys
        for name in names:
            q=p[(p.model==name)&p.apply(lambda r:(r.battery_id,r.cycle_index) in support,axis=1)]
            if len(q):
                per=[regression_metrics(g.actual.to_numpy(),g.predicted.to_numpy()) for _,g in q.groupby('battery_id')]
                common_rows.append({'target':target,'model':name,'n_groups':q.battery_id.nunique(),'n_predictions':len(q),
                                    'macro_mae':float(np.mean([x['mae'] for x in per])), 'macro_rmse':float(np.mean([x['rmse'] for x in per])),
                                    **regression_metrics(q.actual.to_numpy(),q.predicted.to_numpy())})
    pd.DataFrame(common_rows).to_csv(results/'common_support_metrics.csv',index=False,float_format='%.17g')
    comparison['summary']=summary.to_dict('records'); comparison['common_support']=common_rows
    comparison['metric_audit']=audit_saved(results)
    write_json(results/'model_comparison.json',comparison)
    LOGGER.info('Complete: %s',comparison['metric_audit'])
    return comparison


def main() -> int:
    config=ProjectConfig(); parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=config.paths.processed)
    parser.add_argument('--results-dir',type=Path,default=config.paths.results)
    parser.add_argument('--models-dir',type=Path,default=config.paths.models)
    parser.add_argument('--seed',type=int,default=config.random_seed)
    args=parser.parse_args(); logging.basicConfig(level=logging.INFO,format='%(levelname)s %(message)s')
    try:
        if not 0<=args.seed<2**32: raise ValueError('Seed must be in [0, 2**32)')
        run_experiments(args.data_dir,args.results_dir,args.models_dir,args.seed)
    except (FileNotFoundError,ValueError) as exc:
        LOGGER.error('%s',exc); return 2
    return 0

if __name__=='__main__': raise SystemExit(main())

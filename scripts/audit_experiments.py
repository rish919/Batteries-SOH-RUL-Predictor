"""Independently check persisted predictions, split manifests and common-support metrics."""
import argparse
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from ev_battery.experiments import audit_saved, load_target, LABELS, write_json
from ev_battery.evaluation import regression_metrics
from ev_battery.sequences import sequence_indices
from ev_battery.config import ProjectConfig


def audit(data_dir: Path, results: Path) -> dict:
    evidence=audit_saved(results)
    comparison=json.loads((results/'model_comparison.json').read_text())
    oof=pd.read_csv(results/'oof_predictions.csv')
    for target in LABELS:
        eligible,_,_=load_target(data_dir,target)
        for model,predictions in oof[oof.target==target].groupby('model'):
            joined=predictions.merge(eligible[['battery_id','cycle_index',LABELS[target]]],on=['battery_id','cycle_index'],how='left',validate='one_to_one')
            np.testing.assert_allclose(joined.actual,joined[LABELS[target]],rtol=1e-12,atol=1e-12)
            if model=='lstm':
                _,ends=sequence_indices(eligible); expected=eligible.iloc[ends]
            else: expected=eligible
            assert set(map(tuple,expected[['battery_id','cycle_index']].to_numpy()))==set(map(tuple,predictions[['battery_id','cycle_index']].to_numpy())), (target,model,'missing/extra predictions')
    for records in comparison['tuning'].values():
        for fold in records:
            test=fold['test_group']; outer=set(fold['outer_train_groups'])
            assert test not in outer
            for trial in fold.get('trials',[]):
                for split in trial['splits']:
                    tr=set(split['train_groups']); va=set(split['validation_groups'])
                    assert not tr&va and tr|va==outer and test not in tr|va
            if 'validation_groups' in fold:
                assert set(fold['validation_groups']).issubset(outer)
                assert not set(fold['validation_groups'])&set(fold['training_groups'])
    common=pd.read_csv(results/'common_support_metrics.csv')
    for target,p in oof.groupby('target'):
        keys=[set(map(tuple,g[['battery_id','cycle_index']].to_numpy())) for _,g in p.groupby('model')]
        shared=set.intersection(*keys)
        for model,g in p.groupby('model'):
            g=g[g.apply(lambda r:(r.battery_id,r.cycle_index) in shared,axis=1)]
            row=common[(common.target==target)&(common.model==model)].iloc[0]
            metrics=regression_metrics(g.actual.to_numpy(),g.predicted.to_numpy())
            for name in ['mae','rmse','r2']:
                assert (metrics[name] is None and pd.isna(row[name])) or np.isclose(row[name],metrics[name],rtol=1e-9,atol=1e-9)
            battery=[regression_metrics(v.actual.to_numpy(),v.predicted.to_numpy()) for _,v in g.groupby('battery_id')]
            for name in ['mae','rmse']: assert np.isclose(row['macro_'+name],np.mean([v[name] for v in battery]),rtol=1e-9,atol=1e-9)
            assert row.n_groups==g.battery_id.nunique() and row.n_predictions==len(g)
    restored=[]
    for key,status in comparison['final_fits'].items():
        if status['status']!='saved': continue
        metadata_path=Path(status['metadata']); metadata=json.loads(metadata_path.read_text())
        directory=metadata_path.parent; target,name=key.split('/')
        frame,features,_=load_target(data_dir,target)
        assert metadata['features']==features
        if name=='lstm':
            import torch
            from ev_battery.sequences import network
            windows,_=sequence_indices(frame)
            prep=joblib.load(directory/f'{target}_{name}_preprocessing.joblib')
            model=network(len(features),metadata['sequence_settings']['hidden_size'])
            model.load_state_dict(torch.load(directory/f'{target}_{name}.pt',map_location='cpu',weights_only=True)); model.eval()
            X=prep['preprocessor'].transform(frame[features])[windows[:2]]
            with torch.no_grad(): values=model(torch.tensor(X,dtype=torch.float32)).numpy()
            values=prep['target_scaler'].inverse_transform(values.reshape(-1,1)).ravel()
        else:
            model=joblib.load(directory/f'{target}_{name}.joblib')
            values=model.predict(frame[features].iloc[:2])
        assert np.isfinite(values).all(), key
        restored.append(key)
    evidence['final_artifacts_restored_and_predictions_finite']=restored
    evidence.update({'actual_values_match_prepared_data':True,'exact_expected_prediction_coverage':True,
                     'inner_and_outer_groups_disjoint':True,'common_support_metrics_recomputed':True})
    write_json(results/'experiment_audit.json',evidence)
    return evidence

if __name__=='__main__':
    config=ProjectConfig(); parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=config.paths.processed)
    parser.add_argument('--results-dir',type=Path,default=config.paths.results)
    args=parser.parse_args(); print(json.dumps(audit(args.data_dir,args.results_dir),indent=2))

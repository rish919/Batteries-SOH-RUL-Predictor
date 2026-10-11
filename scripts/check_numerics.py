"""Verify real Ridge OOF values with an alternative solver and replay one LSTM fold."""
from pathlib import Path
from collections import Counter
import json
import warnings
import numpy as np
import pandas as pd
from ev_battery.config import ProjectConfig
from ev_battery.experiments import load_target, LABELS, write_json
from ev_battery.models import make_model
from ev_battery.sequences import SequenceModel, sequence_indices


def main() -> None:
    config=ProjectConfig(); results=config.paths.results
    comparison=json.loads((results/'model_comparison.json').read_text())
    oof=pd.read_csv(results/'oof_predictions.csv'); checks=[]
    for target in LABELS:
        frame,features,_=load_target(config.paths.processed,target); differences=[]; warning_count=0
        for bid in sorted(frame.battery_id.unique()):
            train=frame[frame.battery_id!=bid]; test=frame[frame.battery_id==bid]
            alternate=make_model('ridge',comparison['seed'])
            alternate.set_params(regressor__model__solver='svd')
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                alternate.fit(train[features],train[LABELS[target]])
                # Explicit elementwise sum avoids the inference matrix-multiply path.
                pipe=alternate.regressor_; z=pipe[:-1].transform(test[features]); linear=pipe.named_steps['model']
                standardized=np.sum(z*linear.coef_,axis=1)+linear.intercept_
                predicted=alternate.transformer_.inverse_transform(standardized.reshape(-1,1)).ravel()
            warning_count+=len(captured)
            saved=oof[(oof.target==target)&(oof.model=='ridge')&(oof.battery_id==bid)].sort_values('cycle_index').predicted.to_numpy()
            np.testing.assert_allclose(predicted,saved,rtol=1e-9,atol=1e-9)
            differences.append(float(np.max(np.abs(predicted-saved))))
        checks.append({'target':target,'check':'Ridge OOF compared with independent SVD refit and explicit elementwise prediction',
                       'folds':len(differences),'max_absolute_difference':max(differences),'runtime_warning_count_during_check':warning_count})
    if comparison['optional_dependencies']['lstm']['available']:
        frame,features,_=load_target(config.paths.processed,'soh')
        tuning=comparison['tuning']['soh/lstm'][0]; bid=tuning['test_group']
        train=frame[frame.battery_id!=bid].reset_index(drop=True); test=frame[frame.battery_id==bid].reset_index(drop=True)
        windows,_=sequence_indices(test)
        model=SequenceModel(comparison['seed']).fit(train,features,LABELS['soh'],tuning['epochs'])
        predicted=model.predict_windows(test[features].to_numpy()[windows])
        saved=oof[(oof.target=='soh')&(oof.model=='lstm')&(oof.battery_id==bid)].sort_values('cycle_index').predicted.to_numpy()
        np.testing.assert_allclose(predicted,saved,rtol=1e-9,atol=1e-9)
        checks.append({'target':'soh','check':'replayed actual LSTM outer fold','test_battery':bid,'epochs':tuning['epochs'],
                       'max_absolute_difference':float(np.max(np.abs(predicted-saved)))})
    warning_messages=Counter(line.split('RuntimeWarning: ',1)[1] for line in (results/'experiment_run.log').read_text().splitlines() if 'RuntimeWarning: ' in line)
    evidence={'status':'passed','checks':checks,'observed_runtime_warnings':dict(warning_messages),
              'interpretation':'Matrix-operation warnings occurred in this runtime; finite saved Ridge predictions match alternative SVD fits and explicit sums within tolerance. No warning suppression is applied to the experiment pipeline.'}
    write_json(results/'numerical_audit.json',evidence)
    print(json.dumps(evidence,indent=2))

if __name__=='__main__': main()

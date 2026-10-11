"""Deterministic local preprocessing CLI with auditable filtering and feature manifests."""
import argparse
from collections import Counter
from pathlib import Path
import json
import logging
import platform
import numpy as np
import pandas as pd
import scipy
from ev_battery.config import ProjectConfig
from ev_battery.battery_loader import load_batteries
from ev_battery.feature_engineering import build_features, SOH_FEATURES, RUL_FEATURES, validate_features
from ev_battery.targets import add_targets, detect_eol, capacity_to_soh


def prepare(raw: Path, output: Path, reports: Path, figures: Path, rated_capacity: float=2., reference: str='rated', threshold: float=.7, persistence: int=1) -> dict:
    capacity_to_soh(np.array([]),rated_capacity)
    detect_eol([],rated_capacity*threshold,persistence)
    if reference not in {"rated", "first_valid"} or not 0<threshold<1:
        raise ValueError("Invalid reference or EOL threshold fraction")
    batteries,manifest=load_batteries(raw.resolve())
    df=add_targets(build_features(batteries),rated_capacity,reference,threshold,persistence)
    report={'schema_version':1,'inputs':manifest,'settings':{'rated_capacity_ah':rated_capacity,'reference':reference,'threshold_fraction':threshold,'persistence':persistence,'eol_rule':'start of first consecutive qualifying run; missing breaks run'},
            'runtime':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__},
            'feature_sets':{'soh':SOH_FEATURES,'rul':RUL_FEATURES},'batteries':{},'filtering':{},
            'policies':{'observation_time':'end of current discharge','impedance':'separate; no alignment inferred','missing_predictors':'retained as blank; fit any imputation only on training folds','energy':'not computed; no validated integration protocol across test regimes','SOH':'capacity, energy and duration excluded from predictor allowlist','RUL':'current capacity is an observed diagnostic; censored labels remain missing'}}
    output.mkdir(parents=True,exist_ok=True); reports.mkdir(parents=True,exist_ok=True)
    def save(frame: pd.DataFrame,name: str) -> None:
        frame.to_csv(output/name,index=False,float_format='%.12g',na_rep='',lineterminator='\n')
    save(df,'cycle_features.csv')
    for task, features, target in [('soh',SOH_FEATURES,'soh_pct'),('rul',RUL_FEATURES,'rul_cycles')]:
        validate_features(features,task)
        has_signal=df[[c for c in features if c not in ['cycle_index','ambient_temperature']]].notna().any(axis=1)
        keep=has_signal & (df[target].notna() if task=='soh' else (df.eol_censored | df.rul_cycles.notna()))
        columns=['battery_id']+features+[target]
        if task=='rul': columns+=['eol_censored']
        save(df.loc[keep,columns],f'{task}_dataset.csv')
        report['filtering'][task]={'retained':int(keep.sum()),'excluded':int((~keep).sum()),'excluded_rows':[{'battery_id':row.battery_id,'cycle_index':int(row.cycle_index),'reason':'no usable signals' if not has_signal.loc[i] else 'missing SOH target or post-EOL/unavailable-reference RUL'} for i,row in df.loc[~keep].iterrows()]}
    operation_rows=[]
    for bid,ops in sorted(batteries.items()):
        g=df[df.battery_id==bid]
        issues=[{'record_index':o.record_index,'operation':o.operation,'warnings':o.warnings} for o in ops if o.warnings]
        report['batteries'][bid]={'operations':dict(Counter(o.operation for o in ops)),'discharge_cycles':len(g),'at_or_below_threshold_at_first_record':bool(len(g) and g.capacity_ah.iloc[0] <= g.reference_capacity_ah.iloc[0]*threshold),'invalid_capacity_rows':int(g.capacity_ah.isna().sum()),'censored':bool(g.eol_censored.iloc[0]) if len(g) else True,'eol_cycle_index':int(g.eol_cycle_index.iloc[0]) if len(g) and pd.notna(g.eol_cycle_index.iloc[0]) else None,'feature_nonmissing_counts':{c:int(g[c].notna().sum()) for c in SOH_FEATURES+RUL_FEATURES[ len(SOH_FEATURES):]},'warnings':issues}
        for o in ops:
            row={'battery_id':bid,'record_index':o.record_index,'operation':o.operation,'timestamp':o.timestamp,'ambient_temperature':o.ambient_temperature,'warnings':'; '.join(o.warnings)}
            for name,a in o.measurements.items():
                row[name+'_samples']=len(a); row[name+'_nonfinite']=int((~np.isfinite(a)).sum())
                if a.size==1 and np.isfinite(a[0]):
                    row[name+'_real']=float(a[0].real)
                    if np.iscomplexobj(a): row[name+'_imag']=float(a[0].imag)
            operation_rows.append(row)
    operations=pd.DataFrame(operation_rows)
    save(operations,'operation_inventory.csv')
    save(operations[operations.operation=='impedance'],'impedance_records.csv')
    report['total_rows']=len(df)
    (reports/'data_quality_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    from ev_battery.plotting import exploratory_figures
    exploratory_figures(df,figures)
    return report


def main() -> None:
    config=ProjectConfig(); p=argparse.ArgumentParser(description=__doc__)
    for name,default in [('raw-dir',config.paths.raw),('output-dir',config.paths.processed),('reports-dir',config.paths.results),('figures-dir',config.paths.figures)]: p.add_argument('--'+name,type=Path,default=default)
    p.add_argument('--rated-capacity',type=float,default=2.); p.add_argument('--reference',choices=['rated','first_valid'],default='rated')
    p.add_argument('--eol-threshold',type=float,default=.7); p.add_argument('--persistence',type=int,default=1)
    a=p.parse_args(); logging.basicConfig(level=logging.INFO)
    r=prepare(a.raw_dir,a.output_dir,a.reports_dir,a.figures_dir,a.rated_capacity,a.reference,a.eol_threshold,a.persistence)
    print(f'Prepared {r["total_rows"]} discharge rows across {len(r["batteries"])} batteries; report: {a.reports_dir / "data_quality_report.json"}')

if __name__=='__main__': main()

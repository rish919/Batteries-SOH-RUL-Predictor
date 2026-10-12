"""Local interactive dashboard for actual NASA battery experiment artifacts."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'app'))
from ev_battery.config import ProjectConfig,OFFICIAL_REPOSITORY_URL,OFFICIAL_ARCHIVE_URL
from ev_battery.feature_engineering import SOH_FEATURES,RUL_FEATURES
from components.data import cached_csv,cached_json,readiness,setup_message,validate_upload,final_model
from components.charts import line_chart,scatter_actual,residual_chart

st.set_page_config(page_title='EV Battery Health & Prognostics',page_icon='',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>
.block-container{padding-top:1.7rem;padding-bottom:3rem;max-width:1500px}
[data-testid="stMetric"]{background:#fff;border:1px solid #dce5eb;padding:14px 18px;border-radius:12px;box-shadow:0 2px 8px #102a4310}
[data-testid="stMetricLabel"]{color:#526174}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#102a43 0%,#173b50 100%)}
section[data-testid="stSidebar"] *{color:#f3f7fa!important}
section[data-testid="stSidebar"] [data-testid="stRadio"] label{padding:7px 10px;border-radius:8px}
h1,h2,h3{letter-spacing:-.025em}
[data-testid="stAlert"]{border-radius:10px}
</style>''',unsafe_allow_html=True)
config=ProjectConfig(); P=config.paths; ready,missing=readiness(config)

@st.cache_data(show_spinner=False)
def load_data() -> dict[str, pd.DataFrame | None]:
    return {name:cached_csv(path) for name,path in {
      'cycles':P.processed/'cycle_features.csv','soh':P.processed/'soh_dataset.csv','rul':P.processed/'rul_dataset.csv',
      'ops':P.processed/'operation_inventory.csv','impedance':P.processed/'impedance_records.csv',
      'oof':P.results/'oof_predictions.csv','folds':P.results/'fold_metrics.csv',
      'soh_metrics':P.results/'soh_metrics.csv','rul_metrics':P.results/'rul_metrics.csv',
      'importance':P.results/'feature_importance.csv','common':P.results/'common_support_metrics.csv'}.items()}

data=load_data(); quality=cached_json(P.results/'data_quality_report.json') or {}; comparison=cached_json(P.results/'model_comparison.json') or {}

st.sidebar.markdown('## ML Based SOH-RUL predictor in Li-ion batteries of EVs')
st.sidebar.caption('NASA Ames PCoE • SOH & RUL')
section=st.sidebar.radio('Navigation',['Project Overview','Battery Data Explorer','Model Performance','Prediction Explorer','Feature Importance','Methodology & Limitations'])
st.sidebar.divider();st.sidebar.caption('Laboratory aging benchmark. Results do not establish EV fleet performance.')
if missing: setup_message(missing)


def kpis(items):
    cols=st.columns(len(items))
    for col,(label,value,helptext) in zip(cols,items): col.metric(label,value,help=helptext)

def no_data(message): st.info(message)

def safe_num(v,digits=2): return f'{v:,.{digits}f}' if pd.notna(v) and np.isfinite(v) else '—'

if section=='Project Overview':
    st.title('Machine Learning-Based State-of-Health and Remaining Useful Life Prediction of Li-ion Batteries for EVs')
    st.markdown('An auditable battery prognostics study using NASA PCoE laboratory aging data. Explore measured degradation, battery-held-out model results and per-cycle predictions.')
    with st.container(border=True):
        st.subheader('How to use this dashboard')
        st.markdown('**Start here:** Battery Data Explorer shows what was measured. Model Performance shows how models did on batteries they had never seen. Prediction Explorer lets you inspect a specific held-out estimate. For RUL, select **RUL**, a model and a battery with saved predictions to see predicted cycles remaining beside the actual label.')
    if data['cycles'] is not None:
        c=data['cycles']; q=quality.get('batteries',{}); n_batt=c.battery_id.nunique(); n_cycles=len(c)
        parsed=sum(1 for b in q.values() if b.get('operations')) if q else n_batt
        kpis([('Parsed batteries',f'{parsed:,}','Distinct battery structures in preprocessing report'),('Discharge records',f'{n_cycles:,}','One row per recorded discharge; invalid capacities remain auditable'),('RUL observed EOL groups',f'{sum(not x.get("censored",True) for x in q.values()):,}' if q else '—','Censored batteries are not assigned a fabricated EOL'),('Data readiness','Ready' if ready else 'Partial','Prepared cycle table and model results status')])
    else: no_data('Prepared battery tables are not available. Add raw NASA files and run `python scripts/prepare_data.py`.')
    st.subheader('Study definitions')
    a,b=st.columns(2);a.info('**SOH (%)** = measured discharge capacity ÷ selected reference capacity × 100. Default reference: 2.0 Ah.');b.info('**RUL (cycles)** = observed EOL discharge-cycle index − current discharge-cycle index. Default EOL: 70% of reference capacity.')
    st.subheader('Evaluation at a glance')
    if data['soh_metrics'] is not None and data['rul_metrics'] is not None:
        m1=data['soh_metrics'];m2=data['rul_metrics']; best1=m1.loc[m1.macro_mae.idxmin()];best2=m2.loc[m2.macro_mae.idxmin()]
        kpis([('SOH best observed macro MAE',f"{best1.macro_mae:.2f} pp",best1.model),('RUL best observed macro MAE',f"{best2.macro_mae:.2f} cycles",best2.model),('SOH held-out groups',str(int(m1.n_groups.max())),'Leave-one-battery-out'),('RUL held-out groups',str(int(m2.n_groups.max())),'Observed EOL subset')])
    else:no_data('Model results are unavailable; run `python scripts/run_experiments.py`.')
    st.warning('**Scope:** NASA cells were tested in laboratory aging protocols. These results do not establish performance, safety or generalization on a real EV fleet or battery pack.')

elif section=='Battery Data Explorer':
    st.title('Battery Data Explorer');st.caption('Cycle-level views are based on recorded discharge operations. Missing values remain missing.')
    if data['cycles'] is None: no_data('Cycle features are missing. Run `python scripts/prepare_data.py`.')
    else:
        df=data['cycles']; ids=sorted(df.battery_id.dropna().unique()); bid=st.selectbox('Battery',ids); b=df[df.battery_id==bid].sort_values('cycle_index')
        q=quality.get('batteries',{}).get(bid,{})
        kpis([('Discharge records',f'{len(b):,}','Recorded cycles, including invalid-capacity rows'),('Valid capacities',f'{b.capacity_ah.notna().sum():,}','Measured discharge capacity'),('EOL cycle',str(q.get('eol_cycle_index') or 'Not observed'),'First qualifying crossing'),('RUL status','Censored' if q.get('censored',True) else 'Observed','Data-quality report')])
        st.plotly_chart(line_chart(b,'cycle_index','capacity_ah',f'{bid} • measured capacity','Capacity (Ah)'),use_container_width=True)
        soh_col='soh_rated_pct' if 'soh_rated_pct' in b else 'soh_pct'
        st.plotly_chart(line_chart(b,'cycle_index',soh_col,f'{bid} • SOH trajectory','SOH (%)'),use_container_width=True)
        stats=['discharge_voltage_mean','discharge_current_mean','discharge_temperature_mean','charge_voltage_mean','charge_current_mean','charge_temperature_mean']
        selected=st.multiselect('Available signal summaries', [x for x in stats if x in b], default=[x for x in stats[:3] if x in b])
        if selected:
            long=b.melt(id_vars=['cycle_index'],value_vars=selected,var_name='feature',value_name='value').dropna()
            st.plotly_chart(px.line(long,x='cycle_index',y='value',color='feature',labels={'cycle_index':'Discharge cycle','value':'Summary value','feature':'Feature'},title='Measured signal summaries'),use_container_width=True)
        st.subheader('Feature distribution and correlation')
        candidates=[c for c in SOH_FEATURES+['capacity_ah','soh_pct'] if c in b and pd.api.types.is_numeric_dtype(b[c]) and b[c].notna().any()]
        chosen=st.multiselect('Features',candidates,default=[c for c in ['capacity_ah','discharge_voltage_mean','discharge_current_mean','discharge_temperature_mean'] if c in candidates])
        if chosen:
            left,right=st.columns(2)
            with left: st.plotly_chart(px.histogram(b,x=chosen[0],nbins=30,title=f'Distribution • {chosen[0]}'),use_container_width=True)
            with right:
                corr=b[chosen].corr(min_periods=3)
                st.plotly_chart(px.imshow(corr,text_auto='.2f',color_continuous_scale='RdBu_r',zmin=-1,zmax=1,title='Within-battery feature correlation'),use_container_width=True)
        with st.expander('Data quality and source operations'):
            st.json({'operations':q.get('operations',{}),'invalid_capacity_rows':q.get('invalid_capacity_rows'),'warning_count':len(q.get('warnings',[])),'warnings':q.get('warnings',[])[:30],'raw_quality_report':str(P.results/'data_quality_report.json')})
            if data['ops'] is not None: st.dataframe(data['ops'][data['ops'].battery_id==bid],use_container_width=True)
            if data['impedance'] is not None: st.caption(f"{len(data['impedance'][data['impedance'].battery_id==bid])} impedance records shown separately; no array-index alignment was inferred.")

elif section=='Model Performance':
    st.title('Model Performance');st.caption('Primary comparisons use held-out battery predictions. Macro metrics weight each test battery equally.')
    target=st.radio('Prediction target',['SOH','RUL'],horizontal=True); key=target.lower(); metrics=data[f'{key}_metrics']; pred=data['oof']; folds=data['folds']
    if metrics is None or pred is None or folds is None:no_data('Experiment outputs are unavailable. Run `python scripts/run_experiments.py`.')
    else:
        metric=metrics.sort_values('macro_mae'); units='percentage points' if key=='soh' else 'cycles'
        best=metric.iloc[0];kpis([('Eligible batteries',f"{int(metrics.n_groups.max())}",'Independent battery groups'),('Best macro MAE',f"{best.macro_mae:.2f} {('pp' if key=='soh' else 'cycles')}",best.model),('Best macro RMSE',f"{best.macro_rmse:.2f} {('pp' if key=='soh' else 'cycles')}",best.model),('Evaluated models',str(len(metrics)),'Includes baseline where available')])
        st.dataframe(metric[['model','n_groups','n_predictions','macro_mae','macro_rmse','macro_r2','pooled_mae','pooled_rmse','pooled_r2','defined_r2_folds']].style.format(precision=3),use_container_width=True,hide_index=True)
        model=st.multiselect('Models to compare',metric.model.tolist(),default=metric.model.tolist())
        p=pred[(pred.target==key)&pred.model.isin(model)]; f=folds[(folds.target==key)&folds.model.isin(model)]
        st.plotly_chart(scatter_actual(p,f'{target}: out-of-fold predictions','% points' if key=='soh' else 'cycles'),use_container_width=True)
        st.plotly_chart(residual_chart(p,units),use_container_width=True)
        st.subheader('Fold-level held-out scores')
        st.dataframe(f[['model','fold_id','test_battery','status','n_rows','mae','rmse','r2','reason']].sort_values(['model','test_battery']),use_container_width=True,hide_index=True)
        st.caption('Negative R² can occur when predictions are worse than predicting the held-out battery mean. Fold R² is undefined for constant/singleton targets; see blank values.')
        common=data['common']
        if common is not None: st.info('Use the saved common-support table when comparing LSTM with tabular models; sequence predictions omit early or gapped cycles.')

elif section=='Prediction Explorer':
    st.title('Prediction Explorer');st.caption('Every displayed score below comes from a leave-one-battery-out held-out prediction.')
    st.info('**Where is RUL predicted?** Choose the RUL target below. The model uses the current cycle measurements and past capacity history, then estimates discharge cycles until the recorded 70% end-of-life threshold. The result is shown as predicted RUL beside actual RUL. This is a held-out estimate, not a live EV forecast.')
    p=data['oof']
    if p is None:no_data('Out-of-fold predictions are missing. Run `python scripts/run_experiments.py`.')
    else:
        target=st.radio('Target',['SOH','RUL'],horizontal=True).lower(); subset=p[p.target==target]
        if subset.empty:no_data(f'No saved {target.upper()} predictions.');st.stop()
        model=st.selectbox('Model',sorted(subset.model.unique()));subset=subset[subset.model==model]
        bid=st.selectbox('Held-out battery',sorted(subset.battery_id.unique()));subset=subset[subset.battery_id==bid]
        cycle=int(st.selectbox('Cycle with saved held-out prediction',sorted(subset.cycle_index.astype(int).unique())))
        row=subset[subset.cycle_index==cycle].iloc[0]; err=row.predicted-row.actual;unit='percentage points' if target=='soh' else 'cycles'
        st.success(f"**Held-out prediction** • {row.model} • {row.fold_id} • battery {row.battery_id} was excluded from training and tuning for this fold.")
        kpis([('Actual',f'{row.actual:.2f} {unit}','Observed target'),('Predicted',f'{row.predicted:.2f} {unit}','Out-of-fold model prediction'),('Prediction error',f'{err:+.2f} {unit}','Predicted minus actual'),('Absolute error',f'{abs(err):.2f} {unit}','Held-out error')])
        st.dataframe(subset[['cycle_index','actual','predicted','fold_id']].assign(error=lambda x:x.predicted-x.actual),use_container_width=True,hide_index=True)
    st.divider();st.subheader('Optional uploaded feature table')
    st.caption('Predictions use final models fit on all eligible batteries. They are demonstration predictions, not independent evaluation.')
    upload=st.file_uploader('Upload CSV with one row per cycle and exact model feature column names',type=['csv'])
    if upload:
        try:
            frame=pd.read_csv(upload); target_up=st.selectbox('Inference target',['SOH','RUL'],key='upload_target').lower()
            model_up=st.selectbox('Final model',['ridge','random_forest','xgboost','svr','mean_baseline','lstm'],key='upload_model')
            X,features=validate_upload(frame,target_up)
            meta_path=P.models/f'{target_up}_{model_up}_schema.json'
            schema=json.loads(meta_path.read_text()) if meta_path.is_file() else {}
            units={'cycle_index':'discharge cycles','ambient_temperature':'°C','capacity_ah':'Ah','capacity_lag1_ah':'Ah','capacity_past3_mean_ah':'Ah','capacity_change_ah':'Ah'}
            for col in features:
                if 'voltage' in col: units[col]='V'
                elif 'current' in col: units[col]='A'
                elif 'temperature' in col: units[col]='°C'
            st.caption(f'Features are reordered to the saved schema: {", ".join(features)}. Supply these units: '+', '.join(f'{c}={units.get(c,"unitless")}' for c in features)+f'. Output: {"percentage points" if target_up=="soh" else "cycles"}. Missing numeric values use training-fitted median imputation.')
            acknowledged=st.checkbox('I confirm the uploaded feature values use the displayed units and represent measurements available at the prediction cycle.',key='units_ack')
            if X.isna().all(axis=1).any(): st.warning(f'{X.isna().all(axis=1).sum()} row(s) have all features missing; a numerical output would not be informative. Remove or repair these rows.')
            model_path=P.models/f'{target_up}_{model_up}.joblib'
            if not meta_path.is_file(): st.error(f'Final model/schema is unavailable: {meta_path}')
            elif model_up!='lstm' and not model_path.is_file(): st.error(f'Final model is unavailable: {model_path}')
            elif st.button('Generate demonstration predictions',disabled=bool(not acknowledged or X.isna().all(axis=1).any())):
                result=frame.copy()
                if model_up=='lstm':
                    import torch,joblib
                    from ev_battery.sequences import network,sequence_indices
                    if not {'battery_id','cycle_index'}.issubset(frame.columns): st.error('LSTM inference requires battery_id and cycle_index columns for contiguous five-cycle window validation.')
                    elif frame.duplicated(['battery_id','cycle_index']).any(): st.error('LSTM inference requires unique battery_id/cycle_index rows.')
                    else:
                        windows,ends=sequence_indices(frame[['battery_id','cycle_index']].copy(),schema['sequence_settings']['window'])
                        prep=joblib.load(P.models/f'{target_up}_lstm_preprocessing.joblib'); net=network(len(features),schema['sequence_settings']['hidden_size']);net.load_state_dict(torch.load(P.models/f'{target_up}_lstm.pt',map_location='cpu',weights_only=True));net.eval()
                        if len(ends):
                            transformed=prep['preprocessor'].transform(X[features]); seq=torch.tensor(transformed[windows],dtype=torch.float32)
                            with torch.no_grad(): raw=net(seq).numpy()
                            values=prep['target_scaler'].inverse_transform(raw.reshape(-1,1)).ravel()
                            result[f'{target_up}_prediction']=np.nan
                            result.loc[list(ends),f'{target_up}_prediction']=values
                        else: st.warning('No contiguous five-cycle sequences were found; no predictions generated.')
                else:
                    estimator=final_model(str(model_path),model_path.stat().st_mtime);result[f'{target_up}_prediction']=estimator.predict(X[features])
                if f'{target_up}_prediction' in result: st.dataframe(result,use_container_width=True)
        except Exception as exc: st.error(f'Upload validation/prediction failed: {exc}')

elif section=='Feature Importance':
    st.title('Feature Importance & Explainability');st.caption('Importance describes trained models; it is not a causal effect or feature-selection result.')
    imp=data['importance']
    if imp is None:no_data('Importance outputs are unavailable. Run `python scripts/run_experiments.py`.')
    else:
        target=st.radio('Target',['SOH','RUL'],horizontal=True).lower(); x=imp[(imp.target==target)&imp.method.eq('heldout_permutation_mae_increase')]
        if x.empty:no_data('No held-out permutation importance is available for this target.')
        else:
            models=sorted(x.model.unique());model=st.selectbox('Model',models);q=x[x.model==model]
            agg=q.groupby('feature').agg(importance=('importance','mean'),fold_sd=('importance','std'),folds=('fold_id','nunique')).sort_values('importance').tail(20).reset_index()
            units='percentage points MAE' if target=='soh' else 'cycles MAE'
            st.plotly_chart(px.bar(agg,x='importance',y='feature',orientation='h',error_x='fold_sd',labels={'importance':f'Mean increase in held-out {units}','feature':'Feature','fold_sd':'Across-fold SD'},title=f'{model} • held-out permutation importance'),use_container_width=True)
            st.caption('Computed on held-out predictions, at most 64 sampled cycles per fold with three permutations. Correlated features can share or mask importance.')
        st.subheader('SHAP summary')
        s=imp[(imp.target==target)&imp.method.eq('final_demo_mean_abs_shap')]
        status=comparison.get('shap',{})
        available=[k for k,v in status.items() if k.startswith(target+'/') and v.get('status')=='available']
        if s.empty: st.info('No SHAP values were saved for this target. See model_comparison.json for per-model availability or failure details.')
        else:
            model_shap=st.selectbox('TreeSHAP model',sorted(s.model.unique()),key='shap_model');z=s[s.model==model_shap].sort_values('importance').tail(20)
            st.plotly_chart(px.bar(z,x='importance',y='feature',orientation='h',labels={'importance':'Mean absolute SHAP (target units)','feature':'Feature'},title=f'{model_shap} • final-model TreeSHAP'),use_container_width=True)
            st.caption('Descriptive TreeSHAP on up to 128 rows from the full-data demonstration fit. Not cross-validated or independent test evidence.')
        with st.expander('Feature definitions and units'):
            st.dataframe(pd.DataFrame([{'feature':c,'description':c.replace('_',' ').capitalize(),'unit':('V' if 'voltage' in c else 'A' if 'current' in c else '°C' if 'temperature' in c else 'cycle' if 'cycle' in c else 'Ah' if 'capacity' in c else '°C ambient' if c=='ambient_temperature' else 'unitless') } for c in SOH_FEATURES+RUL_FEATURES[len(SOH_FEATURES):]]),use_container_width=True,hide_index=True)

else:
    st.title('Methodology & Limitations')
    st.subheader('Data source and scope')
    st.markdown(f'NASA Ames PCoE [Li-ion Battery Aging Dataset]({OFFICIAL_REPOSITORY_URL}) ([official archive]({OFFICIAL_ARCHIVE_URL})). Operations are parsed from MATLAB files. The primary cycle index counts recorded discharge operations; impedance records remain separate because array-index alignment is not established.')
    st.subheader('Targets and configured thresholds')
    kpis([('Rated capacity',f'{config.nominal_capacity_ah:.2f} Ah','Default benchmark reference'),('EOL threshold',f'{100*config.eol_soh_fraction:.0f}% SOH',f'{config.eol_capacity_ah:.2f} Ah at rated capacity'),('Persistence rule',str(quality.get('settings',{}).get('persistence',1))+' cycle(s)','First qualifying run; missing values break the run'),('Reference mode',quality.get('settings',{}).get('reference','rated'),'Rated capacity or first valid discharge')])
    st.markdown('SOH = measured discharge capacity / reference capacity × 100. RUL = observed first EOL discharge index − current discharge index. Censored batteries remain unlabelled for regression; RUL results apply only to the observed-EOL subset.')
    st.subheader('Models and validation')
    st.markdown('Ridge, mean baseline, Random Forest, XGBoost, SVR and a modest LSTM are compared. The primary protocol leaves one complete battery out at a time. Hyperparameter searches use only group-wise splits among training batteries; imputers/scalers are fitted within each training fold. Macro MAE/RMSE/R² give each battery equal weight; pooled metrics weight longer trajectories more heavily. R² is undefined for constant or single-point folds.')
    st.subheader('Limitations')
    st.markdown('The dataset contains a small number of independent laboratory cells, uneven test lengths, protocol variation, censored trajectories and records already below threshold at their first observed cycle. LOBO folds share most training groups, so fold variation is descriptive rather than a confidence interval. LSTM has reduced coverage because it needs five contiguous cycles. Laboratory cell results do not establish performance on an EV fleet, pack, chemistry or in-field duty cycle.')
    if comparison: st.json({'protocol':comparison.get('protocol'),'dataset_groups':{k:v.get('eligible_groups') for k,v in comparison.get('datasets',{}).items()},'optional_dependencies':comparison.get('optional_dependencies'),'metric_audit':comparison.get('metric_audit')})

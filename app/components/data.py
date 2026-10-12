"""Cached loading, readiness checks and safe final-model inference."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from ev_battery.config import ProjectConfig,OFFICIAL_ARCHIVE_URL,OFFICIAL_REPOSITORY_URL
from ev_battery.experiments import load_target

@st.cache_data(show_spinner=False)
def csv_file(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def json_file(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text())

def cached_csv(path: Path) -> pd.DataFrame | None:
    return csv_file(str(path),path.stat().st_mtime) if path.is_file() else None

def cached_json(path: Path):
    return json_file(str(path),path.stat().st_mtime) if path.is_file() else None

def readiness(config=None):
    config=config or ProjectConfig(); required=[config.paths.processed/'cycle_features.csv',config.paths.results/'oof_predictions.csv',config.paths.results/'soh_metrics.csv',config.paths.results/'rul_metrics.csv']
    return all(p.is_file() for p in required),[p for p in required if not p.is_file()]

def setup_message(missing):
    st.warning('Some project artifacts are missing. The dashboard will show only sections supported by available files.')
    if any('data/raw' in str(p) for p in missing):
        st.markdown(f'Add the official NASA dataset from the [PCoE repository]({OFFICIAL_REPOSITORY_URL}) or [NASA archive]({OFFICIAL_ARCHIVE_URL}) under `data/raw/`, then run `python scripts/prepare_data.py`.')
    if any('processed' in str(p) for p in missing): st.code('python scripts/prepare_data.py',language='bash')
    if any('results' in str(p) for p in missing): st.code('python scripts/run_experiments.py',language='bash')

def validate_upload(frame: pd.DataFrame,target: str):
    _,features,_=load_target(ProjectConfig().paths.processed,target)
    missing=[c for c in features if c not in frame.columns]
    if missing: raise ValueError(f'Missing required model features: {missing}')
    nonnumeric=[c for c in features if not pd.api.types.is_numeric_dtype(frame[c])]
    if nonnumeric: raise ValueError(f'Features must be numeric: {nonnumeric}')
    ordered=frame[features].replace([np.inf,-np.inf],np.nan)
    return ordered,features

@st.cache_resource(show_spinner=False)
def final_model(model_path: str,mtime:float):
    return joblib.load(model_path)

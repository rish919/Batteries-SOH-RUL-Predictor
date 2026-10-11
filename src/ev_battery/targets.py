"""Reference capacity, first crossing with optional persistence, and censored RUL."""
import numpy as np
import pandas as pd


def capacity_to_soh(capacity: np.ndarray, reference: float) -> np.ndarray:
    if not np.isfinite(reference) or reference<=0: raise ValueError('Reference capacity must be finite and positive')
    a = np.asarray(capacity,dtype=float)
    return np.where(np.isfinite(a)&(a>0),100*a/reference,np.nan)


def detect_eol(capacity: np.ndarray, threshold: float, persistence: int = 1) -> int | None:
    """Return zero-based START of first consecutive qualifying run; gaps break runs."""
    if not np.isfinite(threshold) or threshold<=0 or type(persistence) is not int or persistence<1: raise ValueError('Invalid EOL threshold/persistence')
    run = 0
    for i,q in enumerate(capacity):
        run = run+1 if np.isfinite(q) and 0<q<=threshold else 0
        if run>=persistence: return i-persistence+1
    return None


def add_targets(df: pd.DataFrame, rated_capacity: float = 2., reference: str = 'rated', threshold: float = .7, persistence: int = 1) -> pd.DataFrame:
    if reference not in {'rated','first_valid'} or not 0<threshold<1: raise ValueError('Invalid reference or threshold')
    capacity_to_soh(np.array([]),rated_capacity)
    out = []
    for _, frame in df.groupby('battery_id',sort=True):
        f=frame.sort_values('cycle_index').copy(); q=f.capacity_ah.to_numpy()
        valid=q[np.isfinite(q)&(q>0)]; first=valid[0] if len(valid) else np.nan
        ref=rated_capacity if reference=='rated' else first
        f['soh_rated_pct']=capacity_to_soh(q,rated_capacity)
        f['soh_first_valid_pct']=capacity_to_soh(q,first) if np.isfinite(first) else np.nan
        f['soh_pct']=f.soh_rated_pct if reference=='rated' else f.soh_first_valid_pct
        idx=detect_eol(q,ref*threshold,persistence) if np.isfinite(ref) else None
        eol=float(f.iloc[idx].cycle_index) if idx is not None else np.nan
        f['reference_capacity_ah']=ref; f['eol_cycle_index']=eol
        f['eol_censored']=idx is None; f['rul_cycles']=eol-f.cycle_index
        f.loc[f.rul_cycles<0,'rul_cycles']=np.nan
        # The first-valid reference cannot be used before it has been observed.
        if reference=='first_valid' and len(valid):
            first_idx=np.flatnonzero(np.isfinite(q)&(q>0))[0]
            f.loc[f.index[:first_idx],'rul_cycles']=np.nan
        out.append(f)
    return pd.concat(out,ignore_index=True)

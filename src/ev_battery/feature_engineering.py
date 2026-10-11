"""Discharge-indexed features at end-of-discharge observation time."""
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from ev_battery.battery_loader import Operation, scalar

SIGNALS = {'Voltage_measured':'voltage','Current_measured':'current','Temperature_measured':'temperature'}
STATS = ('min','max','mean','std')
BASE_FEATURES = ['cycle_index','ambient_temperature'] + [f'{op}_{signal}_{stat}' for op in ('charge','discharge') for signal in SIGNALS.values() for stat in STATS]
# Conservative SOH list excludes duration/energy and all capacity-derived features.
SOH_FEATURES = BASE_FEATURES.copy()
RUL_FEATURES = BASE_FEATURES + ['capacity_ah','capacity_lag1_ah','capacity_past3_mean_ah','capacity_change_ah']


def validate_features(features: list[str], task: str) -> None:
    allowed = SOH_FEATURES if task == 'soh' else RUL_FEATURES if task == 'rul' else []
    if not features or not set(features).issubset(allowed): raise ValueError(f'Unsafe {task} feature list: {set(features)-set(allowed)}')


def summaries(op: Operation) -> dict[str,float]:
    result = {}
    for field, signal in SIGNALS.items():
        a = op.measurements.get(field,np.array([]))
        a = a[np.isfinite(a)] if not np.iscomplexobj(a) else np.array([])
        for stat in STATS:
            result[f'{op.operation}_{signal}_{stat}'] = float(getattr(np,stat)(a)) if a.size else np.nan
    t = op.measurements.get('Time',np.array([]))
    valid = t.size >= 2 and not np.iscomplexobj(t) and np.isfinite(t).all() and np.all(np.diff(t)>0)
    result[f'{op.operation}_duration_s'] = float(t[-1]-t[0]) if valid else np.nan
    if not valid: op.warnings.append('Time: duration unavailable; requires finite strictly increasing seconds')
    return result


def build_features(batteries: dict[str,list[Operation]]) -> pd.DataFrame:
    rows = []
    for bid, operations in sorted(batteries.items()):
        pending = []; cycle = 0
        for op in operations:
            if op.operation == 'charge': pending.append(op)
            if op.operation != 'discharge': continue
            cycle += 1
            row = {'battery_id':bid,'cycle_index':cycle,'record_index':op.record_index,
                   'timestamp':op.timestamp,'ambient_temperature':op.ambient_temperature,
                   'capacity_ah':scalar(op.measurements.get('Capacity',np.array([]))),
                   'charge_record_index':np.nan, **summaries(op)}
            if not row['capacity_ah'] > 0: row['capacity_ah'] = np.nan
            if len(pending)==1 and op.timestamp and pending[0].timestamp:
                charge = pending[0]; cs = summaries(charge)
                duration = cs['charge_duration_s']
                t = charge.measurements.get('Time',np.array([]))
                # Absolute end time must be known and precede discharge start.
                if np.isfinite(duration) and t[0]>=0 and datetime.fromisoformat(charge.timestamp)+timedelta(seconds=float(t[-1])) <= datetime.fromisoformat(op.timestamp):
                    row.update(cs); row['charge_record_index'] = charge.record_index
                else: op.warnings.append('charge pairing rejected: timing invalid or overlapping')
            else: op.warnings.append('charge pairing unavailable: requires exactly one preceding charge and valid timestamps')
            pending = []; rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty: raise ValueError('No discharge records; cannot prepare targets')
    for name in BASE_FEATURES:
        if name not in df: df[name] = np.nan
    g = df.groupby('battery_id',sort=False)['capacity_ah']
    df['capacity_lag1_ah'] = g.shift(1)
    df['capacity_past3_mean_ah'] = g.transform(lambda s:s.shift(1).rolling(3,min_periods=3).mean())
    df['capacity_change_ah'] = df.capacity_ah-df.capacity_lag1_ah
    return df

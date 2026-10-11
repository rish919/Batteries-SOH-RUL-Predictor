"""NASA operation parser; retains numeric measurements and explicit quality issues."""
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import hashlib
import re
import numpy as np
from ev_battery.data_loader import discover_mat_files, read_mat_file, DatasetError
from ev_battery.inspection import _fields, _cycle_entries, _text_label

@dataclass
class Operation:
    battery_id: str
    record_index: int
    operation: str
    timestamp: str | None
    ambient_temperature: float
    measurements: dict[str, np.ndarray]
    warnings: list[str]


def numeric(value: Any, name: str, warnings: list[str]) -> np.ndarray:
    """Accept scalar/row/column vectors, preserving complex impedance values."""
    if value is None:
        warnings.append(f'{name}: missing')
        return np.array([], dtype=float)
    a = np.asarray(value)
    if a.dtype.kind not in 'biufc' or a.ndim > 2 or (a.ndim == 2 and min(a.shape) > 1):
        warnings.append(f'{name}: malformed numeric shape/type {a.shape}/{a.dtype}')
        return np.array([], dtype=float)
    a = a.reshape(-1)
    if not a.size: warnings.append(f'{name}: empty')
    if not np.isfinite(a).all(): warnings.append(f'{name}: {int((~np.isfinite(a)).sum())} nonfinite samples')
    return a


def scalar(a: np.ndarray) -> float:
    return float(a[0]) if a.size == 1 and not np.iscomplexobj(a) and np.isfinite(a[0]) else float('nan')


def parse_operation(battery_id: str, index: int, value: Any) -> Operation:
    f = _fields(value); warnings: list[str] = []
    kind = (_text_label(f.get('type')) or 'unknown').lower()
    if kind not in {'charge', 'discharge', 'impedance'}: warnings.append('unknown operation type')
    ambient = scalar(numeric(f.get('ambient_temperature'), 'ambient_temperature', warnings))
    if not np.isfinite(ambient): warnings.append('ambient_temperature: invalid scalar')
    t = numeric(f.get('time'), 'timestamp', warnings); timestamp = None
    try:
        if t.size != 6 or not np.isfinite(t).all() or np.iscomplexobj(t): raise ValueError()
        if not np.equal(t[:5], np.floor(t[:5])).all() or not 0 <= t[5] < 60: raise ValueError()
        timestamp = (datetime(*map(int, t[:5])) + timedelta(seconds=float(t[5]))).isoformat()
    except (ValueError, TypeError, OverflowError): warnings.append('timestamp: invalid date vector')
    fields = _fields(f.get('data'))
    if not fields: warnings.append('data: missing or malformed structure')
    measurements = {name: numeric(v, name, warnings) for name,v in fields.items()}
    required = ['Voltage_measured','Current_measured','Temperature_measured','Time'] if kind in {'charge','discharge'} else []
    if kind == 'discharge': required += ['Capacity']
    for name in required:
        if name not in measurements: measurements[name] = numeric(None, name, warnings)
        if np.iscomplexobj(measurements[name]):
            warnings.append(f'{name}: unexpected complex values; unusable')
            measurements[name] = np.array([])
    if kind == 'discharge' and not scalar(measurements['Capacity']) > 0:
        warnings.append('Capacity: requires one finite positive Ah value')
    if kind in {'charge','discharge'}:
        lengths = [len(measurements[n]) for n in required if n != 'Capacity']
        if len(set(lengths)) > 1: warnings.append('measurement lengths differ; no sample-wise alignment inferred')
    return Operation(battery_id,index,kind,timestamp,ambient,measurements,warnings)


def load_batteries(raw_dir: Path) -> tuple[dict[str,list[Operation]], list[dict[str,Any]]]:
    """Inspect every file; skip hash-identical copies, fail on conflicting IDs."""
    raw_dir = Path(raw_dir).expanduser().resolve()
    files = discover_mat_files(raw_dir)
    if not files: raise DatasetError(f'No MATLAB files found in {raw_dir}')
    batteries = {}; manifest = []; hashes = {}
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        item = {'path':str(path.relative_to(raw_dir)), 'sha256':digest}
        if digest in hashes:
            manifest.append({**item, 'duplicate_of':hashes[digest]}); continue
        hashes[digest] = item['path']
        _, contents = read_mat_file(path)
        found = False
        for name,value in contents.items():
            fields = _fields(value)
            if 'cycle' not in fields: continue
            battery_id = name if re.fullmatch(r'B\d+',name) else path.stem
            if battery_id in batteries: raise DatasetError(f'Conflicting nonidentical files for {battery_id}: {path}; explicit reconciliation required')
            batteries[battery_id] = [parse_operation(battery_id,i,e) for i,e in enumerate(_cycle_entries(fields['cycle']),1)]
            previous = None
            for op in batteries[battery_id]:
                if op.timestamp and previous and op.timestamp < previous:
                    op.warnings.append("timestamp precedes prior record; source record order retained")
                if op.timestamp: previous = op.timestamp
            found = True
        if not found: raise DatasetError(f'{path}: no top-level structure with cycle field; variables: {list(contents)}')
        manifest.append(item)
    return batteries, manifest

"""Target, parser, causal-feature and leakage regression tests; artificial unit fixtures only."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import savemat
from ev_battery.battery_loader import parse_operation, load_batteries, Operation
from ev_battery.feature_engineering import build_features, validate_features, SOH_FEATURES, RUL_FEATURES
from ev_battery.targets import capacity_to_soh, detect_eol, add_targets

class PreprocessingTests(unittest.TestCase):
    def frame(self,q):
        return pd.DataFrame({'battery_id':['test']*len(q),'cycle_index':range(1,len(q)+1),'capacity_ah':q})
    def test_soh(self):
        np.testing.assert_allclose(capacity_to_soh([2,1.4,0,np.nan],2),[100,70,np.nan,np.nan],equal_nan=True)
        with self.assertRaises(ValueError): capacity_to_soh([2],0)
    def test_eol_first_crossing_and_persistence(self):
        q=[2,1.3,1.6,1.4,1.3]
        self.assertEqual(detect_eol(q,1.4),1)
        self.assertEqual(detect_eol(q,1.4,2),3)
        self.assertIsNone(detect_eol([1.3,np.nan,1.3],1.4,2))
    def test_rul_and_post_eol(self):
        f=add_targets(self.frame([2,1.6,1.4,1.3]))
        np.testing.assert_allclose(f.rul_cycles,[2,1,0,np.nan],equal_nan=True)
        self.assertFalse(f.eol_censored.any())
    def test_censored_and_missing(self):
        for q in ([2,1.8,np.nan],[np.nan,np.nan]):
            f=add_targets(self.frame(q)); self.assertTrue(f.eol_censored.all()); self.assertTrue(f.rul_cycles.isna().all())
    def test_reference_comparison(self):
        f=add_targets(self.frame([1.8,1.3,1.25]),reference='first_valid')
        self.assertAlmostEqual(f.soh_pct.iloc[0],100)
        self.assertAlmostEqual(f.soh_rated_pct.iloc[0],90)
        self.assertEqual(f.eol_cycle_index.iloc[0],3)
    def test_feature_allowlists(self):
        validate_features(SOH_FEATURES,'soh'); validate_features(RUL_FEATURES,'rul')
        for name in ['capacity_ah','capacity_lag1_ah','soh_pct','eol_cycle_index','battery_id','rul_cycles','discharge_duration_s']:
            with self.subTest(name=name),self.assertRaises(ValueError): validate_features(SOH_FEATURES+[name],'soh')
        with self.assertRaises(ValueError): validate_features(RUL_FEATURES+['eol_cycle_index'],'rul')
    def test_missing_field_actual_mat_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'B9999.mat'
            savemat(path,{'B9999':{'cycle':{'type':'discharge','ambient_temperature':24,'time':[2020,1,1,0,0,0],'data':{'Capacity':[[1.8]]}}}})
            b,_=load_batteries(Path(d)); op=b['B9999'][0]
            self.assertIn('Voltage_measured: missing',op.warnings)
            self.assertEqual(op.measurements['Voltage_measured'].size,0)
    def test_causality(self):
        def records(last):
            return {'test':[Operation('test',i,'discharge',None,24,{'Capacity':np.array([q]),'Voltage_measured':np.array([3.,4.]),'Time':np.array([0.,1.])},[]) for i,q in enumerate([2.,1.9,1.8,last],1)]}
        a=build_features(records(1.7)); b=build_features(records(100))
        pd.testing.assert_frame_equal(a.iloc[:3],b.iloc[:3])
        self.assertAlmostEqual(a.capacity_past3_mean_ah.iloc[3],1.9)
    def test_duplicate_files_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'B9999.mat'; savemat(p,{'B9999':{'cycle':{'type':'impedance','data':{'Re':[[1]]}}}})
            (Path(d)/'copy.mat').write_bytes(p.read_bytes())
            b,m=load_batteries(Path(d)); self.assertEqual(len(b),1); self.assertEqual(sum('duplicate_of' in x for x in m),1)

if __name__=='__main__': unittest.main()

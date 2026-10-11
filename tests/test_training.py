"""Small artificial fixtures for evaluation invariants, not reported experiments."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from ev_battery.models import make_model
from ev_battery.evaluation import regression_metrics,tune_model
from ev_battery.sequences import sequence_indices
from ev_battery.experiments import load_target

class TrainingTests(unittest.TestCase):
    def test_metrics_and_undefined_r2(self):
        m=regression_metrics(np.array([1.,2.,3.]),np.array([1.,2.,4.]))
        self.assertAlmostEqual(m['mae'],1/3); self.assertAlmostEqual(m['rmse'],np.sqrt(1/3)); self.assertAlmostEqual(m['r2'],.5)
        self.assertIsNone(regression_metrics(np.array([1]),np.array([2]))['r2'])
        self.assertIsNone(regression_metrics(np.array([1,1]),np.array([1,2]))['r2'])
        with self.assertRaises(ValueError): regression_metrics(np.array([np.nan]),np.array([0]))
    def test_preprocessing_train_only_and_empty_feature(self):
        X=pd.DataFrame({'x':[1.,3.,np.nan],'empty':[np.nan]*3})
        model=make_model('ridge').fit(X,[10,20,30])
        before=model.regressor_.named_steps['imputer'].statistics_.copy()
        pred=model.predict(pd.DataFrame({'x':[1e8],'empty':[99]}))
        np.testing.assert_array_equal(before,model.regressor_.named_steps['imputer'].statistics_)
        self.assertEqual(before[0],2); self.assertEqual(before[1],0); self.assertTrue(np.isfinite(pred).all())
    def test_group_tuning_splits_and_reproducibility(self):
        X=pd.DataFrame({'x':np.arange(24,dtype=float)})
        y=pd.Series(np.arange(24,dtype=float)); groups=pd.Series(np.repeat(['a','b','c','d'],6))
        p,a=tune_model('svr',X,y,groups,42); q,b=tune_model('svr',X,y,groups,42)
        self.assertEqual(p,q); self.assertEqual(a,b)
        for trial in a['trials']:
            for split in trial['splits']: self.assertFalse(set(split['train_groups'])&set(split['validation_groups']))
        _,fallback=tune_model('svr',X.iloc[:12],y.iloc[:12],groups.iloc[:12],42)
        self.assertEqual(fallback['method'],'fixed')
    def test_sequences_respect_groups_gaps_and_endpoints(self):
        frame=pd.DataFrame({'battery_id':['a']*6+['b']*5,'cycle_index':[1,2,3,5,6,7,1,2,3,4,5]})
        windows,ends=sequence_indices(frame,3)
        for window,end in zip(windows,ends):
            self.assertEqual(frame.iloc[window].battery_id.nunique(),1)
            self.assertTrue((np.diff(frame.iloc[window].cycle_index)==1).all())
            self.assertEqual(window[-1],end)
        self.assertEqual(len(ends),5)
    def test_missing_prepared_data(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(FileNotFoundError,'prepare_data.py'): load_target(Path(d),'soh')
    def test_lstm_determinism_when_available(self):
        try: import torch
        except ImportError: self.skipTest('optional torch unavailable')
        from ev_battery.sequences import train_network
        X=np.arange(120,dtype=float).reshape(12,5,2)/120; y=np.arange(12)/12
        a,_=train_network(X,y,42,2); b,_=train_network(X,y,42,2)
        a.eval();b.eval()
        with torch.no_grad():
            np.testing.assert_array_equal(a(torch.tensor(X,dtype=torch.float32)).numpy(),b(torch.tensor(X,dtype=torch.float32)).numpy())


class OptionalAndEligibilityTests(unittest.TestCase):
    def test_missing_optional_libraries_are_reported(self):
        from unittest.mock import patch
        import builtins
        from ev_battery.experiments import optional_availability
        original=builtins.__import__
        def missing(name,*args,**kwargs):
            if name in {'torch','xgboost'}: raise ImportError('deliberately unavailable for unit test')
            return original(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=missing):
            status=optional_availability()
        self.assertFalse(status['lstm']['available']); self.assertIn('unavailable',status['xgboost']['reason'])
    def test_rul_censor_and_singleton_exclusions(self):
        from ev_battery.feature_engineering import RUL_FEATURES
        records=[]
        for bid in ['a','b','c']:
            for cycle,rul in enumerate([2.,1.,0.],1):
                records.append({**dict.fromkeys(RUL_FEATURES,1.),'battery_id':bid,'cycle_index':cycle,'rul_cycles':rul,'eol_censored':False})
        records.append({**dict.fromkeys(RUL_FEATURES,1.),'battery_id':'single','rul_cycles':0.,'eol_censored':False})
        records.append({**dict.fromkeys(RUL_FEATURES,1.),'battery_id':'censored','rul_cycles':np.nan,'eol_censored':True})
        with tempfile.TemporaryDirectory() as d:
            pd.DataFrame(records).to_csv(Path(d)/'rul_dataset.csv',index=False)
            frame,_,report=load_target(Path(d),'rul')
        self.assertEqual(set(frame.battery_id),{'a','b','c'}); self.assertEqual(report['excluded_rows'],2)

if __name__=='__main__': unittest.main()

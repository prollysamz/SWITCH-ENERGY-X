import json
import pickle
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pandas as pd
from model3 import derived, forward, build_Y
from reconstruction_repair import RepairedParams, map_latents, LOW, HIGH
from frozen_target import freeze,predict

ROOT=Path(__file__).resolve().parents[1]


class RepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old=pickle.loads((ROOT/'work/params_final.pkl').read_bytes())
        cls.params=RepairedParams.from_original(cls.old,json.loads((ROOT/'work/wind_repair_v2.json').read_text()))

    def test_wind_support_and_join(self):
        low=np.linspace(0,14,101)
        np.testing.assert_array_equal(self.params.wind(low),self.old.wind(low))
        tail=self.params.wind(np.linspace(15,30,200))
        self.assertTrue(np.all(tail>0) and np.all(np.diff(tail)<0))
        self.assertNotAlmostEqual(self.params.wind(18),self.params.wind(21))
        for join in [14,15]:
            self.assertLess(abs(self.params.wind(join+1e-6)-self.params.wind(join-1e-6)),1e-4)

    def test_known_high_wind_failure(self):
        df=pd.read_csv(ROOT/'Dataset/test.csv')
        row=df[df.row_id==438963]
        th,cov,info=map_latents(build_Y(row),self.params,need_cov=True,return_info=True)
        panel=float(derived(th,self.params)[1][0])
        sensor_panel=(1-row.feature_15.iloc[0])/self.params.beta+25
        self.assertLess(abs(panel-sensor_panel),3.)
        self.assertTrue(LOW[1]<th[0,1]<HIGH[1])
        self.assertFalse(info['still_inconsistent'][0])
        self.assertGreater(np.linalg.eigvalsh(cov[0]).min(),0)

    def test_corrupt_sensor_is_flagged_not_hidden(self):
        truth=np.array([[700.,28.,60.,6.,1013.,.3,.6,.65]])
        Y=forward(truth,self.params)
        from model3 import COL
        Y[0,COL[4]]=60.
        th,cov,info=map_latents(Y,self.params,need_cov=True,return_info=True)
        self.assertTrue(info['retry'][0] and info['accepted'][0])
        self.assertTrue(info['still_inconsistent'][0])
        self.assertTrue(np.all(th>=LOW) and np.all(th<=HIGH))
        self.assertGreater(np.linalg.eigvalsh(cov[0]).min(),0)

    def test_nonconverged_fallback_is_not_accepted(self):
        from model3 import COL
        truth=np.array([[700.,28.,60.,6.,1013.,.3,.6,.65]])
        Y=forward(truth,self.params)
        Y[0,COL[4]]=60.
        failed=SimpleNamespace(success=False,x=truth[0].copy())
        with patch('reconstruction_repair.original_map',return_value=(truth.copy(),None)), \
             patch('reconstruction_repair.least_squares',return_value=failed):
            th,_,info=map_latents(Y,self.params,return_info=True)
        self.assertTrue(info['failed_fallback'][0])
        self.assertFalse(info['accepted'][0])
        self.assertTrue(info['still_inconsistent'][0])
        np.testing.assert_array_equal(th,truth)

    def test_failed_fallback_cannot_leave_invalid_latents(self):
        truth=np.array([[700.,28.,60.,6.,1013.,.3,.6,.65]])
        invalid=truth.copy()
        invalid[0,1]=90.
        failed=SimpleNamespace(success=False,x=truth[0].copy())
        with patch('reconstruction_repair.original_map',return_value=(invalid,None)), \
             patch('reconstruction_repair.least_squares',return_value=failed):
            with self.assertRaises(RuntimeError):
                map_latents(forward(truth,self.params),self.params)

    def test_frozen_weights_reproduce_scored_best(self):
        train=np.load(ROOT/'work/blocks_train.npz')
        test=np.load(ROOT/'work/blocks_test.npz')
        h=json.loads((ROOT/'candidates/manifest.json').read_text())
        plane=json.loads((ROOT/'candidates/refined_plane_manifest.json').read_text())
        f=freeze(train,test,h,plane)
        expected=pd.read_csv(ROOT/'candidates/submission_best_0p78886.csv').prediction.to_numpy()
        np.testing.assert_allclose(predict(test,f),expected,rtol=1e-12,atol=1e-12)


if __name__=='__main__':
    unittest.main()

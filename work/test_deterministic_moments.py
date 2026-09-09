import json
import pickle
import unittest
from pathlib import Path
import numpy as np
from model3 import derived
from deterministic_moments import normal_nodes,moments
from reconstruction_repair import RepairedParams

ROOT=Path(__file__).resolve().parents[1]


class DeterministicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=pickle.loads((ROOT/'work/params_final.pkl').read_bytes())
        cls.p=RepairedParams.from_original(p,json.loads((ROOT/'work/wind_repair_v2.json').read_text()))
        cls.th=np.array([[700.,28.,60.,6.,1013.,.3,.6,.65],[800.,30.,65.,7.,1015.,.4,.7,.75]])
        cls.cov=np.broadcast_to(np.diag([20,.7,1,.1,1,.01,.01,.01])**2,(2,8,8)).copy()

    def test_affine_temperature_mean_is_exact(self):
        z=moments(self.th,self.cov,self.p,normal_nodes(8,104))
        np.testing.assert_allclose(z['panel'],derived(self.th,self.p)[1],atol=1e-12,rtol=0)

    def test_batch_size_and_row_order_do_not_change_results(self):
        nodes=normal_nodes(6,104)
        all_rows=moments(self.th,self.cov,self.p,nodes)
        reverse=moments(self.th[::-1],self.cov[::-1],self.p,nodes)
        first=moments(self.th[:1],self.cov[:1],self.p,nodes)
        for key in all_rows:
            np.testing.assert_array_equal(all_rows[key],reverse[key][::-1])
            np.testing.assert_array_equal(all_rows[key][:1],first[key])


if __name__=='__main__':unittest.main()

import unittest
from unittest.mock import patch
from types import SimpleNamespace
import json
import pickle
import numpy as np
from learned_energy_distillation import ROOT, stable_map
from reconstruction_repair import RepairedParams
from model3 import forward


class ExtendedSolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original = pickle.loads((ROOT/'work/params_final.pkl').read_bytes())
        cls.params = RepairedParams.from_original(original,
            json.loads((ROOT/'work/wind_repair_v2.json').read_text()))
        cls.truth = np.array([[700., 28., 60., 6., 1013., .3, .6, .65]])

    def test_extended_solver_retains_row_and_positive_covariance(self):
        with patch('learned_energy_distillation.map_latents', side_effect=RuntimeError('retry')):
            th, cov, info = stable_map(forward(self.truth, self.params), self.params)
        self.assertEqual(th.shape, self.truth.shape)
        self.assertTrue(info['accepted'][0])
        self.assertFalse(info['failed_fallback'][0])
        self.assertGreater(np.linalg.eigvalsh(cov[0]).min(), 0)
        np.testing.assert_allclose(th, self.truth, rtol=.01, atol=.01)

    def test_nonconverged_extended_fit_is_rejected(self):
        failed = SimpleNamespace(success=False, x=self.truth[0])
        with patch('learned_energy_distillation.map_latents', side_effect=RuntimeError('retry')), \
             patch('learned_energy_distillation.least_squares', return_value=failed):
            with self.assertRaises(RuntimeError):
                stable_map(forward(self.truth, self.params), self.params)


if __name__ == '__main__':
    unittest.main()

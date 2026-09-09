"""Verify inferred score surfaces against independently known sample residuals."""
import unittest
import numpy as np
from refine_scored_plane import recover_line_moment,fit_surface,surface_mse


class PlaneTests(unittest.TestCase):
    def test_recovers_two_direction_surface_and_optimum(self):
        rng=np.random.default_rng(908)
        base=rng.normal(3,4,6000)
        d=rng.normal(size=(6000,2))+[.2,-.3]
        d[:,1]+=.4*d[:,0]
        target=base+d@np.array([2.1,.4])+rng.normal(0,.8,len(base))
        rmse=lambda p:np.sqrt(np.mean((p-target)**2))
        r0=rmse(base)
        r1=rmse(base+d[:,0])
        rt=rmse(base+2.28*d[:,0])
        q=recover_line_moment(r0,r1,rt,2.28)
        gram=d.T@d/len(d)
        self.assertAlmostEqual(q,gram[0,0],places=11)
        linear,weights=fit_surface(r0,[r1,rmse(base+d[:,1])],gram)
        actual_weights=np.linalg.lstsq(d,target-base,rcond=None)[0]
        np.testing.assert_allclose(weights,actual_weights,rtol=1e-11,atol=1e-11)
        for w in [np.array([0.,0.]),np.array([1.,0.]),np.array([-.5,3.]),weights]:
            self.assertAlmostEqual(surface_mse(r0,linear,gram,w),rmse(base+d@w)**2,places=10)

    def test_rejects_unidentifiable_inputs(self):
        for t in [0,1,np.nan]:
            with self.assertRaises(ValueError):
                recover_line_moment(2,1,1,t)
        with self.assertRaises(ValueError):
            fit_surface(2,[1,1],np.ones((2,2)))


if __name__=='__main__':
    unittest.main()

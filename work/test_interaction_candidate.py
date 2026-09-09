import unittest
import numpy as np
from build_interaction_candidate import orthogonal_effect


class InteractionTests(unittest.TestCase):
    def test_projection_removes_existing_redundant_directions(self):
        rng=np.random.default_rng(924)
        x=rng.normal(size=(5000,3))
        p=np.column_stack([x[:,0]+4,x[:,1]-2,2*x[:,0]+x[:,1]+7])
        raw=3*x[:,0]-2*x[:,1]+.5*x[:,2]+10
        effect,info=orthogonal_effect(raw,p)
        self.assertEqual(info['existing_prediction_rank'],2)
        self.assertAlmostEqual(effect.mean(),0,places=12)
        self.assertAlmostEqual(effect.std(),1,places=12)
        np.testing.assert_allclose((p-p.mean(axis=0)).T@effect/len(p),0,atol=1e-12)

    def test_rejects_redundant_effect(self):
        x=np.arange(1000,dtype=float)
        with self.assertRaises(ValueError):
            orthogonal_effect(3*x+2,np.column_stack([x,x*2]))


if __name__=='__main__':
    unittest.main()

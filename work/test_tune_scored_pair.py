"""Check score-based extrapolation against known synthetic row-level errors."""
import unittest
import numpy as np
from tune_scored_pair import line_mse, optimal_weight


class ScoreDirectionTests(unittest.TestCase):
    def test_recovers_actual_error_curve_and_optimum(self):
        rng=np.random.default_rng(908)
        p0=rng.normal(4,3,2000)
        direction=rng.normal(.4,1,2000)
        target=p0+2.2*direction+rng.normal(0,.7,2000)
        p1=p0+direction
        r0=np.sqrt(np.mean((p0-target)**2))
        r1=np.sqrt(np.mean((p1-target)**2))
        q=np.mean(direction**2)
        for weight in [-1,0,.5,1,2.3,4]:
            actual=np.mean((p0+weight*direction-target)**2)
            self.assertAlmostEqual(line_mse(weight,r0,r1,q),actual,places=11)
        actual_opt=np.dot(direction,target-p0)/np.dot(direction,direction)
        self.assertAlmostEqual(optimal_weight(r0,r1,q),actual_opt,places=12)

    def test_rejects_inconsistent_or_degenerate_inputs(self):
        for values in [(1,3,.1),(1,1,0),(-1,1,1),(1,1,np.nan),(1,1,9)]:
            with self.assertRaises(ValueError):
                optimal_weight(*values)


if __name__=='__main__':
    unittest.main()

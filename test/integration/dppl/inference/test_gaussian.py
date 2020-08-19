
from .harness import MCMCTest
from pprint import pprint
import numpy as np

def test_gaussian():
    t_gaussian = MCMCTest(
        name='gaussian',
        model_file='deepppl/tests/good/gaussian.stan'
    )
    
    res = t_gaussian.run()
    theta = t_gaussian.pyro_samples['theta']
    # assert np.abs(theta.mean() - 1000) < 1
    # assert np.abs(theta.std() - 1.0) < 0.1
    
    return res
    
if __name__ == "__main__":
    pprint(test_gaussian())
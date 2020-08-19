
from .harness import MCMCTest, Config
from pprint import pprint
import numpy as np

def test_gaussian_log_density(config=Config()):
    t_gaussian_log_density = MCMCTest(
        name='gaussian_log_density',
        model_file='deepppl/tests/good/gaussian_log_density.stan',
        config=config
    )
    
    res = t_gaussian_log_density.run()
    theta = t_gaussian_log_density.pyro_samples['theta']
    # assert np.abs(theta.mean() - 1000) < 1
    # assert np.abs(theta.std() - 1.0) < 0.1
    
    return res
    
if __name__ == "__main__":
    pprint(test_gaussian_log_density())
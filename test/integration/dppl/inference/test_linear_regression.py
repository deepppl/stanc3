
from .harness import MCMCTest, Config
from pprint import pprint
import pyro.distributions as dist
import numpy as np


def test_linear_regression(config=Config()):
    data = {}
    data['N'] = 10
    data['x'] = np.arange(data['N'])
    data['y'] = np.random.normal(data['x'], 0.1)

    t_linear_regression = MCMCTest(
        name='linear_regression',
        model_file='deepppl/tests/good/linear_regression_array.stan',
        data=data,
        config=config
    )
    
    return t_linear_regression.run()

if __name__ == "__main__":
    pprint(test_linear_regression())

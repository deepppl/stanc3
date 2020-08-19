
from .harness import MCMCTest, Config
import numpy as np
from pprint import pprint

def test_schools(config=Config()):
    data = {}
    data['N'] = 8
    data['y'] = np.array([28, 8, -3, 7, -1, 1, 18, 12])
    data['sigma_y'] = np.array([15, 10, 16, 11, 9, 11, 10, 18])

    test_schools = MCMCTest(
        name='schools',
        model_file='good/schools.stan',
        # pyro_file='naive/schools_naive.py',
        data=data,
        config=config
    )

    return test_schools.run()

if __name__ == "__main__":
    pprint(test_schools())

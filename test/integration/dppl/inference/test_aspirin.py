
from .harness import MCMCTest, Config
import numpy as np
from pprint import pprint

def test_aspirin(config=Config()):
    data = {}
    data['y'] = np.array([2.77, 2.50, 1.84, 2.56, 2.31, -1.15])
    data['s'] = np.array([1.65, 1.31, 2.34, 1.67, 1.98, 0.90])
    data['N'] = len(data['y'])
    data['mu_loc'] = np.mean(data['y'])
    data['mu_scale'] = 5 * np.std(data['y'])
    data['tau_scale'] = 2.5 * np.std(data['y'])
    data['tau_df'] = 4

    t_aspirin = MCMCTest(
        name='aspirin',
        model_file='good/aspirin.stan',
        # pyro_file='naive/aspirin_naive.py',
        data=data,
        config=config
    )

    return t_aspirin.run()
    
if __name__ == "__main__":
    pprint(test_aspirin())

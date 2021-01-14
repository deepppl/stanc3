
from .harness import MCMCTest, Config
from pprint import pprint
import numpy as np

def test_coin(config=Config()):
    data = {}
    data['N'] = 10
    data['x'] = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 1])

    t_coin = MCMCTest(
        name='coin',
        model_file='good/coin.stan',
        data=data,
        config=config
    )

    return t_coin.run()

if __name__ == "__main__":
    pprint(test_coin())

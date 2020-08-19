
from .harness import MCMCTest, Config
from pprint import pprint

def test_seeds(config=Config()):
    data = {}
    data['I'] = 21
    data['n'] = [10, 23, 23, 26, 17, 5, 53, 55, 32,
                 46, 10, 8, 10, 8, 23, 0, 3, 22, 15, 32, 3]
    data['N'] = [39, 62, 81, 51, 39, 6, 74, 72, 51,
                 79, 13, 16, 30, 28, 45, 4, 12, 41, 30, 51, 7]
    data['x1'] = [0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    data['x2'] = [0, 0, 0, 0, 0, 1, 1, 1, 1,
                  1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    test_seeds = MCMCTest(
        name='seeds',
        model_file='deepppl/tests/good/seeds.stan',
        pyro_file='deepppl/tests/naive/seeds_naive.py',
        data=data,
        config=config
    )
    
    return test_seeds.run()
    
if __name__ == "__main__":
    pprint(test_seeds())
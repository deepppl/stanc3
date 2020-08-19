
from .harness import MCMCTest, Config
from pprint import pprint

def test_double_gaussian(config=Config()):
    t_double_gaussian = MCMCTest(
        name='double_gaussian',
        model_file='deepppl/tests/good/double_gaussian.stan',
        config=config
    )
    
    return t_double_gaussian.run()
    
if __name__ == "__main__":
    pprint(test_double_gaussian())
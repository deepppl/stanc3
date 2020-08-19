
from .harness import MCMCTest, Config
from pprint import pprint

def test_neal_funnel(config=Config()):   
    test_neal_funnel = MCMCTest(
        name='neal_funnel',
        model_file='deepppl/tests/good/neal_funnel.stan',
        config=config
    )
    
    return test_neal_funnel.run()
    
if __name__ == "__main__":
    pprint(test_neal_funnel())
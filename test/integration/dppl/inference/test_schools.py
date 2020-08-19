
from .harness import MCMCTest, Config
from pprint import pprint

def test_schools(config=Config()):
    data = {}
    data['N'] = 8
    data['y'] = [28, 8, -3, 7, -1, 1, 18, 12]
    data['sigma_y'] = [15, 10, 16, 11, 9, 11, 10, 18]
        
    test_schools = MCMCTest(
        name='schools',
        model_file='deepppl/tests/good/schools.stan',
        pyro_file='deepppl/tests/naive/schools_naive.py',
        data=data,
        config=config
    )
    
    return test_schools.run()
    
if __name__ == "__main__":
    pprint(test_schools())
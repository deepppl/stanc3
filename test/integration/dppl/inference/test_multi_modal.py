
from .harness import MCMCTest
from pprint import pprint

def test_multi_modal():
    data = {}
        
    t_multi_modal = MCMCTest(
        name='multi_modal',
        model_file='deepppl/tests/good/multimodal.stan',
        data=data,
        with_numpyro=False # Numpyro cannot compile boolean condition
    )
    
    return t_multi_modal.run()
    
if __name__ == "__main__":
    pprint(test_multi_modal())
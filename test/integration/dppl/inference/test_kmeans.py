
from .harness import MCMCTest
from pprint import pprint
from sklearn.datasets import make_blobs

def test_kmeans():
    data = {}
    data['N'] = 6
    data['D'] = 2
    data['K'] = 3
    
    X, _ = make_blobs(n_samples=data['N'], random_state=170)
    
    data['y'] = X
    
    t_kmeans = MCMCTest(
        name='kmeans',
        model_file='deepppl/tests/good/kmeans.stan',
        data=data,
        with_numpyro=False # In place mutation not supported in Numpyro
    )
    
    return t_kmeans.run()
    
if __name__ == "__main__":
    pprint(test_kmeans())
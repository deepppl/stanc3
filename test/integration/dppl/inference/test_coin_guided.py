
import pyro
import deepppl
import numpy as np


def test_coin_guided_inference():
    model = deepppl.PyroModel(
        model_file='deepppl/tests/good/coin_guide.stan')
    svi = model.svi(params={'lr': 0.1})
    N = 10
    x = [0, 0, 0, 0, 0, 0, 1, 0, 0, 1]
    for step in range(10000):
        svi.step(N, x)
        if step % 100 == 0:
            print('.', end='')
    alpha_q = pyro.param("alpha_q").item()
    beta_q = pyro.param("beta_q").item()
    
    print(f'alpha: {alpha_q} beta: {beta_q}')

    # The posterior distribution should be a Beta(1 + 2, 1 + 8)
    assert np.abs(alpha_q - (1 + 2)) < 2.0
    assert np.abs(beta_q - (1 + 8)) < 2.0

if __name__ == "__main__":
    test_coin_guided_inference()

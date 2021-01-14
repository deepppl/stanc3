
import pyro
import numpy as np
import torch
from runtimes.dppl import Model


def test_coin_guided_inference():
    model = Model('pyro', 'good/coin_guide.stan', True, 'mixed')
    svi = model.svi(params={'lr': 0.1})
    N = 10
    x = torch.Tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 1])
    for step in range(10000):
        svi.step(N=N, x=x)
        if step % 100 == 0:
            print('.', end='', flush=True)
    alpha_q = pyro.param("alpha_q").item()
    beta_q = pyro.param("beta_q").item()
    print(f'\nalpha: {alpha_q} beta: {beta_q}')

    # The posterior distribution should be a Beta(1 + 2, 1 + 8)
    assert np.abs(alpha_q - (1 + 2)) < 2.0
    assert np.abs(beta_q - (1 + 8)) < 2.0

if __name__ == "__main__":
    test_coin_guided_inference()

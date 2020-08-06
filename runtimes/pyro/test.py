from dppl import PyroModel
from torch import Tensor

test = PyroModel('/tmp/a.py')
mcmc = test.mcmc(samples=100)
mcmc.run(N=10, x=Tensor([0 for _ in range(10)]))

print(mcmc.get_samples())
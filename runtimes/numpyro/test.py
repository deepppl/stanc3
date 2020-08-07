from dppl import NumpyroModel
from numpy import array 

test = NumpyroModel('/tmp/a.stan')
mcmc = test.mcmc(warmups=10, samples=100)
mcmc.run(N=10, x=array([0 for _ in range(10)]))

print(mcmc.get_samples())
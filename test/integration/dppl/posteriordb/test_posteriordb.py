from dataclasses import dataclass, field
from posteriordb import PosteriorDatabase
from os.path import splitext, basename
import os
from runtimes.pyro.dppl import PyroModel
from runtimes.numpyro.dppl import NumpyroModel

import numpy as np
from jax import numpy as jnp
from torch import Tensor


pdb_root = "/Users/lmandel/stan/posteriordb"
pdb_path = os.path.join(pdb_root, "posterior_database")
my_pdb = PosteriorDatabase(pdb_path)

@dataclass
class Config:
    iterations: int = 100
    warmups: int = 10
    chains: int = 1
    thin: int = 2


def _convert_to_tensor(value):
    if isinstance(value, (list, np.ndarray)):
        return Tensor(value)
    elif isinstance(value, dict):
        return {k: _convert_to_tensor(v) for k, v in value.items()}
    else:
        return value

def test(posterior, config):
    model = posterior.model
    data = posterior.data
    stanfile = model.code_file_path("stan")
    pythonfile = os.path.join(os.getcwd(), splitext(basename(stanfile))[0] + ".py")
    pyro_model = PyroModel(stanfile, pyfile=pythonfile)
    mcmc = pyro_model.mcmc(config.iterations,
                        warmups=config.warmups,
                        chains=config.chains,
                        thin=config.thin)
    data = {k: _convert_to_tensor(v) for k, v in data.values().items()}
    mcmc.run(**data)
    return mcmc.get_samples()

posterior = my_pdb.posterior("eight_schools-eight_schools_centered")
config = Config()
samples = test(posterior, config)
print(samples)
# print(dir(posterior.gold_standard))
# print(posterior.posterior_info)
# print(posterior.reference_posterior)

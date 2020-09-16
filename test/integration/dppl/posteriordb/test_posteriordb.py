from dataclasses import dataclass, field
from posteriordb import PosteriorDatabase
from os.path import splitext, basename
import os
from runtimes.pyro.dppl import PyroModel
from runtimes.numpyro.dppl import NumpyroModel

import numpy as np
from jax import numpy as jnp
from torch import tensor
import torch

pdb_root = "/Users/lmandel/stan/posteriordb"
pdb_path = os.path.join(pdb_root, "posterior_database")

@dataclass
class Config:
    # iterations: int = 100
    # warmups: int = 10
    # chains: int = 1
    # thin: int = 2
    iterations: int = 10
    warmups: int = 1
    chains: int = 1
    thin: int = 1


def _convert_to_tensor(value):
    if isinstance(value, (list, np.ndarray)):
        # if all(isinstance(n, int) for n in value):
        #     return tensor(value, dtype=torch.long)
        return tensor(value)
    elif isinstance(value, dict):
        return {k: _convert_to_tensor(v) for k, v in value.items()}
    else:
        return value

def test(posterior, config):
    model = posterior.model
    data = posterior.data
    stanfile = model.code_file_path("stan")
    pythonfile = os.path.join(os.getcwd(), splitext(basename(stanfile))[0] + ".py")
    try:
        pyro_model = PyroModel(stanfile, pyfile=pythonfile)
    except Exception as e:
        return { 'code': 1, 'msg': f'compilation error: {model.name}', 'exn': e }
    try:
        mcmc = pyro_model.mcmc(config.iterations,
                               warmups=config.warmups,
                               chains=config.chains,
                               thin=config.thin)
        inputs = {k: _convert_to_tensor(v) for k, v in data.values().items()}
        mcmc.run(**inputs)
    except Exception as e:
        return { 'code': 2, 'msg': f'Inference error: {model.name}({data.name})', 'exn': e }
    return { 'code': 0, 'samples': mcmc.get_samples() }


my_pdb = PosteriorDatabase(pdb_path)

# posterior = my_pdb.posterior('radon_mn-radon_variable_intercept_centered')
# config = Config()
# res = test(posterior, config)
# print(res['code'])
# print(res['msg'])
# print(res['exn'])


success = 0
compile_error = 0
inference_error = 0

for name in my_pdb.posterior_names():
    # posterior = my_pdb.posterior("eight_schools-eight_schools_centered")
    print(f'Test {name}')
    posterior = my_pdb.posterior(name)
    config = Config()
    res = test(posterior, config)
    if res['code'] == 0:
        success = success + 1
    elif res['code'] == 1:
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        print(res['msg'])
        print(res['exn'])
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        compile_error = compile_error + 1
    elif res['code'] == 2:
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        print(res['msg'])
        print(res['exn'])
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        inference_error = inference_error + 1
    print(f'success: {success}, compile errors: {compile_error}, inference errors: {inference_error}, total: {success + compile_error + inference_error}')
# print(dir(posterior.gold_standard))
# print(posterior.posterior_info)
# print(posterior.reference_posterior)

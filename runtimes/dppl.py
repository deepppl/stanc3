import os
import sys
import inspect
import pathlib
import importlib
import subprocess
import inspect
import jax
from os.path import splitext, basename, dirname
from pandas import DataFrame, Series
from collections import defaultdict
from functools import partial


def _flatten_dict(d):
    def _flatten(name, a):
        if len(a.shape) == 0:
            return {name: a.tolist()}
        else:
            return {
                k: v
                for d in (_flatten(name + f"[{i+1}]", v) for i, v in enumerate(a))
                for k, v in d.items()
            }

    return {
        fk: fv for f in (_flatten(k, v) for k, v in d.items()) for fk, fv in f.items()
    }


def _exec(cmd):
    try:
        output = subprocess.check_output(
            cmd, stderr=subprocess.PIPE, universal_newlines=True
        )
        if output:
            print(output, file=sys.stdout)
    except subprocess.CalledProcessError as exc:
        print(f"Error {exc.returncode}: {exc.stderr}", file=sys.stderr)
        assert False


def compile(backend, mode, stanfile, pyfile, compiler):
    _exec(compiler +
        [
            f"--{backend}",
            "--mode",
            mode,
            "--o",
            pyfile,
            stanfile,
        ]
    )


class Model:
    def __init__(self, pyro, tensor, stanfile, recompile=True, mode="mixed", compiler=["dune","exec","stanc","--"]):
        self.pyro = pyro
        self.tensor = tensor
        self.pyro_backend = pyro.__name__

        if not os.path.exists("_tmp"):
            os.makedirs("_tmp")
            pathlib.Path("_tmp/__init__.py").touch()

        self.name = splitext(basename(stanfile))[0]
        self.pyfile = f"_tmp/{self.name}.py"

        if recompile:
            compile(self.pyro_backend, mode, stanfile, self.pyfile, compiler)

        modname = f"_tmp.{self.name}"
        self.module = importlib.import_module(modname)
        if modname in sys.modules:
            importlib.reload(sys.modules[modname])

    def mcmc(self, samples, warmups=0, chains=1, thin=1, kernel=None, **kwargs):
        if kernel is None:
            kernel = self.pyro.infer.NUTS(
                self.module.model,
                adapt_step_size=True,
                init_strategy=self.pyro.init_to_sample,
            )

        # HACK pyro an numpyro MCMC do not have the same parameters...
        if self.pyro_backend == "numpyro":
            rng_key, _ = jax.random.split(jax.random.PRNGKey(0))
            mcmc = self.pyro.infer.MCMC(
                kernel, warmups, samples - warmups, num_chains=chains, **kwargs
            )
            mcmc.run = partial(mcmc.run, rng_key)
        elif self.pyro_backend == "pyro":
            kwargs = {"mp_context": "forkserver", **kwargs}
            mcmc = self.pyro.infer.MCMC(
                kernel,
                samples - warmups,
                warmup_steps=warmups,
                num_chains=chains,
                **kwargs,
            )
        else:
            assert False, "Invalid Pyro implementation"
        mcmc.thin = thin
        return MCMCProxy(mcmc, self.module, self.tensor)

    def svi(
        self, optimizer=None, loss=None, params={"lr": 0.0005, "betas": (0.90, 0.999)}
    ):
        optimizer = optimizer if optimizer else self.pyro.optim.Adam(params)
        loss = loss if loss is not None else self.pyro.infer.Trace_ELBO()
        svi = self.pyro.infer.SVI(self.module.model, self.module.guide, optimizer, loss)
        return SVIProxy(svi, self.module)


class MCMCProxy:
    def __init__(
        self,
        mcmc,
        module,
        tensor,
    ):
        self.mcmc = mcmc
        self.module = module
        self.tensor = tensor
        self.kwargs = {}

    def _sample_model(self):
        samples = self.mcmc.get_samples()
        return {x: samples[x][:: self.mcmc.thin] for x in samples}

    def _sample_generated(self, samples):
        kwargs = self.kwargs
        res = defaultdict(list)
        num_samples = len(list(samples.values())[0])
        for i in range(num_samples):
            kwargs.update({x: samples[x][i] for x in samples})
            if hasattr(self.module, "generated_quantities"):
                d = self.module.generated_quantities(kwargs)
                for k, v in d.items():
                    res[k].append(v)
        return {k: self.tensor.stack(v) for k, v in res.items()}

    def run(self, **kwargs):
        self.kwargs = self.module.convert_inputs(kwargs)
        if hasattr(self.module, "transformed_data"):
            self.kwargs.update(self.module.transformed_data(**self.kwargs))
        self.mcmc.run(**self.kwargs)
        self.samples = self._sample_model()
        if hasattr(self.module, "generated_quantities"):
            gen = self._sample_generated(self.samples)
            self.samples.update(gen)

    def get_samples(self):
        return self.samples

    def summary(self):
        d_mean = _flatten_dict(
            {
                k: self.tensor.mean(
                    self.tensor.array(v, dtype=self.tensor.float), axis=0
                )
                for k, v in self.samples.items()
            }
        )
        d_std = _flatten_dict(
            {
                k: self.tensor.std(
                    self.tensor.array(v, dtype=self.tensor.float), axis=0
                )
                for k, v in self.samples.items()
            }
        )
        return DataFrame({"mean": Series(d_mean), "std": Series(d_std)})


class SVIProxy(object):
    def __init__(self, svi, module):
        self.svi = svi
        self.module = module
        self.args = []
        self.kwargs = {}

    def posterior(self, n, **kwargs):
        # signature = inspect.signature(self.svi.guide)
        # kwargs = {k : None for k in signature.parameters}
        return [self.svi.guide(**kwargs) for _ in range(n)]

    def step(self, *args, **kwargs):
        self.kwargs = kwargs
        if hasattr(self.module, "transformed_data"):
            self.kwargs.update(self.module.transformed_data(**self.kwargs))
        return self.svi.step(**kwargs)


import pyro
import torch

torch.array = torch.tensor
pyro.init_to_sample = pyro.infer.autoguide.initialization.init_to_sample

import numpyro
import jax.numpy as jnp

jnp.long = jnp.dtype("int32")
jnp.float = jnp.dtype("float32")
numpyro.init_to_sample = numpyro.infer.initialization.init_to_sample

PyroModel = partial(Model, pyro, torch)
NumpyroModel = partial(Model, numpyro, jnp)

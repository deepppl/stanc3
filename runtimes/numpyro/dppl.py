import os
import sys
import inspect
import pathlib
import importlib
import subprocess
import jax
import inspect
import numpyro as pyro
import jax.numpy as tensor
from os.path import splitext, basename, dirname
from pandas import DataFrame, Series
from collections import defaultdict
from functools import partial

tensor.long = tensor.dtype("int32")
tensor.float = tensor.dtype("float32")

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


def _compile(mode, stanfile, pyfile):
    _exec(
        [
            "dune",
            "exec",
            "stanc",
            "--",
            "--numpyro",
            "--mode",
            mode,
            "--o",
            pyfile,
            stanfile,
        ]
    )


class NumpyroModel:
    def __init__(self, stanfile, compile, mode):
        if not os.path.exists("_tmp"):
            os.makedirs("_tmp")
            pathlib.Path("_tmp/__init__.py").touch()

        self.name = splitext(basename(stanfile))[0]
        self.pyfile = f"_tmp/{self.name}.py"
        if compile:
            _compile(mode, stanfile, self.pyfile)
        modname = f"_tmp.{self.name}"
        self.module = importlib.import_module(modname)
        if modname in sys.modules:
            importlib.reload(sys.modules[modname])

    def mcmc(self, samples, warmups=0, chains=1, thin=1, kernel=None, **kwargs):
        if kernel is None:
            kernel = pyro.infer.NUTS(self.module.model, adapt_step_size=True)

        rng_key, _ = jax.random.split(jax.random.PRNGKey(0))
        mcmc = pyro.infer.MCMC(
            kernel, 
            warmups, 
            samples - warmups, 
            num_chains=chains, 
            **kwargs,
        )
        mcmc.run = partial(mcmc.run, rng_key)
        mcmc.thin = thin
        return MCMCProxy(mcmc, self.module)

class MCMCProxy:
    def __init__(
        self,
        mcmc,
        module,
    ):
        self.mcmc = mcmc
        self.module = module
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
        return {k: tensor.stack(v) for k, v in res.items()}

    def run(self, **kwargs):
        self.kwargs = kwargs
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
                k: tensor.mean(tensor.array(v, dtype=tensor.float), axis=0)
                for k, v in self.samples.items()
            }
        )
        d_std = _flatten_dict(
            {
                k: tensor.std(tensor.array(v, dtype=tensor.float), axis=0)
                for k, v in self.samples.items()
            }
        )
        return DataFrame({"mean": Series(d_mean), "std": Series(d_std)})

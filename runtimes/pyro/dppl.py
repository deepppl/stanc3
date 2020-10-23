from os.path import splitext, basename, dirname
import importlib.util
import pyro
import torch
from pandas import DataFrame, Series
from collections import defaultdict
import subprocess
import inspect


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


class PyroModel:
    def __init__(self, stanfile, pyfile=None, compile=True, mode="comprehensive"):
        self.name = basename(stanfile)
        self.pyfile = splitext(stanfile)[0] + ".py" if pyfile == None else pyfile
        if compile:
            subprocess.check_call(
                [
                    "dune",
                    "exec",
                    "stanc",
                    "--",
                    "--pyro",
                    "--mode",
                    mode,
                    "--o",
                    self.pyfile,
                    stanfile,
                ]
            )
        module = importlib.import_module(splitext(self.pyfile)[0])
        self.convert_inputs = module.convert_inputs
        self._model = module.model
        self._transformed_data = None
        self._generated_quantities = None
        if hasattr(module, "transformed_data"):
            self._transformed_data = module.transformed_data
        if hasattr(module, "generated_quantities"):
            self._generated_quantities = module.generated_quantities
        if hasattr(module, "guide"):
            self._guide = module.guide

    def mcmc(self, samples, warmups=0, chains=1, thin=1, kernel=None):
        if kernel is None:
            kernel = pyro.infer.NUTS(self._model, adapt_step_size=True)
        mcmc = pyro.infer.MCMC(
            kernel,
            samples - warmups,
            warmup_steps=warmups,
            num_chains=chains,
        )
        return MCMCProxy(mcmc, self._generated_quantities, self._transformed_data, thin)

    def svi(
        self, optimizer=None, loss=None, params={"lr": 0.0005, "betas": (0.90, 0.999)}
    ):
        optimizer = optimizer if optimizer else pyro.optim.Adam(params)
        loss = loss if loss is not None else pyro.infer.Trace_ELBO()
        svi = pyro.infer.SVI(self._model, self._guide, optimizer, loss)
        return SVIProxy(svi, self._generated_quantities, self._transformed_data)


class MCMCProxy:
    def __init__(
        self,
        mcmc,
        generated_quantities=None,
        transformed_data=None,
        thin=1,
    ):
        self.mcmc = mcmc
        self.transformed_data = transformed_data
        self.generated_quantities = generated_quantities
        self.thin = thin
        self.kwargs = {}

    def _sample_model(self):
        samples = self.mcmc.get_samples()
        return {x: samples[x][:: self.thin] for x in samples}

    def _sample_generated(self, samples):
        kwargs = self.kwargs
        res = defaultdict(list)
        num_samples = len(list(samples.values())[0])
        for i in range(num_samples):
            kwargs.update({x: samples[x][i] for x in samples})
            if self.generated_quantities:
                d = self.generated_quantities(kwargs)
                for k, v in d.items():
                    res[k].append(v)
        return {k: torch.stack(v) for k, v in res.items()}

    def run(self, **kwargs):
        self.kwargs = kwargs
        if self.transformed_data:
            self.kwargs.update(self.transformed_data(**self.kwargs))
        self.mcmc.run(**kwargs)
        self.samples = self._sample_model()
        if self.generated_quantities:
            gen = self._sample_generated(self.samples)
            self.samples.update(gen)

    def get_samples(self):
        return self.samples

    def summary(self):
        d_mean = _flatten_dict(
            {k: torch.mean(v, axis=0) for k, v in self.samples.items()}
        )
        d_std = _flatten_dict(
            {k: torch.std(v, axis=0) for k, v in self.samples.items()}
        )
        return DataFrame({"mean": Series(d_mean), "std": Series(d_std)})


class SVIProxy(object):
    def __init__(self, svi, generated_quantities=None, transformed_data=None):
        self.svi = svi
        self.transformed_data = transformed_data
        self.generated_quantities = generated_quantities
        self.args = []
        self.kwargs = {}

    def posterior(self, n):
        signature = inspect.signature(self.svi.guide)
        args = [None for i in range(len(signature.parameters))]
        return [self.svi.guide(*args) for _ in range(n)]

    def step(self, *args, **kwargs):
        self.kwargs = kwargs
        if self.transformed_data:
            self.kwargs.update(self.transformed_data(**self.kwargs))
        return self.svi.step(**kwargs)

import time
from typing import Any, Callable, ClassVar, Dict, Optional, List
from dataclasses import dataclass, field

import pystan
from runtimes.pyro.dppl import PyroModel
from runtimes.numpyro.dppl import NumpyroModel

from scipy.stats import entropy, ks_2samp
import numpy as np
from jax import numpy as jnp
from torch import Tensor

def _ks(s1, s2):
    s, p = ks_2samp(s1, s2)
    return {"statistic": s, "pvalue": p}


def _distance(pyro_samples, stan_samples, dist):
    if len(pyro_samples.shape) == 1:
        return dist(stan_samples, pyro_samples)
    if len(pyro_samples.shape) == 2:
        res = {}
        for i, (p, s) in enumerate(zip(pyro_samples.T, stan_samples.T)):
            res[i] = dist(p, s)
        return res
    # Don't know what to compute here. Too many dimensions.
    return {}


def _compare(res, ref, compare_params, dist):
    divergence = {}
    for k, a in res.items():
        if not compare_params or k in compare_params:
            assert k in ref, f"{k} is not in Stan results"
            b = ref[k]
            assert (
                a.shape == b.shape
            ), f"Shape mismatch for {k}, Pyro {a.shape}, Stan {b.shape}"
            divergence[k] = _distance(a, b, dist)
    return divergence

def _convert_to_tensor(value):
    if isinstance(value, (list, np.ndarray)):
        return Tensor(value)
    elif isinstance(value, dict):
        return {k: _convert_to_tensor(v) for k, v in value.items()}
    else:
        return value

def _convert_to_np(value):
    if type(value) == Tensor:
        return value.cpu().numpy()
    elif isinstance(value, dict):
        return {k: _convert_to_np(v) for k, v in value.items()}
    elif isinstance(value, list):
        return np.array([ _convert_to_np(v) for v in value])
    else:
        return value


@dataclass
class TimeIt:
    name: str
    timers: Dict[str, float]

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, *exc_info):
        self.timers[self.name] = time.perf_counter() - self.start


@dataclass
class Config:
    iterations: int = 100
    warmups: int = 10
    chains: int = 1
    thin: int = 2


@dataclass
class MCMCTest:
    name: str
    model_file: str
    pyro_file: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    compare_params: Optional[List[str]] = None
    config: Config = Config()
    with_pyro: bool = True
    with_numpyro: bool = True

    pyro_samples: Dict[str, Any] = field(init=False)
    pyro__naive_samples: Dict[str, Any] = field(init=False)
    numpyro_samples: Dict[str, Any] = field(init=False)
    numpyro_naive_samples: Dict[str, Any] = field(init=False)
    stan_samples: Dict[str, Any] = field(init=False)
    timers: Dict[str, float] = field(init=False, default_factory=dict)
    divergences: Dict[str, Any] = field(init=False, default_factory=dict)

    def run_pyro(self):
        assert self.with_pyro or self.with_numpyro, "Should run either Pyro or Numpyro"
        if self.with_pyro:
            with TimeIt("Pyro_Compilation", self.timers):
                model = PyroModel(self.model_file)
            with TimeIt("Pyro_Runtime", self.timers):
                mcmc = model.mcmc(
                    self.config.iterations,
                    warmups=self.config.warmups,
                    chains=self.config.chains,
                    thin=self.config.thin,
                )
                data = {k: _convert_to_tensor(v) for k, v in self.data.items()}
                mcmc.run(**data)
                self.pyro_samples = mcmc.get_samples()
        if self.with_numpyro:
            with TimeIt("Numpyro_Compilation", self.timers):
                model = NumpyroModel(self.model_file)
            with TimeIt("Numpyro_Runtime", self.timers):
                mcmc = model.mcmc(
                    self.config.iterations,
                    warmups=self.config.warmups,
                    chains=self.config.chains,
                    thin=self.config.thin,
                )
                mcmc.run(**self.data)
                self.numpyro_samples = mcmc.get_samples()

    def run_naive_pyro(self):
        assert self.with_pyro or self.with_numpyro, "Should run either Pyro or Numpyro"
        if self.with_pyro:
            model = PyroModel(self.pyro_file)
            with TimeIt("Pyro_naive_Runtime", self.timers):
                mcmc = model.mcmc(
                    self.config.iterations,
                    warmups=self.config.warmups,
                    chains=self.config.chains,
                    thin=self.config.thin,
                )
                data = {k: _convert_to_tensor(v) for k, v in self.data.items()}
                mcmc.run(**data)
                self.pyro_naive_samples = mcmc.get_samples()
        if self.with_numpyro:
            model = NumpyroModel(self.pyro_file)
            with TimeIt("Numpyro_naive_Runtime", self.timers):
                mcmc = model.mcmc(
                    self.config.iterations,
                    warmups=self.config.warmups,
                    chains=self.config.chains,
                    thin=self.config.thin,
                )
                mcmc.run(**self.data)
                self.numpyro_naive_samples = mcmc.get_samples()

    def run_stan(self):
        with TimeIt("Stan_Compilation", self.timers):
            mcmc = pystan.StanModel(file=self.model_file)
        with TimeIt("Stan_Runtime", self.timers):
            fit = mcmc.sampling(
                data=self.data,
                iter=self.config.iterations,
                chains=self.config.chains,
                warmup=self.config.warmups,
                thin=self.config.thin,
            )
            self.stan_samples = fit.extract(permuted=True)

    def compare(self):
        self.divergences = {
            "pyro": {},
            "numpyro": {},
            "pyro_naive": {},
            "numpyro_naive": {},
        }
        if self.with_pyro:
            self.divergences["pyro"]["ks"] = _compare(
                _convert_to_np(self.pyro_samples),
                self.stan_samples,
                self.compare_params,
                _ks
            )
            if self.pyro_file:
                self.divergences["pyro_naive"]["ks"] = _compare(
                    _convert_to_np(self.pyro_naive_samples),
                    self.stan_samples,
                    self.compare_params,
                    _ks
                )
        if self.with_numpyro:
            self.divergences["numpyro"]["ks"] = _compare(
                _convert_to_np(self.numpyro_samples),
                self.stan_samples,
                self.compare_params,
                _ks
            )
            if self.pyro_file:
                self.divergences["numpyro_naive"]["ks"] = _compare(
                    _convert_to_np(self.numpyro_naive_samples),
                    self.stan_samples,
                    self.compare_params,
                    _ks,
                )

    def run(self) -> Dict[str, Dict[str, Any]]:
        self.run_pyro()
        self.run_stan()
        if self.pyro_file:
            self.run_naive_pyro()
        self.compare()
        return {"divergences": self.divergences, "timers": self.timers}

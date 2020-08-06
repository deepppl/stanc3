from os.path import splitext, basename, dirname
import importlib.util
import pyro
from collections import defaultdict


class PyroModel:
    def __init__(self, filepath):
        self.name = basename(filepath)
        spec = importlib.util.spec_from_file_location(self.name, filepath)
        Module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(Module)
        self._model = Module.model
        self._transformed_data = None
        self._generated_quantities = None
        if hasattr(Module, "transformed_data"):
            self._transformed_data = Module.transformed_data
        if hasattr(Module, "generated_quantities"):
            self._generated_quantities = Module.generated_quantities

    def mcmc(self, samples, warmups=0, chains=1, thin=1, kernel=None):
        if kernel is None:
            kernel = pyro.infer.NUTS(self._model, adapt_step_size=True)
        mcmc = pyro.infer.MCMC(
            kernel, samples - warmups, warmup_steps=warmups, num_chains=chains,
        )
        return MCMCProxy(mcmc, self._generated_quantities, self._transformed_data, thin)


class MCMCProxy:
    def __init__(
        self, mcmc, generated_quantities=None, transformed_data=None, thin=1,
    ):
        self.mcmc = mcmc
        self.transformed_data = transformed_data
        self.generated_quantities = generated_quantities
        self.thin = thin
        self.kwargs = {}

    def run(self, **kwargs):
        self.kwargs = kwargs
        if self.transformed_data:
            self.kwargs.update(self.transformed_data(**self.kwargs))
        self.mcmc.run(**kwargs)

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
                d = self.generated_quantities(**kwargs)
                for k, v in d.items():
                    res[k].append(v)
        return res

    def get_samples(self):
        samples = self._sample_model()
        if self.generated_quantities:
            gen = self._sample_generated(samples)
            samples.update(gen)
        return samples

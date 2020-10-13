from dataclasses import dataclass, field
import os
from runtimes.pyro.dppl import PyroModel
from runtimes.numpyro.dppl import NumpyroModel
import numpy
from pandas import DataFrame, Series
from posteriordb import PosteriorDatabase
from os.path import splitext, basename

pdb_root = "/Users/gbdrt/Projects/deepstan/posteriordb-mandel"
pdb_path = os.path.join(pdb_root, "posterior_database")


@dataclass
class Config:
    iterations: int
    warmups: int
    chains: int
    thin: int


def parse_config(posterior):
    args = posterior.reference_draws_info()["inference"]["method_arguments"]
    return Config(
        iterations=args["iter"]//100,
        warmups=args["warmup"]//100,
        chains=args["chains"],
        thin=args["thin"],
    )


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


def gold_summary(posterior):
    """
    Summary for pdb reference_draws
    - Aggregate all chains and compute mean, std for all params
    - Flatten results in a DataFrame
    """
    samples = posterior.reference_draws()
    if isinstance(samples, list):
        # Multiple chains
        assert len(samples) > 0
        res = samples[0]
        for c in samples[1:]:
            res = {k: v + c[k] for k, v in res.items()}
    else:
        # Only one chain
        assert isinstance(samples, dict)
        res = samples
    d_mean = _flatten_dict({k: numpy.mean(v, axis=0) for k, v in res.items()})
    d_std = _flatten_dict({k: numpy.std(v, axis=0) for k, v in res.items()})
    return DataFrame({"mean": Series(d_mean), "std": Series(d_std)})


def parse_gold_summary(gold, gold_path):
    """
    Parse CmdStan gold summary format: param, mean, std
    where param.1.2 means param[1][2]
    """
    with open(os.path.join(gold_path, gold)) as f:
        d_mean = {}
        d_std = {}
        for line in f:
            param, avg, stdev = line.split()
            if "." in param:
                ps = param.split(".")
                param = ps[0] + "".join(f"[{i}]" for i in ps[1:])
            d_mean[param] = float(avg)
            d_std[param] = float(stdev)
        return DataFrame({"mean": Series(d_mean), "std": Series(d_std)})


def run_model(posterior, config: Config):
    model = posterior.model
    data = posterior.data
    stanfile = model.code_file_path("stan")
    pythonfile = os.path.join(os.getcwd(), splitext(basename(stanfile))[0] + ".py")
    pyro_model = PyroModel(stanfile, pyfile=pythonfile)
    mcmc = pyro_model.mcmc(
        config.iterations,
        warmups=config.warmups,
        chains=config.chains,
        thin=config.thin,
    )
    inputs = pyro_model.convert_inputs(data.values())
    mcmc.run(**inputs)
    return mcmc.summary()


def compare(posterior):
    config = parse_config(posterior)
    sg = gold_summary(posterior)
    sm = run_model(posterior, config=config)
    sm["err"] = abs(sm["mean"] - sg["mean"])
    sm = sm.dropna()
    # perf_cmdstan condition: err > 0.0001 and (err / stdev) > 0.3
    comp = sm[(sm["err"] > 0.0001) & (sm["err"] / sm["std"] > 0.3)].dropna()
    if not comp.empty:
        print(f"Failed {posterior.name}")
        print(comp)
    else:
        print(f"Success {posterior.name}")


# Stan gold models
golds = [
    "arK",
    "arma",
    "eight_schools",
    "garch",
    "gp_pois_regr",
    "gp_regr",
    "irt_2pl",
    "low_dim_corr_gauss",
    "low_dim_gauss_mix_collapse",
    "low_dim_gauss_mix",
    "sir",
]

if __name__ == "__main__":

    my_pdb = PosteriorDatabase(pdb_path)

    # # Launch only the eight_school model
    # posterior = my_pdb.posterior("eight_schools-eight_schools_centered")
    # compare(posterior)

    for name in my_pdb.posterior_names():
        if name.startswith(tuple(golds)):
            try:
                compare(my_pdb.posterior(name))
            except Exception as e:
                print(f"Failed {name} with {s}")
from dataclasses import dataclass, field
import os
from runtimes.pyro.dppl import PyroModel
from runtimes.numpyro.dppl import NumpyroModel
import numpy
from pandas import DataFrame, Series
from posteriordb import PosteriorDatabase

pdb_root = "/Users/gbdrt/Projects/deepstan/posteriordb-mandel"
gold_path = "test/integration/dppl/golds"
pdb_path = os.path.join(pdb_root, "posterior_database")

@dataclass
class Config:
    # iterations: int = 100
    # warmups: int = 10
    # chains: int = 1
    # thin: int = 2
    iterations: int = 100
    warmups: int = 1
    chains: int = 1
    thin: int = 1



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
        fk:fv 
        for f in (_flatten(k, v) for k, v in d.items())
        for fk, fv in f.items()
    }

def pdb_summary(samples):
    """
    Summary for pdb reference_draws
    - Aggregate all chains and compute mean, std for all params
    - Flatten results in a DataFrame
    """
    if isinstance(samples, list):
        # Multiple chains
        assert len(samples) > 0
        res = samples[0]
        for c in samples:
            res = {k:v + c[k] for k, v in res.items()}
        d_mean = _flatten_dict(
            {k: numpy.mean(v, axis=0) for k, v in res.items()}
        )
        d_std = _flatten_dict(
            {k: numpy.std(v, axis=0) for k, v in res.items()}
        )
        return DataFrame({"mean": Series(d_mean), "std": Series(d_std)})


def parse_gold_summary(gold):
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


def run_model(gold_model: str, data_file: str, config:Config=Config()):
    model = PyroModel(os.path.join(gold_path, gold_model))
    mcmc = model.mcmc(
        config.iterations,
        warmups=config.warmups,
        chains=config.chains,
        thin=config.thin,
    )
    try:
        with open(os.path.join(gold_path, data_file), "r") as f:
            data = eval(f.read())
    except FileNotFoundError:
        data = {}
    inputs = model.convert_inputs(data)
    mcmc.run(**inputs)
    return mcmc.summary()


def compare(gold_model, gold_data, gold, config=Config):
    sg = parse_gold_summary(gold)
    sm = run_model(gold_model, gold_data, config=config)
    sm["err"]  = abs(sm["mean"] - sg["mean"])
    # perf_cmdstan condition: err > 0.0001 and (err / stdev) > 0.3
    comp = sm[(sm["err"] > 0.0001) & (sm["err"] / sm["std"] > 0.3)]
    if not comp.empty:
        print(f"Failed {gold_model}")
        print(comp)
    else:
        print(f"Success {gold_model}")
       
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
    posterior = my_pdb.posterior("eight_schools-eight_schools_centered")
    g = pdb_summary(posterior.reference_draws())
    print(g)

#     for g in golds:
#         try:
#             compare(f"{g}.stan", f"{g}.data.py", f"{g}_{g}.gold")
#         except Exception as s:
#             print(f"Failed {g} with {s}")

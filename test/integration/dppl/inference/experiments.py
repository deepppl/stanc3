import os
import multiprocessing
import time
from datetime import datetime
from pprint import pprint
from pathlib import Path
import argparse
import pandas as pd
from pytablewriter import MarkdownTableWriter

from .test_aspirin import test_aspirin
from .test_cockroaches import test_cockroaches
from .test_coin import test_coin
from .test_double_gaussian import test_double_gaussian
from .test_gaussian_log_density import test_gaussian_log_density
from .test_linear_regression import test_linear_regression
from .test_neal_funnel import test_neal_funnel
from .test_schools import test_schools
from .test_seeds import test_seeds
from .harness import Config

tests = [
    test_aspirin,
    test_cockroaches,
    test_coin,
    test_double_gaussian,
    test_gaussian_log_density,
    test_linear_regression,
    test_neal_funnel,
    test_schools,
    test_seeds,
]

name_map = {
    "coin": "coin",
    "double_gaussian": "double normal",
    "gaussian_log_density": "gaussian target",
    "neal_funnel": "reparameterization",
    "linear_regression": "linear regression",
    "aspirin": "aspirin",
    "cockroaches": "roaches",
    "schools": "8 schools",
    "seeds": "seeds",
}


def run(exp):
    f, i, config, logdir = exp
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    name = f.__name__
    Path(logdir).mkdir(parents=True, exist_ok=True)
    filename = os.path.join(logdir, f"{name}_{i}_{now}.log")
    try:
        results = f(config)
        with open(filename, "w") as file:
            pprint(results, file)
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        print(f"ERROR test {name}")
        raise


def run_all(config=Config(), logdir="logs", n_runs=5):
    experiments = [(t, i, config, logdir) for t in tests for i in range(n_runs)]
    n_cpu = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(n_cpu)
    pool.map(run, experiments)
    pool.close()
    pool.join()


def flatten_scalar(d):
    res = {}
    for k, v in d.items():
        if set(v.keys()) == {"statistic", "pvalue"}:  # leaf values
            res[k] = v["pvalue"]
        else:
            for kk, vv in flatten_scalar(v).items():
                res[f"{k}[{kk}]"] = vv
    return res


def clean_log(f):
    with open(f, "r") as log_file:
        raw = log_file.read()
        log = eval(raw)
        res = {}
        res["stan"] = {
            "compilation": log["timers"]["Stan_Compilation"],
            "runtime": log["timers"]["Stan_Runtime"],
        }
        if "numpyro" in log["divergences"]:
            res["numpyro"] = flatten_scalar(log["divergences"]["numpyro"]["ks"])
            if "NumPyro_Compilation" in log["timers"]:
                res["numpyro"]["compilation"] = log["timers"]["NumPyro_Compilation"]
            res["numpyro"]["runtime"] = log["timers"]["NumPyro_Runtime"]
        if "pyro" in log["divergences"]:
            res["pyro"] = flatten_scalar(log["divergences"]["pyro"]["ks"])
            if "Pyro_Compilation" in log["timers"]:
                res["pyro"]["compilation"] = log["timers"]["Pyro_Compilation"]
            res["pyro"]["runtime"] = log["timers"]["Pyro_Runtime"]
        if (
            "numpyro_naive" in log["divergences"]
            and log["divergences"]["numpyro_naive"]
        ):
            res["numpyro_naive"] = flatten_scalar(
                log["divergences"]["numpyro_naive"]["ks"]
            )
            res["numpyro_naive"]["runtime"] = log["timers"]["NumPyro_naive_Runtime"]
        if "pyro_naive" in log["divergences"] and log["divergences"]["pyro_naive"]:
            res["pyro_naive"] = flatten_scalar(log["divergences"]["pyro_naive"]["ks"])
            res["pyro_naive"]["runtime"] = log["timers"]["Pyro_naive_Runtime"]
    return res


def to_frame(dirname):
    summary = {}
    for x in name_map:
        summary[x] = {}
        logs = [
            clean_log(os.path.join(dirname, f))
            for f in os.listdir(dirname)
            if f.startswith(f"test_{x}")
        ]
        data = {}
        for k in logs[0].keys():
            data[k] = pd.DataFrame([pd.Series(d[k]) for d in logs])
        summary[x] = pd.Series(data)
    return pd.Series(summary)


def backend_results(df):
    df = df.mean()
    params = df.drop(labels=["runtime", "compilation"], errors="ignore")
    res = df.reindex(["compilation", "runtime"])
    res["pvalue"] = params.min()
    res["param"] = params.idxmin()
    return res


def example_results(df):
    s = df.stan.mean()
    df = df.dropna().drop("stan")
    df = df.apply(backend_results).T
    df["stan"] = s
    return df


def summarize(dirname):
    data = to_frame(dirname)
    return data.apply(example_results)


def to_md(s):
    writer = MarkdownTableWriter()
    writer.headers = (
        [""]
        + ["Stan", ""]
        + ["DS(pyro)", "", "", ""]
        + ["DS(numpyro)", "", "", ""]
        + ["Pyro", "", ""]
        + ["Numpyro", "", ""]
    )
    writer.value_matrix = [
        [""]
        + ["comp", "run",]
        + ["comp", "run", "p-value", "param",] * 2
        + ["run", "p-value", "param",] * 2
    ]
    writer.value_matrix += [
        [name_map[x]]
        + [f"{s[x].stan.compilation:,.0f}", f"{s[x].stan.runtime:,.2f}",]
        + [
            c
            for b in ["pyro", "numpyro"]
            for c in [
                f"{s[x][b].compilation:,.2f}",
                f"{s[x][b].runtime:,.0f}",
                f"{s[x][b].pvalue:,.2f}",
                s[x][b].param,
            ]
        ]
        + [
            c
            for b in ["pyro_naive", "numpyro_naive"]
            for c in (
                [f"{s[x][b].runtime:,.0f}", f"{s[x][b].pvalue:,.2f}", s[x][b].param,]
                if b in s[x]
                else ["", "", ""]
            )
        ]
        for x in s.index
    ]
    writer.write_table()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare the output of NUTS for DeepStan (with Pyro and Numpyro) and Stan on the following experiments: coin, double normal, reparameterization, linear regression, aspirin, roaches, 8 schools, seeds"
    )

    parser.add_argument(
        "--logdir",
        action="store",
        dest="logdir",
        default="logs",
        help="Directory name to store the results",
    )

    parser.add_argument(
        "--iterations",
        action="store",
        dest="iterations",
        type=int,
        default=10000,
        help="Total number of iterations",
    )

    parser.add_argument(
        "--warmups",
        action="store",
        dest="warmups",
        type=int,
        default=1000,
        help="Number of warmup steps (included in iterations)",
    )

    parser.add_argument(
        "--thin",
        action="store",
        dest="thin",
        type=int,
        default=10,
        help="Thining factor",
    )

    parser.add_argument(
        "--runs",
        action="store",
        dest="n_runs",
        type=int,
        default=10,
        help="Number of run for each experiment",
    )

    parser.add_argument(
        "--no-run",
        action="store_true",
        dest="no_run",
        default=False,
        help="Analyse logdir without re-running the experiments",
    )

    args = parser.parse_args()
    config = Config(iterations=args.iterations, warmups=args.warmups, thin=args.thin)

    if not args.no_run:
        start = time.perf_counter()
        run_all(config, args.logdir, n_runs=args.n_runs)
        print(f"Total experiment time {time.perf_counter() - start}")
        print(f"Config: {config}")

    res = summarize(args.logdir)
    to_md(res)

import datetime
from dataclasses import dataclass, field
from posteriordb import PosteriorDatabase
import os, sys, argparse

from stanpyro.dppl import PyroModel
from stannumpyro.dppl import NumPyroModel
import jax.random

stanc = "stanc"
pdb_root = "/Users/lmandel/stan/posteriordb"
pdb_path = os.path.join(pdb_root, "posterior_database")
my_pdb = PosteriorDatabase(pdb_path)


@dataclass
class Config:
    iterations: int = 1
    warmups: int = 0
    chains: int = 1
    thin: int = 1


def test(posterior, config, backend="pyro", mode="mixed"):
    model = posterior.model
    data = posterior.data
    stanfile = model.code_file_path("stan")
    try:
        if backend == "pyro":
            pyro_model = PyroModel(
                stanfile, mode=mode, compiler=[stanc], recompile=True
            )
        else:
            pyro_model = NumPyroModel(
                stanfile, mode=mode, compiler=[stanc], recompile=True
            )
    except Exception as e:
        return {
            "code": 1,
            "msg": f"compilation error ({posterior.name}): {model.name}",
            "exn": e,
        }
    try:
        mcmc = pyro_model.mcmc(
            config.iterations,
            warmups=config.warmups,
            chains=config.chains,
            thin=config.thin,
        )
        if backend == "pyro":
            mcmc.run(data.values())
        else:
            mcmc.run(jax.random.PRNGKey(0), data.values())
    except Exception as e:
        return {
            "code": 2,
            "msg": f"Inference error ({posterior.name}): {model.name}({data.name})",
            "exn": e,
        }
    return {"code": 0, "samples": mcmc.get_samples()}


def log(logfile, res):
    test = res["test"]
    code = res["code"]
    exn = res["exn"] if res["code"] != 0 else ""
    # print(f"{test}, {code}, {exn}", file=logfile, flush=True)
    if code != 0:
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        print(res['msg'])
        print(res['exn'])
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXX')


def test_all(backend, mode, config):
    today = datetime.datetime.now()
    # logpath = f"{today.strftime('%y%m%d_%H%M')}_{args.backend}_{args.mode}.csv"
    success = 0
    compile_error = 0
    inference_error = 0
    logfile=sys.stdout
    # with open(logpath, "a") as logfile:
    print("test,exit code,exn", file=logfile, flush=True)
    for name in my_pdb.posterior_names():
        print(f"- Test {backend} {mode}: {name}")
        posterior = my_pdb.posterior(name)
        res = test(posterior, config, backend, mode)
        res["test"] = name
        log(logfile, res)
        if res["code"] == 0:
            success = success + 1
        elif res["code"] == 1:
            compile_error = compile_error + 1
        elif res["code"] == 2:
            inference_error = inference_error + 1
        print(
            f"success: {success}, compile errors: {compile_error}, inference errors: {inference_error}, total: {success + compile_error + inference_error}"
        )
    return {
        "success": success,
        "compile_error": compile_error,
        "inference_error": inference_error,
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run experiments on PosteriorDB models."
    )
    parser.add_argument(
        "--backend",
        help="inference backend (pyro, numpyro)",
        required=True,
    )
    parser.add_argument(
        "--mode",
        help="compilation mode (generative, comprehensive, mixed)",
        default="mixed",
    )
    parser.add_argument("--iterations", type=int, help="number of iterations")
    parser.add_argument("--warmups", type=int, help="warmups steps")
    parser.add_argument("--chains", type=int, help="number of chains")
    parser.add_argument("--thin", type=int, help="thinning factor")

    args = parser.parse_args()

    config = Config()
    if args.iterations is not None:
        config.iterations = args.iterations
    if args.warmups is not None:
        config.warmups = args.warmups
    if args.chains is not None:
        config.chains = args.chains
    if args.thin is not None:
        config.thin = args.thin

    # posterior = my_pdb.posterior('mesquite-logmesquite_logvas')
    # posterior = my_pdb.posterior('radon_mn-radon_variable_intercept_centered')
    # posterior = my_pdb.posterior('irt_2pl-irt_2pl')
    # posterior = my_pdb.posterior('mcycle_gp-accel_gp')
    # posterior = my_pdb.posterior('election88-election88_full')
    # posterior = my_pdb.posterior('dogs-dogs_log')
    # posterior = my_pdb.posterior('sblri-blr')
    # posterior = my_pdb.posterior('rstan_downloads-prophet')
    # posterior = my_pdb.posterior('ecdc0401-covid19imperial_v2')
    # posterior = my_pdb.posterior('ecdc0501-covid19imperial_v2')
    # posterior = my_pdb.posterior('dogs-dogs')
    # posterior = my_pdb.posterior('garch-garch11')
    # posterior = my_pdb.posterior('prostate-logistic_regression_rhs')
    # posterior = my_pdb.posterior('low_dim_gauss_mix_collapse')
    # posterior = my_pdb.posterior('sat-hier_2pl')
    # posterior = my_pdb.posterior('nes1984-nes')
    # posterior = my_pdb.posterior('diamonds-diamonds')
    # posterior = my_pdb.posterior('butterfly-multi_occupancy')
    # posterior = my_pdb.posterior('hmm_example-hmm_example')
    # posterior = my_pdb.posterior('bball_drive_event_1-hmm_drive_1')
    # posterior = my_pdb.posterior('sat-hier_2pl')
    # posterior = my_pdb.posterior('normal_5-normal_mixture_k')
    # posterior = my_pdb.posterior('mcycle_splines-accel_splines')
    # posterior = my_pdb.posterior('hudson_lynx_hare-lotka_volterra')
    # posterior = my_pdb.posterior('sir-sir')
    # posterior = my_pdb.posterior('mnist_100-nn_rbm1bJ10')
    # posterior = my_pdb.posterior('arma-arma11')
    # posterior = my_pdb.posterior('arK-arK')
    # posterior = my_pdb.posterior('bball_drive_event_0-hmm_drive_0')
    # posterior = my_pdb.posterior('radon_all-radon_county_intercept')
    # posterior = my_pdb.posterior('eight_schools-eight_schools_noncentered')
    # posterior = my_pdb.posterior('rstan_downloads-prophet')
    # posterior = my_pdb.posterior('radon_mn-radon_hierarchical_intercept_noncentered')
    # posterior = my_pdb.posterior('one_comp_mm_elim_abs-one_comp_mm_elim_abs')
    # posterior = my_pdb.posterior('gp_pois_regr-gp_regr')

    # res = test(posterior, config, args.backend, args.mode)
    # print(res['code'])
    # print(res['msg'])
    # print(res['exn'])
    # exit(0)

    res = test_all(args.backend, args.mode, config)

    print("Summary")
    print("-------")
    print(f"{args.backend} {args.mode}: {res}")

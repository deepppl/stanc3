open Core_kernel
open Frontend
open Ast
open Middle

type clone_type = Tclone | Tnoclone

let pyro_dppllib =
  [ "sample"; "param"; "observe"; "factor"; "array"; "zeros"; "ones"; "empty";
    "matmul"; "true_divide"; "floor_divide"; "transpose";
    "dtype_long"; "dtype_float";
    "vmap"; ]
let numpyro_dppllib =
  [ "ops_index"; "ops_index_update";
    "lax_cond";
    "lax_while_loop";
    "lax_fori_loop";
    "fori_loop";
    "foreach_loop";
    "lax_foreach_loop";
    "jit"; ] @ pyro_dppllib
let dppllib_networks = [ "register_network"; "random_module"; ]

let no_constraint _ =
  Program.Identity

let lower_constraint args =
  match List.map ~f:untyped_expression_of_typed_expression args with
  | lb :: _ -> Program.Lower lb
  | _ -> assert false

let upper_constraint args =
  match List.map ~f:untyped_expression_of_typed_expression args with
  | ub :: _ -> Program.Upper ub
  | _ -> assert false

let lower_upper_constraint args =
  match List.map ~f:untyped_expression_of_typed_expression args with
  | lb :: ub :: _ -> Program.LowerUpper (lb, ub)
  | _ -> assert false

let simplex_constraint _ =
  Program.Simplex

let unit_constraint _ =
  Program.UnitVector

let ordered_constraint _ =
  Program.Ordered

let positive_ordered_constraint _ =
  Program.PositiveOrdered

let cholesky_corr_constraint _ =
  Program.CholeskyCorr

let cholesky_cov_constraint _ =
  Program.CholeskyCov

let covariance_constraint _ =
  Program.Covariance

let correlation_constraint _ =
  Program.Correlation

let offset_constraint args =
  match List.map ~f:untyped_expression_of_typed_expression args with
  | [ e ] -> Program.Offset e
  | _ -> assert false

let multiplier_constraint args =
  match List.map ~f:untyped_expression_of_typed_expression args with
  | [ e ] -> Program.Multiplier e
  | _ -> assert false

let offset_multiplier_constraint args =
  match List.map ~f:untyped_expression_of_typed_expression args with
  | [ e1; e2 ] -> Program.OffsetMultiplier (e1, e2)
  | _ -> assert false

let inteval_constraint lb ub _ =
  let lb =
    mk_untyped_expression ~expr:(RealNumeral lb) ~loc:Location_span.empty
  in
  let ub =
    mk_untyped_expression ~expr:(RealNumeral ub) ~loc:Location_span.empty
  in
  Program.LowerUpper (lb, ub)

let grather_than_constraint lb _ =
  let lb =
    mk_untyped_expression ~expr:(RealNumeral lb) ~loc:Location_span.empty
  in
  Program.Lower lb

let distributions =
  [ "improper_uniform", (Tnoclone, no_constraint);
    "lower_constrained_improper_uniform", (Tnoclone, lower_constraint);
    "upper_constrained_improper_uniform", (Tnoclone, upper_constraint);
    "simplex_constrained_improper_uniform", (Tnoclone, simplex_constraint);
    "unit_constrained_improper_uniform", (Tnoclone, unit_constraint);
    "ordered_constrained_improper_uniform", (Tnoclone, ordered_constraint);
    "positive_ordered_constrained_improper_uniform", (Tnoclone,
    positive_ordered_constraint);
    "cholesky_factor_corr_constrained_improper_uniform", (Tnoclone,
    cholesky_corr_constraint);
    "cholesky_factor_cov_constrained_improper_uniform", (Tnoclone,
    cholesky_cov_constraint);
    "cov_constrained_improper_uniform", (Tnoclone, covariance_constraint);
    "corr_constrained_improper_uniform", (Tnoclone, correlation_constraint);
    "offset_constrained_improper_uniform", (Tnoclone, offset_constraint);
    "multiplier_constrained_improper_uniform", (Tnoclone, multiplier_constraint);
    "offset_multiplier_constrained_improper_uniform", (Tnoclone,
    offset_multiplier_constraint);
    (* 12 Binary Distributions *)
    (* 12.1 Bernoulli Distribution *)
    "bernoulli", (Tnoclone, inteval_constraint "0" "1");
    "bernoulli_lpmf", (Tnoclone, no_constraint);
    "bernoulli_lupmf", (Tnoclone, no_constraint);
    "bernoulli_cdf", (Tnoclone, no_constraint);
    "bernoulli_lcdf", (Tnoclone, no_constraint);
    "bernoulli_lccdf", (Tnoclone, no_constraint);
    "bernoulli_rng", (Tnoclone, no_constraint);
    (* 12.2 Bernoulli Distribution, Logit Parameterization *)
    "bernoulli_logit", (Tnoclone, inteval_constraint "0" "1");
    "bernoulli_logit_lpmf", (Tnoclone, no_constraint);
    "bernoulli_logit_lupmf", (Tnoclone, no_constraint);
    (* 12.3 Bernoulli-logit generalized linear model (Logistic Regression) *)
    "bernoulli_logit_glm", (Tnoclone, inteval_constraint "0" "1");
    "bernoulli_logit_glm_lpmf", (Tnoclone, no_constraint);
    "bernoulli_logit_glm_lupmf", (Tnoclone, no_constraint);
    (* 13 Bounded Discrete Distributions *)
    (* 13.1 Binomial distribution *)
    "binomial", (Tnoclone, no_constraint);
    "binomial_lpmf", (Tnoclone, no_constraint);
    "binomial_lupmf", (Tnoclone, no_constraint);
    "binomial_cdf", (Tnoclone, no_constraint);
    "binomial_lcdf", (Tnoclone, no_constraint);
    "binomial_lccdf", (Tnoclone, no_constraint);
    "binomial_rng", (Tnoclone, no_constraint);
    (* 13.2 Binomial Distribution, Logit Parameterization *)
    "binomial_logit", (Tnoclone, no_constraint);
    "binomial_logit_lpmf", (Tnoclone, no_constraint);
    "binomial_logit_lupmf", (Tnoclone, no_constraint);
    (* 13.3 Beta-binomial distribution *)
    "beta_binomial", (Tnoclone, no_constraint);
    "beta_binomial_lpmf", (Tnoclone, no_constraint);
    "beta_binomial_lupmf", (Tnoclone, no_constraint);
    "beta_binomial_cdf", (Tnoclone, no_constraint);
    "beta_binomial_lcdf", (Tnoclone, no_constraint);
    "beta_binomial_lccdf", (Tnoclone, no_constraint);
    "beta_binomial_rng", (Tnoclone, no_constraint);
    (* 13.4 Hypergeometric distribution *)
    "hypergeometric", (Tnoclone, no_constraint);
    "hypergeometric_lpmf", (Tnoclone, no_constraint);
    "hypergeometric_lupmf", (Tnoclone, no_constraint);
    "hypergeometric_rng", (Tnoclone, no_constraint);
    (* 13.5 Categorical Distribution *)
    "categorical", (Tnoclone, no_constraint);
    "categorical_lpmf", (Tnoclone, no_constraint);
    "categorical_lupmf", (Tnoclone, no_constraint);
    "categorical_rng", (Tnoclone, no_constraint);
    "categorical_logit", (Tnoclone, no_constraint);
    "categorical_logit_lpmf", (Tnoclone, no_constraint);
    "categorical_logit_lupmf", (Tnoclone, no_constraint);
    "categorical_logit_rng", (Tnoclone, no_constraint);
    (* 13.6 Categorical logit generalized linear model (softmax regression) *)
    "categorical_logit_glm", (Tnoclone, no_constraint);
    "categorical_logit_glm_lpmf", (Tnoclone, no_constraint);
    "categorical_logit_glm_lupmf", (Tnoclone, no_constraint);
    (* 13.7 Discrete range distribution *)
    "discrete_range", (Tnoclone, lower_upper_constraint);
    "discrete_range_lpmf", (Tnoclone, no_constraint);
    "discrete_range_lupmf", (Tnoclone, no_constraint);
    "discrete_range_cdf", (Tnoclone, no_constraint);
    "discrete_range_lcdf", (Tnoclone, no_constraint);
    "discrete_range_lccdf", (Tnoclone, no_constraint);
    "discrete_range_rng", (Tnoclone, no_constraint);
    (* 13.8 Ordered logistic distribution *)
    "ordered_logistic", (Tnoclone, no_constraint);
    "ordered_logistic_lpmf", (Tnoclone, no_constraint);
    "ordered_logistic_lupmf", (Tnoclone, no_constraint);
    "ordered_logistic_rng", (Tnoclone, no_constraint);
    (* 13.9 Ordered logistic generalized linear model (ordinal regression) *)
    "ordered_logistic_glm", (Tnoclone, no_constraint);
    "ordered_logistic_glm_lpmf", (Tnoclone, no_constraint);
    "ordered_logistic_glm_lupmf", (Tnoclone, no_constraint);
    (* 13.10 Ordered probit distribution *)
    "ordered_probit", (Tnoclone, no_constraint);
    "ordered_probit_lpmf", (Tnoclone, no_constraint);
    "ordered_probit_lupmf", (Tnoclone, no_constraint);
    "ordered_probit_rng", (Tnoclone, no_constraint);
    (* 14 Unbounded Discrete Distributions *)
    (* 14.1 Negative binomial distribution *)
    "neg_binomial", (Tnoclone, no_constraint);
    "neg_binomial_lpmf", (Tnoclone, no_constraint);
    "neg_binomial_lupmf", (Tnoclone, no_constraint);
    "neg_binomial_cdf", (Tnoclone, no_constraint);
    "neg_binomial_lcdf", (Tnoclone, no_constraint);
    "neg_binomial_lccdf", (Tnoclone, no_constraint);
    "neg_binomial_rng", (Tnoclone, no_constraint);
    (* 14.2 Negative Binomial Distribution (alternative parameterization) *)
    "neg_binomial_2", (Tnoclone, no_constraint);
    "neg_binomial_2_lpmf", (Tnoclone, no_constraint);
    "neg_binomial_2_lupmf", (Tnoclone, no_constraint);
    "neg_binomial_2_cdf", (Tnoclone, no_constraint);
    "neg_binomial_2_lcdf", (Tnoclone, no_constraint);
    "neg_binomial_2_lccdf", (Tnoclone, no_constraint);
    "neg_binomial_2_rng", (Tnoclone, no_constraint);
    (* 14.3 Negative binomial distribution (log alternative parameterization) *)
    "neg_binomial_2_log", (Tnoclone, no_constraint);
    "neg_binomial_2_log_lpmf", (Tnoclone, no_constraint);
    "neg_binomial_2_log_lupmf", (Tnoclone, no_constraint);
    "neg_binomial_2_log_rng", (Tnoclone, no_constraint);
    (* 14.4 Negative-binomial-2-log generalized linear model (negative binomial regression) *)
    "neg_binomial_2_log_glm", (Tnoclone, no_constraint);
    "neg_binomial_2_log_glm_lpmf", (Tnoclone, no_constraint);
    "neg_binomial_2_log_glm_lupmf", (Tnoclone, no_constraint);
    (* 14.5 Poisson Distribution *)
    "poisson", (Tnoclone, grather_than_constraint "0");
    "poisson_lpmf", (Tnoclone, no_constraint);
    "poisson_lupmf", (Tnoclone, no_constraint);
    "poisson_cdf", (Tnoclone, no_constraint);
    "poisson_lcdf", (Tnoclone, no_constraint);
    "poisson_lccdf", (Tnoclone, no_constraint);
    "poisson_rng", (Tnoclone, no_constraint);
    (* 14.6 Poisson Distribution, Log Parameterization *)
    "poisson_log", (Tnoclone, grather_than_constraint "0");
    "poisson_log_lpmf", (Tnoclone, no_constraint);
    "poisson_log_lupmf", (Tnoclone, no_constraint);
    "poisson_log_rng", (Tnoclone, no_constraint);
    (* 14.7 Poisson-log generalized linear model (Poisson regression) *)
    "poisson_log_glm", (Tnoclone, grather_than_constraint "0");
    "poisson_log_glm_lpmf", (Tnoclone, no_constraint);
    "poisson_log_glm_lpmf", (Tnoclone, no_constraint);
    (* 15 Multivariate Discrete Distributions *)
    (* 15.1 Multinomial distribution *)
    "multinomial", (Tnoclone, no_constraint);
    "multinomial_lpmf", (Tnoclone, no_constraint);
    "multinomial_lupmf", (Tnoclone, no_constraint);
    "multinomial_rng", (Tnoclone, no_constraint);
    (* 15.2 Multinomial distribution, logit parameterization *)
    "multinomial_logit", (Tnoclone, no_constraint);
    "multinomial_logit_lpmf", (Tnoclone, no_constraint);
    "multinomial_logit_lupmf", (Tnoclone, no_constraint);
    "multinomial_logit_rng", (Tnoclone, no_constraint);
    (* 16 Unbounded Continuous Distributions *)
    (* 16.1 Normal Distribution *)
    "normal", (Tnoclone, no_constraint);
    "normal_lpdf", (Tnoclone, no_constraint);
    "normal_lupdf", (Tnoclone, no_constraint);
    "normal_cdf", (Tnoclone, no_constraint);
    "normal_lcdf", (Tnoclone, no_constraint);
    "normal_lccdf", (Tnoclone, no_constraint);
    "normal_rng", (Tnoclone, no_constraint);
    "std_normal", (Tnoclone, no_constraint);
    "std_normal_lpdf", (Tnoclone, no_constraint);
    "std_normal_lupdf", (Tnoclone, no_constraint);
    "std_normal_cdf", (Tnoclone, no_constraint);
    "std_normal_lcdf", (Tnoclone, no_constraint);
    "std_normal_lccdf", (Tnoclone, no_constraint);
    "std_normal_rng", (Tnoclone, no_constraint);
    (* 16.2 Normal-id generalized linear model (linear regression) *)
    "normal_id_glm", (Tnoclone, no_constraint);
    "normal_id_glm_lpdf", (Tnoclone, no_constraint);
    "normal_id_glm_lupdf", (Tnoclone, no_constraint);
    (* 16.5 Student-T Distribution *)
    "student_t", (Tnoclone, no_constraint);
    "student_t_lpdf", (Tnoclone, no_constraint);
    "student_t_cdf", (Tnoclone, no_constraint);
    "student_t_lcdf", (Tnoclone, no_constraint);
    "student_t_lccdf", (Tnoclone, no_constraint);
    "student_t_rng", (Tnoclone, no_constraint);
    (* 16.6 Cauchy Distribution *)
    "cauchy", (Tnoclone, no_constraint);
    "cauchy_lpdf", (Tnoclone, no_constraint);
    "cauchy_cdf", (Tnoclone, no_constraint);
    "cauchy_lcdf", (Tnoclone, no_constraint);
    "cauchy_lccdf", (Tnoclone, no_constraint);
    "cauchy_rng", (Tnoclone, no_constraint);
    (* 16.7 Double Exponential (Laplace) Distribution *)
    "double_exponential", (Tnoclone, no_constraint);
    "double_exponential_lpdf", (Tnoclone, no_constraint);
    "double_exponential_cdf", (Tnoclone, no_constraint);
    "double_exponential_lcdf", (Tnoclone, no_constraint);
    "double_exponential_lccdf", (Tnoclone, no_constraint);
    "double_exponential_rng", (Tnoclone, no_constraint);
    (* 16.8 Logistic Distribution *)
    "logistic", (Tnoclone, no_constraint);
    "logistic_lpdf", (Tnoclone, no_constraint);
    "logistic_cdf", (Tnoclone, no_constraint);
    "logistic_lcdf", (Tnoclone, no_constraint);
    "logistic_lccdf", (Tnoclone, no_constraint);
    "logistic_rng", (Tnoclone, no_constraint);
    (* 17 Positive Continuous Distributions *)
    (* 17.1 Lognormal Distribution *)
    "lognormal", (Tnoclone, grather_than_constraint "0.0");
    "lognormal_lpdf", (Tnoclone, no_constraint);
    "lognormal_cdf", (Tnoclone, no_constraint);
    "lognormal_lcdf", (Tnoclone, no_constraint);
    "lognormal_lccdf", (Tnoclone, no_constraint);
    "lognormal_rng", (Tnoclone, no_constraint);
    (* 17.5 Exponential Distribution *)
    "exponential", (Tnoclone, grather_than_constraint "0.0");
    "exponential_lpdf", (Tnoclone, no_constraint);
    "exponential_cdf", (Tnoclone, no_constraint);
    "exponential_lcdf", (Tnoclone, no_constraint);
    "exponential_lccdf", (Tnoclone, no_constraint);
    "exponential_rng", (Tnoclone, no_constraint);
    (* 17.6 Gamma Distribution *)
    "gamma", (Tnoclone, grather_than_constraint "0.0");
    "gamma_lpdf", (Tnoclone, no_constraint);
    "gamma_cdf", (Tnoclone, no_constraint);
    "gamma_lcdf", (Tnoclone, no_constraint);
    "gamma_lccdf", (Tnoclone, no_constraint);
    "gamma_rng", (Tnoclone, no_constraint);
    (* 17.7 Inverse Gamma Distribution *)
    "inv_gamma", (Tnoclone, no_constraint);
    "inv_gamma_lpdf", (Tnoclone, no_constraint);
    "inv_gamma_cdf", (Tnoclone, no_constraint);
    "inv_gamma_lcdf", (Tnoclone, no_constraint);
    "inv_gamma_lccdf", (Tnoclone, no_constraint);
    "inv_gamma_rng", (Tnoclone, no_constraint);
    (* 18 Positive Lower-Bounded Distributions *)
    (* 18.1 Pareto Distribution *)
    "pareto", (Tnoclone, lower_constraint);
    "pareto_lpdf", (Tnoclone, no_constraint);
    "pareto_cdf", (Tnoclone, no_constraint);
    "pareto_lcdf", (Tnoclone, no_constraint);
    "pareto_lccdf", (Tnoclone, no_constraint);
    "pareto_rng", (Tnoclone, no_constraint);
    (* 19 Continuous Distributions on [0, 1] *)
    (* 19.1 Beta Distribution *)
    "beta", (Tnoclone, inteval_constraint "0.0" "1.0");
    "beta_lpdf", (Tnoclone, no_constraint);
    "beta_cdf", (Tnoclone, no_constraint);
    "beta_lcdf", (Tnoclone, no_constraint);
    "beta_lccdf", (Tnoclone, no_constraint);
    "beta_rng", (Tnoclone, no_constraint);
    (* 21 Bounded Continuous Probabilities *)
    (* 21.1 Uniform Distribution *)
    "uniform", (Tnoclone, lower_upper_constraint);
    "uniform_lpdf", (Tnoclone, no_constraint);
    "uniform_cdf", (Tnoclone, no_constraint);
    "uniform_lcdf", (Tnoclone, no_constraint);
    "uniform_lccdf", (Tnoclone, no_constraint);
    "uniform_rng", (Tnoclone, no_constraint);
    (* 22 Distributions over Unbounded Vectors *)
    (* 22.1 Multivariate Normal Distribution *)
    "multi_normal", (Tnoclone, no_constraint);
    "multi_normal_lpdf", (Tnoclone, no_constraint);
    "multi_normal_rng", (Tnoclone, no_constraint);
    (* 23 Simplex Distributions *)
    (* 23.1 Dirichlet Distribution *)
    "dirichlet", (Tnoclone, simplex_constraint);
    "dirichlet_lpdf", (Tnoclone, no_constraint);
    "dirichlet_rng", (Tnoclone, no_constraint);
  ]

let stanlib =
  [ (* 1 Void Functions *)
    (* 2 Integer-Valued Basic Functions *)
    (* 2.1 Integer-valued arithmetic operators *)
    (* 2.2 Absolute functions *)
    "abs_int", Tnoclone;
    "int_step_int", Tnoclone;
    "int_step_real", Tnoclone;
    (* 2.3 Bound functions *)
    "min_int_int", Tnoclone;
    "max_int_int", Tnoclone;
    (* 2.4 Size functions *)
    "size_int", Tnoclone;
    "size_real", Tnoclone;
    (* 3 Real-Valued Basic Functions *)
    (* 3.1 Vectorization of real-valued functions *)
    (* 3.2 Mathematical Constants *)
    "pi", Tnoclone;
    "e", Tnoclone;
    "sqrt2", Tnoclone;
    "log2", Tnoclone;
    "log10", Tnoclone;
    (* 3.3 Special Values *)
    "not_a_number", Tnoclone;
    "positive_infinity", Tnoclone;
    "negative_infinity", Tnoclone;
    "machine_precision", Tnoclone;
    (* 3.4 Log probability function *)
    "target", Tnoclone;
    "get_lp", Tnoclone;
    (* 3.5 Logical functions *)
    (* 3.5.1 Comparison operators *)
    (* 3.5.2 Boolean operators *)
    (* 3.5.3 Logical functions *)
    "step_real", Tnoclone;
    "is_inf", Tnoclone;
    "is_nan", Tnoclone;
    (* 3.6 Real-valued arithmetic operators *)
    (* 3.6.1 Binary infix operators *)
    (* 3.6.2 Unary prefix operators *)
    (* 3.7 Step-like Functions *)
    (* 3.7.1 Absolute Value Functions *)
    "mabs", Tnoclone;
    "abs_int", Tnoclone;
    "abs_real", Tnoclone;
    "abs_vector", Tnoclone;
    "abs_rowvector", Tnoclone;
    "abs_matrix", Tnoclone;
    "abs_array", Tnoclone;
    "fdim_real_real", Tnoclone;
    "fdim_int_real", Tnoclone;
    "fdim_real_int", Tnoclone;
    "fdim_int_int", Tnoclone;
    "fdim_vectorized", Tnoclone;
    "fdim_vector_vector", Tnoclone;
    "fdim_rowvector_rowvector", Tnoclone;
    "fdim_matrix_matrix", Tnoclone;
    "fdim_array_array", Tnoclone;
    "fdim_real_vector", Tnoclone;
    "fdim_real_rowvector", Tnoclone;
    "fdim_real_matrix", Tnoclone;
    "fdim_real_array", Tnoclone;
    "fdim_vector_real", Tnoclone;
    "fdim_rowvector_real", Tnoclone;
    "fdim_matrix_real", Tnoclone;
    "fdim_array_real", Tnoclone;
    "fdim_int_vector", Tnoclone;
    "fdim_int_rowvector", Tnoclone;
    "fdim_int_matrix", Tnoclone;
    "fdim_int_array", Tnoclone;
    "fdim_vector_int", Tnoclone;
    "fdim_rowvector_int", Tnoclone;
    "fdim_matrix_int", Tnoclone;
    "fdim_array_int", Tnoclone;
    (* 3.7.2 Bounds Functions *)
    "fmin_real_real", Tnoclone;
    "fmin_int_real", Tnoclone;
    "fmin_real_int", Tnoclone;
    "fmin_int_int", Tnoclone;
    "fmin_vectorized", Tnoclone;
    "fmin_vector_vector", Tnoclone;
    "fmin_rowvector_rowvector", Tnoclone;
    "fmin_matrix_matrix", Tnoclone;
    "fmin_array_array", Tnoclone;
    "fmin_real_vector", Tnoclone;
    "fmin_real_rowvector", Tnoclone;
    "fmin_real_matrix", Tnoclone;
    "fmin_real_array", Tnoclone;
    "fmin_vector_real", Tnoclone;
    "fmin_rowvector_real", Tnoclone;
    "fmin_matrix_real", Tnoclone;
    "fmin_array_real", Tnoclone;
    "fmin_int_vector", Tnoclone;
    "fmin_int_rowvector", Tnoclone;
    "fmin_int_matrix", Tnoclone;
    "fmin_int_array", Tnoclone;
    "fmin_vector_int", Tnoclone;
    "fmin_rowvector_int", Tnoclone;
    "fmin_matrix_int", Tnoclone;
    "fmin_array_int", Tnoclone;
    "fmax_real_real", Tnoclone;
    "fmax_int_real", Tnoclone;
    "fmax_real_int", Tnoclone;
    "fmax_int_int", Tnoclone;
    "fmax_vectorized", Tnoclone;
    "fmax_vector_vector", Tnoclone;
    "fmax_rowvector_rowvector", Tnoclone;
    "fmax_matrix_matrix", Tnoclone;
    "fmax_array_array", Tnoclone;
    "fmax_real_vector", Tnoclone;
    "fmax_real_rowvector", Tnoclone;
    "fmax_real_matrix", Tnoclone;
    "fmax_real_array", Tnoclone;
    "fmax_vector_real", Tnoclone;
    "fmax_rowvector_real", Tnoclone;
    "fmax_matrix_real", Tnoclone;
    "fmax_array_real", Tnoclone;
    "fmax_int_vector", Tnoclone;
    "fmax_int_rowvector", Tnoclone;
    "fmax_int_matrix", Tnoclone;
    "fmax_int_array", Tnoclone;
    "fmax_vector_int", Tnoclone;
    "fmax_rowvector_int", Tnoclone;
    "fmax_matrix_int", Tnoclone;
    "fmax_array_int", Tnoclone;
    (* 3.7.3 Arithmetic Functions *)
    "fmod_real_real", Tnoclone;
    "fmod_int_real", Tnoclone;
    "fmod_real_int", Tnoclone;
    "fmod_int_int", Tnoclone;
    "fmod_vectorized", Tnoclone;
    "fmod_vector_vector", Tnoclone;
    "fmod_rowvector_rowvector", Tnoclone;
    "fmod_matrix_matrix", Tnoclone;
    "fmod_array_array", Tnoclone;
    "fmod_real_vector", Tnoclone;
    "fmod_real_rowvector", Tnoclone;
    "fmod_real_matrix", Tnoclone;
    "fmod_real_array", Tnoclone;
    "fmod_vector_real", Tnoclone;
    "fmod_rowvector_real", Tnoclone;
    "fmod_matrix_real", Tnoclone;
    "fmod_array_real", Tnoclone;
    "fmod_int_vector", Tnoclone;
    "fmod_int_rowvector", Tnoclone;
    "fmod_int_matrix", Tnoclone;
    "fmod_int_array", Tnoclone;
    "fmod_vector_int", Tnoclone;
    "fmod_rowvector_int", Tnoclone;
    "fmod_matrix_int", Tnoclone;
    "fmod_array_int", Tnoclone;
    (* 3.7.4 Rounding Functions *)
    "floor_int", Tnoclone;
    "floor_real", Tnoclone;
    "floor_vector", Tnoclone;
    "floor_rowvector", Tnoclone;
    "floor_matrix", Tnoclone;
    "floor_array", Tnoclone;
    "ceil_int", Tnoclone;
    "ceil_real", Tnoclone;
    "ceil_vector", Tnoclone;
    "ceil_rowvector", Tnoclone;
    "ceil_matrix", Tnoclone;
    "ceil_array", Tnoclone;
    "round_int", Tnoclone;
    "round_real", Tnoclone;
    "round_vector", Tnoclone;
    "round_rowvector", Tnoclone;
    "round_matrix", Tnoclone;
    "round_array", Tnoclone;
    "trunc_int", Tnoclone;
    "trunc_real", Tnoclone;
    "trunc_vector", Tnoclone;
    "trunc_rowvector", Tnoclone;
    "trunc_matrix", Tnoclone;
    "trunc_array", Tnoclone;
    (* 3.8 Power and Logarithm Functions *)
    "sqrt_int", Tnoclone;
    "sqrt_real", Tnoclone;
    "sqrt_vector", Tnoclone;
    "sqrt_rowvector", Tnoclone;
    "sqrt_matrix", Tnoclone;
    "sqrt_array", Tnoclone;
    "cbrt_int", Tnoclone;
    "cbrt_real", Tnoclone;
    "cbrt_vector", Tnoclone;
    "cbrt_rowvector", Tnoclone;
    "cbrt_matrix", Tnoclone;
    "cbrt_array", Tnoclone;
    "square_int", Tnoclone;
    "square_real", Tnoclone;
    "square_vector", Tnoclone;
    "square_rowvector", Tnoclone;
    "square_matrix", Tnoclone;
    "square_array", Tnoclone;
    "exp_int", Tnoclone;
    "exp_real", Tnoclone;
    "exp_vector", Tnoclone;
    "exp_rowvector", Tnoclone;
    "exp_matrix", Tnoclone;
    "exp_array", Tnoclone;
    "exp2_int", Tnoclone;
    "exp2_real", Tnoclone;
    "exp2_vector", Tnoclone;
    "exp2_rowvector", Tnoclone;
    "exp2_matrix", Tnoclone;
    "exp2_array", Tnoclone;
    "log_int", Tnoclone;
    "log_real", Tnoclone;
    "log_vector", Tnoclone;
    "log_rowvector", Tnoclone;
    "log_matrix", Tnoclone;
    "log_array", Tnoclone;
    "log2_int", Tnoclone;
    "log2_real", Tnoclone;
    "log2_vector", Tnoclone;
    "log2_rowvector", Tnoclone;
    "log2_matrix", Tnoclone;
    "log2_array", Tnoclone;
    "log10_int", Tnoclone;
    "log10_real", Tnoclone;
    "log10_vector", Tnoclone;
    "log10_rowvector", Tnoclone;
    "log10_matrix", Tnoclone;
    "log10_array", Tnoclone;
    "pow_int_int", Tnoclone;
    "pow_int_real", Tnoclone;
    "pow_real_int", Tnoclone;
    "pow_real_real", Tnoclone;
    "pow_vectorized", Tnoclone;
    "pow_vector_vector", Tnoclone;
    "pow_rowvector_rowvector", Tnoclone;
    "pow_matrix_matrix", Tnoclone;
    "pow_array_array", Tnoclone;
    "pow_real_vector", Tnoclone;
    "pow_real_rowvector", Tnoclone;
    "pow_real_matrix", Tnoclone;
    "pow_real_array", Tnoclone;
    "pow_vector_real", Tnoclone;
    "pow_rowvector_real", Tnoclone;
    "pow_matrix_real", Tnoclone;
    "pow_array_real", Tnoclone;
    "pow_int_vector", Tnoclone;
    "pow_int_rowvector", Tnoclone;
    "pow_int_matrix", Tnoclone;
    "pow_int_array", Tnoclone;
    "pow_vector_int", Tnoclone;
    "pow_rowvector_int", Tnoclone;
    "pow_matrix_int", Tnoclone;
    "pow_array_int", Tnoclone;
    "inv_int", Tnoclone;
    "inv_real", Tnoclone;
    "inv_vector", Tnoclone;
    "inv_rowvector", Tnoclone;
    "inv_matrix", Tnoclone;
    "inv_array", Tnoclone;
    "inv_sqrt_int", Tnoclone;
    "inv_sqrt_real", Tnoclone;
    "inv_sqrt_vector", Tnoclone;
    "inv_sqrt_rowvector", Tnoclone;
    "inv_sqrt_matrix", Tnoclone;
    "inv_sqrt_array", Tnoclone;
    "inv_square_int", Tnoclone;
    "inv_square_real", Tnoclone;
    "inv_square_vector", Tnoclone;
    "inv_square_rowvector", Tnoclone;
    "inv_square_matrix", Tnoclone;
    "inv_square_array", Tnoclone;
    (* 3.9 Trigonometric Functions *)
    "hypot_real_real", Tnoclone;
    "cos_int", Tnoclone;
    "cos_real", Tnoclone;
    "cos_vector", Tnoclone;
    "cos_rowvector", Tnoclone;
    "cos_matrix", Tnoclone;
    "cos_array", Tnoclone;
    "sin_int", Tnoclone;
    "sin_real", Tnoclone;
    "sin_vector", Tnoclone;
    "sin_rowvector", Tnoclone;
    "sin_matrix", Tnoclone;
    "sin_array", Tnoclone;
    "tan_int", Tnoclone;
    "tan_real", Tnoclone;
    "tan_vector", Tnoclone;
    "tan_rowvector", Tnoclone;
    "tan_matrix", Tnoclone;
    "tan_array", Tnoclone;
    "acos_int", Tnoclone;
    "acos_real", Tnoclone;
    "acos_vector", Tnoclone;
    "acos_rowvector", Tnoclone;
    "acos_matrix", Tnoclone;
    "acos_array", Tnoclone;
    "asin_int", Tnoclone;
    "asin_real", Tnoclone;
    "asin_vector", Tnoclone;
    "asin_rowvector", Tnoclone;
    "asin_matrix", Tnoclone;
    "asin_array", Tnoclone;
    "atan_int", Tnoclone;
    "atan_real", Tnoclone;
    "atan_vector", Tnoclone;
    "atan_rowvector", Tnoclone;
    "atan_matrix", Tnoclone;
    "atan_array", Tnoclone;
    "atan2_real_real", Tnoclone;
    "atan2_int_real", Tnoclone;
    "atan2_real_int", Tnoclone;
    "atan2_int_int", Tnoclone;
    (* 3.10 Hyperbolic Trigonometric Functions *)
    "cosh_int", Tnoclone;
    "cosh_real", Tnoclone;
    "cosh_vector", Tnoclone;
    "cosh_rowvector", Tnoclone;
    "cosh_matrix", Tnoclone;
    "cosh_array", Tnoclone;
    "sinh_int", Tnoclone;
    "sinh_real", Tnoclone;
    "sinh_vector", Tnoclone;
    "sinh_rowvector", Tnoclone;
    "sinh_matrix", Tnoclone;
    "sinh_array", Tnoclone;
    "tanh_int", Tnoclone;
    "tanh_real", Tnoclone;
    "tanh_vector", Tnoclone;
    "tanh_rowvector", Tnoclone;
    "tanh_matrix", Tnoclone;
    "tanh_array", Tnoclone;
    "acosh_int", Tnoclone;
    "acosh_real", Tnoclone;
    "acosh_vector", Tnoclone;
    "acosh_rowvector", Tnoclone;
    "acosh_matrix", Tnoclone;
    "acosh_array", Tnoclone;
    "asinh_int", Tnoclone;
    "asinh_real", Tnoclone;
    "asinh_vector", Tnoclone;
    "asinh_rowvector", Tnoclone;
    "asinh_matrix", Tnoclone;
    "asinh_array", Tnoclone;
    "atanh_int", Tnoclone;
    "atanh_real", Tnoclone;
    "atanh_vector", Tnoclone;
    "atanh_rowvector", Tnoclone;
    "atanh_matrix", Tnoclone;
    "atanh_array", Tnoclone;
    (* 3.11 Link Functions *)
    "logit_int", Tnoclone;
    "logit_real", Tnoclone;
    "logit_vector", Tnoclone;
    "logit_rowvector", Tnoclone;
    "logit_matrix", Tnoclone;
    "logit_array", Tnoclone;
    "inv_logit_int", Tnoclone;
    "inv_logit_real", Tnoclone;
    "inv_logit_vector", Tnoclone;
    "inv_logit_rowvector", Tnoclone;
    "inv_logit_matrix", Tnoclone;
    "inv_logit_array", Tnoclone;
    "inv_cloglog_int", Tnoclone;
    "inv_cloglog_real", Tnoclone;
    "inv_cloglog_vector", Tnoclone;
    "inv_cloglog_rowvector", Tnoclone;
    "inv_cloglog_matrix", Tnoclone;
    "inv_cloglog_array", Tnoclone;
    (* 3.12 Probability-related functions *)
    (* 3.12.1 Normal cumulative distribution functions *)
    "erf_int", Tnoclone;
    "erf_real", Tnoclone;
    "erf_vector", Tnoclone;
    "erf_rowvector", Tnoclone;
    "erf_matrix", Tnoclone;
    "erf_array", Tnoclone;
    "erfc_int", Tnoclone;
    "erfc_real", Tnoclone;
    "erfc_vector", Tnoclone;
    "erfc_rowvector", Tnoclone;
    "erfc_matrix", Tnoclone;
    "erfc_array", Tnoclone;
    "Phi_int", Tnoclone;
    "Phi_real", Tnoclone;
    "Phi_vector", Tnoclone;
    "Phi_rowvector", Tnoclone;
    "Phi_matrix", Tnoclone;
    "Phi_array", Tnoclone;
    "inv_Phi_int", Tnoclone;
    "inv_Phi_real", Tnoclone;
    "inv_Phi_vector", Tnoclone;
    "inv_Phi_rowvector", Tnoclone;
    "inv_Phi_matrix", Tnoclone;
    "inv_Phi_array", Tnoclone;
    "Phi_approx_int", Tnoclone;
    "Phi_approx_real", Tnoclone;
    "Phi_approx_vector", Tnoclone;
    "Phi_approx_rowvector", Tnoclone;
    "Phi_approx_matrix", Tnoclone;
    "Phi_approx_array", Tnoclone;
    (* 3.12.2 Other probability-related functions *)
    "binary_log_loss_real_real", Tnoclone;
    "binary_log_loss_int_real", Tnoclone;
    "binary_log_loss_real_int", Tnoclone;
    "binary_log_loss_int_int", Tnoclone;
    "binary_log_loss_vectorized", Tnoclone;
    "binary_log_loss_vector_vector", Tnoclone;
    "binary_log_loss_rowvector_rowvector", Tnoclone;
    "binary_log_loss_matrix_matrix", Tnoclone;
    "binary_log_loss_array_array", Tnoclone;
    "binary_log_loss_real_vector", Tnoclone;
    "binary_log_loss_real_rowvector", Tnoclone;
    "binary_log_loss_real_matrix", Tnoclone;
    "binary_log_loss_real_array", Tnoclone;
    "binary_log_loss_vector_real", Tnoclone;
    "binary_log_loss_rowvector_real", Tnoclone;
    "binary_log_loss_matrix_real", Tnoclone;
    "binary_log_loss_array_real", Tnoclone;
    "binary_log_loss_int_vector", Tnoclone;
    "binary_log_loss_int_rowvector", Tnoclone;
    "binary_log_loss_int_matrix", Tnoclone;
    "binary_log_loss_int_array", Tnoclone;
    "binary_log_loss_vector_int", Tnoclone;
    "binary_log_loss_rowvector_int", Tnoclone;
    "binary_log_loss_matrix_int", Tnoclone;
    "binary_log_loss_array_int", Tnoclone;
    "owens_t_real_real", Tnoclone;
    "owens_t_int_real", Tnoclone;
    "owens_t_real_int", Tnoclone;
    "owens_t_int_int", Tnoclone;
    "owens_t_vectorized", Tnoclone;
    "owens_t_vector_vector", Tnoclone;
    "owens_t_rowvector_rowvector", Tnoclone;
    "owens_t_matrix_matrix", Tnoclone;
    "owens_t_array_array", Tnoclone;
    "owens_t_real_vector", Tnoclone;
    "owens_t_real_rowvector", Tnoclone;
    "owens_t_real_matrix", Tnoclone;
    "owens_t_real_array", Tnoclone;
    "owens_t_vector_real", Tnoclone;
    "owens_t_rowvector_real", Tnoclone;
    "owens_t_matrix_real", Tnoclone;
    "owens_t_array_real", Tnoclone;
    "owens_t_int_vector", Tnoclone;
    "owens_t_int_rowvector", Tnoclone;
    "owens_t_int_matrix", Tnoclone;
    "owens_t_int_array", Tnoclone;
    "owens_t_vector_int", Tnoclone;
    "owens_t_rowvector_int", Tnoclone;
    "owens_t_matrix_int", Tnoclone;
    "owens_t_array_int", Tnoclone;
    (* 3.13 Combinatorial functions *)
    "beta_real_real", Tnoclone;
    "beta_int_real", Tnoclone;
    "beta_real_int", Tnoclone;
    "beta_int_int", Tnoclone;
    "beta_vectorized", Tnoclone;
    "beta_vector_vector", Tnoclone;
    "beta_rowvector_rowvector", Tnoclone;
    "beta_matrix_matrix", Tnoclone;
    "beta_array_array", Tnoclone;
    "beta_real_vector", Tnoclone;
    "beta_real_rowvector", Tnoclone;
    "beta_real_matrix", Tnoclone;
    "beta_real_array", Tnoclone;
    "beta_vector_real", Tnoclone;
    "beta_rowvector_real", Tnoclone;
    "beta_matrix_real", Tnoclone;
    "beta_array_real", Tnoclone;
    "beta_int_vector", Tnoclone;
    "beta_int_rowvector", Tnoclone;
    "beta_int_matrix", Tnoclone;
    "beta_int_array", Tnoclone;
    "beta_vector_int", Tnoclone;
    "beta_rowvector_int", Tnoclone;
    "beta_matrix_int", Tnoclone;
    "beta_array_int", Tnoclone;
    "inc_beta_real_real_real", Tnoclone;
    "lbeta_real_real", Tnoclone;
    "lbeta_int_real", Tnoclone;
    "lbeta_real_int", Tnoclone;
    "lbeta_int_int", Tnoclone;
    "lbeta_vectorized", Tnoclone;
    "lbeta_vector_vector", Tnoclone;
    "lbeta_rowvector_rowvector", Tnoclone;
    "lbeta_matrix_matrix", Tnoclone;
    "lbeta_array_array", Tnoclone;
    "lbeta_real_vector", Tnoclone;
    "lbeta_real_rowvector", Tnoclone;
    "lbeta_real_matrix", Tnoclone;
    "lbeta_real_array", Tnoclone;
    "lbeta_vector_real", Tnoclone;
    "lbeta_rowvector_real", Tnoclone;
    "lbeta_matrix_real", Tnoclone;
    "lbeta_array_real", Tnoclone;
    "lbeta_int_vector", Tnoclone;
    "lbeta_int_rowvector", Tnoclone;
    "lbeta_int_matrix", Tnoclone;
    "lbeta_int_array", Tnoclone;
    "lbeta_vector_int", Tnoclone;
    "lbeta_rowvector_int", Tnoclone;
    "lbeta_matrix_int", Tnoclone;
    "lbeta_array_int", Tnoclone;
    "tgamma_int", Tnoclone;
    "tgamma_real", Tnoclone;
    "tgamma_vector", Tnoclone;
    "tgamma_rowvector", Tnoclone;
    "tgamma_matrix", Tnoclone;
    "tgamma_array", Tnoclone;
    "lgamma_int", Tnoclone;
    "lgamma_real", Tnoclone;
    "lgamma_vector", Tnoclone;
    "lgamma_rowvector", Tnoclone;
    "lgamma_matrix", Tnoclone;
    "lgamma_array", Tnoclone;
    "digamma_int", Tnoclone;
    "digamma_real", Tnoclone;
    "digamma_vector", Tnoclone;
    "digamma_rowvector", Tnoclone;
    "digamma_matrix", Tnoclone;
    "digamma_array", Tnoclone;
    "trigamma_int", Tnoclone;
    "trigamma_real", Tnoclone;
    "trigamma_vector", Tnoclone;
    "trigamma_rowvector", Tnoclone;
    "trigamma_matrix", Tnoclone;
    "trigamma_array", Tnoclone;
    "lmgamma_real_real", Tnoclone;
    "lmgamma_int_real", Tnoclone;
    "lmgamma_real_int", Tnoclone;
    "lmgamma_int_int", Tnoclone;
    "lmgamma_vectorized", Tnoclone;
    "lmgamma_vector_vector", Tnoclone;
    "lmgamma_rowvector_rowvector", Tnoclone;
    "lmgamma_matrix_matrix", Tnoclone;
    "lmgamma_array_array", Tnoclone;
    "lmgamma_real_vector", Tnoclone;
    "lmgamma_real_rowvector", Tnoclone;
    "lmgamma_real_matrix", Tnoclone;
    "lmgamma_real_array", Tnoclone;
    "lmgamma_vector_real", Tnoclone;
    "lmgamma_rowvector_real", Tnoclone;
    "lmgamma_matrix_real", Tnoclone;
    "lmgamma_array_real", Tnoclone;
    "lmgamma_int_vector", Tnoclone;
    "lmgamma_int_rowvector", Tnoclone;
    "lmgamma_int_matrix", Tnoclone;
    "lmgamma_int_array", Tnoclone;
    "lmgamma_vector_int", Tnoclone;
    "lmgamma_rowvector_int", Tnoclone;
    "lmgamma_matrix_int", Tnoclone;
    "lmgamma_array_int", Tnoclone;
    "lmgamma_real_real", Tnoclone;
    "lmgamma_int_real", Tnoclone;
    "lmgamma_real_int", Tnoclone;
    "lmgamma_int_int", Tnoclone;
    "lmgamma_vectorized", Tnoclone;
    "lmgamma_vector_vector", Tnoclone;
    "lmgamma_rowvector_rowvector", Tnoclone;
    "lmgamma_matrix_matrix", Tnoclone;
    "lmgamma_array_array", Tnoclone;
    "lmgamma_real_vector", Tnoclone;
    "lmgamma_real_rowvector", Tnoclone;
    "lmgamma_real_matrix", Tnoclone;
    "lmgamma_real_array", Tnoclone;
    "lmgamma_vector_real", Tnoclone;
    "lmgamma_rowvector_real", Tnoclone;
    "lmgamma_matrix_real", Tnoclone;
    "lmgamma_array_real", Tnoclone;
    "lmgamma_int_vector", Tnoclone;
    "lmgamma_int_rowvector", Tnoclone;
    "lmgamma_int_matrix", Tnoclone;
    "lmgamma_int_array", Tnoclone;
    "lmgamma_vector_int", Tnoclone;
    "lmgamma_rowvector_int", Tnoclone;
    "lmgamma_matrix_int", Tnoclone;
    "lmgamma_array_int", Tnoclone;
    "gamma_q_real_real", Tnoclone;
    "gamma_q_int_real", Tnoclone;
    "gamma_q_real_int", Tnoclone;
    "gamma_q_int_int", Tnoclone;
    "gamma_q_vectorized", Tnoclone;
    "gamma_q_vector_vector", Tnoclone;
    "gamma_q_rowvector_rowvector", Tnoclone;
    "gamma_q_matrix_matrix", Tnoclone;
    "gamma_q_array_array", Tnoclone;
    "gamma_q_real_vector", Tnoclone;
    "gamma_q_real_rowvector", Tnoclone;
    "gamma_q_real_matrix", Tnoclone;
    "gamma_q_real_array", Tnoclone;
    "gamma_q_vector_real", Tnoclone;
    "gamma_q_rowvector_real", Tnoclone;
    "gamma_q_matrix_real", Tnoclone;
    "gamma_q_array_real", Tnoclone;
    "gamma_q_int_vector", Tnoclone;
    "gamma_q_int_rowvector", Tnoclone;
    "gamma_q_int_matrix", Tnoclone;
    "gamma_q_int_array", Tnoclone;
    "gamma_q_vector_int", Tnoclone;
    "gamma_q_rowvector_int", Tnoclone;
    "gamma_q_matrix_int", Tnoclone;
    "gamma_q_array_int", Tnoclone;
    "binomial_coefficient_log_real_real", Tnoclone;
    "binomial_coefficient_log_int_real", Tnoclone;
    "binomial_coefficient_log_real_int", Tnoclone;
    "binomial_coefficient_log_int_int", Tnoclone;
    "binomial_coefficient_log_vectorized", Tnoclone;
    "binomial_coefficient_log_vector_vector", Tnoclone;
    "binomial_coefficient_log_rowvector_rowvector", Tnoclone;
    "binomial_coefficient_log_matrix_matrix", Tnoclone;
    "binomial_coefficient_log_array_array", Tnoclone;
    "binomial_coefficient_log_real_vector", Tnoclone;
    "binomial_coefficient_log_real_rowvector", Tnoclone;
    "binomial_coefficient_log_real_matrix", Tnoclone;
    "binomial_coefficient_log_real_array", Tnoclone;
    "binomial_coefficient_log_vector_real", Tnoclone;
    "binomial_coefficient_log_rowvector_real", Tnoclone;
    "binomial_coefficient_log_matrix_real", Tnoclone;
    "binomial_coefficient_log_array_real", Tnoclone;
    "binomial_coefficient_log_int_vector", Tnoclone;
    "binomial_coefficient_log_int_rowvector", Tnoclone;
    "binomial_coefficient_log_int_matrix", Tnoclone;
    "binomial_coefficient_log_int_array", Tnoclone;
    "binomial_coefficient_log_vector_int", Tnoclone;
    "binomial_coefficient_log_rowvector_int", Tnoclone;
    "binomial_coefficient_log_matrix_int", Tnoclone;
    "binomial_coefficient_log_array_int", Tnoclone;
    "choose_real_real", Tnoclone;
    "choose_int_real", Tnoclone;
    "choose_real_int", Tnoclone;
    "choose_int_int", Tnoclone;
    "choose_vectorized", Tnoclone;
    "choose_vector_vector", Tnoclone;
    "choose_rowvector_rowvector", Tnoclone;
    "choose_matrix_matrix", Tnoclone;
    "choose_array_array", Tnoclone;
    "choose_real_vector", Tnoclone;
    "choose_real_rowvector", Tnoclone;
    "choose_real_matrix", Tnoclone;
    "choose_real_array", Tnoclone;
    "choose_vector_real", Tnoclone;
    "choose_rowvector_real", Tnoclone;
    "choose_matrix_real", Tnoclone;
    "choose_array_real", Tnoclone;
    "choose_int_vector", Tnoclone;
    "choose_int_rowvector", Tnoclone;
    "choose_int_matrix", Tnoclone;
    "choose_int_array", Tnoclone;
    "choose_vector_int", Tnoclone;
    "choose_rowvector_int", Tnoclone;
    "choose_matrix_int", Tnoclone;
    "choose_array_int", Tnoclone;
    "bessel_first_kind_real_real", Tnoclone;
    "bessel_first_kind_int_real", Tnoclone;
    "bessel_first_kind_real_int", Tnoclone;
    "bessel_first_kind_int_int", Tnoclone;
    "bessel_first_kind_vectorized", Tnoclone;
    "bessel_first_kind_vector_vector", Tnoclone;
    "bessel_first_kind_rowvector_rowvector", Tnoclone;
    "bessel_first_kind_matrix_matrix", Tnoclone;
    "bessel_first_kind_array_array", Tnoclone;
    "bessel_first_kind_real_vector", Tnoclone;
    "bessel_first_kind_real_rowvector", Tnoclone;
    "bessel_first_kind_real_matrix", Tnoclone;
    "bessel_first_kind_real_array", Tnoclone;
    "bessel_first_kind_vector_real", Tnoclone;
    "bessel_first_kind_rowvector_real", Tnoclone;
    "bessel_first_kind_matrix_real", Tnoclone;
    "bessel_first_kind_array_real", Tnoclone;
    "bessel_first_kind_int_vector", Tnoclone;
    "bessel_first_kind_int_rowvector", Tnoclone;
    "bessel_first_kind_int_matrix", Tnoclone;
    "bessel_first_kind_int_array", Tnoclone;
    "bessel_first_kind_vector_int", Tnoclone;
    "bessel_first_kind_rowvector_int", Tnoclone;
    "bessel_first_kind_matrix_int", Tnoclone;
    "bessel_first_kind_array_int", Tnoclone;
    "bessel_second_kind_real_real", Tnoclone;
    "bessel_second_kind_int_real", Tnoclone;
    "bessel_second_kind_real_int", Tnoclone;
    "bessel_second_kind_int_int", Tnoclone;
    "bessel_second_kind_vectorized", Tnoclone;
    "bessel_second_kind_vector_vector", Tnoclone;
    "bessel_second_kind_rowvector_rowvector", Tnoclone;
    "bessel_second_kind_matrix_matrix", Tnoclone;
    "bessel_second_kind_array_array", Tnoclone;
    "bessel_second_kind_real_vector", Tnoclone;
    "bessel_second_kind_real_rowvector", Tnoclone;
    "bessel_second_kind_real_matrix", Tnoclone;
    "bessel_second_kind_real_array", Tnoclone;
    "bessel_second_kind_vector_real", Tnoclone;
    "bessel_second_kind_rowvector_real", Tnoclone;
    "bessel_second_kind_matrix_real", Tnoclone;
    "bessel_second_kind_array_real", Tnoclone;
    "bessel_second_kind_int_vector", Tnoclone;
    "bessel_second_kind_int_rowvector", Tnoclone;
    "bessel_second_kind_int_matrix", Tnoclone;
    "bessel_second_kind_int_array", Tnoclone;
    "bessel_second_kind_vector_int", Tnoclone;
    "bessel_second_kind_rowvector_int", Tnoclone;
    "bessel_second_kind_matrix_int", Tnoclone;
    "bessel_second_kind_array_int", Tnoclone;
    "modified_bessel_first_kind_real_real", Tnoclone;
    "modified_bessel_first_kind_int_real", Tnoclone;
    "modified_bessel_first_kind_real_int", Tnoclone;
    "modified_bessel_first_kind_int_int", Tnoclone;
    "modified_bessel_first_kind_vectorized", Tnoclone;
    "modified_bessel_first_kind_vector_vector", Tnoclone;
    "modified_bessel_first_kind_rowvector_rowvector", Tnoclone;
    "modified_bessel_first_kind_matrix_matrix", Tnoclone;
    "modified_bessel_first_kind_array_array", Tnoclone;
    "modified_bessel_first_kind_real_vector", Tnoclone;
    "modified_bessel_first_kind_real_rowvector", Tnoclone;
    "modified_bessel_first_kind_real_matrix", Tnoclone;
    "modified_bessel_first_kind_real_array", Tnoclone;
    "modified_bessel_first_kind_vector_real", Tnoclone;
    "modified_bessel_first_kind_rowvector_real", Tnoclone;
    "modified_bessel_first_kind_matrix_real", Tnoclone;
    "modified_bessel_first_kind_array_real", Tnoclone;
    "modified_bessel_first_kind_int_vector", Tnoclone;
    "modified_bessel_first_kind_int_rowvector", Tnoclone;
    "modified_bessel_first_kind_int_matrix", Tnoclone;
    "modified_bessel_first_kind_int_array", Tnoclone;
    "modified_bessel_first_kind_vector_int", Tnoclone;
    "modified_bessel_first_kind_rowvector_int", Tnoclone;
    "modified_bessel_first_kind_matrix_int", Tnoclone;
    "modified_bessel_first_kind_array_int", Tnoclone;
    "log_modified_bessel_first_kind_real_real", Tnoclone;
    "log_modified_bessel_first_kind_int_real", Tnoclone;
    "log_modified_bessel_first_kind_real_int", Tnoclone;
    "log_modified_bessel_first_kind_int_int", Tnoclone;
    "log_modified_bessel_first_kind_vectorized", Tnoclone;
    "log_modified_bessel_first_kind_vector_vector", Tnoclone;
    "log_modified_bessel_first_kind_rowvector_rowvector", Tnoclone;
    "log_modified_bessel_first_kind_matrix_matrix", Tnoclone;
    "log_modified_bessel_first_kind_array_array", Tnoclone;
    "log_modified_bessel_first_kind_real_vector", Tnoclone;
    "log_modified_bessel_first_kind_real_rowvector", Tnoclone;
    "log_modified_bessel_first_kind_real_matrix", Tnoclone;
    "log_modified_bessel_first_kind_real_array", Tnoclone;
    "log_modified_bessel_first_kind_vector_real", Tnoclone;
    "log_modified_bessel_first_kind_rowvector_real", Tnoclone;
    "log_modified_bessel_first_kind_matrix_real", Tnoclone;
    "log_modified_bessel_first_kind_array_real", Tnoclone;
    "log_modified_bessel_first_kind_int_vector", Tnoclone;
    "log_modified_bessel_first_kind_int_rowvector", Tnoclone;
    "log_modified_bessel_first_kind_int_matrix", Tnoclone;
    "log_modified_bessel_first_kind_int_array", Tnoclone;
    "log_modified_bessel_first_kind_vector_int", Tnoclone;
    "log_modified_bessel_first_kind_rowvector_int", Tnoclone;
    "log_modified_bessel_first_kind_matrix_int", Tnoclone;
    "log_modified_bessel_first_kind_array_int", Tnoclone;
    "modified_bessel_second_kind_real_real", Tnoclone;
    "modified_bessel_second_kind_int_real", Tnoclone;
    "modified_bessel_second_kind_real_int", Tnoclone;
    "modified_bessel_second_kind_int_int", Tnoclone;
    "modified_bessel_second_kind_vectorized", Tnoclone;
    "modified_bessel_second_kind_vector_vector", Tnoclone;
    "modified_bessel_second_kind_rowvector_rowvector", Tnoclone;
    "modified_bessel_second_kind_matrix_matrix", Tnoclone;
    "modified_bessel_second_kind_array_array", Tnoclone;
    "modified_bessel_second_kind_real_vector", Tnoclone;
    "modified_bessel_second_kind_real_rowvector", Tnoclone;
    "modified_bessel_second_kind_real_matrix", Tnoclone;
    "modified_bessel_second_kind_real_array", Tnoclone;
    "modified_bessel_second_kind_vector_real", Tnoclone;
    "modified_bessel_second_kind_rowvector_real", Tnoclone;
    "modified_bessel_second_kind_matrix_real", Tnoclone;
    "modified_bessel_second_kind_array_real", Tnoclone;
    "modified_bessel_second_kind_int_vector", Tnoclone;
    "modified_bessel_second_kind_int_rowvector", Tnoclone;
    "modified_bessel_second_kind_int_matrix", Tnoclone;
    "modified_bessel_second_kind_int_array", Tnoclone;
    "modified_bessel_second_kind_vector_int", Tnoclone;
    "modified_bessel_second_kind_rowvector_int", Tnoclone;
    "modified_bessel_second_kind_matrix_int", Tnoclone;
    "modified_bessel_second_kind_array_int", Tnoclone;
    "falling_factorial_real_real", Tnoclone;
    "falling_factorial_int_real", Tnoclone;
    "falling_factorial_real_int", Tnoclone;
    "falling_factorial_int_int", Tnoclone;
    "falling_factorial_vectorized", Tnoclone;
    "falling_factorial_vector_vector", Tnoclone;
    "falling_factorial_rowvector_rowvector", Tnoclone;
    "falling_factorial_matrix_matrix", Tnoclone;
    "falling_factorial_array_array", Tnoclone;
    "falling_factorial_real_vector", Tnoclone;
    "falling_factorial_real_rowvector", Tnoclone;
    "falling_factorial_real_matrix", Tnoclone;
    "falling_factorial_real_array", Tnoclone;
    "falling_factorial_vector_real", Tnoclone;
    "falling_factorial_rowvector_real", Tnoclone;
    "falling_factorial_matrix_real", Tnoclone;
    "falling_factorial_array_real", Tnoclone;
    "falling_factorial_int_vector", Tnoclone;
    "falling_factorial_int_rowvector", Tnoclone;
    "falling_factorial_int_matrix", Tnoclone;
    "falling_factorial_int_array", Tnoclone;
    "falling_factorial_vector_int", Tnoclone;
    "falling_factorial_rowvector_int", Tnoclone;
    "falling_factorial_matrix_int", Tnoclone;
    "falling_factorial_array_int", Tnoclone;
    "lchoose_real_real", Tnoclone;
    "lchoose_int_real", Tnoclone;
    "lchoose_real_int", Tnoclone;
    "lchoose_int_int", Tnoclone;
    "log_falling_factorial_real_real", Tnoclone;
    "log_falling_factorial_int_real", Tnoclone;
    "log_falling_factorial_real_int", Tnoclone;
    "log_falling_factorial_int_int", Tnoclone;
    "rising_factorial_real_real", Tnoclone;
    "rising_factorial_int_real", Tnoclone;
    "rising_factorial_real_int", Tnoclone;
    "rising_factorial_int_int", Tnoclone;
    "rising_factorial_vectorized", Tnoclone;
    "rising_factorial_vector_vector", Tnoclone;
    "rising_factorial_rowvector_rowvector", Tnoclone;
    "rising_factorial_matrix_matrix", Tnoclone;
    "rising_factorial_array_array", Tnoclone;
    "rising_factorial_real_vector", Tnoclone;
    "rising_factorial_real_rowvector", Tnoclone;
    "rising_factorial_real_matrix", Tnoclone;
    "rising_factorial_real_array", Tnoclone;
    "rising_factorial_vector_real", Tnoclone;
    "rising_factorial_rowvector_real", Tnoclone;
    "rising_factorial_matrix_real", Tnoclone;
    "rising_factorial_array_real", Tnoclone;
    "rising_factorial_int_vector", Tnoclone;
    "rising_factorial_int_rowvector", Tnoclone;
    "rising_factorial_int_matrix", Tnoclone;
    "rising_factorial_int_array", Tnoclone;
    "rising_factorial_vector_int", Tnoclone;
    "rising_factorial_rowvector_int", Tnoclone;
    "rising_factorial_matrix_int", Tnoclone;
    "rising_factorial_array_int", Tnoclone;
    "log_rising_factorial_real_real", Tnoclone;
    "log_rising_factorial_int_real", Tnoclone;
    "log_rising_factorial_real_int", Tnoclone;
    "log_rising_factorial_int_int", Tnoclone;
    "log_rising_factorial_vectorized", Tnoclone;
    "log_rising_factorial_vector_vector", Tnoclone;
    "log_rising_factorial_rowvector_rowvector", Tnoclone;
    "log_rising_factorial_matrix_matrix", Tnoclone;
    "log_rising_factorial_array_array", Tnoclone;
    "log_rising_factorial_real_vector", Tnoclone;
    "log_rising_factorial_real_rowvector", Tnoclone;
    "log_rising_factorial_real_matrix", Tnoclone;
    "log_rising_factorial_real_array", Tnoclone;
    "log_rising_factorial_vector_real", Tnoclone;
    "log_rising_factorial_rowvector_real", Tnoclone;
    "log_rising_factorial_matrix_real", Tnoclone;
    "log_rising_factorial_array_real", Tnoclone;
    "log_rising_factorial_int_vector", Tnoclone;
    "log_rising_factorial_int_rowvector", Tnoclone;
    "log_rising_factorial_int_matrix", Tnoclone;
    "log_rising_factorial_int_array", Tnoclone;
    "log_rising_factorial_vector_int", Tnoclone;
    "log_rising_factorial_rowvector_int", Tnoclone;
    "log_rising_factorial_matrix_int", Tnoclone;
    "log_rising_factorial_array_int", Tnoclone;
    (* 3.14 Composed Functions *)
    "expm1_int", Tnoclone;
    "expm1_real", Tnoclone;
    "expm1_vector", Tnoclone;
    "expm1_rowvector", Tnoclone;
    "expm1_matrix", Tnoclone;
    "expm1_array", Tnoclone;
    "fma_real_real_real", Tnoclone;
    "multiply_log_real_real", Tnoclone;
    "multiply_log_int_real", Tnoclone;
    "multiply_log_real_int", Tnoclone;
    "multiply_log_int_int", Tnoclone;
    "multiply_log_vectorized", Tnoclone;
    "multiply_log_vector_vector", Tnoclone;
    "multiply_log_rowvector_rowvector", Tnoclone;
    "multiply_log_matrix_matrix", Tnoclone;
    "multiply_log_array_array", Tnoclone;
    "multiply_log_real_vector", Tnoclone;
    "multiply_log_real_rowvector", Tnoclone;
    "multiply_log_real_matrix", Tnoclone;
    "multiply_log_real_array", Tnoclone;
    "multiply_log_vector_real", Tnoclone;
    "multiply_log_rowvector_real", Tnoclone;
    "multiply_log_matrix_real", Tnoclone;
    "multiply_log_array_real", Tnoclone;
    "multiply_log_int_vector", Tnoclone;
    "multiply_log_int_rowvector", Tnoclone;
    "multiply_log_int_matrix", Tnoclone;
    "multiply_log_int_array", Tnoclone;
    "multiply_log_vector_int", Tnoclone;
    "multiply_log_rowvector_int", Tnoclone;
    "multiply_log_matrix_int", Tnoclone;
    "multiply_log_array_int", Tnoclone;
    "ldexp_real_real", Tnoclone;
    "ldexp_int_real", Tnoclone;
    "ldexp_real_int", Tnoclone;
    "ldexp_int_int", Tnoclone;
    "ldexp_vectorized", Tnoclone;
    "ldexp_vector_vector", Tnoclone;
    "ldexp_rowvector_rowvector", Tnoclone;
    "ldexp_matrix_matrix", Tnoclone;
    "ldexp_array_array", Tnoclone;
    "ldexp_real_vector", Tnoclone;
    "ldexp_real_rowvector", Tnoclone;
    "ldexp_real_matrix", Tnoclone;
    "ldexp_real_array", Tnoclone;
    "ldexp_vector_real", Tnoclone;
    "ldexp_rowvector_real", Tnoclone;
    "ldexp_matrix_real", Tnoclone;
    "ldexp_array_real", Tnoclone;
    "ldexp_int_vector", Tnoclone;
    "ldexp_int_rowvector", Tnoclone;
    "ldexp_int_matrix", Tnoclone;
    "ldexp_int_array", Tnoclone;
    "ldexp_vector_int", Tnoclone;
    "ldexp_rowvector_int", Tnoclone;
    "ldexp_matrix_int", Tnoclone;
    "ldexp_array_int", Tnoclone;
    "lmultiply_real_real", Tnoclone;
    "lmultiply_int_real", Tnoclone;
    "lmultiply_real_int", Tnoclone;
    "lmultiply_int_int", Tnoclone;
    "lmultiply_vectorized", Tnoclone;
    "lmultiply_vector_vector", Tnoclone;
    "lmultiply_rowvector_rowvector", Tnoclone;
    "lmultiply_matrix_matrix", Tnoclone;
    "lmultiply_array_array", Tnoclone;
    "lmultiply_real_vector", Tnoclone;
    "lmultiply_real_rowvector", Tnoclone;
    "lmultiply_real_matrix", Tnoclone;
    "lmultiply_real_array", Tnoclone;
    "lmultiply_vector_real", Tnoclone;
    "lmultiply_rowvector_real", Tnoclone;
    "lmultiply_matrix_real", Tnoclone;
    "lmultiply_array_real", Tnoclone;
    "lmultiply_int_vector", Tnoclone;
    "lmultiply_int_rowvector", Tnoclone;
    "lmultiply_int_matrix", Tnoclone;
    "lmultiply_int_array", Tnoclone;
    "lmultiply_vector_int", Tnoclone;
    "lmultiply_rowvector_int", Tnoclone;
    "lmultiply_matrix_int", Tnoclone;
    "lmultiply_array_int", Tnoclone;
    "log1p_int", Tnoclone;
    "log1p_real", Tnoclone;
    "log1p_vector", Tnoclone;
    "log1p_rowvector", Tnoclone;
    "log1p_matrix", Tnoclone;
    "log1p_array", Tnoclone;
    "log1m_int", Tnoclone;
    "log1m_real", Tnoclone;
    "log1m_vector", Tnoclone;
    "log1m_rowvector", Tnoclone;
    "log1m_matrix", Tnoclone;
    "log1m_array", Tnoclone;
    "log1p_exp_int", Tnoclone;
    "log1p_exp_real", Tnoclone;
    "log1p_exp_vector", Tnoclone;
    "log1p_exp_rowvector", Tnoclone;
    "log1p_exp_matrix", Tnoclone;
    "log1p_exp_array", Tnoclone;
    "log1m_exp_int", Tnoclone;
    "log1m_exp_real", Tnoclone;
    "log1m_exp_vector", Tnoclone;
    "log1m_exp_rowvector", Tnoclone;
    "log1m_exp_matrix", Tnoclone;
    "log1m_exp_array", Tnoclone;
    "log_diff_exp_real_real", Tnoclone;
    "log_diff_exp_int_real", Tnoclone;
    "log_diff_exp_real_int", Tnoclone;
    "log_diff_exp_int_int", Tnoclone;
    "log_diff_exp_vectorized", Tnoclone;
    "log_diff_exp_vector_vector", Tnoclone;
    "log_diff_exp_rowvector_rowvector", Tnoclone;
    "log_diff_exp_matrix_matrix", Tnoclone;
    "log_diff_exp_array_array", Tnoclone;
    "log_diff_exp_real_vector", Tnoclone;
    "log_diff_exp_real_rowvector", Tnoclone;
    "log_diff_exp_real_matrix", Tnoclone;
    "log_diff_exp_real_array", Tnoclone;
    "log_diff_exp_vector_real", Tnoclone;
    "log_diff_exp_rowvector_real", Tnoclone;
    "log_diff_exp_matrix_real", Tnoclone;
    "log_diff_exp_array_real", Tnoclone;
    "log_diff_exp_int_vector", Tnoclone;
    "log_diff_exp_int_rowvector", Tnoclone;
    "log_diff_exp_int_matrix", Tnoclone;
    "log_diff_exp_int_array", Tnoclone;
    "log_diff_exp_vector_int", Tnoclone;
    "log_diff_exp_rowvector_int", Tnoclone;
    "log_diff_exp_matrix_int", Tnoclone;
    "log_diff_exp_array_int", Tnoclone;
    "log_mix_real_real_real", Tnoclone;
    "log_sum_exp_real_real", Tnoclone;
    "log_sum_exp_int_real", Tnoclone;
    "log_sum_exp_real_int", Tnoclone;
    "log_sum_exp_int_int", Tnoclone;
    "log_inv_logit_int", Tnoclone;
    "log_inv_logit_real", Tnoclone;
    "log_inv_logit_vector", Tnoclone;
    "log_inv_logit_rowvector", Tnoclone;
    "log_inv_logit_matrix", Tnoclone;
    "log_inv_logit_array", Tnoclone;
    "log1m_inv_logit_int", Tnoclone;
    "log1m_inv_logit_real", Tnoclone;
    "log1m_inv_logit_vector", Tnoclone;
    "log1m_inv_logit_rowvector", Tnoclone;
    "log1m_inv_logit_matrix", Tnoclone;
    "log1m_inv_logit_array", Tnoclone;
    (* 4 Array Operations *)
    (* 4.1 Reductions *)
    (* 4.1.1 Minimum and Maximum *)
    "min_array", Tnoclone;
    "max_array", Tnoclone;
    (* 4.1.2 Sum, Product, and Log Sum of Exp *)
    "sum_array", Tnoclone;
    "prod_array", Tnoclone;
    "log_sum_exp_array", Tclone;
    (* 4.1.3 Sample Mean, Variance, and Standard Deviation *)
    "mean_array", Tnoclone;
    "variance_array", Tnoclone;
    "sd_array", Tnoclone;
    (* 4.1.4 Euclidean Distance and Squared Distance *)
    "distance_vector_vector", Tnoclone;
    "distance_vector_rowvector", Tnoclone;
    "distance_rowvector_vector", Tnoclone;
    "distance_rowvector_rowvector", Tnoclone;
    "squared_distance_vector_vector", Tnoclone;
    "squared_distance_vector_rowvector", Tnoclone;
    "squared_distance_rowvector_vector", Tnoclone;
    "squared_distance_rowvector_rowvector", Tnoclone;
    (* 4.2 Array Size and Dimension Function *)
    "dims_int", Tnoclone;
    "dims_real", Tnoclone;
    "dims_vector", Tnoclone;
    "dims_rowvector", Tnoclone;
    "dims_matrix", Tnoclone;
    "dims_array", Tnoclone;
    "num_elements_array", Tnoclone;
    "size_array", Tnoclone;
    (* 4.3 Array Broadcasting *)
    "rep_array_int_int", Tnoclone;
    "rep_array_real_int", Tnoclone;
    "rep_array_int_int_int", Tnoclone;
    "rep_array_real_int_int", Tnoclone;
    "rep_array_int_int_int_int", Tnoclone;
    "rep_array_real_int_int_int", Tnoclone;
    (* 4.4 Array concatenation *)
    "append_array_array_array", Tnoclone;
    (* 4.5 Sorting functions *)
    "sort_asc_array", Tnoclone;
    "sort_desc_array", Tnoclone;
    "sort_indices_asc_array", Tnoclone;
    "sort_indices_desc_array", Tnoclone;
    "rank_array", Tnoclone;
    (* 4.6 Reversing functions *)
    "reverse_array", Tnoclone;
    (* 5 Matrix Operations *)
    (* 5.1 Integer-Valued Matrix Size Functions *)
    "num_elements_vector", Tnoclone;
    "num_elements_rowvector", Tnoclone;
    "num_elements_matrix", Tnoclone;
    "rows_vector", Tnoclone;
    "rows_rowvector", Tnoclone;
    "rows_matrix", Tnoclone;
    "cols_vector", Tnoclone;
    "cols_rowvector", Tnoclone;
    "cols_matrix", Tnoclone;
    (* 5.2 Matrix arithmetic operators *)
    (* 5.2.1 Negation prefix operators *)
    (* 5.2.2 Infix matrix operators *)
    (* 5.2.3 Broadcast infix operators *)
    (* 5.3 Transposition operator *)
    (* 5.4 Elementwise functions *)
    (* 5.5 Dot Products and Specialized Products *)
    "dot_product_vector_vector", Tnoclone;
    "dot_product_vector_rowvector", Tnoclone;
    "dot_product_rowvector_vector", Tnoclone;
    "dot_product_rowvector_rowvector", Tnoclone;
    "columns_dot_product_vector_vector", Tnoclone;
    "columns_dot_product_rowvector_rowvector", Tnoclone;
    "columns_dot_product_matrix_matrix", Tnoclone;
    "rows_dot_product_vector_vector", Tnoclone;
    "rows_dot_product_rowvector_rowvector", Tnoclone;
    "rows_dot_product_matrix_matrix", Tnoclone;
    "dot_self_vector", Tnoclone;
    "dot_self_rowvector", Tnoclone;
    "columns_dot_self_vector", Tnoclone;
    "columns_dot_self_rowvector", Tnoclone;
    "columns_dot_self_matrix", Tnoclone;
    "rows_dot_self_vector", Tnoclone;
    "rows_dot_self_rowvector", Tnoclone;
    "rows_dot_self_matrix", Tnoclone;
    (* 5.5.1 Specialized Products *)
    "tcrossprod_matrix", Tnoclone;
    "crossprod_matrix", Tnoclone;
    "quad_form_matrix_matrix", Tnoclone;
    "quad_form_matrix_vector", Tnoclone;
    "quad_form_diag_matrix_vector", Tnoclone;
    "quad_form_diag_matrix_row_vector", Tnoclone;
    "quad_form_sym_matrix_matrix", Tnoclone;
    "quad_form_sym_matrix_vector", Tnoclone;
    "trace_quad_form_matrix_matrix", Tnoclone;
    "trace_gen_quad_form_matrix_matrix_matrix", Tnoclone;
    "multiply_lower_tri_self_matrix", Tnoclone;
    "diag_pre_multiply_vector_matrix", Tnoclone;
    "diag_pre_multiply_rowvector_matrix", Tnoclone;
    "diag_post_multiply_matrix_vector", Tnoclone;
    "diag_post_multiply_matrix_rowvector", Tnoclone;
    (* 5.6 Reductions *)
    (* 5.6.1 Log Sum of Exponents *)
    "log_sum_exp_vector", Tnoclone;
    "log_sum_exp_rowvector", Tnoclone;
    "log_sum_exp_matrix", Tnoclone;
    (* 5.6.2 Minimum and Maximum *)
    "min_vector", Tnoclone;
    "min_rowvector", Tnoclone;
    "min_matrix", Tnoclone;
    "max_vector", Tnoclone;
    "max_rowvector", Tnoclone;
    "max_matrix", Tnoclone;
    (* 5.6.3 Sums and Products *)
    "sum_vector", Tnoclone;
    "sum_rowvector", Tnoclone;
    "sum_matrix", Tnoclone;
    "prod_vector", Tnoclone;
    "prod_rowvector", Tnoclone;
    "prod_matrix", Tnoclone;
    (* 5.6.4 Sample Moments *)
    "mean_vector", Tnoclone;
    "mean_rowvector", Tnoclone;
    "mean_matrix", Tnoclone;
    "variance_vector", Tnoclone;
    "variance_rowvector", Tnoclone;
    "variance_matrix", Tnoclone;
    "sd_vector", Tnoclone;
    "sd_rowvector", Tnoclone;
    "sd_matrix", Tnoclone;
    (* 5.7 Broadcast Functions *)
    "rep_vector_real_int", Tnoclone;
    "rep_vector_int_int", Tnoclone;
    "rep_row_vector_real_int", Tnoclone;
    "rep_row_vector_int_int", Tnoclone;
    "rep_matrix_real_int_int", Tnoclone;
    "rep_matrix_int_int_int", Tnoclone;
    "rep_matrix_vector_int", Tnoclone;
    "rep_matrix_rowvector_int", Tnoclone;
    (* 5.8 Diagonal Matrix Functions *)
    "add_diag_matrix_rowvector", Tnoclone;
    "add_diag_matrix_vector", Tnoclone;
    "add_diag_matrix_real", Tnoclone;
    "diagonal_matrix", Tnoclone;
    "diag_matrix_vector", Tnoclone;
    "identity_matrix_int", Tnoclone;
    (* 5.9 Container construction functions *)
    "linspaced_array_int_real_real", Tnoclone;
    "linspaced_array_int_int_real", Tnoclone;
    "linspaced_array_int_real_int", Tnoclone;
    "linspaced_array_int_int_int", Tnoclone;
    "linspaced_int_array_int_int_int", Tnoclone;
    "linspaced_vector_int_real_real", Tnoclone;
    "linspaced_vector_int_int_real", Tnoclone;
    "linspaced_vector_int_real_int", Tnoclone;
    "linspaced_vector_int_int_int", Tnoclone;
    "linspaced_row_vector_int_real_real", Tnoclone;
    "linspaced_row_vector_int_int_real", Tnoclone;
    "linspaced_row_vector_int_real_int", Tnoclone;
    "linspaced_row_vector_int_int_int", Tnoclone;
    "one_hot_int_array_int_int", Tnoclone;
    "one_hot_array_int_int", Tnoclone;
    "one_hot_vector_int_int", Tnoclone;
    "one_hot_row_vector_int_int", Tnoclone;
    "ones_int_array_int", Tnoclone;
    "ones_array_int", Tnoclone;
    "ones_vector_int", Tnoclone;
    "ones_row_vector_int", Tnoclone;
    "zeros_int_array_int", Tnoclone;
    "zeros_array_int", Tnoclone;
    "zeros_vector_int", Tnoclone;
    "zeros_row_vector_int", Tnoclone;
    "uniform_simplex_int", Tnoclone;
    (* 5.10 Slicing and Blocking Functions *)
    (* 5.10.1 Columns and Rows *)
    "col_matrix_int", Tnoclone;
    "row_matrix_int", Tnoclone;
    (* 5.10.2 Block Operations *)
    (* 5.10.2.1 Matrix Slicing Operations *)
    "block_matrix_int_int_int_int", Tnoclone;
    "sub_col_matrix_int_int_int", Tnoclone;
    "sub_row_matrix_int_int_int", Tnoclone;
    (* 5.10.2.2 Vector and Array Slicing Operations *)
    "head_vector_int", Tnoclone;
    "head_rowvector_int", Tnoclone;
    "head_array_int", Tnoclone;
    "tail_vector_int", Tnoclone;
    "tail_rowvector_int", Tnoclone;
    "tail_array_int", Tnoclone;
    "segment_vector_int_int", Tnoclone;
    "segment_rowvector_int_int", Tnoclone;
    "segment_array_int_int", Tnoclone;
    (* 5.11 Matrix Concatenation *)
    (* 5.11.0.1 Horizontal concatenation *)
    "append_col_matrix_matrix", Tnoclone;
    "append_col_matrix_vector", Tnoclone;
    "append_col_vector_matrix", Tnoclone;
    "append_col_vector_vector", Tnoclone;
    "append_col_rowvector_rowvector", Tnoclone;
    "append_col_real_rowvector", Tnoclone;
    "append_col_int_rowvector", Tnoclone;
    "append_col_rowvector_real", Tnoclone;
    "append_col_rowvector_int", Tnoclone;
    (* 5.11.0.2 Vertical concatenation *)
    "append_row_matrix_matrix", Tnoclone;
    "append_row_matrix_rowvector", Tnoclone;
    "append_row_rowvector_matrix", Tnoclone;
    "append_row_rowvector_rowvector", Tnoclone;
    "append_row_vector_vector", Tnoclone;
    "append_row_real_vector", Tnoclone;
    "append_row_int_vector", Tnoclone;
    "append_row_vector_real", Tnoclone;
    "append_row_vector_int", Tnoclone;
    (* 5.12 Special Matrix Functions *)
    (* 5.12.1 Softmax *)
    "softmax_vector", Tnoclone;
    "log_softmax_vector", Tnoclone;
    (* 5.12.2 Cumulative Sums *)
    "cumulative_sum_array", Tnoclone;
    "cumulative_sum_vector", Tnoclone;
    "cumulative_sum_rowvector", Tnoclone;
    (* 5.13 Covariance Functions *)
    (* 5.13.1 Exponentiated quadratic covariance function *)
    "cov_exp_quad_rowvector_real_real", Tnoclone;
    "cov_exp_quad_vector_real_real", Tnoclone;
    "cov_exp_quad_array_real_real", Tnoclone;
    "cov_exp_quad_rowvector_rowvector_real_real", Tnoclone;
    "cov_exp_quad_vector_vector_real_real", Tnoclone;
    "cov_exp_quad_array_array_real_real", Tnoclone;
    (* 5.14 Linear Algebra Functions and Solvers *)
    (* 5.14.1.1 Matrix division operators *)
    (* 5.14.1.2 Lower-triangular matrix division functions *)
    "mdivide_left_tri_low_matrix_vector", Tnoclone;
    "mdivide_left_tri_low_matrix_matrix", Tnoclone;
    "mdivide_right_tri_low_row_vector_matrix", Tnoclone;
    "mdivide_right_tri_low_matrix_matrix", Tnoclone;
    (* 5.14.2 Symmetric positive-definite matrix division functions *)
    "mdivide_left_spd_matrix_vector", Tnoclone;
    "mdivide_left_spd_matrix_matrix", Tnoclone;
    "mdivide_right_spd_row_vector_matrix", Tnoclone;
    "mdivide_right_spd_matrix_matrix", Tnoclone;
    (* 5.14.3 Matrix exponential *)
    "matrix_exp_matrix", Tnoclone;
    "matrix_exp_multiply_matrix_matrix", Tnoclone;
    "scale_matrix_exp_multiply_real_matrix_matrix", Tnoclone;
    "scale_matrix_exp_multiply_int_matrix_matrix", Tnoclone;
    (* 5.14.4 Matrix power *)
    "matrix_power_matrix_int", Tnoclone;
    (* 5.14.5 Linear algebra functions *)
    (* 5.14.5.1 Trace *)
    "trace_matrix", Tnoclone;
    (* 5.14.5.2 Determinants *)
    "determinant_matrix", Tnoclone;
    "log_determinant_matrix", Tnoclone;
    (* 5.14.5.3 Inverses *)
    "inverse_matrix", Tnoclone;
    "inverse_spd_matrix", Tnoclone;
    (* 5.14.5.4 Generalized Inverse *)
    "generalized_inverse_matrix", Tnoclone;
    (* 5.14.5.5 Eigendecomposition *)
    "eigenvalues_sym_matrix", Tnoclone;
    "eigenvectors_sym_matrix", Tnoclone;
    (* 5.14.5.6 QR decomposition *)
    "qr_thin_Q_matrix", Tnoclone;
    "qr_thin_R_matrix", Tnoclone;
    "qr_Q_matrix", Tnoclone;
    "qr_R_matrix", Tnoclone;
    (* 5.14.5.7 Cholesky decomposition *)
    "cholesky_decompose_matrix", Tnoclone;
    (* 5.14.5.8 Singular value decomposition *)
    "singular_values_matrix", Tnoclone;
    "svd_U_matrix", Tnoclone;
    "svd_V_matrix", Tnoclone;
    (* 5.15 Sort functions *)
    "sort_asc_vector", Tnoclone;
    "sort_asc_row_vector", Tnoclone;
    "sort_desc_vector", Tnoclone;
    "sort_desc_row_vector_row_vector", Tnoclone;
    "sort_indices_asc_vector", Tnoclone;
    "sort_indices_asc_row_vector", Tnoclone;
    "sort_indices_desc_vector", Tnoclone;
    "sort_indices_desc_row_vector", Tnoclone;
    "rank_vector_int", Tnoclone;
    "rank_row_vector_int", Tnoclone;
    (* 5.16 Reverse functions *)
    "reverse_vector", Tnoclone;
    "reverse_row_vector", Tnoclone;
    (* 6 Sparse Matrix Operations *)
    (* 6.1 Compressed row storage *)
    (* 6.2 Conversion functions *)
    (* 6.2.1 Dense to sparse conversion *)
    "csr_extract_w_matrix", Tnoclone;
    "csr_extract_v_matrix", Tnoclone;
    "csr_extract_u_matrix", Tnoclone;
    (* 6.2.2 Sparse to dense conversion *)
    "csr_to_dense_matrix_int_int_vector_int_int", Tnoclone;
    (* 6.3 Sparse matrix arithmetic *)
    (* 6.3.1 Sparse matrix multiplication *)
    "csr_matrix_times_vector_int_int_vector_int_int_vector", Tnoclone;
    (* 7. Mixed Operations *)
    "to_matrix_matrix", Tnoclone;
    "to_matrix_vector", Tnoclone;
    "to_matrix_rowvector", Tnoclone;
    "to_matrix_matrix_int_int", Tnoclone;
    "to_matrix_vector_int_int", Tnoclone;
    "to_matrix_rowvector_int_int", Tnoclone;
    "to_matrix_matrix_int_int_int", Tnoclone;
    "to_matrix_vector_int_int_int", Tnoclone;
    "to_matrix_rowvector_int_int_int", Tnoclone;
    "to_matrix_array_int_int", Tnoclone;
    "to_matrix_array_int_int_int", Tnoclone;
    "to_matrix_array", Tnoclone;
    "to_vector_matrix", Tnoclone;
    "to_vector_vector", Tnoclone;
    "to_vector_rowvector", Tnoclone;
    "to_vector_array", Tnoclone;
    "to_row_vector_matrix", Tnoclone;
    "to_row_vector_vector", Tnoclone;
    "to_row_vector_rowvector", Tnoclone;
    "to_row_vector_array", Tnoclone;
    "to_array_2d_matrix", Tnoclone;
    "to_array_1d_vector", Tnoclone;
    "to_array_1d_rowvector", Tnoclone;
    "to_array_1d_matrix", Tnoclone;
    "to_array_1d_array", Tnoclone;
    (* 8 Compound Arithmetic and Assignment *)
    (* 9 Higher-Order Functions *)
    (* 9.1 Algebraic equation solver *)
    (* 9.1.1 Specifying an algebraic equation as a function *)
    (* 9.1.2 Call to the algebraic solver *)
    "algebra_solver_function_vector_vector_array_array", Tnoclone;
    "algebra_solver_function_vector_vector_array_array_real_real_int", Tnoclone;
    "algebra_solver_newton_function_vector_vector_array_array", Tnoclone;
    "algebra_solver_newton_function_vector_vector_array_array_real_real_int", Tnoclone;
    (* 9.2 Ordinary Differential Equation (ODE) Solvers *)
    (* 9.2.1 Non-stiff solver *)
    "ode_rk45_function_vector_real_array", Tnoclone;
    "ode_rk45_tol_function_vector_real_array_real_real_int", Tnoclone;
    "ode_adams_function_vector_real_array", Tnoclone;
    "ode_adams_tol_function_vector_real_array_real_real_int", Tnoclone;
    (* 9.2.2 Stiff solver *)
    "ode_bdf_function_real_array", Tnoclone;
    "ode_bdf_tol_function_vector_real_array_real_real_int", Tnoclone;
    (* 9.3 1D integrator *)
    (* 9.3.1 Specifying an integrand as a function *)
    (* 9.3.2 Call to the 1D integrator *)
    "integrate_1d_function_real_real_array_array_array", Tnoclone;
    "integrate_1d_function_real_real_array_array_array_real", Tnoclone;
    (* 9.4 Reduce-sum function *)
    (* 9.4.1 Specifying the reduce-sum function *)
    "reduce_sum_function", Tnoclone;
    "reduce_sum_static_function", Tnoclone;
    (* 9.5 Map-rect function *)
    (* 9.5.1 Specifying the mapped function *)
    (* 9.5.2 Rectangular map *)
    "map_rect_function", Tnoclone;
    (* 10 Deprecated Functions *)
    (* 10.1 integrate_ode_rk45, integrate_ode_adams, integrate_ode_bdf ODE integrators *)
    (* 10.1.1 Specifying an ordinary differential equation as a function *)
    (* 10.1.2 Non-stiff solver *)
    "integrate_ode_rk45_function_array_real_array_array_array_array", Tnoclone;
    "integrate_ode_rk45_function_array_int_array_array_array_array", Tnoclone;
    "integrate_ode_rk45_function_array_real_array_array_array_array_real_real_int", Tnoclone;
    "integrate_ode_rk45_function_array_int_array_array_array_array_real_real_int", Tnoclone;
    "integrate_ode_rk45_function_array_int_array_array_array_array_real_real_real", Tnoclone;
    "integrate_ode_function_array_real_array_array_array_array", Tnoclone;
    "integrate_ode_adams_function_array_real_array_array_array_array", Tnoclone;
    "integrate_ode_adams_function_array_real_array_array_array_array_real_real_int", Tnoclone;
    (* 10.1.3 Stiff solver *)
    "integrate_ode_bdf_function_array_real_array_array_array_array", Tnoclone;
    "integrate_ode_bdf_function_array_int_array_array_array_array", Tnoclone;
    "integrate_ode_bdf_function_array_real_array_array_array_array_real_real_int", Tnoclone;
    "integrate_ode_bdf_function_array_int_array_array_array_array_real_real_int", Tnoclone;
  ]

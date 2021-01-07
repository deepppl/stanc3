open Core_kernel
open Ast
open Middle
open Format

module SSet = Set.Make(String)

type backend =
  | Pyro
  | Numpyro
  | Pyro_cuda

type mode =
  | Comprehensive
  | Generative
  | Mixed

type block =
  | Model
  | Guide
  | GeneratedQuantities

type context =
  { ctx_prog: typed_program
  ; ctx_backend: backend
  ; ctx_mode: mode
  ; ctx_block: block option
  ; ctx_loops: string list
  ; ctx_mutation: bool
  ; ctx_params: (string, unit) Hashtbl.t
  ; ctx_to_clone: bool
  ; ctx_to_clone_vars: SSet.t }

type clone_type = Tclone | Tnoclone

let set_ctx_mutation ctx =
  { ctx with ctx_mutation = true }

let set_to_clone ctx =
  { ctx with ctx_to_clone = true }

let unset_to_clone ctx =
  { ctx with ctx_to_clone = false }

let print_warning loc message =
  Fmt.pf Fmt.stderr
    "@[<v>@,Warning: %s:@,%s@]@."
    (Location.to_string loc.Location_span.begin_loc) message

let to_string pp args =
  let buff = Buffer.create 1024 in
  let sff = formatter_of_buffer buff in
  fprintf sff "@[<v0>%a@,@]@." pp args;
  Format.pp_print_flush sff ();
  Buffer.contents buff

let pp_print_nothing _ () = ()

let print_list_comma printer ff l =
  fprintf ff "@[<hov 0>%a@]"
    (pp_print_list ~pp_sep:(fun ff () -> fprintf ff ",@ ") printer)
    l

let print_list_newline ?(eol=false) printer ff l =
  fprintf ff "@[<v 0>%a%a@]"
  (pp_print_list ~pp_sep:(fun ff () -> fprintf ff "@,") printer)
  l
  (if eol && l <> [] then pp_print_cut else pp_print_nothing) ()

let pyro_dppllib =
  [ "sample"; "param"; "observe"; "factor"; "array"; "zeros"; "ones"; "empty";
    "matmul"; "true_divide"; "floor_divide"; "transpose";
    "dtype_long"; "dtype_float"; ]
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

let distribution =
  [ "improper_uniform", Tnoclone;
    "lower_constrained_improper_uniform", Tnoclone;
    "upper_constrained_improper_uniform", Tnoclone;
    "simplex_constrained_improper_uniform", Tnoclone;
    "unit_constrained_improper_uniform", Tnoclone;
    "ordered_constrained_improper_uniform", Tnoclone;
    "positive_ordered_constrained_improper_uniform", Tnoclone;
    "cholesky_factor_corr_constrained_improper_uniform", Tnoclone;
    "cholesky_factor_cov_constrained_improper_uniform", Tnoclone;
    "cov_constrained_improper_uniform", Tnoclone;
    "corr_constrained_improper_uniform", Tnoclone;
    "offset_constrained_improper_uniform", Tnoclone;
    "multiplier_constrained_improper_uniform", Tnoclone;
    "offset_multiplier_constrained_improper_uniform", Tnoclone;
    (* 19 Continuous Distributions on [0, 1] *)
    (* 19.1 Beta Distribution *)
    "beta", Tnoclone;
    "beta_lpdf", Tnoclone;
    "beta_cdf", Tnoclone;
    "beta_lcdf", Tnoclone;
    "beta_lccdf", Tnoclone;
    "beta_rng", Tnoclone;
    (* 12 Binary Distributions *)
    (* 12.1 Bernoulli Distribution *)
    "bernoulli", Tnoclone;
    "bernoulli_lpmf", Tnoclone;
    "bernoulli_cdf", Tnoclone;
    "bernoulli_lcdf", Tnoclone;
    "bernoulli_lccdf", Tnoclone;
    "bernoulli_rng", Tnoclone;
    (* 12.2 Bernoulli Distribution, Logit Parameterization *)
    "bernoulli_logit", Tnoclone;
    "bernoulli_logit_lpmf", Tnoclone;
    (* 13 Bounded Discrete Distributions *)
    (* 13.2 Binomial Distribution, Logit Parameterization *)
    "binomial_logit", Tnoclone;
    "binomial_logit_lpmf", Tnoclone;
    (* 13.5 Categorical Distribution *)
    "categorical", Tnoclone;
    "categorical_lpmf", Tnoclone;
    "categorical_rng", Tnoclone;
    "categorical_logit", Tnoclone;
    "categorical_logit_lpmf", Tnoclone;
    "categorical_logit_rng", Tnoclone;
    (* 14 Unbounded Discrete Distributions *)
    (* 14.2 Negative Binomial Distribution (alternative parameterization) *)
    "neg_binomial_2", Tnoclone;
    "neg_binomial_2_lpmf", Tnoclone;
    "neg_binomial_2_cdf", Tnoclone;
    "neg_binomial_2_lcdf", Tnoclone;
    "neg_binomial_2_lccdf", Tnoclone;
    "neg_binomial_2_rng", Tnoclone;
    (* 14.5 Poisson Distribution *)
    "poisson", Tnoclone;
    "poisson_lpmf", Tnoclone;
    "poisson_cdf", Tnoclone;
    "poisson_lcdf", Tnoclone;
    "poisson_lccdf", Tnoclone;
    "poisson_rng", Tnoclone;
    (* 14.6 Poisson Distribution, Log Parameterization *)
    "poisson_log", Tnoclone;
    "poisson_log_lpmf", Tnoclone;
    "poisson_log_rng", Tnoclone;
    (* 16 Unbounded Continuous Distributions *)
    (* 16.1 Normal Distribution *)
    "normal", Tnoclone;
    "normal_lpdf", Tnoclone;
    "normal_cdf", Tnoclone;
    "normal_lcdf", Tnoclone;
    "normal_lccdf", Tnoclone;
    "normal_rng", Tnoclone;
    "std_normal", Tnoclone;
    "std_normal_lpdf", Tnoclone;
    "std_normal_cdf", Tnoclone;
    "std_normal_lcdf", Tnoclone;
    "std_normal_lccdf", Tnoclone;
    "std_normal_rng", Tnoclone;
    (* 16.5 Student-T Distribution *)
    "student_t", Tnoclone;
    "student_t_lpdf", Tnoclone;
    "student_t_cdf", Tnoclone;
    "student_t_lcdf", Tnoclone;
    "student_t_lccdf", Tnoclone;
    "student_t_rng", Tnoclone;
    (* 16.6 Cauchy Distribution *)
    "cauchy", Tnoclone;
    "cauchy_lpdf", Tnoclone;
    "cauchy_cdf", Tnoclone;
    "cauchy_lcdf", Tnoclone;
    "cauchy_lccdf", Tnoclone;
    "cauchy_rng", Tnoclone;
    (* 16.7 Double Exponential (Laplace) Distribution *)
    "double_exponential", Tnoclone;
    "double_exponential_lpdf", Tnoclone;
    "double_exponential_cdf", Tnoclone;
    "double_exponential_lcdf", Tnoclone;
    "double_exponential_lccdf", Tnoclone;
    "double_exponential_rng", Tnoclone;
    (* 16.8 Logistic Distribution *)
    "logistic", Tnoclone;
    "logistic_lpdf", Tnoclone;
    "logistic_cdf", Tnoclone;
    "logistic_lcdf", Tnoclone;
    "logistic_lccdf", Tnoclone;
    "logistic_rng", Tnoclone;
    (* 17 Positive Continuous Distributions *)
    (* 17.1 Lognormal Distribution *)
    "lognormal", Tnoclone;
    "lognormal_lpdf", Tnoclone;
    "lognormal_cdf", Tnoclone;
    "lognormal_lcdf", Tnoclone;
    "lognormal_lccdf", Tnoclone;
    "lognormal_rng", Tnoclone;
    (* 17.5 Exponential Distribution *)
    "exponential", Tnoclone;
    "exponential_lpdf", Tnoclone;
    "exponential_cdf", Tnoclone;
    "exponential_lcdf", Tnoclone;
    "exponential_lccdf", Tnoclone;
    "exponential_rng", Tnoclone;
    (* 17.6 Gamma Distribution *)
    "gamma", Tnoclone;
    "gamma_lpdf", Tnoclone;
    "gamma_cdf", Tnoclone;
    "gamma_lcdf", Tnoclone;
    "gamma_lccdf", Tnoclone;
    "gamma_rng", Tnoclone;
    (* 17.7 Inverse Gamma Distribution *)
    "inv_gamma", Tnoclone;
    "inv_gamma_lpdf", Tnoclone;
    "inv_gamma_cdf", Tnoclone;
    "inv_gamma_lcdf", Tnoclone;
    "inv_gamma_lccdf", Tnoclone;
    "inv_gamma_rng", Tnoclone;
    (* 18 Positive Lower-Bounded Distributions *)
    (* 18.1 Pareto Distribution *)
    "pareto", Tnoclone;
    "pareto_lpdf", Tnoclone;
    "pareto_cdf", Tnoclone;
    "pareto_lcdf", Tnoclone;
    "pareto_lccdf", Tnoclone;
    "pareto_rng", Tnoclone;
    (* 21 Bounded Continuous Probabilities *)
    (* 21.1 Uniform Distribution *)
    "uniform", Tnoclone;
    "uniform_lpdf", Tnoclone;
    "uniform_cdf", Tnoclone;
    "uniform_lcdf", Tnoclone;
    "uniform_lccdf", Tnoclone;
    "uniform_rng", Tnoclone;
    (* 22 Distributions over Unbounded Vectors *)
    (* 22.1 Multivariate Normal Distribution *)
    "multi_normal", Tnoclone;
    "multi_normal_lpdf", Tnoclone;
    "multi_normal_rng", Tnoclone;
    (* 23 Simplex Distributions *)
    (* 23.1 Dirichlet Distribution *)
    "dirichlet", Tnoclone;
    "dirichlet_lpdf", Tnoclone;
    "dirichlet_rng", Tnoclone;
  ]

let stanlib =
  [ (* 3.2 Mathematical Constants *)
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
    (* 3.7 Step-like Functions *)
    "abs_int", Tnoclone;
    "abs_real", Tnoclone;
    "abs_vector", Tnoclone;
    "abs_rowvector", Tnoclone;
    "abs_matrix", Tnoclone;
    "abs_array", Tnoclone;
    "fdim_real_real", Tnoclone;
    "fmin_real_real", Tnoclone;
    "fmin_int_real", Tnoclone;
    "fmin_real_int", Tnoclone;
    "fmin_int_int", Tnoclone;
    "fmax_real_real", Tnoclone;
    "fmax_int_real", Tnoclone;
    "fmax_real_int", Tnoclone;
    "fmax_int_int", Tnoclone;
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
    "min_array", Tnoclone;
    "max_array", Tnoclone;
    "sum_array", Tnoclone;
    "prod_array", Tnoclone;
    "log_sum_exp_array", Tclone;
    "mean_array", Tnoclone;
    "variance_array", Tnoclone;
    "sd_array", Tnoclone;
    "distance_vector_vector", Tnoclone;
    "distance_vector_rowvector", Tnoclone;
    "distance_rowvector_vector", Tnoclone;
    "distance_rowvector_rowvector", Tnoclone;
    "squared_distance_vector_vector", Tnoclone;
    "squared_distance_vector_rowvector", Tnoclone;
    "squared_distance_rowvector_vector", Tnoclone;
    "squared_distance_rowvector_rowvector", Tnoclone;
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
    (* 3.14 Composed Functions *)
    "expm1_int", Tnoclone;
    "expm1_real", Tnoclone;
    "expm1_vector", Tnoclone;
    "expm1_rowvector", Tnoclone;
    "expm1_matrix", Tnoclone;
    "expm1_array", Tnoclone;
    "fma_real_real_real", Tnoclone;
    "multiply_log_real_real", Tnoclone;
    "lmultiply_real_real", Tnoclone;
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
    "log_mix_real_real_real", Tnoclone;
    "log_sum_exp_real_real", Tnoclone;
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
    (* 4.1 Reductions *)
    "min_array", Tnoclone;
    "max_array", Tnoclone;
    "sum_array", Tnoclone;
    "prod_array", Tnoclone;
    "log_sum_exp_array", Tnoclone;
    "mean_array", Tnoclone;
    "variance_array", Tnoclone;
    "sd_array", Tnoclone;
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
    "tcrossprod_matrix", Tnoclone;
    "crossprod_matrix", Tnoclone;
    "quad_form_matrix_matrix", Tnoclone;
    "quad_form_matrix_vector", Tnoclone;
    "quad_form_diag_matrix_vector", Tnoclone;
    "quad_form_diag_matrix_row_vector ", Tnoclone;
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
    "log_sum_exp_vector", Tnoclone;
    "log_sum_exp_rowvector", Tnoclone;
    "log_sum_exp_matrix", Tnoclone;
    "min_vector", Tnoclone;
    "min_rowvector", Tnoclone;
    "min_matrix", Tnoclone;
    "max_vector", Tnoclone;
    "max_rowvector", Tnoclone;
    "max_matrix", Tnoclone;
    "sum_vector", Tnoclone;
    "sum_rowvector", Tnoclone;
    "sum_matrix", Tnoclone;
    "prod_vector", Tnoclone;
    "prod_rowvector", Tnoclone;
    "prod_matrix", Tnoclone;
    "mean_vector", Tnoclone;
    "mean_rowvector", Tnoclone;
    "mean_matrix", Tnoclone;
    "variance_vector", Tnoclone;
    "variance_rowvector", Tnoclone;
    "variance_matrix", Tnoclone;
    "sd_vector", Tnoclone;
    "sd_rowvector", Tnoclone;
    "sd_matrix", Tnoclone;
    "rep_vector_real_int", Tnoclone;
    "rep_vector_int_int", Tnoclone;
    "rep_row_vector_real_int", Tnoclone;
    "rep_row_vector_int_int", Tnoclone;
    "rep_matrix_real_int_int", Tnoclone;
    "rep_matrix_int_int_int", Tnoclone;
    "rep_matrix_vector_int", Tnoclone;
    "rep_matrix_rowvector_int", Tnoclone;
    "col_matrix_int", Tnoclone;
    "row_matrix_int", Tnoclone;
    "block_matrix_int_int_int_int", Tnoclone;
    "sub_col_matrix_int_int_int", Tnoclone;
    "sub_row_matrix_int_int_int", Tnoclone;
    "head_vector_int", Tnoclone;
    "head_rowvector_int", Tnoclone;
    "head_array_int", Tnoclone;
    "tail_vector_int", Tnoclone;
    "tail_rowvector_int", Tnoclone;
    "tail_array_int", Tnoclone;
    "segment_vector_int_int", Tnoclone;
    "segment_rowvector_int_int", Tnoclone;
    "segment_array_int_int", Tnoclone;
    "append_col_matrix_matrix", Tnoclone;
    "append_col_matrix_vector", Tnoclone;
    "append_col_vector_matrix", Tnoclone;
    "append_col_vector_vector", Tnoclone;
    "append_col_rowvector_rowvector", Tnoclone;
    "append_col_real_rowvector", Tnoclone;
    "append_col_int_rowvector", Tnoclone;
    "append_col_rowvector_real", Tnoclone;
    "append_col_rowvector_int", Tnoclone;
    "append_row_matrix_matrix", Tnoclone;
    "append_row_matrix_rowvector", Tnoclone;
    "append_row_rowvector_matrix", Tnoclone;
    "append_row_rowvector_rowvector", Tnoclone;
    "append_row_vector_vector", Tnoclone;
    "append_row_real_vector", Tnoclone;
    "append_row_int_vector", Tnoclone;
    "append_row_vector_real", Tnoclone;
    "append_row_vector_int", Tnoclone;
    (* 5.8 Diagonal Matrix Functions *)
    "add_diag_matrix_rowvector", Tnoclone;
    "add_diag_matrix_vector", Tnoclone;
    "add_diag_matrix_real", Tnoclone;
    "diagonal_matrix", Tnoclone;
    "diag_matrix_vector", Tnoclone;
    (* 5.11 Special Matrix Functions *)
    "softmax_vector", Tnoclone;
    "softmax_vector", Tnoclone;
    "cumulative_sum_array", Tnoclone;
    "cumulative_sum_vector", Tnoclone;
    "cumulative_sum_rowvector", Tnoclone;
    (* 5.12 Covariance Functions *)
    "cov_exp_quad_rowvector_real_real", Tnoclone;
    "cov_exp_quad_vector_real_real", Tnoclone;
    "cov_exp_quad_array_real_real", Tnoclone;
    "cov_exp_quad_rowvector_rowvector_real_real", Tnoclone;
    "cov_exp_quad_vector_vector_real_real", Tnoclone;
    "cov_exp_quad_array_array_real_real", Tnoclone;
    (* 5.13 Linear Algebra Functions and Solvers *)
    "cholesky_decompose_matrix", Tnoclone;
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
    (* 9.2 Ordinary Differential Equation (ODE) Solvers *)
    "integrate_ode_rk45_array_real_array_array_array_array", Tnoclone;
    "integrate_ode_rk45_array_int_array_array_array_array", Tnoclone;
    "integrate_ode_rk45_array_real_array_array_array_array_real_real_int", Tnoclone;
    "integrate_ode_rk45_array_int_array_array_array_array_real_real_real", Tnoclone;
  ]


let keywords =
  [ "lambda"; "def"; ]

let avoid =
  keywords @ pyro_dppllib @ numpyro_dppllib @ dppllib_networks @
  (List.map ~f:fst distribution) @ (List.map ~f:fst stanlib)

let trans_name ff name =
  let x =
    if List.mem ~equal:(=) avoid name then name ^ "__"
    else name
  in
  fprintf ff "%s" x

let trans_nn_base ff s =
  fprintf ff "%s_" s

let trans_id ff id =
  match id.path with
  | None -> trans_name ff id.name
  | Some (base :: p) ->
      fprintf ff "%a['%a']"
        trans_nn_base base
        (pp_print_list ~pp_sep:(fun ff () -> fprintf ff ".") pp_print_string) p
  | Some [] -> assert false

let stanlib_id id args =
  let arg_type arg =
    match arg.emeta.type_ with
    | UnsizedType.UInt -> "_int"
    | UReal -> "_real"
    | UVector -> "_vector"
    | URowVector -> "_rowvector"
    | UMatrix -> "_matrix"
    | UArray _ -> "_array"
    | UMathLibraryFunction | UFun _ -> ""
  in
  if
    List.exists
      ~f:(fun x -> String.is_suffix id.name ~suffix:x)
      ["_lpdf"; "_lpmf"; "_lcdf"; "_lccdf"; "_rng"]
 then
   id.name
 else
   List.fold_left ~init:id.name
     ~f:(fun acc arg -> acc ^ (arg_type arg))
     args

let function_id fn_kind id args =
  match fn_kind with
  | StanLib -> stanlib_id id args
  | UserDefined -> to_string trans_id id

let gen_id, gen_name =
  let cpt = ref 0 in
  let gen_id ?(fresh=true) l ff e =
    incr cpt;
    let s =
      match e.expr with
      | Variable {name; _} -> name
      | IntNumeral x | RealNumeral x ->
          if fresh then x else raise_s [%message "Unexpected identifier"]
      | Indexed ({ expr = Variable {name; _}; _ }, _) ->
          if fresh then name else raise_s [%message "Unexpected identifier"]
      | _ -> if fresh then "expr" else raise_s [%message "Unexpected identifier"]
    in
    if fresh then
      match l with
      | [] -> fprintf ff "'%s__%d'" s !cpt
      | _ ->
          fprintf ff "f'%s%a__%d'" s
            (pp_print_list ~pp_sep:(fun _ _ ->())
               (fun ff x -> fprintf ff "__{%s}" x)) l !cpt
    else
      fprintf ff "'%s'" s
  in
  let gen_name s =
    incr cpt;
    s ^ "__" ^ (string_of_int !cpt)
  in
  (gen_id, gen_name)

let without_underscores = String.filter ~f:(( <> ) '_')

let drop_leading_zeros s =
  match String.lfindi ~f:(fun _ c -> c <> '0') s with
  | Some p when p > 0 -> (
    match s.[p] with
    | 'e' | '.' -> String.drop_prefix s (p - 1)
    | _ -> String.drop_prefix s p )
  | Some _ -> s
  | None -> "0"

let format_number s = s |> without_underscores |> drop_leading_zeros

let expr_one =
  { expr = IntNumeral "1";
    emeta = { type_ = UInt; loc = Location_span.empty; ad_level = DataOnly; } }

let is_real t =
  match t with
  | Type.Sized (SizedType.SReal)
  | Unsized (UnsizedType.UReal) -> true
  | _ -> false

let is_unsized_tensor (type_ : UnsizedType.t) =
  match type_ with
  | UInt | UReal -> false
  | _ -> true

let is_tensor (type_ : typed_expression Type.t) =
  match type_ with
  | Sized (SInt | SReal)
  | Unsized (UInt | UReal) -> false
  | _ -> true

let is_path_expr e =
  match e.expr with
  | Variable { path = Some _; _ } -> true
  | _ -> false

let rec dims_of_sizedtype t =
  match t with
  | SizedType.SInt
  | SReal -> []
  | SVector e -> [e]
  | SRowVector e -> [e]
  | SMatrix (e1, e2) -> [e1; e2]
  | SArray (t, e) -> e :: dims_of_sizedtype t

let dims_of_type t =
  match t with
  | Type.Sized t -> dims_of_sizedtype t
  | Unsized _ ->
      raise_s
        [%message "Expecting sized type" (t : typed_expression Type.t)]

let var_of_lval lv =
  let rec var_of_lval acc lv =
    match lv.lval with
    | LVariable id -> id.name
    | _ -> fold_lvalue var_of_lval (fun acc _ -> acc) acc lv.lval
  in
  let x = var_of_lval "" lv in
  assert (x <> "");
  x

let split_lval lv =
  let rec split_lval acc lv =
    match lv.lval with
    | LVariable id -> (id, acc)
    | LIndexed (lv, idx) -> split_lval (idx::acc) lv
  in
  split_lval [] lv

let rec free_vars_expr (bv, fv) e =
  let fv =
    match e.expr with
    | Variable x -> if SSet.mem bv x.name then fv else SSet.add fv x.name
    | _ -> fv
  in
  fold_expression free_vars_expr (fun (bv, fv) _ -> (bv, fv)) (bv, fv) e.expr

let rec free_vars_lval (bv, fv) lv =
  match lv.lval with
  | LVariable id ->
      let fv = if SSet.mem bv id.name then fv else SSet.add fv id.name in
      (bv, fv)
  | _ -> fold_lvalue free_vars_lval free_vars_expr (bv, fv) lv.lval

let rec free_vars_stmt (bv, fv) s =
  let bv =
    match s.stmt with
    | VarDecl { identifier = x; _ }
    | For { loop_variable = x; _ }
    | ForEach (x, _, _) -> SSet.add bv x.name
    | _ -> bv
  in
  fold_statement
    free_vars_expr
    free_vars_stmt
    free_vars_lval
    (fun (bv, fv) _ -> (bv, fv))
    (bv, fv) s.stmt

let free_vars bv stmt =
  let _bv, fv = free_vars_stmt (bv, SSet.empty) stmt in
  fv

let is_variable_sampling x stmt =
  match stmt.stmt with
  | Tilde { arg = { expr = Variable y; _ }; _ } -> x = y.name
  | _ -> false

let is_variable_initialization x stmt =
  match stmt.stmt with
  | Assignment { assign_lhs = { lval = LVariable y; _ };
                 assign_op = (Assign | ArrowAssign); _ } -> x = y.name
  | _ -> false

let merge_decl_sample decl (stmt:typed_statement) =
  match decl.stmt, stmt.stmt with
  | VarDecl { decl_type; _ }, Tilde s ->
      begin match dims_of_type decl_type with
      | [] -> stmt
      | shape ->
          let shape_arg =
            { expr = ArrayExpr shape;
              emeta = { loc = Location_span.empty;
                        ad_level = DataOnly;
                        type_ = UArray UInt; }}
          in
          let args =
            s.args @ [ shape_arg ]
          in
          { stmt with stmt = Tilde { s with args } }
      end
  | _, _ -> assert false

let rec push_prior_stmts (x, decl) stmts =
  match stmts with
  | [] -> Some decl, [ ]
  | stmt :: stmts ->
      if is_variable_sampling x stmt then
        None, merge_decl_sample decl stmt :: stmts
      else if SSet.mem (free_vars SSet.empty stmt) x then
        Some decl, stmt :: stmts
      else
        let prior, stmts = push_prior_stmts (x, decl) stmts in
        prior, stmt ::  stmts

let push_priors priors stmts =
  List.fold_left
    ~f:(fun (priors, stmts) decl ->
        match decl.stmt with
        | VarDecl { identifier = id; initial_value = None; _ } ->
            begin match push_prior_stmts (id.name, decl) stmts with
            | Some prior, stmts -> priors @ [prior], stmts
            | None, stmts -> priors, stmts
            end
        | _ -> assert false)
    ~init:([], stmts) priors

let rec updated_vars_stmt ?(tilde_is_update=false) acc s =
  let acc =
    match s.stmt with
    | VarDecl { identifier = x; _ } -> SSet.add acc x.name
    | Assignment { assign_lhs = lhs; _ } -> SSet.add acc (var_of_lval lhs)
    | Tilde { arg = { expr = Variable x; _ }; _ } -> SSet.add acc x.name
    | For { loop_variable; _} -> SSet.add acc loop_variable.name
    | ForEach (loop_variable, _, _) -> SSet.add acc loop_variable.name
    | _ -> acc
  in
  fold_statement
    (fun acc _ -> acc)
    (updated_vars_stmt ~tilde_is_update)
    (fun acc _ -> acc) (fun acc _ -> acc)
    acc s.stmt

let moveup_stmt stmt stmts =
  let rec moveup (deps, stmt) rev_stmts acc =
    match rev_stmts with
    | [] -> stmt :: acc
    | stmt' :: rev_stmts ->
      let updated_vars =
        updated_vars_stmt ~tilde_is_update:true SSet.empty stmt'
      in
      if SSet.is_empty (SSet.inter deps updated_vars) then
        moveup (deps, stmt) rev_stmts (stmt'::acc)
      else
        List.rev_append rev_stmts (stmt' :: stmt :: acc)
  in
  let fvs = free_vars SSet.empty stmt in
  moveup (fvs, stmt) (List.rev stmts) []

let moveup_observes stmts =
  List.fold_left
    ~f:(fun acc s ->
         match s.stmt with
         | Tilde { arg = { expr = Variable _; _ }; _ } -> moveup_stmt s acc
         | _ -> acc @ [ s ])
    ~init:[] stmts

let merge_decl_stmt decl stmt =
  match decl.stmt, stmt.stmt with
  | VarDecl d, Assignment { assign_rhs = e; _ } ->
    { decl with stmt = VarDecl { d with initial_value = Some e } }
  | _, _ -> assert false

let rec push_vardecl_stmts (x, decl) stmts =
  match stmts with
  | [] -> [ decl ]
  | stmt :: stmts ->
      if is_variable_initialization x stmt then
        merge_decl_stmt decl stmt :: stmts
      else if SSet.mem (free_vars SSet.empty stmt) x then
        decl :: stmt :: stmts
      else stmt :: push_vardecl_stmts (x, decl) stmts

let rec push_vardecls_stmts stmts =
  List.fold_right
    ~f:(fun s stmts ->
        match s.stmt with
        | VarDecl { identifier = id; initial_value = None; _ } ->
            push_vardecl_stmts (id.name, s) stmts
        | _ -> push_vardecls_stmt s :: stmts)
    ~init:[] stmts

and push_vardecls_stmt (s: typed_statement) =
  match s.stmt with
  | Block stmts -> { s with stmt = Block (push_vardecls_stmts stmts) }
  | stmt ->
      let stmt =
        map_statement
          (fun e -> e) push_vardecls_stmt (fun lv -> lv) (fun f -> f)
          stmt
      in
      { s with stmt }

let flatten_stmts stmts =
  List.fold_right
    ~f:(fun s acc ->
        match s.stmt with
        | Block stmts -> stmts @ acc
        | _ -> s :: acc)
    ~init:[] stmts

let rewrite_program f p =
  { functionblock = Option.map ~f p.functionblock
  ; datablock = Option.map ~f p.datablock
  ; transformeddatablock = Option.map ~f p.transformeddatablock
  ; parametersblock = Option.map ~f p.parametersblock
  ; transformedparametersblock = Option.map ~f p.transformedparametersblock
  ; modelblock = Option.map ~f p.modelblock
  ; generatedquantitiesblock = Option.map ~f p.generatedquantitiesblock
  ; networksblock = p.networksblock
  ; guideparametersblock = Option.map ~f p.guideparametersblock
  ; guideblock = Option.map ~f p.guideblock
  }

let simplify_program (p: typed_program) =
  let p = rewrite_program flatten_stmts p in
  let p = rewrite_program push_vardecls_stmts p in
  let p = rewrite_program push_vardecls_stmts p in
  p

let get_var_decl_names stmts =
  List.fold_right
    ~f:(fun stmt acc ->
          match stmt.stmt with
          | VarDecl {identifier; _} -> identifier :: acc
          | _ -> acc)
    ~init:[] stmts

let get_var_decl_names_block block =
  Option.value_map ~default:[] ~f:get_var_decl_names block

let split_parameters parameters =
  let rec add x v l =
    match l with
    | [] -> [ (x, [v]) ]
    | (y, l') :: l when x = y -> (x, v :: l') :: l
    | (y, l') :: l -> (y, l') :: add x v l
  in
  List.fold_right
    ~f:(fun stmt (net_params, params) ->
        match stmt.stmt with
        | VarDecl {identifier = { path = Some p; _ }; _} ->
            begin match p with
            | base :: _ ->
              (add base stmt net_params, params)
            | [] -> assert false
            end
        | _ -> (net_params, stmt :: params))
    ~init:([],[])
    parameters

let split_networks networks parametersblock =
  let networks_params, parameters =
    split_parameters (Option.value ~default:[] parametersblock)
  in
  let register_networks =
    Option.map
      ~f:(List.filter
            ~f:(fun nn ->
                not (List.exists ~f:(fun (y, _) -> nn.net_id.name = y)
                       networks_params)))
      networks
  in
  (register_networks, networks_params, parameters)


let get_stanlib_calls program =
  let rec get_stanlib_calls_in_expr acc e =
    let acc =
      match e.expr with
      | FunApp (StanLib, id, args) | CondDistApp (StanLib, id, args) ->
        let sid = stanlib_id id args in
        if List.Assoc.mem ~equal:(=) stanlib sid then SSet.add acc sid
        else acc
      | _ -> acc
    in
    fold_expression get_stanlib_calls_in_expr (fun acc _ -> acc)
      acc e.expr
  in
  let rec get_stanlib_calls_in_lval acc lv =
    fold_lvalue get_stanlib_calls_in_lval get_stanlib_calls_in_expr
      acc lv.lval
  in
  let rec get_stanlib_calls_in_stmt acc stmt =
    fold_statement
      get_stanlib_calls_in_expr
      get_stanlib_calls_in_stmt
      get_stanlib_calls_in_lval
      (fun acc _ -> acc)
      acc stmt.stmt
  in
  fold_program get_stanlib_calls_in_stmt SSet.empty program

let get_functions_calls stmts =
  let rec get_functions_calls_in_expr acc e =
    let acc =
      match e.expr with
      | FunApp (_, id, _args) -> SSet.add acc id.name
      | _ -> acc
    in
    fold_expression
      get_functions_calls_in_expr
      (fun acc _ -> acc)
      acc e.expr
  in
  let rec get_functions_calls_in_lval acc lv =
    fold_lvalue
      get_functions_calls_in_lval
      get_functions_calls_in_expr
      acc lv.lval
  in
  let rec get_functions_calls_in_stmt acc s =
    fold_statement
      get_functions_calls_in_expr
      get_functions_calls_in_stmt
      get_functions_calls_in_lval
      (fun acc _ -> acc)
      acc s.stmt
  in
  List.fold_left ~init:SSet.empty ~f:get_functions_calls_in_stmt stmts

let get_networks_calls onetworks stmts =
  Option.value_map ~default:SSet.empty
    ~f:(fun networks ->
        let nets =
          List.fold_left ~init:SSet.empty
            ~f:(fun acc n -> SSet.add acc n.net_id.name)
            networks
        in
        let calls = get_functions_calls stmts in
        SSet.inter nets calls)
    onetworks

let get_var_decl_type x prog =
  let o =
    fold_program
      (fun acc s ->
         match s.stmt with
         | VarDecl { identifier = { name; _ }; decl_type; _} when x = name ->
           Some decl_type
         | _ -> acc)
      None prog
  in
  match o with
  | Some t -> t
  | None -> raise_s [%message "Unexpected unbounded variable" (x: string)]


let rec get_updated_arrays_stmt acc s =
  match s.stmt with
  | Assignment { assign_lhs = { lval = LIndexed (lhs, _); _ }; _ } ->
      SSet.add acc (var_of_lval lhs)
  | stmt ->
      let k = (fun acc _ -> acc) in
      fold_statement k get_updated_arrays_stmt k k acc stmt

let free_updated bv stmt =
  let fv = free_vars bv stmt in
  let arrays = updated_vars_stmt SSet.empty stmt in
  SSet.inter fv arrays

let rec trans_expr ctx ff (e: typed_expression) : unit =
  match e.expr with
  | Paren x -> fprintf ff "(%a)" (trans_expr ctx) x
  | BinOp (lhs, op, rhs) ->
      fprintf ff "%a" (trans_binop ctx lhs rhs) op
  | PrefixOp (op, e) | PostfixOp (e, op) ->
      fprintf ff "%a" (trans_unop ctx e) op
  | TernaryIf (cond, ifb, elseb) ->
      fprintf ff "%a if %a else %a"
        (trans_expr ctx) ifb (trans_expr ctx) cond (trans_expr ctx) elseb
  | Variable id ->
      fprintf ff "%a%s"
        trans_id id
        (if is_to_clone ctx e then ".clone()" else "")
  | IntNumeral x -> trans_numeral e.emeta.type_ ff x
  | RealNumeral x -> trans_numeral e.emeta.type_ ff x
  | FunApp (fn_kind, id, args) ->
      let ctx =
        match List.Assoc.find ~equal:(=) stanlib
                (function_id fn_kind id args) with
        | Some Tclone -> set_ctx_mutation (set_to_clone ctx)
        | Some Tnoclone | None -> ctx
      in
      trans_fun_app ctx fn_kind id ff args
  | CondDistApp (fn_kind, id, args) ->
      let ctx =
        match List.Assoc.find ~equal:(=) distribution
                (function_id fn_kind id args) with
        | Some Tclone -> set_ctx_mutation (set_to_clone ctx)
        | Some Tnoclone | None -> ctx
      in
      trans_cond_dist_app ctx fn_kind id ff args
  | GetLP | GetTarget -> fprintf ff "stanlib.target()" (* XXX TODO XXX *)
  | ArrayExpr eles ->
      fprintf ff "array([%a], dtype=%a)"
        (trans_exprs ctx) eles
        dtype_of_unsized_type e.emeta.type_
  | RowVectorExpr eles ->
      fprintf ff "array([%a], dtype=%a)"
        (trans_exprs ctx) eles
        dtype_of_unsized_type e.emeta.type_
  | Indexed (lhs, indices) ->
      if is_to_clone ctx e then
        let ctx' = unset_to_clone ctx in
        fprintf ff "%a[%a].clone()" (trans_expr ctx') lhs
          (print_list_comma (trans_idx ctx)) indices
      else
        fprintf ff "%a[%a]" (trans_expr ctx) lhs
          (print_list_comma (trans_idx ctx)) indices

and trans_numeral type_ ff x =
  begin match type_ with
  | UInt -> fprintf ff "%s" (format_number x)
  | UReal ->
      fprintf ff "array(%s, dtype=%a)"
        (format_number x) dtype_of_unsized_type type_
  | _ ->
      raise_s [%message "Unexpected type for a numeral" (type_ : UnsizedType.t)]
  end

and trans_binop ctx e1 e2 ff op =
    match op with
    | Operator.Plus ->
        fprintf ff "%a + %a" (trans_expr ctx) e1 (trans_expr ctx) e2
    | Minus ->
        fprintf ff "%a - %a" (trans_expr ctx) e1 (trans_expr ctx) e2
    | Times ->
        let ctx = set_ctx_mutation ctx in
        begin match e1.emeta.type_, e2.emeta.type_ with
        | ((UInt | UReal), _) | (_, (UInt | UReal)) ->
            fprintf ff "%a * %a" (trans_expr ctx) e1 (trans_expr ctx) e2
        | _ ->
            fprintf ff "matmul(%a, %a)" (trans_expr ctx) e1 (trans_expr ctx) e2
        end
    | Divide ->
        fprintf ff "true_divide(%a, %a)" (trans_expr ctx) e1 (trans_expr ctx) e2
    | IntDivide ->
        begin match e1.emeta.type_, e2.emeta.type_ with
        | (UInt, UInt) ->
            fprintf ff "%a / %a" (trans_expr ctx) e1 (trans_expr ctx) e2
        | _ ->
            fprintf ff "floor_divide(%a, %a)"
              (trans_expr ctx) e1 (trans_expr ctx) e2
        end
    | Modulo ->
        fprintf ff "%a %s %a" (trans_expr ctx) e1 "%" (trans_expr ctx) e2
    | LDivide ->
        fprintf ff "true_divide(%a, %a)" (trans_expr ctx) e2 (trans_expr ctx) e1
    | EltTimes -> fprintf ff "%a * %a" (trans_expr ctx) e1 (trans_expr ctx) e2
    | EltDivide ->
      fprintf ff "true_divide(%a, %a)" (trans_expr ctx) e1 (trans_expr ctx) e2
    | Pow -> fprintf ff "%a ** %a" (trans_expr ctx) e1 (trans_expr ctx) e2
    | EltPow -> fprintf ff "%a ** %a" (trans_expr ctx) e1 (trans_expr ctx) e2
    | Or -> fprintf ff "%a or %a" (trans_expr ctx) e1 (trans_expr ctx) e2
    | And -> fprintf ff "%a and %a" (trans_expr ctx) e1 (trans_expr ctx) e2
    | Equals -> fprintf ff "%a == %a" (trans_expr ctx) e1 (trans_expr ctx) e2
    | NEquals -> fprintf ff "%a != %a" (trans_expr ctx) e1 (trans_expr ctx) e2
    | Less -> fprintf ff "%a < %a" (trans_expr ctx) e1 (trans_expr ctx) e2
    | Leq -> fprintf ff "%a <= %a" (trans_expr ctx) e1 (trans_expr ctx) e2
    | Greater -> fprintf ff "%a > %a" (trans_expr ctx) e1 (trans_expr ctx) e2
    | Geq -> fprintf ff "%a >= %a" (trans_expr ctx) e1 (trans_expr ctx) e2
    | PNot
    | PPlus
    | PMinus
    | Transpose ->
        raise_s [%message "Binary operator expected" (op: Operator.t)]

and trans_unop ctx e ff op =
  match op with
  | Operator.PNot -> fprintf ff "+ %a" (trans_expr ctx) e
  | PPlus -> fprintf ff "+ %a" (trans_expr ctx) e
  | PMinus -> fprintf ff "- %a" (trans_expr ctx) e
  | Transpose ->
      begin match e.emeta.type_ with
      | UnsizedType.UVector | URowVector -> fprintf ff "%a" (trans_expr ctx) e
      | UMatrix -> fprintf ff "transpose(%a, 0, 1)" (trans_expr ctx) e
      | _ ->
        raise_s [%message "transpose: unexpected type"
            (e.emeta.type_: UnsizedType.t)]
      end
  | Plus
  | Minus
  | Times
  | Divide
  | IntDivide
  | Modulo
  | LDivide
  | EltTimes
  | EltDivide
  | Pow
  | EltPow
  | Or
  | And
  | Equals
  | NEquals
  | Less
  | Leq
  | Greater
  | Geq ->
      raise_s [%message "Unary operator expected" (op: Operator.t)]

and trans_idx ctx ff = function
  | All -> fprintf ff ":"
  | Upfrom e -> fprintf ff "%a - 1:" (trans_expr ctx) e
  | Downfrom e -> fprintf ff ":%a" (trans_expr ctx) e
  | Between (lb, ub) ->
      fprintf ff "%a - 1:%a" (trans_expr ctx) lb (trans_expr ctx) ub
  | Single e -> (
    match e.emeta.type_ with
    | UInt -> fprintf ff "%a - 1" (trans_expr ctx) e
    | UArray _ -> fprintf ff "%a - 1" (trans_expr ctx) e
    | _ ->
        raise_s
          [%message "Expecting int or array" (e.emeta.type_ : UnsizedType.t)] )

and dtype_of_unsized_type ff t =
  match t with
  | UInt -> fprintf ff "dtype_long"
  | UReal -> fprintf ff "dtype_float"
  | UVector | URowVector | UMatrix -> fprintf ff "dtype_float"
  | UArray(t) -> dtype_of_unsized_type ff t
  | UFun _ | UMathLibraryFunction -> assert false

and dtype_of_sized_type ff t =
  match t with
  | SizedType.SInt -> fprintf ff "dtype_long"
  | SReal -> fprintf ff "dtype_float"
  | SVector _ | SRowVector _ | SMatrix _ -> fprintf ff "dtype_float"
  | SArray(t, _) -> dtype_of_sized_type ff t

and dtype_of_type ff t =
  match t with
  | Type.Unsized t -> dtype_of_unsized_type ff t
  | Sized t -> dtype_of_sized_type ff t

and trans_exprs ctx ff exprs =
  fprintf ff "%a" (print_list_comma (trans_expr ctx)) exprs

and trans_fun_app ctx fn_kind id ff args =
  match fn_kind with
  | StanLib ->
      fprintf ff "%s(%a)"
        (stanlib_id id args) (trans_exprs ctx) args
  | UserDefined ->
      fprintf ff "%a(%a)"
        trans_id id (trans_exprs ctx) args

and trans_cond_dist_app ctx fn_kind id ff args =
  match fn_kind with
  | StanLib ->
      fprintf ff "%s(%a)"
        (stanlib_id id args) (trans_exprs ctx) args
  | UserDefined ->
      fprintf ff "%s(%a)"
        id.name (trans_exprs ctx) args

and trans_dims ctx ff (t : typed_expression Type.t) =
  match dims_of_type t with
  | [] -> fprintf ff "[]"
  | l -> fprintf ff "[%a]" (trans_exprs ctx) l

and is_to_clone ctx e =
  let rec can_be_modified e =
    match e.expr with
    | Variable x -> SSet.mem ctx.ctx_to_clone_vars x.name
    | Indexed (lhs, _) -> can_be_modified lhs
    | TernaryIf (_, ifb, elseb) -> can_be_modified ifb || can_be_modified elseb
    | Paren e -> can_be_modified e
    | FunApp (_, _, args) -> List.exists ~f:(is_to_clone ctx) args
    | _ -> false
  in
  match ctx.ctx_backend with
  | Pyro | Pyro_cuda ->
      ctx.ctx_to_clone && ctx.ctx_mutation &&
      ctx.ctx_block <> Some GeneratedQuantities &&
      can_be_modified e
  | Numpyro -> false

let trans_expr_opt ctx (type_ : typed_expression Type.t) ff = function
  | Some e -> trans_expr ctx ff e
  | None ->
      if is_tensor type_ then
        fprintf ff "empty(%a, dtype=%a)"
          (trans_dims ctx) type_
          dtype_of_type type_
      else fprintf ff "None"

let trans_arg ff (_, _, ident) =
  trans_id ff ident

let trans_args ff args =
  fprintf ff "%a" (print_list_comma trans_arg) args

let trans_printables ctx ff (ps : _ printable list) =
  fprintf ff "%a"
    (print_list_comma
       (fun ff -> function
          | PString s -> fprintf ff "%s" s
          | PExpr e -> (trans_expr ctx) ff e))
    ps

type control_kind =
  | CtrlPython
  | CtrlLax
  | CtrlNympyro

let rec trans_stmt ctx ff (ts : typed_statement) =
  let stmt_typed = ts.stmt in
  match stmt_typed with
  | Assignment {assign_lhs; assign_rhs; assign_op} ->
    let rec expr_of_lval = function
      | { lval= LVariable id; lmeta } ->
        {expr = Variable id; emeta = lmeta }
      | { lval= LIndexed (lhs, indices); lmeta } ->
        {expr = Indexed (expr_of_lval lhs, indices); emeta = lmeta; }
    in
    let trans_rhs lhs ff rhs =
      let ctx = set_to_clone ctx in
      match assign_op with
      | Assign | ArrowAssign -> trans_expr ctx ff rhs
      | OperatorAssign op -> trans_binop ctx lhs rhs ff op
    in
    begin match assign_lhs with
      | { lval= LVariable id; _ } ->
          fprintf ff "%a = %a"
            trans_id id
            (trans_rhs (expr_of_lval assign_lhs)) assign_rhs
      | { lval= LIndexed _; _ } ->
          let trans_indices ff l =
            fprintf ff "%a"
              (pp_print_list ~pp_sep:(fun _ () -> ())
                 (fun ff idx ->
                    fprintf ff "[%a]" (print_list_comma (trans_idx ctx)) idx))
              l
          in
          let id, indices = split_lval assign_lhs in
          begin match ctx.ctx_backend with
          | Pyro | Pyro_cuda ->
              fprintf ff "%a%a = %a"
                trans_id id trans_indices indices
                (trans_rhs (expr_of_lval assign_lhs)) assign_rhs
          | Numpyro ->
              fprintf ff "%a = ops_index_update(%a, ops_index%a, %a)"
                trans_id id
                trans_id id trans_indices indices
                (trans_rhs (expr_of_lval assign_lhs)) assign_rhs
          end
    end
  | NRFunApp (fn_kind, id, args) ->
      let ctx =
        match List.Assoc.find ~equal:(=) stanlib
                (function_id fn_kind id args) with
        | Some Tclone -> set_ctx_mutation (set_to_clone ctx)
        | Some Tnoclone | None -> ctx
      in
      trans_fun_app ctx fn_kind id ff args
  | IncrementLogProb e | TargetPE e ->
      let ctx = set_to_clone ctx in
      fprintf ff "factor(%a, %a)"
        (gen_id ctx.ctx_loops) e
        (trans_expr ctx) e
  | Tilde {arg; distribution; args; truncation} ->
      let trans_distribution ff (dist, args) =
        fprintf ff "%s(%a)"
          dist.name
          (print_list_comma (trans_expr ctx)) args
      in
      let trans_truncation _ff = function
        | NoTruncate -> ()
        | _ -> (* XXX TODO XXX *)
          raise_s [%message "Truncations are currently not supported."]
      in
      let is_sample =
        match ctx.ctx_mode with
        | Comprehensive -> false
        | Generative ->
            let _, fv = free_vars_expr (SSet.empty, SSet.empty) arg in
            let contains_param =
              List.exists ~f:(fun x -> SSet.mem fv x.name)
                (get_var_decl_names_block ctx.ctx_prog.parametersblock)
            in
            if contains_param then
              match arg.expr with Variable _ -> true | _ -> false
            else
              false
        | Mixed ->
            begin match arg.expr with
            | Variable {name; _} ->
                let is_parameter =
                  List.exists ~f:(fun x -> x.name = name)
                    (get_var_decl_names_block ctx.ctx_prog.parametersblock)
                in
                if is_parameter && not (Hashtbl.mem ctx.ctx_params name) then
                  (ignore (Hashtbl.add ctx.ctx_params ~key:name ~data:());
                   true)
                else false
            | _ -> false
            end || ctx.ctx_block = Some Guide
      in
      if is_sample then
        match ctx.ctx_block with
        | Some Guide when is_path_expr arg ->
            fprintf ff "%a = %a%a"
              (trans_expr ctx) arg
              trans_distribution (distribution, args)
              trans_truncation truncation
        | _ ->
            fprintf ff "%a = sample(%a, %a)%a"
              (trans_expr ctx) arg
              (gen_id ~fresh:false ctx.ctx_loops) arg
              trans_distribution (distribution, args)
              trans_truncation truncation
      else
        let ctx' = set_to_clone ctx in
        let adustment =
          match distribution.name with
          | "categorical"
          | "categorical_logit" -> " - 1"
          | _ -> ""
        in
        fprintf ff "observe(%a, %a, %a%s)%a"
          (gen_id ~fresh:true ctx.ctx_loops) arg
          trans_distribution (distribution, args)
          (trans_expr ctx') arg
          adustment
          trans_truncation truncation
  | Print ps -> fprintf ff "print(%a)" (trans_printables ctx) ps
  | Reject ps -> fprintf ff "stanlib.reject(%a)" (trans_printables ctx) ps
  | IfThenElse (cond, ifb, None) ->
      let kind =
        match ctx.ctx_backend with
        | Numpyro -> CtrlLax
        | Pyro | Pyro_cuda -> CtrlPython
      in
      begin match kind with
      | CtrlPython ->
          fprintf ff "@[<v 0>@[<v 4>if %a:@,%a@]@]"
            (trans_expr ctx) cond
            (trans_stmt ctx) ifb
      | CtrlLax->
          let name_tt, closure_tt, pp_pack, pp_unpack =
            build_closure ctx "then"
              (free_updated SSet.empty ifb)
              [] ifb
          in
          let name_ff = gen_name "else" in
          let pp_closure_ff ff kind =
            fprintf ff "%a@[<v 4>def %a(acc):@,return acc@]"
              print_jit kind
              trans_name name_ff
          in
          fprintf ff "@[<v 0>%a@,%a@,%a = lax_cond(@[%a,@ %a, %a,@ %a@])@]"
            closure_tt kind
            pp_closure_ff kind
            pp_unpack ()
            (trans_expr ctx) cond
            trans_name name_tt
            trans_name name_ff
            pp_pack ()
      | CtrlNympyro -> assert false
      end
  | IfThenElse (cond, ifb, Some elseb) ->
      let kind =
        match ctx.ctx_backend with
        | Numpyro -> CtrlLax
        | Pyro | Pyro_cuda -> CtrlPython
      in
      begin match kind with
      | CtrlPython ->
          fprintf ff "@[<v 0>@[<v 4>if %a:@,%a@]@,@[<v 4>else:@,%a@]@]"
            (trans_expr ctx) cond
            (trans_stmt ctx) ifb
            (trans_stmt ctx) elseb
      | CtrlLax ->
          let fv_tt = free_updated SSet.empty ifb in
          let fv_ff = free_updated SSet.empty elseb in
          let fv_closure = SSet.union fv_tt fv_ff in
          let name_tt, closure_tt, pp_pack, pp_unpack =
            build_closure ctx "then" fv_closure [] ifb
          in
          let name_ff, closure_ff, _, _ =
            build_closure ctx "else" fv_closure [] elseb
          in
          fprintf ff "@[<v 0>%a@,%a@,%a = lax_cond(@[%a,@ %a, %a,@ %a@])@]"
            closure_tt kind
            closure_ff kind
            pp_unpack ()
            (trans_expr ctx) cond
            trans_name name_tt
            trans_name name_ff
            pp_pack ()
      | CtrlNympyro -> assert false
      end
  | While (cond, body) ->
      let ctx' = ctx_enter_loop ctx "genid()" body in
      let kind =
        match ctx.ctx_backend with
        | Numpyro ->
          begin match ctx.ctx_block with
          | Some Model -> CtrlLax
          | _ -> CtrlPython
          end
        | Pyro | Pyro_cuda -> CtrlPython
      in
      begin match kind with
      | CtrlPython ->
          fprintf ff "@[<v4>while %a:@,%a@]"
            (trans_expr ctx) cond
            (trans_stmt ctx') body
      | CtrlLax ->
          let body_name, closure, pp_pack, pp_unpack =
            build_closure ctx' "while"
              (free_updated SSet.empty body)
              [] body
          in
          let cond_name = gen_name "cond" in
          let pp_cond ff kind =
            let acc_name = gen_name "acc" in
            fprintf ff "%a@[<v 4>def %a(%a):@,%a = %a@,return %a@]"
              print_jit kind
              trans_name cond_name
              trans_name acc_name
              pp_unpack () trans_name acc_name
              (trans_expr ctx) cond
          in
          fprintf ff
            "@[<v 0>%a@,%a@,%a = lax_while_loop(@[%a,@ %a,@ %a@])@]"
            pp_cond kind
            closure kind
            pp_unpack ()
            trans_name cond_name trans_name body_name pp_pack ()
      | CtrlNympyro -> assert false
      end
  | For {loop_variable; lower_bound; upper_bound; loop_body} ->
      let ctx' = ctx_enter_loop ctx loop_variable.name loop_body in
      let kind =
        match ctx.ctx_backend with
        | Numpyro ->
            begin match ctx.ctx_block, ctx.ctx_loops with
            | Some Model, [] ->
               if is_pure loop_body then CtrlLax
               else CtrlNympyro
            | _ -> CtrlLax
            end
        | Pyro | Pyro_cuda -> CtrlPython
      in
      begin match kind with
      | CtrlPython ->
          fprintf ff "@[<v 4>for %a in range(%a,%a + 1):@,%a@]"
            trans_id loop_variable
            (trans_expr ctx) lower_bound
            (trans_expr ctx) upper_bound
            (trans_stmt ctx') loop_body
      | CtrlLax | CtrlNympyro ->
          let body_name, closure, pp_pack, pp_unpack =
            build_closure ctx' "fori"
              (free_updated (SSet.singleton loop_variable.name) loop_body)
              [loop_variable.name] loop_body
          in
          let fori_loop =
            match kind with
            | CtrlLax -> "lax_fori_loop"
            | CtrlNympyro -> "fori_loop"
            | CtrlPython -> assert false
          in
          fprintf ff
            "@[<v 0>%a@,%a = %s(@[%a,@ %a + 1,@ %a,@ %a@])@]"
            closure kind
            pp_unpack ()
            fori_loop
            (trans_expr ctx) lower_bound
            (trans_expr ctx) upper_bound
            trans_name body_name
            pp_pack ()
      end
  | ForEach (loop_variable, iteratee, body) ->
      let ctx' = ctx_enter_loop ctx loop_variable.name body in
      let kind =
        match ctx.ctx_backend with
        | Numpyro ->
            begin match ctx.ctx_block, ctx.ctx_loops with
            | Some Model, [] ->
               if is_pure body then CtrlLax
               else CtrlNympyro
            | _ -> CtrlLax
            end
        | Pyro | Pyro_cuda -> CtrlPython
      in
      begin match kind with
      | CtrlPython ->
        fprintf ff "@[<v4>for %a in %a:@,%a@]"
          trans_id loop_variable
          (trans_expr ctx) iteratee
          (trans_stmt ctx') body
      | CtrlLax | CtrlNympyro ->
          let body_name, closure, pp_pack, pp_unpack =
            build_closure ctx' "for"
              (free_updated (SSet.singleton loop_variable.name) body)
              [loop_variable.name] body
          in
          let foreach_loop =
            match kind with
            | CtrlLax -> "lax_foreach_loop"
            | CtrlNympyro -> "foreach_loop"
            | CtrlPython -> assert false
          in
          fprintf ff
            "@[<v 0>%a@,%a = %s(@[%a,@ %a,@ %a@])@]"
            closure kind
            pp_unpack ()
            foreach_loop
            trans_name body_name
            (trans_expr ctx) iteratee
            pp_pack ()
      end
  | FunDef _ ->
      raise_s
        [%message
          "Found function definition statement outside of function block"]
  | VarDecl {identifier; initial_value; decl_type; _ } ->
      let ctx = set_to_clone ctx in
      fprintf ff "%a = %a"
        trans_id identifier
        (trans_expr_opt ctx decl_type) initial_value
  | Block stmts ->
      fprintf ff "%a" (print_list_newline (trans_stmt ctx)) stmts
  | Return e ->
      fprintf ff "return %a" (trans_expr ctx) e
  | ReturnVoid ->
      fprintf ff "return"
  | Break ->
      fprintf ff "break"
  | Continue ->
      fprintf ff "continue"
  | Skip ->
      fprintf ff "pass"

and ctx_enter_loop ctx i body =
  { ctx with
    ctx_loops = i::ctx.ctx_loops;
    ctx_to_clone_vars = get_updated_arrays_stmt ctx.ctx_to_clone_vars body }

and build_closure ctx fun_name fv args stmt =
  let fun_name = gen_name fun_name in
  let fv = SSet.to_list fv in
  let acc_name = gen_name "acc" in
  let pp_pack ff () =
    match fv with
    | [] -> fprintf ff "None"
    | [ x ] -> fprintf ff "%a" trans_name x
    | vars ->
        fprintf ff "(%a)" (print_list_comma trans_name) vars
  in
  let pp_unpack ff () =
    match fv with
    | [] -> fprintf ff "_"
    | [ x ] -> fprintf ff "%a" trans_name x
    | vars ->
        fprintf ff "(%a)" (print_list_comma trans_name) vars
  in
  let pp_destruct ff () =
    match fv with
    | [] -> pp_print_nothing ff ()
    | _ :: _ -> fprintf ff "%a = %a@," pp_unpack () trans_name acc_name
  in
  let pp_closure ff kind =
    fprintf ff "@[<v 0>%a@[<v 4>def %a(%a%s%a):@,%a%a@,return %a@]@]"
      print_jit kind
      trans_name fun_name
      (print_list_comma pp_print_string) args
      (if args = [] then "" else ", ")
      trans_name acc_name
      pp_destruct ()
      (trans_stmt ctx) stmt
      pp_pack ()
  in
  fun_name, pp_closure, pp_pack, pp_unpack

and is_pure stmt =
  let rec is_pure acc s =
    match s.stmt with
    | TargetPE _ | IncrementLogProb _ | Tilde _ -> false
    | _ ->
      fold_statement (fun b _ -> b)
        is_pure (fun b _ -> b) (fun b _ -> b) acc s.stmt
  in
  is_pure true stmt

and print_jit ff kind =
  match kind with
  | CtrlLax -> fprintf ff "%s@," "@jit"
  | _ -> ()

let trans_stmts ctx ff stmts =
  fprintf ff "%a" (print_list_newline (trans_stmt ctx)) stmts

let trans_fun_def ctx ff (ts : typed_statement) =
  match ts.stmt with
  | FunDef {funname; arguments; body; _} ->
      fprintf ff "@[<v 0>@[<v 4>def %a(%a):@,%a@]@,@]"
        trans_id funname trans_args arguments (trans_stmt ctx) body
  | _ ->
      raise_s
        [%message "Found non-function definition statement in function block"]

let trans_functionblock ctx ff functionblock =
  fprintf ff "@[<v 0>%a@,@]"
    (print_list_newline (trans_fun_def ctx)) functionblock

let trans_block_as_args ff block =
  match get_var_decl_names_block block with
  | [] -> ()
  | args -> fprintf ff "*, %a" (print_list_comma trans_id) args

let trans_block_as_return ?(with_rename=false) ff block =
  fprintf ff "return { %a }"
    (print_list_comma
       (fun ff x ->
          if with_rename then
            fprintf ff "'%a': %a" trans_id x trans_id x
          else
            fprintf ff "'%s': %a" x.name trans_id x))
    (get_var_decl_names_block block)

let trans_block_as_unpack name fvs ff block =
  let unpack ff x = fprintf ff "%a = %s['%s']" trans_id x name x.name in
  let args = get_var_decl_names_block block in
  let used_args = List.filter ~f:(fun x -> SSet.mem fvs x.name) args in
  fprintf ff "%a" (print_list_newline ~eol:true unpack) used_args

let convert_input ff stmt =
  match stmt.stmt with
  | VarDecl {decl_type; identifier; _} ->
      begin match decl_type with
      | Type.Sized SInt | Unsized UInt ->
          fprintf ff "%a = inputs['%s']" trans_id identifier identifier.name
      | Sized SReal | Unsized UReal
      | Sized (SVector _) | Unsized UVector
      | Sized (SRowVector _) | Unsized URowVector
      | Sized (SMatrix _) | Unsized UMatrix
      | Sized (SArray (_, _)) | Unsized (UArray _) ->
          fprintf ff "%a = array(inputs['%s'], dtype=%a)"
            trans_id identifier identifier.name
            dtype_of_type decl_type
      | Type.Unsized (UMathLibraryFunction | UFun _) ->
          raise_s [%message "Unexpected input type: "
              (decl_type: typed_expression Type.t)]
      end
  | _ -> ()

let convert_inputs ff odata =
  Option.iter
    ~f:(fun data -> print_list_newline convert_input ff data)
    odata

let trans_datablock ff data =
  fprintf ff
    "@[<v 0>@[<v 4>def convert_inputs(inputs):@,%a@,%a@]@,@,@]"
    convert_inputs data
    (trans_block_as_return ~with_rename:true) data

let trans_networks_as_arg ff networks =
  match networks with
  | None -> ()
  | Some nets ->
      fprintf ff ", ";
      pp_print_list
        ~pp_sep:(fun ff () -> fprintf ff ", ")
        (fun ff net -> fprintf ff "%s" net.net_id.name)
        ff nets

let trans_prior ctx (decl_type: typed_expression Type.t) ff transformation =
  match transformation with
  | Program.Identity ->
      fprintf ff "improper_uniform(shape=%a)" (trans_dims ctx) decl_type
  | Lower lb ->
      fprintf ff "lower_constrained_improper_uniform(%a, shape=%a)"
        (trans_expr ctx) lb (trans_dims ctx) decl_type
  | Upper ub ->
      fprintf ff "upper_constrained_improper_uniform(%a, shape=%a)"
        (trans_expr ctx) ub (trans_dims ctx) decl_type
  | LowerUpper (lb, ub) ->
      if is_tensor decl_type then
        fprintf ff "uniform(%a * ones(%a), %a)"
          (trans_expr ctx) lb
          (trans_dims ctx) decl_type
          (trans_expr ctx) ub
      else
        fprintf ff "uniform(%a, %a)"
          (trans_expr ctx) lb
          (trans_expr ctx) ub
  | Simplex ->
      fprintf ff "simplex_constrained_improper_uniform(shape=%a)"
        (trans_dims ctx) decl_type
  | UnitVector ->
      fprintf ff "unit_constrained_improper_uniform(shape=%a)"
        (trans_dims ctx) decl_type
  | Ordered ->
      fprintf ff "ordered_constrained_improper_uniform(shape=%a)"
        (trans_dims ctx) decl_type
  | PositiveOrdered ->
      fprintf ff "positive_ordered_constrained_improper_uniform(shape=%a)"
        (trans_dims ctx) decl_type
  | CholeskyCorr ->
      fprintf ff "cholesky_factor_corr_constrained_improper_uniform(shape=%a)"
        (trans_dims ctx) decl_type
  | CholeskyCov ->
      fprintf ff "cholesky_factor_cov_constrained_improper_uniform(shape=%a)"
        (trans_dims ctx) decl_type
  | Covariance ->
      fprintf ff "cov_constrained_improper_uniform(shape=%a)"
        (trans_dims ctx) decl_type
  | Correlation ->
      fprintf ff "corr_constrained_improper_uniform(shape=%a)"
        (trans_dims ctx) decl_type
  | Offset e ->
      fprintf ff "offset_constrained_improper_uniform(%a, shape=%a)"
        (trans_expr ctx) e
        (trans_dims ctx) decl_type
  | Multiplier e ->
      fprintf ff "multiplier_constrained_improper_uniform(%a, shape=%a)"
        (trans_expr ctx) e
        (trans_dims ctx) decl_type
  | OffsetMultiplier (e1, e2) ->
      fprintf ff "offset_multiplier_constrained_improper_uniform(%a, %a, shape=%a)"
        (trans_expr ctx) e1
        (trans_expr ctx) e2
        (trans_dims ctx) decl_type

let trans_block ?(eol=true) comment ctx ff block =
  Option.iter
    ~f:(fun stmts ->
          fprintf ff "@[<v 0># %s@,%a@]%a"
            comment (trans_stmts ctx) stmts
            (if eol then pp_print_cut else pp_print_nothing) ())
    block

let trans_transformeddatablock ctx data ff transformeddata =
  if transformeddata <> None then begin
    fprintf ff
      "@[<v 0>@[<v 4>def transformed_data(%a):@,%a%a@]@,@,@]"
      trans_block_as_args data
      (trans_block "Transformed data" ctx) transformeddata
      (trans_block_as_return ~with_rename:true) transformeddata
  end

let trans_nn_parameter ctx ff p =
  match p.stmt with
  | VarDecl {identifier; initial_value = None; decl_type; transformation; _} ->
      let decl_type =
        match decl_type with
        | Type.Sized (SizedType.SReal) ->
            let meta = { loc = Location_span.empty;
                         ad_level = DataOnly;
                         type_ = UArray UReal; }
            in
            let size = { name="size"; id_loc=Location_span.empty; path=None } in
            let shape =
              { expr= FunApp(StanLib, size,
                             [{ expr= Variable { identifier with path = None };
                                emeta= meta }]);
                emeta= meta }
            in
            Type.Sized (SizedType.SArray (SReal, shape))
        | t -> t
      in
      Hashtbl.add_exn ctx.ctx_params ~key:identifier.name ~data:();
      fprintf ff "%a = %a" trans_id identifier
        (trans_prior ctx decl_type) transformation
  | _ -> assert false


let trans_networks_priors ctx networks data tdata ff parameters =
  let trans_nn_prior ctx network_name args ff network_parameters =
    fprintf ff "@[<v 4>def prior_%s(%a%a):@,"
      network_name
      trans_block_as_args args
      trans_networks_as_arg networks;
    fprintf ff "%a = {}@,%a" trans_nn_base network_name
      (print_list_newline ~eol:true (trans_nn_parameter ctx))
      network_parameters;
    fprintf ff "return random_module('%s', %a, %a)()@]@,"
      network_name trans_name network_name
      trans_nn_base network_name
  in
  let args = Option.merge ~f:(@) data tdata in
  let _, nparameters, _ =
    split_networks networks parameters
  in
  fprintf ff "%a"
    (print_list_newline ~eol:true
       (fun ff (name, params) -> trans_nn_prior ctx name args ff params))
    nparameters

let trans_networks_parameters networks data tdata ff (network_name, _) =
  let args = Option.merge ~f:(@) data tdata in
  fprintf ff "%a = prior_%s(%a%s%a)@,"
    trans_name network_name network_name
    (print_list_comma (fun ff id -> fprintf ff "%a=%a" trans_id id trans_id id))
    (get_var_decl_names_block args)
    (if networks <> None && args <> None then ", " else "")
    (print_list_comma
       (fun ff nn -> fprintf ff "%a=%a"
           trans_id nn.net_id trans_id nn.net_id))
    (Option.value ~default:[] networks);
  fprintf ff "%a = dict(%a.named_parameters())"
    trans_nn_base network_name trans_name network_name

let trans_parameter ctx ff p =
  match p.stmt with
  | VarDecl {identifier; initial_value = None; decl_type; transformation; _} ->
    Hashtbl.add_exn ctx.ctx_params ~key:identifier.name ~data:();
    fprintf ff "%a = sample('%s', %a)" trans_id identifier identifier.name
      (trans_prior ctx decl_type) transformation
  | _ -> assert false

let trans_parametersblock ctx networks data tdata ff (networks_params, params) =
  match ctx.ctx_mode with
  | Generative ->
      if networks_params <> [] || params <> [] then
        raise_s [%message "Non generative feature"]
  | Comprehensive | Mixed ->
      fprintf ff "# Parameters@,%a%a"
        (print_list_newline ~eol:true
           (trans_networks_parameters networks data tdata)) networks_params
        (print_list_newline ~eol:true (trans_parameter ctx)) params

let register_network networks ff ostmts =
  Option.iter
    ~f:(fun stmts ->
        let nets = get_networks_calls networks stmts in
        if not (SSet.is_empty nets) then
          fprintf ff "# Networks@,%a@,"
            (print_list_newline
               (fun ff net -> fprintf ff "register_network('%s', %a)"
                   net trans_name net)) (SSet.to_list nets))
    ostmts

let trans_modelblock ctx networks data tdata parameters tparameters ff model =
  let ctx = { ctx with ctx_block = Some Model } in
  let rnetworks, nparameters, parameters =
    split_networks networks parameters
  in
  match ctx.ctx_mode with
  | Comprehensive ->
      fprintf ff "@[<v 4>def model(%a%a):@,%a%a%a%a@]@,@,@."
        trans_block_as_args (Option.merge ~f:(@) data tdata)
        trans_networks_as_arg networks
        (register_network rnetworks) model
        (trans_parametersblock ctx networks data tdata)
        (nparameters, parameters)
        (trans_block "Transformed parameters" ctx) tparameters
        (trans_block ~eol:false "Model" ctx) model
  | Generative ->
      let rem_parameters, model_ext =
        match Option.merge ~f:(@) tparameters model with
        | None -> parameters, None
        | Some model ->
          let model = moveup_observes model in
          let parameters, model = push_priors parameters model in
          let model = moveup_observes model in
          let parameters, model = push_priors parameters model in
          parameters, Some model
      in
      fprintf ff "@[<v 4>def model(%a%a):@,%a%a%a@]@,@,@."
        trans_block_as_args (Option.merge ~f:(@) data tdata)
        trans_networks_as_arg networks
        (register_network rnetworks) model
        (trans_parametersblock ctx networks data tdata)
        (nparameters, rem_parameters)
        (trans_block ~eol:false "Model" ctx) model_ext
  | Mixed ->
      let rem_parameters, model_ext =
        match Option.merge ~f:(@) tparameters model with
        | None -> parameters, None
        | Some model ->
          let model = moveup_observes model in
          let parameters, model = push_priors parameters model in
          let model = moveup_observes model in
          let parameters, model = push_priors parameters model in
          parameters, Some model
      in
      fprintf ff "@[<v 4>def model(%a%a):@,%a%a%a@]@,@,@."
        trans_block_as_args (Option.merge ~f:(@) data tdata)
        trans_networks_as_arg networks
        (register_network rnetworks) model
        (trans_parametersblock ctx networks data tdata)
        (nparameters, rem_parameters)
        (trans_block ~eol:false "Model" ctx) model_ext

let trans_generatedquantitiesblock ctx data tdata params tparams ff
    genquantities =
  let ctx = { ctx with ctx_block = Some GeneratedQuantities } in
  if tparams <> None || genquantities <> None then begin
    let fvs =
      Option.value_map ~default:SSet.empty
        ~f:(List.fold_left ~init:SSet.empty
              ~f:(fun acc s -> SSet.union (free_vars SSet.empty s) acc))
        (Option.merge ~f:(@) tparams genquantities)
    in
    fprintf ff
      "@[<v 0>@,@[<v 4>def generated_quantities(__inputs__):@,%a%a%a%a"
      (trans_block_as_unpack "__inputs__" fvs)
      Option.(merge ~f:(@) data (merge ~f:(@) tdata params))
      (trans_block "Transformed parameters" ctx) tparams
      (trans_block "Generated quantities" ctx) genquantities
      (trans_block_as_return ~with_rename:false)
      (Option.merge ~f:(@) tparams genquantities);
    fprintf ff "@]@,@]"
  end

let trans_guide_parameter ctx ff p =
  let ctx =
    { ctx with ctx_mode = Generative }
  in
  match p.stmt with
  | VarDecl {identifier; initial_value = None; decl_type; transformation; _} ->
    fprintf ff "%a = param('%s', %a.sample())"
      trans_id identifier identifier.name
      (trans_prior ctx decl_type) transformation
  | VarDecl {identifier; initial_value = Some e; decl_type; _} ->
    if is_real decl_type then
      fprintf ff "%a = param('%s', array(%a))"
        trans_id identifier identifier.name
        (trans_expr ctx) e
    else
      fprintf ff "%a = param('%s', %a)"
        trans_id identifier identifier.name
        (trans_expr ctx) e
  | _ -> assert false

let trans_guideparametersblock ctx ff guide_parameters =
  Option.iter
    ~f:(fprintf ff "# Guide Parameters@,%a@,"
          (print_list_newline (trans_guide_parameter ctx)))
    guide_parameters

let trans_guideblock ctx networks data tdata parameters
    guide_parameters ff guide =
  let rnetworks, nparameters, parameters =
    split_networks networks parameters
  in
  let ctx =
    { ctx with
      ctx_mode = Generative;
      ctx_block = Some Guide;
      ctx_params = Hashtbl.create (module String) }
  in
  if guide_parameters <> None || guide <> None then begin
    fprintf ff "@[<v 4>def guide(%a%a):@,%a%a%a%areturn { %a%s%a }@]@,"
      trans_block_as_args (Option.merge ~f:(@) data tdata)
      trans_networks_as_arg networks
      (register_network rnetworks) guide
      (trans_guideparametersblock ctx) guide_parameters
      (print_list_newline ~eol:true
         (fun ff (nn, _) -> fprintf ff "%a = {}" trans_nn_base nn))
      nparameters
      (trans_block "Guide" ~eol:true ctx) guide
      (print_list_comma
         (fun ff (nn, _) -> fprintf ff "'%s': random_module('%s', %a, %a)()"
             nn nn trans_name nn trans_nn_base nn))
      nparameters
      ((if nparameters <> [] then ", " else ""))
      (print_list_comma
         (fun ff x -> fprintf ff "'%s': %a" x.name trans_id x))
      (get_var_decl_names parameters)
  end

let pp_imports lib ff funs =
  if List.length funs > 0 then
    fprintf ff "from %s import %a@,"
      lib
      (pp_print_list ~pp_sep:(fun ff () -> fprintf ff ", ") pp_print_string)
      funs

let trans_prog backend mode ff (p : typed_program) =
  let p = simplify_program p in
  let ctx =
    { ctx_prog = p
    ; ctx_backend = backend
    ; ctx_mode = mode
    ; ctx_block = None
    ; ctx_loops = []
    ; ctx_mutation = false
    ; ctx_params = Hashtbl.create (module String)
    ; ctx_to_clone = false
    ; ctx_to_clone_vars = SSet.empty }
  in
  let runtime =
    match backend with
    | Pyro -> "pyro"
    | Numpyro -> "numpyro"
    | Pyro_cuda -> "pyro_cuda"
  in
  let dppllib =
    match backend with
    | Pyro -> pyro_dppllib
    | Numpyro -> numpyro_dppllib
    | Pyro_cuda -> pyro_dppllib
  in
  let dppllib =
    if p.networksblock <> None then dppllib @ dppllib_networks else dppllib
  in
  fprintf ff "@[<v 0>%a%a%a@,@]"
    (pp_imports ("runtimes."^runtime^".distributions")) ["*"]
    (pp_imports ("runtimes."^runtime^".dppllib")) dppllib
    (pp_imports ("runtimes."^runtime^".stanlib"))
    (SSet.to_list (get_stanlib_calls p));
  Option.iter ~f:(trans_functionblock ctx ff) p.functionblock;
  fprintf ff "%a" trans_datablock p.datablock;
  fprintf ff "%a"
    (trans_transformeddatablock ctx p.datablock) p.transformeddatablock;
  fprintf ff "%a"
    (trans_networks_priors ctx
       p.networksblock p.datablock p.transformeddatablock)
    p.parametersblock;
  fprintf ff "%a"
    (trans_modelblock ctx
       p.networksblock p.datablock p.transformeddatablock
       p.parametersblock p.transformedparametersblock)
    p.modelblock;
  fprintf ff "%a"
    (trans_guideblock ctx
       p.networksblock p.datablock p.transformeddatablock p.parametersblock
       p.guideparametersblock)
    p.guideblock;
  fprintf ff "%a"
    (trans_generatedquantitiesblock ctx
       p.datablock p.transformeddatablock
       p.parametersblock p.transformedparametersblock)
    p.generatedquantitiesblock;

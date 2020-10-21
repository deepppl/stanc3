open Core_kernel
open Ast
open Middle
open Format

module SSet = Set.Make(String)

type backend =
  | Pyro
  | Numpyro

type mode =
  | Comprehensive
  | Generative
  | Mixed

type context =
  { ctx_prog: typed_program
  ; ctx_backend: backend
  ; ctx_mode: mode
  ; ctx_loops: string list
  ; ctx_params: (string, unit) Hashtbl.t
  ; ctx_to_clone: bool
  ; ctx_to_clone_vars: SSet.t }

let set_to_clone ctx =
  { ctx with ctx_to_clone = true }

let unset_to_clone ctx =
  { ctx with ctx_to_clone = false }

let print_warning loc message =
  Fmt.pf Fmt.stderr
    "@[<v>@,Warning: %s:@,%s@]@."
    (Location.to_string loc.Location_span.begin_loc) message

let print_list_comma printer ff l =
  fprintf ff "@[<hov 0>%a@]"
    (pp_print_list ~pp_sep:(fun ff () -> fprintf ff ",@ ") printer)
    l

let print_list_newline printer ff l =
  fprintf ff "@[<v 0>%a@]"
  (pp_print_list ~pp_sep:(fun ff () -> fprintf ff "@,") printer)
  l

let pyro_dppllib =
  [ "sample"; "param"; "observe"; "factor"; "array"; "zeros"; "ones"; "empty";
    "matmul"; "true_divide"; "floor_divide"; "transpose";
    "dtype_long"; "dtype_float"; "register_network"; ]
let numpyro_dppllib =
  "ops_index" :: "ops_index_update" :: pyro_dppllib

let distribution =
  [ "improper_uniform";
    "lower_constrained_improper_uniform";
    "upper_constrained_improper_uniform";
    "simplex_constrained_improper_uniform";
    "unit_constrained_improper_uniform";
    "ordered_constrained_improper_uniform";
    "positive_ordered_constrained_improper_uniform";
    "cholesky_factor_corr_constrained_improper_uniform";
    (* 19 Continuous Distributions on [0, 1] *)
    (* 19.1 Beta Distribution *)
    "beta";
    "beta_lpdf";
    "beta_cdf";
    "beta_lcdf";
    "beta_lccdf";
    "beta_rng";
    (* 12 Binary Distributions *)
    (* 12.1 Bernoulli Distribution *)
    "bernoulli";
    "bernoulli_lpmf";
    "bernoulli_cdf";
    "bernoulli_lcdf";
    "bernoulli_lccdf";
    "bernoulli_rng";
    (* 12.2 Bernoulli Distribution, Logit Parameterization *)
    "bernoulli_logit";
    "bernoulli_logit_lpmf";
    (* 13 Bounded Discrete Distributions *)
    (* 13.2 Binomial Distribution, Logit Parameterization *)
    "binomial_logit";
    "binomial_logit_lpmf";
    (* 13.5 Categorical Distribution *)
    "categorical";
    "categorical_lpmf";
    "categorical_rng";
    "categorical_logit";
    "categorical_logit_lpmf";
    "categorical_logit_rng";
    (* 14 Unbounded Discrete Distributions *)
    (* 14.2 Negative Binomial Distribution (alternative parameterization) *)
    "neg_binomial_2";
    "neg_binomial_2_lpmf";
    "neg_binomial_2_cdf";
    "neg_binomial_2_lcdf";
    "neg_binomial_2_lccdf";
    "neg_binomial_2_rng";
    (* 14.5 Poisson Distribution *)
    "poisson";
    "poisson_lpmf";
    "poisson_cdf";
    "poisson_lcdf";
    "poisson_lccdf";
    "poisson_rng";
    (* 14.6 Poisson Distribution, Log Parameterization *)
    "poisson_log";
    "poisson_log_lpmf";
    "poisson_log_rng";
    (* 16 Unbounded Continuous Distributions *)
    (* 16.1 Normal Distribution *)
    "normal";
    "normal_lpdf";
    "normal_cdf";
    "normal_lcdf";
    "normal_lccdf";
    "normal_rng";
    "std_normal";
    "std_normal_lpdf";
    "std_normal_cdf";
    "std_normal_lcdf";
    "std_normal_lccdf";
    "std_normal_rng";
    (* 16.5 Student-T Distribution *)
    "student_t";
    "student_t_lpdf";
    "student_t_cdf";
    "student_t_lcdf";
    "student_t_lccdf";
    "student_t_rng";
    (* 16.6 Cauchy Distribution *)
    "cauchy";
    "cauchy_lpdf";
    "cauchy_cdf";
    "cauchy_lcdf";
    "cauchy_lccdf";
    "cauchy_rng";
    (* 16.7 Double Exponential (Laplace) Distribution *)
    "double_exponential";
    "double_exponential_lpdf";
    "double_exponential_cdf";
    "double_exponential_lcdf";
    "double_exponential_lccdf";
    "double_exponential_rng";
    (* 16.8 Logistic Distribution *)
    "logistic";
    "logistic_lpdf";
    "logistic_cdf";
    "logistic_lcdf";
    "logistic_lccdf";
    "logistic_rng";
    (* 17 Positive Continuous Distributions *)
    (* 17.1 Lognormal Distribution *)
    "lognormal";
    "lognormal_lpdf";
    "lognormal_cdf";
    "lognormal_lcdf";
    "lognormal_lccdf";
    "lognormal_rng";
    (* 17.5 Exponential Distribution *)
    "exponential";
    "exponential_lpdf";
    "exponential_cdf";
    "exponential_lcdf";
    "exponential_lccdf";
    "exponential_rng";
    (* 17.6 Gamma Distribution *)
    "gamma";
    "gamma_lpdf";
    "gamma_cdf";
    "gamma_lcdf";
    "gamma_lccdf";
    "gamma_rng";
    (* 17.7 Inverse Gamma Distribution *)
    "inv_gamma";
    "inv_gamma_lpdf";
    "inv_gamma_cdf";
    "inv_gamma_lcdf";
    "inv_gamma_lccdf";
    "inv_gamma_rng";
    (* 18 Positive Lower-Bounded Distributions *)
    (* 18.1 Pareto Distribution *)
    "pareto";
    "pareto_lpdf";
    "pareto_cdf";
    "pareto_lcdf";
    "pareto_lccdf";
    "pareto_rng";
    (* 21 Bounded Continuous Probabilities *)
    (* 21.1 Uniform Distribution *)
    "uniform";
    "uniform_lpdf";
    "uniform_cdf";
    "uniform_lcdf";
    "uniform_lccdf";
    "uniform_rng";
    (* 22 Distributions over Unbounded Vectors *)
    (* 22.1 Multivariate Normal Distribution *)
    "multi_normal";
    "multi_normal_lpdf";
    "multi_normal_rng";
    (* 23 Simplex Distributions *)
    (* 23.1 Dirichlet Distribution *)
    "dirichlet";
    "dirichlet_lpdf";
    "dirichlet_rng";
  ]

let stanlib =
  [ (* 3.2 Mathematical Constants *)
    "pi";
    "e";
    "sqrt2";
    "log2";
    "log10";
    (* 3.3 Special Values *)
    "not_a_number";
    "positive_infinity";
    "negative_infinity";
    "machine_precision";
    (* 3.7 Step-like Functions *)
    "abs_int";
    "abs_real";
    "abs_vector";
    "abs_rowvector";
    "abs_matrix";
    "abs_array";
    "fdim_real_real";
    "fmin_real_real";
    "fmin_int_real";
    "fmin_real_int";
    "fmin_int_int";
    "fmax_real_real";
    "fmax_int_real";
    "fmax_real_int";
    "fmax_int_int";
    "floor_int";
    "floor_real";
    "floor_vector";
    "floor_rowvector";
    "floor_matrix";
    "floor_array";
    "ceil_int";
    "ceil_real";
    "ceil_vector";
    "ceil_rowvector";
    "ceil_matrix";
    "ceil_array";
    "round_int";
    "round_real";
    "round_vector";
    "round_rowvector";
    "round_matrix";
    "round_array";
    "trunc_int";
    "trunc_real";
    "trunc_vector";
    "trunc_rowvector";
    "trunc_matrix";
    "trunc_array";
    (* 3.8 Power and Logarithm Functions *)
    "sqrt_int"; "sqrt_real"; "sqrt_vector"; "sqrt_rowvector";
    "sqrt_matrix"; "sqrt_array";
    "cbrt_int"; "cbrt_real"; "cbrt_vector"; "cbrt_rowvector";
    "cbrt_matrix"; "cbrt_array";
    "square_int"; "square_real"; "square_vector"; "square_rowvector";
    "square_matrix"; "square_array";
    "exp_int"; "exp_real"; "exp_vector"; "exp_rowvector";
    "exp_matrix"; "exp_array";
    "exp2_int"; "exp2_real"; "exp2_vector"; "exp2_rowvector";
    "exp2_matrix"; "exp2_array";
    "log_int"; "log_real"; "log_vector"; "log_rowvector";
    "log_matrix"; "log_array";
    "log2_int"; "log2_real"; "log2_vector"; "log2_rowvector";
    "log2_matrix"; "log2_array";
    "log10_int"; "log10_real"; "log10_vector"; "log10_rowvector";
    "log10_matrix"; "log10_array";
    "pow_int_int"; "pow_int_real"; "pow_real_int"; "pow_real_real";
    "inv_int"; "inv_real"; "inv_vector"; "inv_rowvector";
    "inv_matrix"; "inv_array";
    "inv_sqrt_int"; "inv_sqrt_real"; "inv_sqrt_vector"; "inv_sqrt_rowvector";
    "inv_sqrt_matrix"; "inv_sqrt_array";
    "inv_square_int"; "inv_square_real"; "inv_square_vector";
    "inv_square_rowvector"; "inv_square_matrix"; "inv_square_array";
    "min_array"; "max_array";
    "sum_array"; "prod_array"; "log_sum_exp_array";
    "mean_array"; "variance_array"; "sd_array";
    "distance_vector_vector"; "distance_vector_rowvector";
    "distance_rowvector_vector"; "distance_rowvector_rowvector";
    "squared_distance_vector_vector"; "squared_distance_vector_rowvector";
    "squared_distance_rowvector_vector";
    "squared_distance_rowvector_rowvector";
    (* 3.9 Trigonometric Functions *)
    "hypot_real_real";
    "cos_int";
    "cos_real";
    "cos_vector";
    "cos_rowvector";
    "cos_matrix";
    "cos_array";
    "sin_int";
    "sin_real";
    "sin_vector";
    "sin_rowvector";
    "sin_matrix";
    "sin_array";
    "tan_int";
    "tan_real";
    "tan_vector";
    "tan_rowvector";
    "tan_matrix";
    "tan_array";
    "acos_int";
    "acos_real";
    "acos_vector";
    "acos_rowvector";
    "acos_matrix";
    "acos_array";
    "asin_int";
    "asin_real";
    "asin_vector";
    "asin_rowvector";
    "asin_matrix";
    "asin_array";
    "atan_int";
    "atan_real";
    "atan_vector";
    "atan_rowvector";
    "atan_matrix";
    "atan_array";
    "atan2_real_real";
    (* 3.10 Hyperbolic Trigonometric Functions *)
    "cosh_int";
    "cosh_real";
    "cosh_vector";
    "cosh_rowvector";
    "cosh_matrix";
    "cosh_array";
    "sinh_int";
    "sinh_real";
    "sinh_vector";
    "sinh_rowvector";
    "sinh_matrix";
    "sinh_array";
    "tanh_int";
    "tanh_real";
    "tanh_vector";
    "tanh_rowvector";
    "tanh_matrix";
    "tanh_array";
    "acosh_int";
    "acosh_real";
    "acosh_vector";
    "acosh_rowvector";
    "acosh_matrix";
    "acosh_array";
    "asinh_int";
    "asinh_real";
    "asinh_vector";
    "asinh_rowvector";
    "asinh_matrix";
    "asinh_array";
    "atanh_int";
    "atanh_real";
    "atanh_vector";
    "atanh_rowvector";
    "atanh_matrix";
    "atanh_array";
    (* 3.11 Link Functions *)
    "logit_int";
    "logit_real";
    "logit_vector";
    "logit_rowvector";
    "logit_matrix";
    "logit_array";
    "inv_logit_int";
    "inv_logit_real";
    "inv_logit_vector";
    "inv_logit_rowvector";
    "inv_logit_matrix";
    "inv_logit_array";
    "inv_cloglog_int";
    "inv_cloglog_real";
    "inv_cloglog_vector";
    "inv_cloglog_rowvector";
    "inv_cloglog_matrix";
    "inv_cloglog_array";
    (* 3.14 Composed Functions *)
    "expm1_int";
    "expm1_real";
    "expm1_vector";
    "expm1_rowvector";
    "expm1_matrix";
    "expm1_array";
    "fma_real_real_real";
    "multiply_log_real_real";
    "lmultiply_real_real";
    "log1p_int";
    "log1p_real";
    "log1p_vector";
    "log1p_rowvector";
    "log1p_matrix";
    "log1p_array";
    "log1m_int";
    "log1m_real";
    "log1m_vector";
    "log1m_rowvector";
    "log1m_matrix";
    "log1m_array";
    "log1p_exp_int";
    "log1p_exp_real";
    "log1p_exp_vector";
    "log1p_exp_rowvector";
    "log1p_exp_matrix";
    "log1p_exp_array";
    "log1m_exp_int";
    "log1m_exp_real";
    "log1m_exp_vector";
    "log1m_exp_rowvector";
    "log1m_exp_matrix";
    "log1m_exp_array";
    "log_diff_exp_real_real";
    "log_mix_real_real_real";
    "log_sum_exp_real_real";
    "log_inv_logit_int";
    "log_inv_logit_real";
    "log_inv_logit_vector";
    "log_inv_logit_rowvector";
    "log_inv_logit_matrix";
    "log_inv_logit_array";
    "log1m_inv_logit_int";
    "log1m_inv_logit_real";
    "log1m_inv_logit_vector";
    "log1m_inv_logit_rowvector";
    "log1m_inv_logit_matrix";
    "log1m_inv_logit_array";
    (* 4.1 Reductions *)
    "min_array";
    "max_array";
    "sum_array";
    "prod_array";
    "log_sum_exp_array";
    "mean_array";
    "variance_array";
    "sd_array";
    "distance_vector_vector";
    "distance_vector_rowvector";
    "distance_rowvector_vector";
    "distance_rowvector_rowvector";
    "squared_distance_vector_vector";
    "squared_distance_vector_rowvector";
    "squared_distance_rowvector_vector";
    "squared_distance_rowvector_rowvector";
    (* 4.2 Array Size and Dimension Function *)
    "dims_int"; "dims_real"; "dims_vector"; "dims_rowvector";
    "dims_matrix"; "dims_array";
    "num_elements_array";
    "size_array";
    (* 4.3 Array Broadcasting *)
    "rep_array_int_int";
    "rep_array_real_int";
    "rep_array_int_int_int";
    "rep_array_real_int_int";
    "rep_array_int_int_int_int";
    "rep_array_real_int_int_int";
    (* 5.1 Integer-Valued Matrix Size Functions *)
    "num_elements_vector";
    "num_elements_rowvector";
    "num_elements_matrix";
    "rows_vector";
    "rows_rowvector";
    "rows_matrix";
    "cols_vector";
    "cols_rowvector";
    "cols_matrix";
    (* 5.5 Dot Products and Specialized Products *)
    "dot_product_vector_vector";
    "dot_product_vector_rowvector";
    "dot_product_rowvector_vector";
    "dot_product_rowvector_rowvector";
    "columns_dot_product_vector_vector";
    "columns_dot_product_rowvector_rowvector";
    "columns_dot_product_matrix_matrix";
    "rows_dot_product_vector_vector";
    "rows_dot_product_rowvector_rowvector";
    "rows_dot_product_matrix_matrix";
    "dot_self_vector";
    "dot_self_rowvector";
    "columns_dot_self_vector";
    "columns_dot_self_rowvector";
    "columns_dot_self_matrix";
    "rows_dot_self_vector";
    "rows_dot_self_rowvector";
    "rows_dot_self_matrix";
    "tcrossprod_matrix";
    "crossprod_matrix";
    "quad_form_matrix_matrix";
    "quad_form_matrix_vector";
    "quad_form_diag_matrix_vector";
    "quad_form_diag_matrix_row_vector ";
    "quad_form_sym_matrix_matrix";
    "quad_form_sym_matrix_vector";
    "trace_quad_form_matrix_matrix";
    "trace_gen_quad_form_matrix_matrix_matrix";
    "multiply_lower_tri_self_matrix";
    "diag_pre_multiply_vector_matrix";
    "diag_pre_multiply_rowvector_matrix";
    "diag_post_multiply_matrix_vector";
    "diag_post_multiply_matrix_rowvector";
    (* 5.6 Reductions *)
    "log_sum_exp_vector"; "log_sum_exp_rowvector"; "log_sum_exp_matrix";
    "min_vector"; "min_rowvector"; "min_matrix";
    "max_vector"; "max_rowvector"; "max_matrix";
    "sum_vector"; "sum_rowvector"; "sum_matrix";
    "prod_vector"; "prod_rowvector"; "prod_matrix";
    "mean_vector"; "mean_rowvector"; "mean_matrix";
    "variance_vector"; "variance_rowvector"; "variance_matrix";
    "sd_vector"; "sd_rowvector"; "sd_matrix";
    "rep_vector_real_int"; "rep_vector_int_int";
    "rep_row_vector_real_int"; "rep_row_vector_int_int";
    "rep_matrix_real_int_int"; "rep_matrix_int_int_int";
    "rep_matrix_vector_int";
    "rep_matrix_rowvector_int";
    "col_matrix_int"; "row_matrix_int";
    "block_matrix_int_int_int_int"; "sub_col_matrix_int_int_int";
    "sub_row_matrix_int_int_int";
    "head_vector_int"; "head_rowvector_int"; "head_array_int";
    "tail_vector_int"; "tail_rowvector_int"; "tail_array_int";
    "segment_vector_int_int"; "segment_rowvector_int_int";
    "segment_array_int_int";
    "append_col_matrix_matrix"; "append_col_matrix_vector";
    "append_col_vector_matrix"; "append_col_vector_vector";
    "append_col_rowvector_rowvector"; "append_col_real_rowvector";
    "append_col_int_rowvector"; "append_col_rowvector_real";
    "append_col_rowvector_int";
    "append_row_matrix_matrix"; "append_row_matrix_rowvector";
    "append_row_rowvector_matrix"; "append_row_rowvector_rowvector";
    "append_row_vector_vector"; "append_row_real_vector";
    "append_row_int_vector"; "append_row_vector_real";
    "append_row_vector_int";
    (* 5.8 Diagonal Matrix Functions *)
    "add_diag_matrix_rowvector";
    "add_diag_matrix_vector";
    "add_diag_matrix_real";
    "diagonal_matrix";
    "diag_matrix_vector";
    (* 5.11 Special Matrix Functions *)
    "softmax_vector";
    "softmax_vector";
    "cumulative_sum_array";
    "cumulative_sum_vector";
    "cumulative_sum_rowvector";
    (* 5.12 Covariance Functions *)
    "cov_exp_quad_rowvector_real_real";
    "cov_exp_quad_vector_real_real";
    "cov_exp_quad_array_real_real";
    "cov_exp_quad_rowvector_rowvector_real_real";
    "cov_exp_quad_vector_vector_real_real";
    "cov_exp_quad_array_array_real_real";
    (* 5.13 Linear Algebra Functions and Solvers *)
    "cholesky_decompose_matrix";
    (* 7. Mixed Operations *)
    "to_matrix_matrix";
    "to_matrix_vector";
    "to_matrix_rowvector";
    "to_matrix_matrix_int_int";
    "to_matrix_vector_int_int";
    "to_matrix_rowvector_int_int";
    "to_matrix_matrix_int_int_int";
    "to_matrix_vector_int_int_int";
    "to_matrix_rowvector_int_int_int";
    "to_matrix_array_int_int";
    "to_matrix_array_int_int_int";
    "to_matrix_array";
    "to_vector_matrix";
    "to_vector_vector";
    "to_vector_rowvector";
    "to_vector_array";
    "to_row_vector_matrix";
    "to_row_vector_vector";
    "to_row_vector_rowvector";
    "to_row_vector_array";
    "to_array_2d_matrix";
    "to_array_1d_vector";
    "to_array_1d_rowvector";
    "to_array_1d_matrix";
    "to_array_1d_array";
    (* 9.2 Ordinary Differential Equation (ODE) Solvers *)
    "integrate_ode_rk45_array_real_array_array_array_array";
    "integrate_ode_rk45_array_int_array_array_array_array";
    "integrate_ode_rk45_array_real_array_array_array_array_real_real_int";
    "integrate_ode_rk45_array_int_array_array_array_array_real_real_real";
  ]


let keywords =
  [ "lambda"; "def"; ]

let avoid =
  keywords @ pyro_dppllib @ numpyro_dppllib @ distribution @ stanlib

let trans_id ff id =
  let x =
    if List.mem ~equal:(=) avoid id.name then id.name ^ "__"
    else id.name
  in
  fprintf ff "%s" x

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


let gen_id =
  let cpt = ref 0 in
  fun ?(fresh=true) l ff e ->
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

let rec used_vars_expr acc e =
  let acc =
    match e.expr with
    | Variable x -> SSet.add acc x.name
    | _ -> acc
  in
  fold_expression used_vars_expr (fun acc _ -> acc) acc e.expr

let rec used_vars_lval acc lv =
  match lv.lval with
  | LVariable id -> SSet.add acc id.name
  | _ -> fold_lvalue used_vars_lval used_vars_expr acc lv.lval

let rec used_vars_stmt acc s =
  fold_statement
    used_vars_expr
    used_vars_stmt
    used_vars_lval
    (fun acc _ -> acc)
    acc s.stmt

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
      else if SSet.mem (used_vars_stmt SSet.empty stmt) x then
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

let rec updated_vars_stmt acc s =
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
    (fun acc _ -> acc) updated_vars_stmt (fun acc _ -> acc) (fun acc _ -> acc)
    acc s.stmt

let moveup_stmt stmt stmts =
  let rec moveup (deps, stmt) rev_stmts acc =
    match rev_stmts with
    | [] -> stmt :: acc
    | stmt' :: rev_stmts ->
      let updated_vars = updated_vars_stmt SSet.empty stmt' in
      if SSet.is_empty (SSet.inter deps updated_vars) then
        moveup (deps, stmt) rev_stmts (stmt'::acc)
      else
        List.rev_append rev_stmts (stmt' :: stmt :: acc)
  in
  let used_vars = used_vars_stmt SSet.empty stmt in
  moveup (used_vars, stmt) (List.rev stmts) []

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
      else if SSet.mem (used_vars_stmt SSet.empty stmt) x then
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
  ; networkblock = p.networkblock
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

let get_stanlib_calls program =
  let rec get_stanlib_calls_in_expr acc e =
    let acc =
      match e.expr with
      | FunApp (StanLib, id, args) | CondDistApp (StanLib, id, args) ->
        let sid = stanlib_id id args in
        if List.mem ~equal:(=) stanlib sid then SSet.add acc sid
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

let get_networks_calls networks stmts =
  let rec get_networks_calls_in_expr networks acc e =
    let acc =
      match e.expr with
      | FunApp (_, id, _args) ->
        if List.exists ~f:(fun n -> n.net_id.name = id.name) networks then
          id :: acc
        else acc
      | _ -> acc
    in
    fold_expression
      (get_networks_calls_in_expr networks)
      (fun acc _ -> acc)
      acc e.expr
  in
  let rec get_networks_calls_in_lval networks acc lv =
    fold_lvalue (get_networks_calls_in_lval networks)
      (get_networks_calls_in_expr networks)
      acc lv.lval
  in
  let rec get_networks_calls_in_stmt nets acc s =
    fold_statement
      (get_networks_calls_in_expr nets)
      (get_networks_calls_in_stmt nets)
      (get_networks_calls_in_lval nets)
      (fun acc _ -> acc)
      acc s.stmt
  in
  match networks with
  | Some nets ->
      List.fold_left ~init:[] ~f:(get_networks_calls_in_stmt nets) stmts
  | _ -> []

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

let get_updated_arrays p =
  fold_program get_updated_arrays_stmt SSet.empty p

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
        (if to_clone ctx e then ".clone()" else "")
  | IntNumeral x -> trans_numeral e.emeta.type_ ff x
  | RealNumeral x -> trans_numeral e.emeta.type_ ff x
  | FunApp (fn_kind, id, args) ->
      trans_fun_app ctx fn_kind id ff args
  | CondDistApp (fn_kind, id, args) ->
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
      if to_clone ctx e then
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

and to_clone ctx e =
  let rec can_be_modified e =
    match e.expr with
    | Variable x -> SSet.mem ctx.ctx_to_clone_vars x.name
    | Indexed (lhs, _) ->
      not (SSet.is_empty ctx.ctx_to_clone_vars) &&
      ((* is_unsized_tensor e.emeta.type_ || *) can_be_modified lhs)
    | TernaryIf (_, ifb, elseb) -> can_be_modified ifb || can_be_modified elseb
    | Paren e -> can_be_modified e
    | FunApp (_, _, args) -> List.exists ~f:(to_clone ctx) args
    | _ -> false
  in
  match ctx.ctx_backend with
  | Pyro -> ctx.ctx_to_clone && can_be_modified e
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
      | { lval= LIndexed (_, _indices); _ } as assign_lhs ->
          begin match ctx.ctx_backend with
          | Pyro ->
              let rec trans_lval ff = function
                | { lval= LVariable id; _ } -> trans_id ff id
                | { lval= LIndexed (lhs, indices); _ } ->
                  fprintf ff "%a[%a]" trans_lval lhs
                    (print_list_comma (trans_idx ctx)) indices
              in
              fprintf ff "%a = %a"
                trans_lval assign_lhs
                (trans_rhs (expr_of_lval assign_lhs)) assign_rhs
          | Numpyro ->
              let rec trans_variable ff = function
                | { lval= LVariable id; _ } -> trans_id ff id
                | { lval= LIndexed (lhs, _); _ } -> trans_variable ff lhs
              in
              let trans_updated ff = function
                | { lval= LVariable _; _ } -> assert false
                | { lval= LIndexed ({ lval= LVariable id; _ }, indices); _ } ->
                  fprintf ff "%a, ops_index[%a]" trans_id id
                    (print_list_comma (trans_idx ctx)) indices
                | _ -> assert false (* XXX TODO XXX *)
              in
              fprintf ff "%a = ops_index_update(%a, %a)"
                (* fprintf ff "%a = %a.set(%a)" *)
                trans_variable assign_lhs
                trans_updated assign_lhs
                (trans_rhs (expr_of_lval assign_lhs)) assign_rhs
          end
    end
  | NRFunApp (fn_kind, id, args) ->
      let ctx = set_to_clone ctx in
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
        | Generative | Mixed ->
            begin match arg.expr with
            | Variable {name; _} ->
                let is_parameter =
                  List.exists ~f:(fun x -> x.name = name)
                    (get_var_decl_names_block ctx.ctx_prog.parametersblock)
                in
                if is_parameter &&
                   (ctx.ctx_mode = Generative ||
                    not (Hashtbl.mem ctx.ctx_params name)) then
                  (ignore (Hashtbl.add ctx.ctx_params
                             ~key:name ~data:());
                   true)
                else false
            | _ -> false
            end
      in
      if is_sample then
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
      fprintf ff "@[<v 0>@[<v 4>if %a:@,%a@]@]"
        (trans_expr ctx) cond
        (trans_stmt ctx) ifb
  | IfThenElse (cond, ifb, Some elseb) ->
      fprintf ff "@[<v 0>@[<v 4>if %a:@,%a@]@,@[<v 4>else:@,%a@]@]"
        (trans_expr ctx) cond
        (trans_stmt ctx) ifb
        (trans_stmt ctx) elseb
  | While (cond, body) ->
      let ctx' =
        { ctx with
          ctx_loops = "genid()"::ctx.ctx_loops;
          ctx_to_clone_vars =
            get_updated_arrays_stmt ctx.ctx_to_clone_vars body }
      in
      fprintf ff "@[<v4>while %a:@,%a@]"
        (trans_expr ctx) cond
        (trans_stmt ctx') body
  | For {loop_variable; lower_bound; upper_bound; loop_body} ->
      let ctx' =
        { ctx with
          ctx_loops = loop_variable.name::ctx.ctx_loops;
          ctx_to_clone_vars =
            get_updated_arrays_stmt ctx.ctx_to_clone_vars loop_body }
      in
      fprintf ff "@[<v 4>for %a in range(%a,%a + 1):@,%a@]"
        trans_id loop_variable
        (trans_expr ctx) lower_bound
        (trans_expr ctx) upper_bound
        (trans_stmt ctx') loop_body
  | ForEach (loop_variable, iteratee, body) ->
      let ctx' =
        { ctx with
          ctx_loops = loop_variable.name::ctx.ctx_loops;
          ctx_to_clone_vars =
            get_updated_arrays_stmt ctx.ctx_to_clone_vars body }
      in
      fprintf ff "@[<v4>for %a in %a:@,%a@]"
        trans_id loop_variable
        (trans_expr ctx) iteratee
        (trans_stmt ctx') body
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

let trans_block_as_unpack name ff block =
  let unpack ff x =
    fprintf ff "%a = %s['%s']" trans_id x name x.name
  in
  match get_var_decl_names_block block with
  | [] -> ()
  | args -> fprintf ff "%a@," (print_list_newline unpack) args

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
  | Offset _
  | Multiplier _
  | OffsetMultiplier _ ->
      assert false (* XXX TODO XXX *)
  | CholeskyCov
  | Correlation
  | Covariance ->
      raise_s [%message "Unsupported type constraints"]

let pp_print_nothing _ () = ()

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

let trans_parameter ctx ff p =
  match p.stmt with
  | VarDecl {identifier; initial_value = None; decl_type; transformation; _} ->
    Hashtbl.add_exn ctx.ctx_params ~key:identifier.name ~data:();
    fprintf ff "%a = sample('%s', %a)" trans_id identifier identifier.name
      (trans_prior ctx decl_type) transformation
  | _ -> assert false

let trans_parametersblock ctx ff parameters =
  match ctx.ctx_mode with
  | Generative -> ()
  | Comprehensive | Mixed ->
      Option.iter
        ~f:(fprintf ff "# Parameters@,%a@,"
              (print_list_newline (trans_parameter ctx)))
        parameters

let register_network networks ff ostmts =
  Option.iter
    ~f:(fun stmts ->
        let nets = get_networks_calls networks stmts in
        if nets <> [] then
          fprintf ff "# Networks@,%a@,"
            (print_list_newline
               (fun ff net -> fprintf ff "register_network('%s', %a)"
                   net.name trans_id net)) nets)
    ostmts

let trans_modelblock ctx networks data tdata parameters tparameters ff model =
  match ctx.ctx_mode with
  | Comprehensive | Generative ->
      fprintf ff "@[<v 4>def model(%a%a):@,%a%a%a%a@]@,@,@."
        trans_block_as_args (Option.merge ~f:(@) data tdata)
        trans_networks_as_arg networks
        (register_network networks) model
        (trans_parametersblock ctx) parameters
        (trans_block "Transformed parameters" ctx) tparameters
        (trans_block ~eol:false "Model" ctx) model
  | Mixed ->
      let rem_parameters, model_ext =
        match parameters, (Option.merge ~f:(@) tparameters model) with
        | None, None -> None, None
        | Some parameters, None -> Some parameters, None
        | None, Some model -> None, Some model
        | Some parameters, Some model ->
          let model = moveup_observes model in
          let parameters, model = push_priors parameters model in
          let model = moveup_observes model in
          let parameters, model = push_priors parameters model in
          Some parameters, Some model
      in
      fprintf ff "@[<v 4>def model(%a%a):@,%a%a%a@]@,@,@."
        trans_block_as_args (Option.merge ~f:(@) data tdata)
        trans_networks_as_arg networks
        (register_network networks) model
        (trans_parametersblock ctx) rem_parameters
        (trans_block ~eol:false "Model" ctx) model_ext

let trans_generatedquantitiesblock ctx data tdata params tparams ff
    genquantities =
  if tparams <> None || genquantities <> None then begin
    fprintf ff
      "@[<v 0>@,@[<v 4>def generated_quantities(__inputs__):@,%a%a%a%a"
      (trans_block_as_unpack "__inputs__")
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

let trans_guideblock ctx networks data tdata guide_parameters ff guide =
  let ctx =
    { ctx with
      ctx_mode = Generative;
      ctx_params = Hashtbl.create (module String) }
  in
  if guide_parameters <> None || guide <> None then begin
    fprintf ff "@[<v 4>def guide(%a%a):@,%a%a%a@]@."
      trans_block_as_args (Option.merge ~f:(@) data tdata)
      trans_networks_as_arg networks
      (register_network networks) guide
      (trans_guideparametersblock ctx) guide_parameters
      (trans_block "Guide" ~eol:false ctx) guide
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
    ; ctx_loops = []
    ; ctx_params = Hashtbl.create (module String)
    ; ctx_to_clone = false
    ; ctx_to_clone_vars = SSet.empty }
  in
  let runtime =
    match backend with
    | Pyro -> "pyro"
    | Numpyro -> "numpyro"
  in
  let dppllib =
    match backend with
    | Pyro -> pyro_dppllib
    | Numpyro -> numpyro_dppllib
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
    (trans_modelblock ctx
       p.networkblock p.datablock p.transformeddatablock
       p.parametersblock p.transformedparametersblock)
    p.modelblock;
  fprintf ff "%a"
    (trans_generatedquantitiesblock ctx
       p.datablock p.transformeddatablock
       p.parametersblock p.transformedparametersblock)
    p.generatedquantitiesblock;
  fprintf ff "%a"
    (trans_guideblock ctx
       p.networkblock p.datablock p.transformeddatablock
       p.guideparametersblock)
    p.guideblock;

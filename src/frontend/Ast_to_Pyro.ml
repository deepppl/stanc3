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

type 'a context =
  { ctx_prog: typed_program
  ; ctx_backend: backend
  ; ctx_mode: mode
  ; ctx_ext: 'a }

type stmt_ext =
  { ext_loops: string list
  ; ext_params: (string, unit) Hashtbl.t }

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
  [ "machine_precision";
    (* 3.2 Mathematical Constants *)
    "pi";
    "e";
    "sqrt2";
    "log2";
    "log10";
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
    "dims_int"; "dims_real"; "dims_vector"; "dims_rowvector";
    "dims_matrix"; "dims_array";
    "num_elements_array";
    "size_array";
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

let used_vars_stmt stmt =
  fold_statement_with
    used_vars_expr
    (fun acc _ -> acc)
    used_vars_lval
    (fun acc _ -> acc)
    SSet.empty stmt

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
      else if SSet.mem (used_vars_stmt stmt) x then
        Some decl, stmt :: stmts
      else
        let prior, stmts = push_prior_stmts (x, decl) stmts in
        prior, stmt ::  stmts

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
      else if SSet.mem (used_vars_stmt stmt) x then
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

and push_vardecls_stmt (stmt: typed_statement) =
  match stmt.stmt with
  | Block stmts -> { stmt with stmt = Block (push_vardecls_stmts stmts) }
  | _ -> stmt
      (* map_statement_with *)
      (*   (fun e -> e) (fun m -> m) (fun lv -> lv) (fun f -> f) *)
      (*   stmt *)

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
  p

let get_var_decl_names stmts =
  List.fold_right
    ~f:(fun stmt acc ->
          match stmt.stmt with
          | VarDecl {identifier; _} -> identifier :: acc
          | _ -> acc)
    ~init:[] stmts

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
  let get_stanlib_calls_in_stmt acc stmt =
    fold_statement_with
      get_stanlib_calls_in_expr
      (fun acc _ -> acc)
      get_stanlib_calls_in_lval
      (fun acc _ -> acc)
      acc stmt
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
  match networks with
  | Some nets ->
      List.fold_left ~init:[]
        ~f:(fun acc stmt ->
            fold_statement_with
              (get_networks_calls_in_expr nets)
              (fun acc _ -> acc)
              (get_networks_calls_in_lval nets)
              (fun acc _ -> acc)
              acc stmt)
        stmts
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

let rec trans_expr ctx ff ({expr; emeta }: typed_expression) : unit =
  match expr with
  | Paren x -> fprintf ff "(%a)" (trans_expr ctx) x
  | BinOp (lhs, op, rhs) ->
      fprintf ff "%a" (trans_binop ctx lhs rhs) op
  | PrefixOp (op, e) | PostfixOp (e, op) ->
      fprintf ff "%a" (trans_unop ctx e) op
  | TernaryIf (cond, ifb, elseb) ->
      fprintf ff "%a if %a else %a"
        (trans_expr ctx) ifb (trans_expr ctx) cond (trans_expr ctx) elseb
  | Variable id -> trans_id ff id
  | IntNumeral x -> trans_numeral emeta.type_ ff x
  | RealNumeral x -> trans_numeral emeta.type_ ff x
  | FunApp (fn_kind, id, args) ->
      trans_fun_app ctx fn_kind id ff args
  | CondDistApp (fn_kind, id, args) ->
      trans_cond_dist_app ctx fn_kind id ff args
  | GetLP | GetTarget -> fprintf ff "stanlib.target()" (* XXX TODO XXX *)
  | ArrayExpr eles ->
      fprintf ff "array([%a], dtype=%a)"
        (trans_exprs ctx) eles
        dtype_of_unsized_type emeta.type_
  | RowVectorExpr eles ->
      fprintf ff "array([%a], dtype=%a)"
        (trans_exprs ctx) eles
        dtype_of_unsized_type emeta.type_
  | Indexed (lhs, indices) ->
      begin match ctx.ctx_backend with
      | Pyro ->
          fprintf ff "%a[%a].clone()" (trans_expr ctx) lhs
            (print_list_comma (trans_idx ctx)) indices
      | Numpyro ->
          fprintf ff "%a[%a]" (trans_expr ctx) lhs
            (print_list_comma (trans_idx ctx)) indices
      end

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

let trans_expr_opt ctx (type_ : typed_expression Type.t) ff = function
  | Some e -> trans_expr ctx ff e
  | None ->
      if is_tensor type_ then fprintf ff "ones(%a)" (trans_dims ctx) type_
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

(* These types signal the context for a declaration during statement translation.
   They are only interpreted by trans_decl.*)
(* type constrainaction = Check | Constrain | Unconstrain [@@deriving sexp] *)

(* let constrainaction_fname c =
  Internal_fun.to_string
    ( match c with
    | Check -> FnCheck
    | Constrain -> FnConstrain
    | Unconstrain -> FnUnconstrain ) *)

(* type decl_context =
  {dconstrain: constrainaction option; dadlevel: UnsizedType.autodifftype} *)

(* let check_constraint_to_string t (c : constrainaction) =
  match t with
  | Program.Ordered -> "ordered"
  | PositiveOrdered -> "positive_ordered"
  | Simplex -> "simplex"
  | UnitVector -> "unit_vector"
  | CholeskyCorr -> "cholesky_factor_corr"
  | CholeskyCov -> "cholesky_factor"
  | Correlation -> "corr_matrix"
  | Covariance -> "cov_matrix"
  | Lower _ -> (
    match c with
    | Check -> "greater_or_equal"
    | Constrain | Unconstrain -> "lb" )
  | Upper _ -> (
    match c with Check -> "less_or_equal" | Constrain | Unconstrain -> "ub" )
  | LowerUpper _ -> (
    match c with
    | Check ->
        raise_s
          [%message "LowerUpper is really two other checks tied together"]
    | Constrain | Unconstrain -> "lub" )
  | Offset _ | Multiplier _ | OffsetMultiplier _ -> (
    match c with Check -> "" | Constrain | Unconstrain -> "offset_multiplier" )
  | Identity -> "" *)

(* let constrain_constraint_to_string t (c : constrainaction) =
  match t with
  | Program.CholeskyCorr -> "cholesky_corr"
  | _ -> check_constraint_to_string t c *)

(* let constraint_forl = function
  | Program.Identity | Offset _ | Multiplier _ | OffsetMultiplier _ | Lower _
   |Upper _ | LowerUpper _ ->
      Stmt.Helpers.for_scalar
  | Ordered | PositiveOrdered | Simplex | UnitVector | CholeskyCorr
   |CholeskyCov | Correlation | Covariance ->
      Stmt.Helpers.for_eigen *)

(* let same_shape decl_id decl_var id var meta =
  if UnsizedType.is_scalar_type (Expr.Typed.type_of var) then []
  else
    [ Stmt.
        { Fixed.pattern=
            NRFunApp
              ( StanLib
              , "check_matching_dims"
              , Expr.Helpers.
                  [str "constraint"; str decl_id; decl_var; str id; var] )
        ; meta } ] *)

(* let check_transform_shape decl_id decl_var meta = function
  | Program.Offset e -> same_shape decl_id decl_var "offset" e meta
  | Multiplier e -> same_shape decl_id decl_var "multiplier" e meta
  | Lower e -> same_shape decl_id decl_var "lower" e meta
  | Upper e -> same_shape decl_id decl_var "upper" e meta
  | OffsetMultiplier (e1, e2) ->
      same_shape decl_id decl_var "offset" e1 meta
      @ same_shape decl_id decl_var "multiplier" e2 meta
  | LowerUpper (e1, e2) ->
      same_shape decl_id decl_var "lower" e1 meta
      @ same_shape decl_id decl_var "upper" e2 meta
  | Covariance | Correlation | CholeskyCov | CholeskyCorr | Ordered
   |PositiveOrdered | Simplex | UnitVector | Identity ->
      [] *)

(* let copy_indices indexed (var : Expr.Typed.t) =
  if UnsizedType.is_scalar_type var.meta.type_ then var
  else
    match Expr.Helpers.collect_indices indexed with
    | [] -> var
    | indices ->
        Expr.Fixed.
          { pattern= Indexed (var, indices)
          ; meta=
              { var.meta with
                type_=
                  Expr.Helpers.infer_type_of_indexed var.meta.type_ indices }
          } *)

(* let extract_transform_args var = function
  | Program.Lower a | Upper a -> [copy_indices var a]
  | Offset a ->
      [copy_indices var a; {a with Expr.Fixed.pattern= Lit (Int, "1")}]
  | Multiplier a -> [{a with pattern= Lit (Int, "0")}; copy_indices var a]
  | LowerUpper (a1, a2) | OffsetMultiplier (a1, a2) ->
      [copy_indices var a1; copy_indices var a2]
  | Covariance | Correlation | CholeskyCov | CholeskyCorr | Ordered
   |PositiveOrdered | Simplex | UnitVector | Identity ->
      [] *)

(* let extra_constraint_args st = function
  | Program.Lower _ | Upper _ | Offset _ | Multiplier _ | LowerUpper _
   |OffsetMultiplier _ | Ordered | PositiveOrdered | Simplex | UnitVector
   |Identity ->
      []
  | Covariance | Correlation | CholeskyCorr ->
      [List.hd_exn (SizedType.dims_of st)]
  | CholeskyCov -> SizedType.dims_of st

let param_size transform sizedtype =
  let rec shrink_eigen f st =
    match st with
    | SizedType.SArray (t, d) -> SizedType.SArray (shrink_eigen f t, d)
    | SVector d | SMatrix (d, _) -> SVector (f d)
    | SInt | SReal | SRowVector _ ->
        raise_s
          [%message
            "Expecting SVector or SMatrix, got " (st : Expr.Typed.t SizedType.t)]
  in
  let rec shrink_eigen_mat f st =
    match st with
    | SizedType.SArray (t, d) -> SizedType.SArray (shrink_eigen_mat f t, d)
    | SMatrix (d1, d2) -> SVector (f d1 d2)
    | SInt | SReal | SRowVector _ | SVector _ ->
        raise_s
          [%message "Expecting SMatrix, got " (st : Expr.Typed.t SizedType.t)]
  in
  let k_choose_2 k =
    Expr.Helpers.(binop (binop k Times (binop k Minus (int 1))) Divide (int 2))
  in
  match transform with
  | Program.Identity | Lower _ | Upper _
   |LowerUpper (_, _)
   |Offset _ | Multiplier _
   |OffsetMultiplier (_, _)
   |Ordered | PositiveOrdered | UnitVector ->
      sizedtype
  | Simplex ->
      shrink_eigen (fun d -> Expr.Helpers.(binop d Minus (int 1))) sizedtype
  | CholeskyCorr | Correlation -> shrink_eigen k_choose_2 sizedtype
  | CholeskyCov ->
      (* (N * (N + 1)) / 2 + (M - N) * N *)
      shrink_eigen_mat
        (fun m n ->
          Expr.Helpers.(
            binop
              (binop (k_choose_2 n) Plus n)
              Plus
              (binop (binop m Minus n) Times n)) )
        sizedtype
  | Covariance ->
      shrink_eigen
        (fun k -> Expr.Helpers.(binop k Plus (k_choose_2 k)))
        sizedtype *)

(* let remove_possibly_exn pst action loc =
  match pst with
  | Type.Sized st -> st
  | Unsized _ ->
      raise_s
        [%message
          "Error extracting sizedtype" ~action ~loc:(loc : Location_span.t)] *)

(* let constrain_decl st dconstrain t decl_id decl_var smeta =
  let mkstring = mkstring (Expr.Typed.loc_of decl_var) in
  match Option.map ~f:(constrain_constraint_to_string t) dconstrain with
  | None | Some "" -> []
  | Some constraint_str ->
      let dc = Option.value_exn dconstrain in
      let fname = constrainaction_fname dc in
      let extra_args =
        match dconstrain with
        | Some Constrain -> extra_constraint_args st t
        | _ -> []
      in
      let args var =
        (var :: mkstring constraint_str :: extract_transform_args var t)
        @ extra_args
      in
      let constrainvar var =
        { var with
          Expr.Fixed.pattern= FunApp (CompilerInternal, fname, args var) }
      in
      let unconstrained_decls, decl_id, ut =
        let ut = SizedType.to_unsized (param_size t st) in
        match dconstrain with
        | Some Unconstrain when SizedType.to_unsized st <> ut ->
            ( [ Stmt.Fixed.
                  { pattern=
                      Decl
                        { decl_adtype= DataOnly
                        ; decl_id= decl_id ^ "_free__"
                        ; decl_type= Sized (param_size t st) }
                  ; meta= smeta } ]
            , decl_id ^ "_free__"
            , ut )
        | _ -> ([], decl_id, SizedType.to_unsized st)
      in
      unconstrained_decls
      @ [ (constraint_forl t) st
            (Stmt.Helpers.assign_indexed ut decl_id smeta constrainvar)
            decl_var smeta ] *)

(* let rec check_decl var decl_type' decl_id decl_trans smeta adlevel =
  let decl_type = remove_possibly_exn decl_type' "check" smeta in
  let chk fn var =
    let check_id id =
      let id_str = Expr.Helpers.str (Fmt.strf "%a" Expr.Typed.pp id) in
      let args = extract_transform_args id decl_trans in
      Stmt.Helpers.internal_nrfunapp FnCheck (fn :: id_str :: id :: args) smeta
    in
    [(constraint_forl decl_trans) decl_type check_id var smeta]
  in
  match decl_trans with
  | Identity | Offset _ | Multiplier _ | OffsetMultiplier (_, _) -> []
  | LowerUpper (lb, ub) ->
      check_decl var decl_type' decl_id (Lower lb) smeta adlevel
      @ check_decl var decl_type' decl_id (Upper ub) smeta adlevel
  | _ -> chk (mkstring smeta (check_constraint_to_string decl_trans Check)) var *)

(* let check_sizedtype name =
  let check x = function
    | {Expr.Fixed.pattern= Lit (Int, i); _} when float_of_string i >= 0. -> []
    | n ->
        [ Stmt.Helpers.internal_nrfunapp FnValidateSize
            Expr.Helpers.
              [str name; str (Fmt.strf "%a" Pretty_printing.pp_expression x); n]
            n.meta.loc ]
  in
  let rec sizedtype = function
    | SizedType.(SInt | SReal) as t -> ([], t)
    | SVector s ->
        let e = trans_expr s in
        (check s e, SizedType.SVector e)
    | SRowVector s ->
        let e = trans_expr s in
        (check s e, SizedType.SRowVector e)
    | SMatrix (r, c) ->
        let er = trans_expr r in
        let ec = trans_expr c in
        (check r er @ check c ec, SizedType.SMatrix (er, ec))
    | SArray (t, s) ->
        let e = trans_expr s in
        let ll, t = sizedtype t in
        (check s e @ ll, SizedType.SArray (t, e))
  in
  function
  | Type.Sized st ->
      let ll, st = sizedtype st in
      (ll, Type.Sized st)
  | Unsized ut -> ([], Unsized ut) *)

(* let trans_decl (*{dconstrain; dadlevel}*) smeta decl_type transform identifier
    initial_value =
  let decl_id = identifier.name in
  let rhs = Option.map ~f:trans_expr initial_value in
  let size_checks, dt = check_sizedtype identifier.name decl_type in
  let decl_adtype = dadlevel in
  let decl_var =
    Expr.
      { Fixed.pattern= Var decl_id
      ; meta=
          Typed.Meta.create ~adlevel:dadlevel ~loc:smeta
            ~type_:(Type.to_unsized decl_type)
            () }
  in
  let decl =
    Stmt.
      {Fixed.pattern= Decl {decl_adtype; decl_id; decl_type= dt}; meta= smeta}
  in
  let rhs_assignment =
    Option.map
      ~f:(fun e ->
        Stmt.Fixed.
          {pattern= Assignment ((decl_id, e.meta.type_, []), e); meta= smeta}
        )
      rhs
    |> Option.to_list
  in
  if Utils.is_user_ident decl_id then
    let constrain_checks =
      match dconstrain with
      | Some Constrain | Some Unconstrain ->
          raise_s [%message "This should never happen."]
      | Some Check ->
          check_transform_shape decl_id decl_var smeta transform
          @ check_decl decl_var dt decl_id transform smeta dadlevel
      | None -> []
    in
    size_checks @ (decl :: rhs_assignment) @ constrain_checks
  else size_checks @ (decl :: rhs_assignment) *)

(* let unwrap_block_or_skip = function
  | [({Stmt.Fixed.pattern= Block _; _} as b)] | [({pattern= Skip; _} as b)] ->
      b
  | x ->
      raise_s
        [%message "Expecting a block or skip, not" (x : Stmt.Located.t list)]

let dist_name_suffix udf_names name =
  let is_udf_name s = List.exists ~f:(fun (n, _) -> n = s) udf_names in
  match
    Middle.Utils.distribution_suffices
    |> List.filter ~f:(fun sfx ->
           Stan_math_signatures.is_stan_math_function_name (name ^ sfx)
           || is_udf_name (name ^ sfx) )
    |> List.hd
  with
  | Some hd -> hd
  | None -> raise_s [%message "Couldn't find distribution " name] *)

(* let%expect_test "dist name suffix" =
  dist_name_suffix [] "normal" |> print_endline ;
  [%expect {| _lpdf |}] *)

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
      trans_fun_app ctx fn_kind id ff args
  | IncrementLogProb e | TargetPE e ->
      fprintf ff "factor(%a, %a)"
        (gen_id ctx.ctx_ext.ext_loops) e
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
                if Hashtbl.mem ctx.ctx_ext.ext_params name then false
                else (Hashtbl.add_exn ctx.ctx_ext.ext_params ~key:name ~data:();
                      true)
            | _ -> false
          end
      in
      if is_sample then
        fprintf ff "%a = sample(%a, %a)%a"
          (trans_expr ctx) arg
          (gen_id ~fresh:false ctx.ctx_ext.ext_loops) arg
          trans_distribution (distribution, args)
          trans_truncation truncation
      else
        fprintf ff "observe(%a, %a, %a)%a"
          (gen_id ~fresh:true ctx.ctx_ext.ext_loops) arg
          trans_distribution (distribution, args)
          (trans_expr ctx) arg
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
      let ctx_ext' =
        { ctx.ctx_ext with ext_loops = "genid()"::ctx.ctx_ext. ext_loops }
      in
      let ctx' = { ctx with ctx_ext = ctx_ext' } in
      fprintf ff "@[<v4>while %a:@,%a@]"
        (trans_expr ctx) cond
        (trans_stmt ctx') body
  | For {loop_variable; lower_bound; upper_bound; loop_body} ->
      let ctx_ext' =
        { ctx.ctx_ext with
          ext_loops = loop_variable.name::ctx.ctx_ext. ext_loops }
      in
      let ctx' = { ctx with ctx_ext = ctx_ext' } in
      fprintf ff "@[<v 4>for %a in range(%a,%a + 1):@,%a@]"
        trans_id loop_variable
        (trans_expr ctx) lower_bound
        (trans_expr ctx) upper_bound
        (trans_stmt ctx') loop_body
  | ForEach (loop_variable, iteratee, body) ->
      let ctx_ext' =
        { ctx.ctx_ext with
          ext_loops = loop_variable.name::ctx.ctx_ext. ext_loops }
      in
      let ctx' = { ctx with ctx_ext = ctx_ext' } in
      fprintf ff "@[<v4>for %a in %a:@,%a@]"
        trans_id loop_variable
        (trans_expr ctx) iteratee
        (trans_stmt ctx') body
  | FunDef _ ->
      raise_s
        [%message
          "Found function definition statement outside of function block"]
  | VarDecl {identifier; initial_value; decl_type; _ } ->
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

(* let get_block block prog =
  match block with
  | Program.Parameters -> prog.parametersblock
  | TransformedParameters -> prog.transformedparametersblock
  | GeneratedQuantities -> prog.generatedquantitiesblock

let trans_sizedtype_decl declc tr name =
  let check fn x n =
    Stmt.Helpers.internal_nrfunapp fn
      Expr.Helpers.
        [str name; str (Fmt.strf "%a" Pretty_printing.pp_expression x); n]
      n.meta.loc
  in
  let grab_size fn n = function
    | ({expr= IntNumeral i; _}) as s when float_of_string i >= 2. ->
        ([], trans_expr s)
    | ({expr= IntNumeral _; _} | {expr= Variable _; _}) as s ->
        let e = trans_expr s in
        ([check fn s e], e)
    | s ->
        let e = trans_expr s in
        let decl_id = Fmt.strf "%s_%ddim__" name n in
        let decl =
          { Stmt.Fixed.pattern=
              Decl {decl_type= Sized SInt; decl_id; decl_adtype= DataOnly}
          ; meta= e.meta.loc }
        in
        let assign =
          { Stmt.Fixed.pattern= Assignment ((decl_id, UInt, []), e)
          ; meta= e.meta.loc }
        in
        let var =
          Expr.
            { Fixed.pattern= Var decl_id
            ; meta=
                Typed.Meta.
                  { type_= s.emeta.type_
                  ; adlevel= s.emeta.ad_level
                  ; loc= s.emeta.loc } }
        in
        ([decl; assign; check fn s var], var)
  in
  let rec go n = function
    | SizedType.(SInt | SReal) as t -> ([], t)
    | SVector s ->
        let fn =
          match (declc.dconstrain, tr) with
          | Some Constrain, Program.Simplex ->
              Internal_fun.FnValidateSizeSimplex
          | Some Constrain, UnitVector -> FnValidateSizeUnitVector
          | _ -> FnValidateSize
        in
        let l, s = grab_size fn n s in
        (l, SizedType.SVector s)
    | SRowVector s ->
        let l, s = grab_size FnValidateSize n s in
        (l, SizedType.SRowVector s)
    | SMatrix (r, c) ->
        let l1, r = grab_size FnValidateSize n r in
        let l2, c = grab_size FnValidateSize (n + 1) c in
        let cf_cov =
          match (declc.dconstrain, tr) with
          | Some Constrain, CholeskyCov ->
              [ { Stmt.Fixed.pattern=
                    NRFunApp
                      ( StanLib
                      , "check_greater_or_equal"
                      , Expr.Helpers.
                          [ str ("cholesky_factor_cov " ^ name)
                          ; str
                              "num rows (must be greater or equal to num cols)"
                          ; r; c ] )
                ; meta= r.Expr.Fixed.meta.Expr.Typed.Meta.loc } ]
          | _ -> []
        in
        (l1 @ l2 @ cf_cov, SizedType.SMatrix (r, c))
    | SArray (t, s) ->
        let l, s = grab_size FnValidateSize n s in
        let ll, t = go (n + 1) t in
        (l @ ll, SizedType.SArray (t, s))
  in
  go 1 *)

(* let trans_block ud_dists declc block prog =
  let f stmt (accum1, accum2, accum3) =
    match stmt with
    | { stmt=
          VarDecl
            { decl_type= Sized type_
            ; identifier
            ; transformation
            ; initial_value
            ; is_global= true }
      ; smeta } ->
        let decl_id = identifier.name in
        let transform = Program.map_transformation trans_expr transformation in
        let rhs = Option.map ~f:trans_expr initial_value in
        let size, type_ =
          trans_sizedtype_decl declc transform identifier.name type_
        in
        let decl_adtype = declc.dadlevel in
        let decl_var =
          Expr.
            { Fixed.pattern= Var decl_id
            ; meta=
                Typed.Meta.create ~adlevel:declc.dadlevel ~loc:smeta.loc
                  ~type_:(SizedType.to_unsized type_)
                  () }
        in
        let decl =
          Stmt.
            { Fixed.pattern= Decl {decl_adtype; decl_id; decl_type= Sized type_}
            ; meta= smeta.loc }
        in
        let rhs_assignment =
          Option.map
            ~f:(fun e ->
              Stmt.Fixed.
                { pattern= Assignment ((decl_id, e.meta.type_, []), e)
                ; meta= smeta.loc } )
            rhs
          |> Option.to_list
        in
        let outvar =
          ( identifier.name
          , Program.
              { out_constrained_st= type_
              ; out_unconstrained_st= param_size transform type_
              ; out_block= block
              ; out_trans= transform } )
        in
        let stmts =
          if Utils.is_user_ident decl_id then
            let constrain_checks =
              match declc.dconstrain with
              | Some Constrain | Some Unconstrain ->
                  check_transform_shape decl_id decl_var smeta.loc transform
                  @ constrain_decl type_ declc.dconstrain transform decl_id
                      decl_var smeta.loc
              | Some Check ->
                  check_transform_shape decl_id decl_var smeta.loc transform
                  @ check_decl decl_var (Sized type_) decl_id transform
                      smeta.loc declc.dadlevel
              | None -> []
            in
            (decl :: rhs_assignment) @ constrain_checks
          else decl :: rhs_assignment
        in
        (outvar :: accum1, size @ accum2, stmts @ accum3)
    | stmt -> (accum1, accum2, trans_stmt ud_dists declc stmt @ accum3)
  in
  Option.value ~default:[] (get_block block prog)
  |> List.fold_right ~f ~init:([], [], [])

let migrate_checks_to_end_of_block stmts =
  let is_check = Stmt.Helpers.contains_fn FnCheck in
  let checks, not_checks = List.partition_tf ~f:is_check stmts in
  not_checks @ checks *)

let trans_block_as_args ff block =
  Option.iter
    ~f:(fun stmts ->
          match get_var_decl_names stmts with
          | [] -> ()
          | args ->
              fprintf ff "*, %a"
                (print_list_comma trans_id) args)
    block


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


let trans_datablock ff odata =
  match odata with
  | Some data ->
      fprintf ff
        "@[<v 0>@[<v 4>def convert_inputs(inputs):@,%a@,return { %a }@]@,@,@]"
        (print_list_newline convert_input) data
        (print_list_comma
           (fun ff x -> fprintf ff "'%a': %a" trans_id x trans_id x))
        (get_var_decl_names data)
  | None ->
      fprintf ff
        "@[<v 0>@[<v 4>def convert_inputs(inputs):@,return { }@]@,@,@]"


let trans_networks_as_arg ff networks =
  match networks with
  | None -> ()
  | Some nets ->
      fprintf ff ", ";
      pp_print_list
        ~pp_sep:(fun ff () -> fprintf ff ", ")
        (fun ff net -> fprintf ff "%s" net.net_id.name)
        ff nets

let trans_block_as_return ff block =
  Option.iter
    ~f:(fun stmts ->
          fprintf ff "return { %a }"
            (print_list_comma
               (fun ff x -> fprintf ff "'%s': %a" x.name trans_id x))
            (get_var_decl_names stmts))
    block

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
      trans_block_as_return transformeddata
  end

let trans_parameter ctx ff p =
  match p.stmt with
  | VarDecl {identifier; initial_value = None; decl_type; transformation; _} ->
    Hashtbl.add_exn ctx.ctx_ext.ext_params ~key:identifier.name ~data:();
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
          let priors, model = push_priors parameters model in
          Some priors, Some model
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
      "@[<v 0>@,@[<v 4>def generated_quantities(%a):@,%a%a%a"
      trans_block_as_args Option.(merge ~f:(@) data (merge ~f:(@) tdata params))
      (trans_block "Transformed parameters" ctx) tparams
      (trans_block "Generated quantities" ctx) genquantities
      trans_block_as_return (Option.merge ~f:(@) tparams genquantities);
    fprintf ff "@]@,@]"
  end

let trans_guide_parameter ctx ff p =
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
    ; ctx_ext = { ext_loops = []
                ; ext_params = Hashtbl.create (module String) } }
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

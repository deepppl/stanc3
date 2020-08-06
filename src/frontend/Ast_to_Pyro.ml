open Core_kernel
open Ast
open Middle
open Format

let print_list_comma printer ff l =
  fprintf ff "@[<hov 0>%a@]"
    (pp_print_list ~pp_sep:(fun ff () -> fprintf ff ",@ ") printer)
    l

let print_list_newline printer ff l =
  fprintf ff "@[<v 0>%a@]"
  (pp_print_list ~pp_sep:(fun ff () -> fprintf ff "@,") printer)
  l

let gen_id =
  let cpt = ref 0 in
  fun l ff e ->
    incr cpt;
    let s =
      match e.expr with
      | Variable {name; _} -> name
      | _ -> "expr"
    in
    match l with
    | [] -> fprintf ff "'%s__%d'" s !cpt
    | _ -> fprintf ff "f'%s%a__%d'" s (pp_print_list (fun ff x -> fprintf ff "__{%s}" x)) l !cpt

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

let rec base_type ff = function
  | UnsizedType.UInt -> fprintf ff "long"
  | UReal
  | UVector
  | URowVector
  | UMatrix -> fprintf ff "float"
  | UArray t -> base_type ff t
  | _ -> raise_s [%message "Unexpected base type"]

let rec dims_of_sizedtype t =
  match t with
  | SizedType.SInt
  | SReal -> []
  | SVector e -> [e]
  | SRowVector e -> [e; 1]
  | SMatrix (e1, e2) -> [e1; e2]
  | SArray (t, e) -> e :: dims_of_sizedtype t

let dims ff _t =
  fprintf ff "XXX TODO XXX"

let trans_numeral (type_: UnsizedType.t) ff x =
  begin match type_ with
  | UInt -> fprintf ff "%s" (format_number x)
  | UReal -> fprintf ff "%s" (format_number x)
  | UVector (* s *) ->
      fprintf ff "%s * ones(%a)" (format_number x) (fun ff () -> fprintf ff "XXX TODO XXX") ()
  | URowVector (* s *) ->
      fprintf ff "%s * ones(1, %a)" (format_number x) (fun ff () -> fprintf ff "XXX TODO XXX") ()
  | UMatrix (* s1, s2 *) ->
      fprintf ff "%s * ones(%a)" (format_number x) (fun ff () -> fprintf ff "XXX TODO XXX") ()
  | UArray t ->
      fprintf ff "%s * ones(%a, dtype=%a)"
        x dims t base_type t
  | _ ->
     raise_s [%message "Unexpected type for a numeral" (type_ : UnsizedType.t)]
  end

let rec trans_expr ff ({expr; emeta }: typed_expression) : unit =
  match expr with
  | Paren x -> fprintf ff "(%a)" trans_expr x
  | BinOp (lhs, op, rhs) -> fprintf ff "%a" (trans_binop lhs rhs) op
  | PrefixOp (op, e) | PostfixOp (e, op) -> fprintf ff "%a" (trans_unop e) op
  | TernaryIf (cond, ifb, elseb) ->
      fprintf ff "%a if %a else %a"
        trans_expr ifb trans_expr cond trans_expr elseb
  | Variable {name; _} -> fprintf ff "%s" name
  | IntNumeral x -> trans_numeral emeta.type_ ff x
  | RealNumeral x -> trans_numeral emeta.type_ ff x
  | FunApp (fn_kind, {name; _}, args) | CondDistApp (fn_kind, {name; _}, args)
    ->
      trans_fun_app ff fn_kind name args
  | GetLP | GetTarget -> fprintf ff "stanlib.target()" (* XXX TODO XXX *)
  | ArrayExpr eles ->
      fprintf ff "array([%a])" trans_exprs eles
  | RowVectorExpr eles ->
      fprintf ff "array([%a])" trans_exprs eles
  | Indexed (lhs, indices) ->
      fprintf ff "%a%a" trans_expr lhs (pp_print_list trans_idx) indices

and trans_binop e1 e2 ff op =
    match op with
    | Operator.Plus -> fprintf ff "%a + %a" trans_expr e1 trans_expr e2
    | Minus -> fprintf ff "%a - %a" trans_expr e1 trans_expr e2
    | Times -> fprintf ff "%a * %a" trans_expr e1 trans_expr e2
    | Divide -> fprintf ff "%a / %a" trans_expr e1 trans_expr e2
    | IntDivide -> fprintf ff "%a // %a" trans_expr e1 trans_expr e2
    | Modulo -> fprintf ff "%a %s %a" trans_expr e1 "%" trans_expr e2
    | LDivide -> fprintf ff "%a / %a" trans_expr e2 trans_expr e1
    | EltTimes -> fprintf ff "%a * %a" trans_expr e1 trans_expr e2 (* XXX TODO: check XXX *)
    | EltDivide -> fprintf ff "%a / %a" trans_expr e1 trans_expr e2 (* XXX TODO: check XXX *)
    | Pow -> fprintf ff "%a ** %a" trans_expr e1 trans_expr e2
    | EltPow -> fprintf ff "%a ** %a" trans_expr e1 trans_expr e2 (* XXX TODO: check XXX *)
    | Or -> fprintf ff "%a || %a" trans_expr e1 trans_expr e2
    | And -> fprintf ff "%a && %a" trans_expr e1 trans_expr e2
    | Equals -> fprintf ff "%a == %a" trans_expr e1 trans_expr e2
    | NEquals -> fprintf ff "%a != %a" trans_expr e1 trans_expr e2
    | Less -> fprintf ff "%a < %a" trans_expr e1 trans_expr e2
    | Leq -> fprintf ff "%a <= %a" trans_expr e1 trans_expr e2
    | Greater -> fprintf ff "%a > %a" trans_expr e1 trans_expr e2
    | Geq -> fprintf ff "%a >= %a" trans_expr e1 trans_expr e2
    | PNot
    | PPlus
    | PMinus
    | Transpose ->
        raise_s [%message "Binary operator expexter" (op: Operator.t)]

and trans_unop e ff op =
  match op with
  | Operator.PNot -> fprintf ff "+ %a" trans_expr e
  | PPlus -> fprintf ff "+ %a" trans_expr e
  | PMinus -> fprintf ff "- %a" trans_expr e
  | Transpose -> fprintf ff "torch.transpose(%a)" trans_expr e
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
      raise_s [%message "Unary operator expexter" (op: Operator.t)]

and trans_idx ff = function
  | All -> fprintf ff "[:]"
  | Upfrom e -> fprintf ff "[%a - 1:]" trans_expr e
  | Downfrom e -> fprintf ff "[:%a]" trans_expr e
  | Between (lb, ub) -> fprintf ff "[%a - 1:%a]" trans_expr lb trans_expr ub
  | Single e -> (
    match e.emeta.type_ with
    | UInt -> fprintf ff "[%a - 1]" trans_expr e
    | UArray _ -> fprintf ff "[%a - 1]" trans_expr e
    | _ ->
        raise_s
          [%message "Expecting int or array" (e.emeta.type_ : UnsizedType.t)] )

and trans_exprs ff exprs =
  fprintf ff "%a" (print_list_comma trans_expr) exprs

and trans_fun_app ff fn_kind name args =
  fprintf ff "%s%s(%a)"
    (match fn_kind with StanLib -> "stanlib." | UserDefined -> "")
    name trans_exprs args

let trans_sizedtype = SizedType.map trans_expr

let trans_expr_opt ff = function
  | Some e -> trans_expr ff e
  | None -> fprintf ff "None"

(* let neg_inf =
  Expr.
    { Fixed.pattern= FunApp (StanLib, Internal_fun.to_string FnNegInf, [])
    ; meta=
        Typed.Meta.{type_= UReal; loc= Location_span.empty; adlevel= DataOnly}
    } *)

let trans_arg ff (_, _, ident) =
  pp_print_string ff ident.name

let trans_args ff args =
  fprintf ff "%a" (print_list_comma trans_arg) args

(* let truncate_dist ud_dists (id : identifier) ast_obs ast_args t =
  let cdf_suffices = ["_lcdf"; "_cdf_log"] in
  let ccdf_suffices = ["_lccdf"; "_ccdf_log"] in
  let find_function_info sfx =
    let possible_names =
      List.map ~f:(( ^ ) id.name) sfx |> String.Set.of_list
    in
    match List.find ~f:(fun (n, _) -> Set.mem possible_names n) ud_dists with
    | Some (name, tp) -> (UserDefined, name, tp)
    | None ->
        ( StanLib
        , Set.to_list possible_names |> List.hd_exn
        , if Stan_math_signatures.is_stan_math_function_name (id.name ^ "_lpmf")
          then UnsizedType.UInt
          else UnsizedType.UReal (* close enough *) )
  in
  let trunc cond_op (x : typed_expression) y =
    let smeta = x.emeta.loc in
    { Stmt.Fixed.meta= smeta
    ; pattern=
        IfElse
          ( op_to_funapp cond_op [ast_obs; x]
          , {Stmt.Fixed.meta= smeta; pattern= TargetPE neg_inf}
          , Some y ) }
  in
  let targetme loc e =
    {Stmt.Fixed.meta= loc; pattern= TargetPE (op_to_funapp Operator.PMinus [e])}
  in
  let funapp meta kind name args =
    { emeta= meta
    ; expr= FunApp (kind, {name; id_loc= Location_span.empty}, args) }
  in
  let inclusive_bound tp (lb : typed_expression) =
    let emeta = lb.emeta in
    if UnsizedType.is_int_type tp then
      
        { emeta
        ; expr= BinOp (lb, Operator.Minus, {emeta; expr= IntNumeral "1"})
        }
    else lb
  in
  match t with
  | NoTruncate -> []
  | TruncateUpFrom lb ->
      let fk, fn, tp = find_function_info ccdf_suffices in
      [ trunc Less lb
          (targetme lb.emeta.loc
             (funapp lb.emeta fk fn (inclusive_bound tp lb :: ast_args))) ]
  | TruncateDownFrom ub ->
      let fk, fn, _ = find_function_info cdf_suffices in
      [ trunc Greater ub
          (targetme ub.emeta.loc (funapp ub.emeta fk fn (ub :: ast_args))) ]
  | TruncateBetween (lb, ub) ->
      let fk, fn, tp = find_function_info cdf_suffices in
      [ trunc Less lb
          (trunc Greater ub
             (targetme ub.emeta.loc
                (funapp ub.emeta StanLib "log_diff_exp"
                   [ funapp ub.emeta fk fn (ub :: ast_args)
                   ; funapp ub.emeta fk fn (inclusive_bound tp lb :: ast_args)
                   ]))) ] *)
  
(* let unquote s =
  if s.[0] = '"' && s.[String.length s - 1] = '"' then
    String.drop_suffix (String.drop_prefix s 1) 1
  else s *)

(* hack(sean): strings aren't real
   XXX add UString to MIR and maybe 
*)
(* let mkstring loc s =
  Expr.
    { Fixed.pattern= Lit (Str, s)
    ; meta= Typed.Meta.create ~type_:UReal ~loc ~adlevel:DataOnly () } *)

let trans_printables ff (ps : typed_expression printable list) =
  fprintf ff "%a"
    (print_list_comma
       (fun ff -> function
          | PString s -> fprintf ff "%s" s
          | PExpr e -> trans_expr ff e))
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
      | { lval= LVariable ident; lmeta } ->
          { expr = Variable ident; emeta = lmeta } 
      | { lval= LIndexed (l, i); lmeta } ->
          { expr= Indexed (expr_of_lval l, i); emeta=lmeta }
      in
      let trans_rhs lhs ff rhs =
        match assign_op with
        | Assign | ArrowAssign -> trans_expr ff rhs
        | OperatorAssign op -> trans_binop lhs rhs ff op
      in
      let lhs = expr_of_lval assign_lhs in
      fprintf ff "%a = %a" trans_expr lhs (trans_rhs lhs) assign_rhs
  | NRFunApp (fn_kind, {name; _}, args) ->
      trans_fun_app ff fn_kind name args
  | IncrementLogProb e | TargetPE e ->
      fprintf ff "factor(%a, %a)"
        (gen_id ctx) e
        trans_expr e
  | Tilde {arg; distribution; args; truncation} ->
      let trans_distribution ff dist =
        fprintf ff "%s" dist.name
      in
      let trans_truncation _ff = function
        | NoTruncate -> ()
        | _ -> assert false (* XXX TODO XXX *)
      in
      fprintf ff "observe(%a, %a(%a), %a)%a"
        (gen_id ctx) arg
        trans_distribution distribution
        trans_exprs args
        trans_expr arg
        trans_truncation truncation
  | Print ps -> fprintf ff "print(%a)" trans_printables ps
  | Reject ps -> fprintf ff "stanlib.reject(%a)" trans_printables ps
  | IfThenElse (cond, ifb, None) ->
      fprintf ff "@[<v 0>@[<v 4>if %a:@,%a@]@]"
        trans_expr cond
        (trans_stmt ctx) ifb
  | IfThenElse (cond, ifb, Some elseb) ->
      fprintf ff "@[<v 0>@[<v 4>if %a:@,%a@]@,@[<v 4>else:@,%a@]@]"
        trans_expr cond
        (trans_stmt ctx) ifb
        (trans_stmt ctx) elseb
  | While (cond, body) ->
      fprintf ff "@[<v4>while %a:@,%a@]"
        trans_expr cond
        (trans_stmt ("genid()"::ctx)) body
  | For {loop_variable; lower_bound; upper_bound; loop_body} ->
      fprintf ff "@[<v 4>for %s in range(%a,%a + 1):@,%a@]"
        loop_variable.name
        trans_expr lower_bound
        trans_expr upper_bound
        (trans_stmt (loop_variable.name :: ctx)) loop_body
  | ForEach (loopvar, iteratee, body) ->
      fprintf ff "@[<v4>for %s in %a:@,%a@]"
        loopvar.name
        trans_expr iteratee
        (trans_stmt (loopvar.name :: ctx)) body
  | FunDef _ ->
      raise_s
        [%message
          "Found function definition statement outside of function block"]
  | VarDecl {identifier; initial_value; _} ->
      fprintf ff "%s = %a" identifier.name
        trans_expr_opt initial_value
  | Block stmts ->
      fprintf ff "%a" (print_list_newline (trans_stmt ctx)) stmts
  | Return e ->
      fprintf ff "return %a" trans_expr e
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

let trans_fun_def ff (ts : typed_statement) =
  match ts.stmt with
  | FunDef {funname; arguments; body; _} ->
      fprintf ff "@[<v 0>@[<v 4>def %s(%a):@,%a@]@,@]"
        funname.name trans_args arguments (trans_stmt []) body
  | _ ->
      raise_s
        [%message "Found non-function definition statement in function block"]

let trans_functionblock ff functionblock =
  fprintf ff "@[<v 0>%a@,@]" (print_list_newline trans_fun_def) functionblock

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

let get_var_decl_names stmts =
  List.fold_left
    ~f:(fun acc stmt ->
          match stmt.stmt with
          | VarDecl {identifier; _} -> identifier.name :: acc
          | _ -> acc ) ~init:[] stmts

let trans_block_as_args ff block =
  Option.iter
    ~f:(fun stmts ->
          fprintf ff "*, %a"
            (print_list_comma pp_print_string) (get_var_decl_names stmts))
    block

let trans_block_as_return ff block =
  Option.iter
    ~f:(fun stmts ->
          fprintf ff "return { %a }"
            (print_list_comma (fun ff x -> fprintf ff "'%s': %s" x x))
            (get_var_decl_names stmts))
    block

let trans_prior (decl_type: _ Type.t) ff transformation =
  match transformation with
  | Program.Identity -> fprintf ff "ImproperUniform('%a')" dims decl_type
  | Lower lb ->
     fprintf ff "LowerConstrainedImproperUniform(%a, shape='%a')"
       trans_expr lb dims decl_type
  | Upper ub ->
     fprintf ff "UpperConstrainedImproperUniform(%a, shape='%a')"
       trans_expr ub dims decl_type
  | LowerUpper (lb, ub) ->
      fprintf ff "uniform(%a, %a)" (* XXX TODO: shape XXX *)
        trans_expr lb trans_expr ub
  | Offset _
  | Multiplier _
  | OffsetMultiplier _
  | Ordered
  | PositiveOrdered
  | Simplex
  | UnitVector
  | CholeskyCorr
  | CholeskyCov
  | Correlation
  | Covariance -> assert false (* XXX TODO XXX *)

let pp_print_nothing _ () = ()

let trans_block ?(eol=true) comment ff block =
  Option.iter
    ~f:(fun stmts ->
          fprintf ff "@[<v 0># %s@,%a@]%a"
            comment (trans_stmts []) stmts
            (if eol then pp_print_cut else pp_print_nothing) ())
    block

let trans_transformeddatablock ff data transformeddata =
  if transformeddata <> None then begin
    fprintf ff
      "@[<v 0>@[<v 4>def transformed_data(%a):@,%a%a@]@,@,@]"
      trans_block_as_args data
      (trans_block "Transformed data") transformeddata
      trans_block_as_return transformeddata
  end

let trans_parameter ff p =
  match p.stmt with
  | VarDecl {identifier; initial_value = None; decl_type; transformation; _} ->
    fprintf ff "%s = sample('%s', %a)" identifier.name identifier.name
      (trans_prior decl_type) transformation
  | _ -> assert false

let trans_parametersblock ff parameters =
  Option.iter
    ~f:(fprintf ff "@[<v 0># Parameters@,%a@]"
          (print_list_newline trans_parameter))
    parameters

let trans_modelblock ff data tdata parameters tparameters model =
  fprintf ff "@[<v 4>def model(%a):@,%a@,%a%a@]@."
    trans_block_as_args (Option.merge ~f:(@) data tdata)
    trans_parametersblock parameters
    (trans_block "Transformed parameters") tparameters
    (trans_block ~eol:false "Model") model

let trans_generatedquantitiesblock ff data tdata params tparams genquantities =
  if tparams <> None || genquantities <> None then begin
    fprintf ff
      "@[<v 0>@,@[<v 4>def generated_quantities(%a):@,%a%a%a"
      trans_block_as_args Option.(merge ~f:(@) data (merge ~f:(@) tdata params))
      (trans_block "Transformed parameters") tparams
      (trans_block "Generated quantities") genquantities
      trans_block_as_return (Option.merge ~f:(@) tparams genquantities);
    fprintf ff "@]@,@]"
  end

let trans_prog ff (p : typed_program) =
  fprintf ff "@[<v 0>%s@,%s@,@,@]"
    "from distributions import *"
    "from dppllib import sample, observe, factor";
  Option.iter ~f:(trans_functionblock ff) p.functionblock;
  trans_transformeddatablock ff p.datablock p.transformeddatablock;
  trans_modelblock ff
    p.datablock p.transformeddatablock
    p.parametersblock p.transformedparametersblock p.modelblock;
  trans_generatedquantitiesblock ff
    p.datablock p.transformeddatablock
    p.parametersblock p.transformedparametersblock p.generatedquantitiesblock
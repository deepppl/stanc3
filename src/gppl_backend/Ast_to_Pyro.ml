open Core_kernel
open Frontend
open Ast
open Middle
open Format
open Gppl_utils

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

let set_ctx_mutation ctx =
  { ctx with ctx_mutation = true }

let set_to_clone ctx =
  { ctx with ctx_to_clone = true }

let unset_to_clone ctx =
  { ctx with ctx_to_clone = false }


(** {1 Translation of expressions} *)

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
        match List.Assoc.find ~equal:(=) Pyro_lib.stanlib
                (function_id fn_kind id args) with
        | Some Tclone -> set_ctx_mutation (set_to_clone ctx)
        | Some Tnoclone | None -> ctx
      in
      trans_fun_app ctx fn_kind id ff args
  | CondDistApp (fn_kind, id, args) ->
      let ctx =
        match List.Assoc.find ~equal:(=) Pyro_lib.distributions
                (function_id fn_kind id args) with
        | Some (Tclone, _) -> set_ctx_mutation (set_to_clone ctx)
        | Some (Tnoclone, _) | None -> ctx
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

and trans_dims ctx ff (t : typed_expression Type.t) =
  match dims_of_type t with
  | [] -> fprintf ff "[]"
  | l -> fprintf ff "[%a]" (trans_exprs ctx) l

let trans_expr_opt ctx (type_ : typed_expression Type.t) ff = function
  | Some e -> trans_expr ctx ff e
  | None ->
      if is_tensor type_ then
        fprintf ff "empty(%a, dtype=%a)"
          (trans_dims ctx) type_
          dtype_of_type type_
      else fprintf ff "None"


(** {1 Translation of statements} *)

(** Kind of control structure to generate (e.g., [for], [fori_loop],
    [lax_fori_loop] )*)
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
          let id, indices = split_lval assign_lhs in
          begin match ctx.ctx_backend with
          | Pyro | Pyro_cuda ->
              let trans_indices ff l =
                fprintf ff "%a"
                  (pp_print_list ~pp_sep:(fun _ () -> ())
                     (fun ff idx ->
                        fprintf ff "[%a]"
                          (print_list_comma (trans_idx ctx)) idx))
                  l
              in
              fprintf ff "%a%a = %a"
                trans_id id trans_indices indices
                (trans_rhs (expr_of_lval assign_lhs)) assign_rhs
          | Numpyro ->
              let trans_indices ff l =
                match l with
                | [ idx ] -> print_list_comma (trans_idx ctx) ff idx
                | _ ->
                    fprintf ff "%a"
                      (print_list_comma
                         (fun ff idx ->
                            fprintf ff "ops_index[%a]"
                              (print_list_comma (trans_idx ctx)) idx))
                      l
              in
              fprintf ff "%a = ops_index_update(%a, ops_index[%a], %a)"
                trans_id id
                trans_id id trans_indices indices
                (trans_rhs (expr_of_lval assign_lhs)) assign_rhs
          end
    end
  | NRFunApp (fn_kind, id, args) ->
      let ctx =
        match List.Assoc.find ~equal:(=) Pyro_lib.stanlib
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
        | Numpyro -> CtrlLax
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
            begin match ctx.ctx_block with
            | Some Model -> if is_pure loop_body then CtrlLax else CtrlNympyro
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
            begin match ctx.ctx_block with
            | Some Model -> if is_pure body then CtrlLax else CtrlNympyro
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
  | Profile (x, stmts) ->
      fprintf ff "@[<v 0># profile(%s)@,%a@]"
        x (print_list_newline (trans_stmt ctx)) stmts
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

and trans_printables ctx ff (ps : _ printable list) =
  fprintf ff "%a"
    (print_list_comma
       (fun ff -> function
          | PString s -> fprintf ff "%s" s
          | PExpr e -> (trans_expr ctx) ff e))
    ps

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

and print_jit ff kind =
  match kind with
  | CtrlLax -> fprintf ff "%s@," "@jit"
  | _ -> ()

let trans_stmts ctx ff stmts =
  fprintf ff "%a" (print_list_newline (trans_stmt ctx)) stmts

(** {1 Scheduling for mixed compilation} *)

let compare_untyped_expression' e1 e2 =
  match compare_untyped_expression e1 e2 with
  | 0 -> 0
  | n ->
    begin try match e1.expr, e2.expr with
      | IntNumeral n1, IntNumeral n2 ->
        compare (int_of_string n1) (int_of_string n2)
      | RealNumeral n1, RealNumeral n2
      | RealNumeral n1, IntNumeral n2
      | IntNumeral n1, RealNumeral n2 ->
        compare (float_of_string n1) (float_of_string n2)
      | _ -> n
      with _ -> n
    end

let same_support ctx distribution args transformation =
  ctx.ctx_mode = Generative ||
  match List.Assoc.find ~equal:(=) Pyro_lib.distributions distribution.name with
  | Some (_, cstr) ->
    let trans = cstr args in
    let transformation =
      Program.map_transformation
        untyped_expression_of_typed_expression transformation
    in
    Program.compare_transformation compare_untyped_expression'
      trans transformation = 0
  | None -> false

let is_variable_sampling ctx x transformation stmt =
  match stmt.stmt with
  | Tilde { arg = { expr = Variable y; _ }; distribution; args; _ } ->
    x = y.name && same_support ctx distribution args transformation
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

let rec push_prior_stmts ctx (x, decl, transformation) stmts =
  match stmts with
  | [] -> Some decl, [ ]
  | stmt :: stmts ->
      if is_variable_sampling ctx x transformation stmt then
        None, merge_decl_sample decl stmt :: stmts
      else if SSet.mem (free_vars SSet.empty stmt) x then
        Some decl, stmt :: stmts
      else
        let prior, stmts =
          push_prior_stmts ctx (x, decl, transformation) stmts
        in
        prior, stmt ::  stmts

let push_priors ctx priors stmts =
  List.fold_left
    ~f:(fun (priors, stmts) decl ->
        match decl.stmt with
        | VarDecl { identifier = id; initial_value = None;
                    transformation = trans; _ } ->

            begin match push_prior_stmts ctx (id.name, decl, trans) stmts with
            | Some prior, stmts -> priors @ [prior], stmts
            | None, stmts -> priors, stmts
            end
        | _ -> assert false)
    ~init:([], stmts) priors

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

(** {1 Translation of functions} *)

let trans_arg ff (_, _, ident) =
  trans_id ff ident

let trans_args ff args =
  fprintf ff "%a" (print_list_comma trans_arg) args

let trans_fun_def ctx ff (ts : typed_statement) =
  match ts.stmt with
  | FunDef {funname; arguments; body; _} ->
      fprintf ff "@[<v 0>@[<v 4>def %a(%a):@,%a@]@,@]"
        trans_id funname trans_args arguments (trans_stmt ctx) body
  | _ ->
      raise_s
        [%message "Found non-function definition statement in function block"]


(** {1 Translation of priors} *)

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

(** {1 Utility functions for block translations} *)

let trans_block_as_args ?(named=true) ff block =
  match get_var_decl_names_block block with
  | [] -> ()
  | args -> fprintf ff "%s%a" (if named then "*, " else "")
              (print_list_comma trans_id) args

let trans_block_as_kwargs ff block =
  match get_var_decl_names_block block with
  | [] -> ()
  | args -> fprintf ff "%a"
              (print_list_comma
                 (fun ff x -> fprintf ff "%a=%a" trans_id x trans_id x)) args

let trans_ids_as_return ?(with_rename=false) ff ids =
  fprintf ff "return { %a }"
    (print_list_comma
       (fun ff x ->
          if with_rename then
            fprintf ff "'%a': %a" trans_id x trans_id x
          else
            fprintf ff "'%s': %a" x.name trans_id x))
    ids

let trans_block_as_return ?(with_rename=false) ff block =
  trans_ids_as_return ~with_rename ff (get_var_decl_names_block block)

let trans_block_as_unpack name ff block =
  let unpack ff x = fprintf ff "%s['%s']" name x.name in
  let args = get_var_decl_names_block block in
  fprintf ff "%a" (print_list_comma unpack) args

let trans_block ?(eol=true) comment ctx ff block =
  Option.iter
    ~f:(fun stmts ->
          fprintf ff "@[<v 0># %s@,%a@]%a"
            comment (trans_stmts ctx) stmts
            (if eol then pp_print_cut else pp_print_nothing) ())
    block


(** {1 Translation of the networks block} *)

let trans_networks_as_arg ff networks =
  match networks with
  | None -> ()
  | Some nets ->
      fprintf ff ", ";
      pp_print_list
        ~pp_sep:(fun ff () -> fprintf ff ", ")
        (fun ff net -> fprintf ff "%s" net.net_id.name)
        ff nets

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
      (trans_block_as_args ~named:true) args
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


(** {1 Translation of the functions block} *)

let trans_functionblock ctx ff functionblock =
  fprintf ff "@[<v 0>%a@,@]"
    (print_list_newline (trans_fun_def ctx)) functionblock


(** {1 Translation of the data block} *)

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

let convert_networks_inputs ff onetworks =
  let convert_network_input ff net =
    fprintf ff "%a = inputs['%s']" trans_id net.net_id net.net_id.name
  in
  Option.iter
    ~f:(fun nets -> print_list_newline ~eol:true convert_network_input ff nets)
    onetworks

let trans_datablock networks ff data =
  let ids =
    (get_var_decl_names_block data) @ (id_list_of_networks networks)
  in
  fprintf ff
    "@[<v 0>@[<v 4>def convert_inputs(inputs):@,%a@,%a%a@]@,@,@]"
    convert_inputs data
    convert_networks_inputs networks
    (trans_ids_as_return ~with_rename:true) ids


(** {1 Translation of the transformed data block} *)

let trans_transformeddatablock ctx data ff transformeddata =
  if transformeddata <> None then begin
    fprintf ff
      "@[<v 0>@[<v 4>def transformed_data(%a):@,%a%a@]@,@,@]"
      (trans_block_as_args ~named:true) data
      (trans_block "Transformed data" ctx) transformeddata
      (trans_block_as_return ~with_rename:true) transformeddata
  end


(** {1 Translation of the parameters block} *)

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


(** {1 Translation of the model block} *)

let trans_modelblock ctx networks data tdata parameters tparameters ff model =
  let ctx = { ctx with ctx_block = Some Model } in
  let rnetworks, nparameters, parameters =
    split_networks networks parameters
  in
  match ctx.ctx_mode with
  | Comprehensive ->
      fprintf ff "@[<v 4>def model(%a%a):@,%a%a%a%a@]@,@,@."
        (trans_block_as_args ~named:true) (Option.merge ~f:(@) data tdata)
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
          let parameters, model = push_priors ctx parameters model in
          let model = moveup_observes model in
          let parameters, model = push_priors ctx parameters model in
          parameters, Some model
      in
      fprintf ff "@[<v 4>def model(%a%a):@,%a%a%a@]@,@,@."
        (trans_block_as_args ~named:true) (Option.merge ~f:(@) data tdata)
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
          let parameters, model = push_priors ctx parameters model in
          let model = moveup_observes model in
          let parameters, model = push_priors ctx parameters model in
          parameters, Some model
      in
      fprintf ff "@[<v 4>def model(%a%a):@,%a%a%a@]@,@,@."
        (trans_block_as_args ~named:true) (Option.merge ~f:(@) data tdata)
        trans_networks_as_arg networks
        (register_network rnetworks) model
        (trans_parametersblock ctx networks data tdata)
        (nparameters, rem_parameters)
        (trans_block ~eol:false "Model" ctx) model_ext


(** {1 Translation of the generated quantities block} *)

let trans_generatedquantitiesblock ctx networks data tdata params tparams ff
    genquantities =
  let ctx = { ctx with ctx_block = Some GeneratedQuantities } in
  if tparams <> None || genquantities <> None then begin
    fprintf ff
      "@[<v 0>@,@[<v 4>def generated_quantities(%a%a):@,%a%a%a"
      (trans_block_as_args ~named:true)
      Option.(merge ~f:(@) data (merge ~f:(@) tdata params))
      trans_networks_as_arg networks
      (trans_block "Transformed parameters" ctx) tparams
      (trans_block "Generated quantities" ctx) genquantities
      (trans_block_as_return ~with_rename:false)
      (Option.merge ~f:(@) tparams genquantities);
    fprintf ff "@]@,@]";
    fprintf ff
      "@[<v 0>@,@[<v 4>def map_generated_quantities(_samples, %a%a):@,"
      (trans_block_as_args ~named:true) (Option.merge ~f:(@) data tdata)
      trans_networks_as_arg networks;
    fprintf ff "@[<v 4>def _generated_quantities(%a):@,return generated_quantities(%a%a)@]@,"
      (trans_block_as_args ~named:false) params
      trans_block_as_kwargs
      Option.(merge ~f:(@) data (merge ~f:(@) tdata params))
      trans_networks_as_arg networks;
    begin match ctx.ctx_backend with
    | Numpyro -> fprintf ff "_f = jit(vmap(_generated_quantities))@,"
    | Pyro | Pyro_cuda -> fprintf ff "_f = vmap(_generated_quantities)@,"
    end;
    fprintf ff "return _f(%a)"
      (trans_block_as_unpack "_samples") params;
    fprintf ff "@]@,@]";
  end


(** {1 Translation of the guide parameters block} *)

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


(** {1 Translation of the guide block} *)

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
      (trans_block_as_args ~named:true) (Option.merge ~f:(@) data tdata)
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

(** {1 Translation of the program} *)

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
    | Pyro -> Pyro_lib.pyro_dppllib
    | Numpyro -> Pyro_lib.numpyro_dppllib
    | Pyro_cuda -> Pyro_lib.pyro_dppllib
  in
  let dppllib =
    let open Pyro_lib in
    if p.networksblock <> None then dppllib @ dppllib_networks else dppllib
  in
  fprintf ff "@[<v 0>%a%a%a@,@]"
    (pp_imports ("stan"^runtime^".distributions")) ["*"]
    (pp_imports ("stan"^runtime^".dppllib")) dppllib
    (pp_imports ("stan"^runtime^".stanlib"))
    (SSet.to_list (get_stanlib_calls p));
  Option.iter ~f:(trans_functionblock ctx ff) p.functionblock;
  fprintf ff "%a" (trans_datablock p.networksblock) p.datablock;
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
       p.networksblock p.datablock p.transformeddatablock
       p.parametersblock p.transformedparametersblock)
    p.generatedquantitiesblock;

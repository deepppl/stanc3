open Core_kernel
open Ast
open Middle
open SizedType

type vectorized_type = size SizedType.t
and size =
  | SExpr of vectorized_expression
  | SConst of int
  (* | SIndex of size index *)
  | Sigma of int
and vectorized_expression = (vectorized_expr_meta, fun_kind) expr_with
and vectorized_expr_meta =
    { loc: Location_span.t sexp_opaque [@compare.ignore]
    ; type_: vectorized_type }
  [@@deriving sexp, compare, map]

type vectorized_lval =
  (vectorized_expression, vectorized_expr_meta) lval_with
[@@deriving sexp, compare, map]

type vectorized_statement =
  ( vectorized_expression
  , stmt_typed_located_meta
  , vectorized_lval
  , fun_kind )
  statement_with
[@@deriving sexp, compare, map]


module SEnv = Map.Make(String)
module IEnv = Map.Make(Int)
module SizeSet = Set.Make(struct
  type t = size
  let compare = compare_size
  let sexp_of_t = sexp_of_size
  let t_of_sexp = size_of_sexp
end)

let opt_map_snd f x opt_y =
  match opt_y with
  | None -> x, None
  | Some y -> let (x, y) = f x y in x, Some y

let fresh_sigma =
  let cpt = ref 0 in
  fun cstrs ->
  incr cpt;
  let s = !cpt in
  let cstrs = IEnv.add_exn cstrs ~key:s ~data:(ref SizeSet.empty) in
  (cstrs, Sigma (!cpt))

let rec vectorized_type_of_unsized_type (cstrs: 'a) (ut: UnsizedType.t) : 'a * 'b =
  match ut with
  | UInt -> cstrs, SInt
  | UReal -> cstrs, SReal
  | UVector ->
      let cstrs, s = fresh_sigma cstrs in
      cstrs, SVector s
  | URowVector ->
      let cstrs, s = fresh_sigma cstrs in
      cstrs, SRowVector s
  | UMatrix ->
      let cstrs, s1 = fresh_sigma cstrs in
      let cstrs, s2 = fresh_sigma cstrs in
      cstrs, SMatrix (s1, s2)
  | UArray t ->
      let cstrs, s = fresh_sigma cstrs in
      let cstrs, t = vectorized_type_of_unsized_type cstrs t in
      cstrs, SArray (t, s)
  | UMathLibraryFunction -> assert false (* XXX TODO XXX *)
  | UFun _ -> assert false (* XXX TODO XXX *)


let unify_size cstrs actual expected =
  match actual, expected with
  | SConst n1, SConst n2 ->
      if n1 = n2 then
        cstrs, SConst n1
      else
        (* XXX TODO: error message XXX *)
        raise_s [%message "Sizes are not equal" (n1:int) (n2:int)]
  | Sigma s, SExpr e | SExpr e, Sigma s ->
      let s_cstrs =  IEnv.find_exn cstrs s in
      s_cstrs := SizeSet.add !s_cstrs (SExpr e);
      cstrs, Sigma s
  | Sigma s, SConst n | SConst n, Sigma s ->
      let s_cstrs =  IEnv.find_exn cstrs s in
      s_cstrs := SizeSet.add !s_cstrs (SConst n);
      cstrs, SConst n
  | Sigma s1, Sigma s2 ->
      let s1_cstrs = IEnv.find_exn cstrs s1 in
      let s2_cstrs = IEnv.find_exn cstrs s2 in
      s1_cstrs := SizeSet.union !s1_cstrs !s2_cstrs;
      let cstrs = IEnv.update cstrs s2 ~f:(function _ -> s1_cstrs) in
      cstrs, Sigma s1
  | SExpr e1, SExpr e2 ->
      begin match compare_vectorized_expression e1 e2 with
      | 0 -> cstrs, SExpr e1
      | _ -> (* XXX TODO: improve XXX *)
        Fmt.pf Fmt.stderr "Warning: you must check that %s = %s@."
          (Sexp.to_string (sexp_of_vectorized_expression e1))
          (Sexp.to_string (sexp_of_vectorized_expression e2));
        cstrs, SExpr e1
      end
  | SExpr e, SConst n | SConst n, SExpr e ->
      begin match e.expr with
      | IntNumeral x ->
          if x = string_of_int n then
            cstrs, SConst n
          else
            (* XXX TODO: error message XXX *)
            raise_s [%message "Sizes are not equal" (x:string) (n:int)]
      | _ ->
        Fmt.pf Fmt.stderr "Warning: you must check that %s = %d@."
          (Sexp.to_string (sexp_of_vectorized_expression e))
          n;
        cstrs, SConst n
      end

let rec unify cstrs actual expected =
  match actual, expected with
  | SInt, SInt -> cstrs, SInt
  | (SInt | SReal), SReal -> cstrs, SReal
  | SVector s1, SVector s2 ->
      let cstrs, s = unify_size cstrs s1 s2 in
      cstrs, SVector s
  | SRowVector s1, SRowVector s2 ->
      let cstrs, s = unify_size cstrs s1 s2 in
      cstrs, SRowVector s
  | SMatrix (s11,s12), SMatrix (s21,s22) ->
      let cstrs, s1 = unify_size cstrs s11 s21 in
      let cstrs, s2 = unify_size cstrs s12 s22 in
      cstrs, SMatrix (s1, s2)
  | SReal, (SVector _ | SRowVector _ | SMatrix _) -> cstrs, expected
  | SArray (t1, s1), SArray (t2, s2) ->
      let cstrs, t = unify cstrs t1 t2 in
      let cstrs, s = unify_size cstrs s1 s2 in
      cstrs, SArray(t, s)
  | t1, SArray (t2, s) ->
      let cstrs, t = unify cstrs t1 t2 in
      cstrs, SArray (t, s)
  | SInt, (SVector _ | SRowVector _ | SMatrix _) ->
      raise_s
        [%message "Integer cannot be cast into " (expected:vectorized_type)]
  | SVector _, (SInt | SRowVector _ | SMatrix _) ->
      raise_s
        [%message "Vector cannot be cast into " (expected:vectorized_type)]
  | SRowVector _, (SInt | SVector _ | SMatrix _) ->
      raise_s
        [%message "Row vector cannot be cast into " (expected:vectorized_type)]
  | SMatrix _, (SInt | SVector _ | SRowVector _) ->
      raise_s
        [%message "Vector cannot be cast into " (expected:vectorized_type)]
  | SReal, SInt ->
      raise_s
        [%message "real cannot be cast into int"]
  | SArray _, (SInt | SReal | SVector _ | SRowVector _ | SMatrix (_, _)) ->
      raise_s
        [%message "Array cannot be cast into " (expected:vectorized_type)]
  | (SVector _ | SRowVector _ | SMatrix _), SReal ->
      raise_s
        [%message (actual:vectorized_type) " cannot be cast into real"]


let annotate_list annot env cstrs l =
  let env, cstrs, rev_xs =
    List.fold_left
      ~f:(fun (env, cstrs, acc) x ->
          let env, cstrs, x = annot env cstrs x in
          (env, cstrs, x :: acc))
      ~init:(env, cstrs, [])
      l
  in
  env, cstrs, List.rev rev_xs

let rec annotate_expr:
  vectorized_type SEnv.t ->
  SizeSet.t ref IEnv.t ->
  typed_expression ->
  vectorized_type -> SizeSet.t ref IEnv.t * vectorized_expression =
  fun env cstrs texpr expected ->
  let vect_expr expr t =
    { expr; emeta = { type_ = t; loc = texpr.emeta.loc }; }
  in
  match texpr.expr with
  | Paren e ->
      let cstrs, e = annotate_expr env cstrs e expected in
      cstrs, vect_expr (Paren e) e.emeta.type_
  | BinOp (lhs, op, rhs) ->
      let  cstrs, expr = annotate_binop env cstrs lhs op rhs expected in
      cstrs, vect_expr expr expected
  | PrefixOp (op, e) ->
      let cstrs, e, t = annotate_unop env cstrs op e expected in
      cstrs, vect_expr (PrefixOp(op, e)) t
  | PostfixOp (e, op) ->
      let cstrs, e, t = annotate_unop env cstrs op e expected in
      cstrs, vect_expr (PostfixOp(e, op)) t
  | TernaryIf (cond, ifb, elseb) ->
      let cstrs, cond = annotate_expr env cstrs cond SReal in
      let cstrs, ifb = annotate_expr env cstrs ifb expected in
      let cstrs, elseb = annotate_expr env cstrs elseb expected in
      cstrs, vect_expr (TernaryIf (cond, ifb, elseb)) expected
  | Variable {name; id_loc } ->
      let actual = SEnv.find_exn env name in
      let cstrs, t = unify cstrs actual expected in
      cstrs, vect_expr (Variable {name; id_loc}) t
  | IntNumeral x ->
      assert (texpr.emeta.type_ = UnsizedType.UInt);
      cstrs, vect_expr (IntNumeral x) SInt
  | RealNumeral x ->
      assert (texpr.emeta.type_ = UnsizedType.UReal);
      let cstrs, t = unify cstrs SReal expected in
      cstrs, vect_expr (RealNumeral x) t
  (* | FunApp (fn_kind, {name; _}, args) | CondDistApp (fn_kind, {name; _}, args) *)
  (*   -> *)
  (*     trans_fun_app ff fn_kind name args *)
  | GetLP | GetTarget ->
      let cstrs, t = unify cstrs SReal expected in
      cstrs, vect_expr GetTarget t
  | ArrayExpr eles ->
      let t, s =
        match expected with
        | SArray(t, s) -> t, s
        | s -> raise_s [%message "Array type expected" (s: vectorized_type)]
      in
      let cstrs, s = unify_size cstrs s (SConst (List.length eles)) in
      let cstrs, eles =
        List.fold_map
          ~f:(fun cstrs e -> annotate_expr env cstrs e t)
          ~init:cstrs
          eles
      in
      cstrs, vect_expr (ArrayExpr eles) (SArray (t, s))
  (* | RowVectorExpr eles -> *)
  (*     fprintf ff "array([%a])" trans_exprs eles *)
  (* | Indexed (lhs, indices) -> *)
  (*     let cstrs, expected = *)
  (*       vectorized_type_of_unsized_type cstrs lhs.emeta.type_ *)
  (*     in *)
  (*     let cstrs, lhs = annotate_expr env cstrs lhs expected in *)
  (*     let _, cstrs, indices = *)
  (*       annotate_list *)
  (*         (fun env cstrs e -> *)
  (*            let cstrs, e = annotate_expr env cstrs e SInt in *)
  (*            env, cstrs, e) *)
  (*         env cstrs indices *)
  (*     in *)
  (*     let cstrs, expected = *)
  (*       vectorized_type_of_unsized_type cstrs texpr.emeta.type_ *)
  (*     in *)
  (*     cstrs, vect_expr (Indexed (lhs, indices)) expected *)
  | e ->
    (* XXX TODO XXX *)
    Format.eprintf "XXXX %a" UnsizedType.pp texpr.emeta.type_;
    let cstrs, expected =
      vectorized_type_of_unsized_type cstrs texpr.emeta.type_
    in
    let hack = ref cstrs in
    let e =
      map_expression
        (fun (texpr: typed_expression) ->
           let cstrs = !hack in
           let cstrs, expected =
             vectorized_type_of_unsized_type cstrs texpr.emeta.type_
           in
           let cstrs, expr = annotate_expr env cstrs texpr expected in
           hack := cstrs;
           expr)
        (fun kind -> kind)
        e
    in
    let cstrs = !hack in
    cstrs, vect_expr e expected

and annotate_binop env cstrs e1 op e2 expected =
  (* XXX TODO XXX *)
  let cstrs, e1 = annotate_expr env cstrs e1 expected in
  let cstrs, e2 = annotate_expr env cstrs e2 expected in
  cstrs, BinOp (e1, op, e2)

and annotate_unop env cstrs _op e expected =
  (* XXX TODO XXX *)
  let cstrs, e = annotate_expr env cstrs e expected in
  cstrs, e, expected


and annotate_stmt (env: 'a) (cstrs: 'b) (tstmt: typed_statement) : 'a * 'b * vectorized_statement =
  let vect_stmt stmt : vectorized_statement =
    { stmt; smeta = tstmt.smeta }
  in
  match tstmt.stmt with
  | Assignment {assign_lhs; assign_rhs; assign_op} ->
      let cstrs, lhs = annotate_lval env cstrs assign_lhs in
      let lhst = lhs.lmeta.type_ in
      let cstrs, rhs = annotate_expr env cstrs assign_rhs lhst in
      let stmt' = Assignment { assign_lhs = lhs; assign_rhs = rhs; assign_op } in
      env, cstrs, vect_stmt stmt'
  (* | NRFunApp (fn_kind, {name; _}, args) -> *)
  (*     trans_fun_app ff fn_kind name args *)
  | IncrementLogProb e | TargetPE e ->
     let cstrs, e = annotate_expr env cstrs e SReal in
     env, cstrs, vect_stmt (TargetPE e)
  (* | Tilde {arg; distribution; args; truncation} -> *)
  (*     let trans_distribution ff dist = *)
  (*       fprintf ff "%s" dist.name *)
  (*     in *)
  (*     let trans_truncation _ff = function *)
  (*       | NoTruncate -> () *)
  (*       | _ -> assert false (\* XXX TODO XXX *\) *)
  (*     in *)
  (*     fprintf ff "observe(%a, %a(%a), %a)%a" *)
  (*       (gen_id ctx) arg *)
  (*       trans_distribution distribution *)
  (*       trans_exprs args *)
  (*       trans_expr arg *)
  (*       trans_truncation truncation *)
  | Print ps ->
      let cstrs, ps = annotate_printables env cstrs ps in
      env, cstrs, vect_stmt (Print ps)
  | Reject ps ->
      let cstrs, ps = annotate_printables env cstrs ps in
      env, cstrs, vect_stmt (Reject ps)
  | IfThenElse (cond, ifb, None) ->
      let cstrs, cond = annotate_expr env cstrs cond SReal in
      let env, cstrs, ifb = annotate_stmt env cstrs ifb in
      env, cstrs, vect_stmt (IfThenElse (cond, ifb, None))
  | IfThenElse (cond, ifb, Some elseb) ->
      let cstrs, cond = annotate_expr env cstrs cond SReal in
      let env, cstrs, ifb = annotate_stmt env cstrs ifb in
      let env, cstrs, elseb = annotate_stmt env cstrs elseb in
      env, cstrs, vect_stmt (IfThenElse (cond, ifb, Some elseb))
  | While (cond, body) ->
      let cstrs, cond = annotate_expr env cstrs cond SReal in
      let env, cstrs, body = annotate_stmt env cstrs body in
      env, cstrs, vect_stmt (While (cond, body))
  | For {loop_variable; lower_bound; upper_bound; loop_body} ->
      let cstrs, lower_bound = annotate_expr env cstrs lower_bound SInt in
      let cstrs, upper_bound = annotate_expr env cstrs upper_bound SInt in
      let env_body =
        SEnv.add_exn env ~key:loop_variable.name ~data:SInt
      in
      let _, cstrs, loop_body = annotate_stmt env_body cstrs loop_body in
      let stmt =
        vect_stmt (For {loop_variable; lower_bound; upper_bound; loop_body})
      in
      env, cstrs, stmt
  (* | ForEach (loopvar, iteratee, body) -> *)
  (*     fprintf ff "@[<v4>for %s in %a:@,%a@]" *)
  (*       loopvar.name *)
  (*       trans_expr iteratee *)
  (*       (trans_stmt (loopvar.name :: ctx)) body *)
  (* | FunDef _ (\* {funname; arguments; body; _} *\) -> *)
  (*     assert false (\* XXX TODO XXX *\) *)
  | VarDecl { identifier; initial_value; decl_type; is_global; transformation } ->
      let cstrs, decl_type, expected =
        vectorized_type_of_type env cstrs decl_type
      in
      let cstrs, initial_value =
        opt_map_snd
          (fun cstrs e -> annotate_expr env cstrs e expected)
          cstrs initial_value
      in
      let cstrs, transformation =
        annotate_transformation env cstrs transformation
      in
      let env = SEnv.add_exn env ~key:identifier.name ~data:expected in
      let stmt =
        vect_stmt
          (VarDecl
             { identifier; initial_value; decl_type; is_global; transformation; })
      in
      env, cstrs, stmt
  | Block stmts ->
      let env, cstrs, stmts = annotate_list annotate_stmt env cstrs stmts in
      env, cstrs, vect_stmt (Block stmts)
  | Return e ->
      let cstrs, t = vectorized_type_of_unsized_type cstrs e.emeta.type_ in
      let cstrs, e = annotate_expr env cstrs e t in
      env, cstrs, vect_stmt (Return e)
  | ReturnVoid ->
      env, cstrs, vect_stmt ReturnVoid
  | Break ->
      env, cstrs, vect_stmt Break
  | Continue ->
      env, cstrs, vect_stmt Continue
  | Skip ->
      env, cstrs, vect_stmt Skip
  | s ->
    (* XXX TODO XXX *)
    let hack = ref (env, cstrs) in
    let s =
      map_statement
        (fun (texpr: typed_expression) ->
           let env, cstrs = !hack in
           let cstrs, expected =
             vectorized_type_of_unsized_type cstrs texpr.emeta.type_
           in
           let cstrs, expr = annotate_expr env cstrs texpr expected in
           hack := (env, cstrs);
           expr)
        (fun stmt ->
           let env, cstrs = !hack in
           let env, cstrs, stmt = annotate_stmt env cstrs stmt in
           hack := (env, cstrs);
           stmt)
        (fun lv ->
           let env, cstrs = !hack in
           let _, lv = annotate_lval env cstrs lv in
           hack := (env, cstrs);
           lv)
        (fun kind -> kind)
        s
    in
    let env, cstrs = !hack in
    env, cstrs, vect_stmt s
    (* map_typed_statement (fun _stmt -> assert false) stmt *)

and annotate_printables env cstrs ps =
  List.fold_map
    ~f:(fun cstrs -> function
        | PString s -> cstrs, PString s
        | PExpr e ->
            let cstrs, expected =
              vectorized_type_of_unsized_type cstrs e.emeta.type_
            in
            let cstrs, e = annotate_expr env cstrs e expected in
            cstrs, PExpr e)
    ~init:cstrs
    ps

and annotate_transformation env cstrs transformation =
  match transformation with
  | Program.Identity -> cstrs, Program.Identity
  | Lower lb ->
    let cstrs, lb = annotate_expr env cstrs lb SReal in
     cstrs, Lower lb
  | Upper ub ->
    let cstrs, ub = annotate_expr env cstrs ub SReal in
     cstrs, Upper ub
  | LowerUpper (lb, ub) ->
    let cstrs, lb = annotate_expr env cstrs lb SReal in
    let cstrs, ub = annotate_expr env cstrs ub SReal in
     cstrs, LowerUpper (lb, ub)
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

and vectorized_type_of_type env cstrs type_ =
  match type_ with
  | Type.Unsized ut ->
      let cstrs, t = vectorized_type_of_unsized_type cstrs ut in
      cstrs, Type.Unsized ut,  t
  | Sized t ->
      let cstrs, d, t = vectorized_expr_type_of_sized_type env cstrs t in
      cstrs, Type.Sized d,  t

and vectorized_expr_type_of_sized_type env cstrs t =
  match t with
  | SInt -> cstrs, SInt, SInt
  | SReal -> cstrs, SReal, SReal
  | SVector e ->
      let cstrs, s = fresh_sigma cstrs in
      let cstrs, e = annotate_expr env cstrs e (SVector s) in
      cstrs, SVector e, SVector (SExpr e)
  | SRowVector e ->
      let cstrs, s = fresh_sigma cstrs in
      let cstrs, e = annotate_expr env cstrs e (SVector s) in
      cstrs, SRowVector e, SRowVector (SExpr e)
  | SMatrix (e1, e2) ->
      let cstrs, s1 = fresh_sigma cstrs in
      let cstrs, s2 = fresh_sigma cstrs in
      let cstrs, e1 = annotate_expr env cstrs e1 (SVector s1) in
      let cstrs, e2 = annotate_expr env cstrs e2 (SVector s2) in
      cstrs, SMatrix (e1, e2), SMatrix (SExpr e1, SExpr e2)
  | SArray (t, e) ->
      let cstrs, s = fresh_sigma cstrs in
      let cstrs, d, t = vectorized_expr_type_of_sized_type env cstrs t in
      let cstrs, e = annotate_expr env cstrs e (SArray (t, s)) in
      cstrs, SArray (d, e), SArray (t, SExpr e)

and annotate_lval env cstrs = function
  | { lval = LVariable ident; lmeta } ->
    let actual = SEnv.find_exn env ident.name in
    let cstrs, expected = vectorized_type_of_unsized_type cstrs lmeta.type_ in
    let cstrs, t = unify cstrs actual expected in
    cstrs, { lval = LVariable ident; lmeta = { type_ = t; loc = lmeta.loc } }
  | { lval = LIndexed (_l, _i); lmeta = _ } ->
    (* let cstrs, lactual, l = annotate_lval env cstrs l in *)
    (* let t, sizes = split_type lactual i in *)
    (* let i = assert false (\* XXX TODO XXX *\) in *)
    (* { lval = LIndexed (l, i); lmeta }, SVector(Expr i) *)
    assert false (* XXX TODO XXX *)


let annotate_block env cstrs block =
  match block with
  | None -> (env, cstrs, None)
  | Some stmts ->
      let env, cstrs, stmts = annotate_list annotate_stmt env cstrs stmts in
      (env, cstrs, Some stmts)

let annotate_prog p =
  let env = SEnv.empty in
  let cstrs = IEnv.empty in
  let env, cstrs, functionblock =
    annotate_block env cstrs p.functionblock
  in
  let env, cstrs, datablock = annotate_block env cstrs p.datablock in
  let env, cstrs, transformeddatablock =
    annotate_block env cstrs p.transformeddatablock
  in
  let env, cstrs, parametersblock =
    annotate_block env cstrs p.parametersblock
  in
  let env, cstrs, transformedparametersblock =
    annotate_block env cstrs p.transformedparametersblock
  in
  let env, cstrs, modelblock = annotate_block env cstrs p.modelblock in
  let env, cstrs, generatedquantitiesblock =
    annotate_block env cstrs p.generatedquantitiesblock
  in
  let env, cstrs, guideparametersblock = 
    annotate_block env cstrs p.guideparametersblock
  in
  let _, _, guideblock = 
    annotate_block env cstrs p.guideblock
  in
  { functionblock; datablock; transformeddatablock;
    parametersblock; transformedparametersblock;
    modelblock; generatedquantitiesblock; 
    guideparametersblock; guideblock; }

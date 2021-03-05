open Core_kernel
open Frontend
open Ast
open Middle
open Format

module SSet = Set.Make(String)

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

(** {1 Name handling} *)

let cpt = ref 0
let gen_id, gen_name =
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
      | [] -> fprintf ff "'_%s__%d'" s !cpt
      | _ ->
          fprintf ff "f'_%s%a__%d'" s
            (pp_print_list ~pp_sep:(fun _ _ ->())
               (fun ff x -> fprintf ff "__{%s}" x)) l !cpt
    else
      fprintf ff "'%s'" s
  in
  let gen_name s =
    incr cpt;
    "_" ^ s ^ "__" ^ (string_of_int !cpt)
  in
  (gen_id, gen_name)

let keywords =
  [ "lambda"; "def"; ]

let avoid =
  let open Pyro_lib in
  "_f" ::
  keywords @ pyro_dppllib @ numpyro_dppllib @
  (List.map ~f:fst distributions) @ (List.map ~f:fst stanlib)

let trans_name ff name =
  let x =
    if List.mem ~equal:(=) avoid name then name ^ "__"
    else name
  in
  fprintf ff "%s" x

let trans_nn_base ff s =
  fprintf ff "%s_" s

let trans_id ff id =
  trans_name ff id.name

let stanlib_id id args =
  let arg_type arg =
    match arg.emeta.type_ with
    | UnsizedType.UInt -> "_int"
    | UReal -> "_real"
    | UVector -> "_vector"
    | URowVector -> "_rowvector"
    | UMatrix -> "_matrix"
    | UArray _ -> "_array"
    | UMathLibraryFunction | UFun _ -> "_function"
  in
  if
    List.exists
      ~f:(fun x -> String.is_suffix id.name ~suffix:x)
      ["_lpdf"; "_lupdf"; "_lpmf"; "_lupmf"; "_lcdf"; "_lccdf"; "_rng"]
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

(** {1 Number handling} *)

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


(** {1 Left values} *)

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

(** {1 Ast tests} *)

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

let is_pure stmt =
  let rec is_pure acc s =
    match s.stmt with
    | TargetPE _ | IncrementLogProb _ | Tilde _ -> false
    | _ ->
      fold_statement (fun b _ -> b)
        is_pure (fun b _ -> b) (fun b _ -> b) acc s.stmt
  in
  is_pure true stmt


(** {1 Type dimensions} *)

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

(** {1 Access functions} *)

let get_stanlib_calls program =
  let rec get_stanlib_calls_in_expr acc e =
    let acc =
      match e.expr with
      | FunApp (StanLib, id, args) | CondDistApp (StanLib, id, args) ->
        let sid = stanlib_id id args in
        if List.Assoc.mem ~equal:(=) Pyro_lib.stanlib sid then SSet.add acc sid
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

(** {1 Variable handling} *)

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

let get_var_decl_names stmts =
  List.fold_right
    ~f:(fun stmt acc ->
          match stmt.stmt with
          | VarDecl {identifier; _} -> identifier :: acc
          | _ -> acc)
    ~init:[] stmts

let get_var_decl_names_block block =
  Option.value_map ~default:[] ~f:get_var_decl_names block

let free_updated bv stmt =
  let fv = free_vars bv stmt in
  let arrays = updated_vars_stmt SSet.empty stmt in
  SSet.inter fv arrays

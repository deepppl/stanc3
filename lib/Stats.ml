(** Compute some statistics on Stan programs. *)

open Core_kernel
open Ast

module SSet = struct
  include Set.Make(String)

  type string_list = string list
  [@@deriving yojson]

  let to_yojson s =
    let l = to_list s in
    string_list_to_yojson l

  let of_yojson j =
    begin match string_list_of_yojson j with
    | Ok l -> Ok (of_list l)
    | Error msg -> Error msg
    end

end


type stats =
  { target: int
  ; left_expr: int
  ; functions: SSet.t
  ; unsampled: SSet.t
  ; sampled_under: (string * SSet.t) list
  ; sampled_over: (string * SSet.t) list
  ; resampled: (string * SSet.t) list
  ; improper_prior: bool
  ; parameters_transformation: bool }
[@@deriving yojson]

module Target : sig
  val untyped_program : stats -> untyped_program -> stats
end = struct

  let untyped_expression stats _ = stats

  let rec untyped_statement stats stmt =
    begin match stmt.stmt_untyped with
    | IncrementLogProb _
    | TargetPE _ ->  { stats with target = stats.target + 1 }
    | Assignment _
    | NRFunApp (_, _)
    | Tilde _
    | Break
    | Continue
    | Return _
    | ReturnVoid
    | Print _
    | Reject _
    | Skip
    | IfThenElse (_, _, _)
    | While (_, _)
    | For _
    | ForEach (_, _, _)
    | Block _
    | VarDecl _
    | FunDef _ ->
      fold_untyped_statement untyped_expression untyped_statement stats stmt
    end

  let untyped_program stats prog =
    fold_untyped_program untyped_statement stats prog

end

module Left_expr : sig
  val untyped_program : stats -> untyped_program -> stats
end = struct

  let is_lvalue e =
    begin match e with
    | Variable _
    | Indexed _ -> true
    | TernaryIf _
    | BinOp _
    | PrefixOp _
    | PostfixOp _
    | IntNumeral _
    | RealNumeral _
    | FunApp _
    | CondDistApp _
    | GetLP
    | GetTarget
    | ArrayExpr _
    | RowVectorExpr _
    | Paren _-> false
    end

  let untyped_expression stats _ = stats

  let rec untyped_statement stats stmt =
    begin match stmt.stmt_untyped with
    | Tilde { arg = e; _ } ->
      if is_lvalue e.expr_untyped then stats
      else { stats with left_expr = stats.left_expr + 1; }
    | Assignment _
    | NRFunApp (_, _)
    | TargetPE _
    | IncrementLogProb _
    | Break
    | Continue
    | Return _
    | ReturnVoid
    | Print _
    | Reject _
    | Skip
    | IfThenElse (_, _, _)
    | While (_, _)
    | For _
    | ForEach (_, _, _)
    | Block _
    | VarDecl _
    | FunDef _ ->
      fold_untyped_statement untyped_expression untyped_statement stats stmt
    end

  let untyped_program stats prog =
    fold_untyped_program untyped_statement stats prog

end


module Calls : sig
  val untyped_program : stats -> untyped_program -> stats
end = struct

  let rec untyped_expression stats expr =
    begin match expr.expr_untyped with
    | FunApp (f, _) ->
      let stats = { stats with functions = SSet.add stats.functions f.name } in
      fold_untyped_expression untyped_expression stats expr
    | TernaryIf _
    | BinOp _
    | PrefixOp _
    | PostfixOp _
    | Variable _
    | IntNumeral _
    | RealNumeral _
    | CondDistApp _
    | GetLP
    | GetTarget
    | ArrayExpr _
    | RowVectorExpr _
    | Paren _
    | Indexed _ ->
      fold_untyped_expression untyped_expression stats expr
    end

  let rec untyped_statement stats stmt =
    begin match stmt.stmt_untyped with
    | NRFunApp (f, _) ->
      let stats = { stats with functions = SSet.add stats.functions f.name } in
      fold_untyped_statement untyped_expression untyped_statement stats stmt
    | Assignment _
    | TargetPE _
    | IncrementLogProb _
    | Tilde _
    | Break
    | Continue
    | Return _
    | ReturnVoid
    | Print _
    | Reject _
    | Skip
    | IfThenElse (_, _, _)
    | While (_, _)
    | For _
    | ForEach (_, _, _)
    | Block _
    | VarDecl _
    | FunDef _ ->
      fold_untyped_statement untyped_expression untyped_statement stats stmt
    end

  let untyped_program stats prog =
    fold_untyped_program untyped_statement stats prog

end


module Resampling : sig
  val untyped_program : stats -> untyped_program -> stats
end = struct

  let fv =
    let rec fv acc expr =
      begin match expr.expr_untyped with
        | Variable x -> SSet.add acc x.name
        | TernaryIf _
        | BinOp _
        | PrefixOp _
        | PostfixOp _
        | IntNumeral _
        | RealNumeral _
        | FunApp _
        | CondDistApp _
        | GetLP
        | GetTarget
        | ArrayExpr _
        | RowVectorExpr _
        | Paren _
        | Indexed _ ->
          fold_untyped_expression fv acc expr
      end
    in
    fv SSet.empty

  let deps =
    let rec deps acc expr =
      begin match expr.expr_untyped with
        | Variable x -> (x.name, SSet.empty) :: acc
        | Indexed (e, l) ->
          let d = deps [] e in
          let idx =
            List.fold_left l
              ~init:SSet.empty
              ~f:(fun acc x ->
                  fold_index (fun acc e -> SSet.union (fv e) acc) acc x)
          in
          List.fold_left d
            ~init:acc
            ~f:(fun acc (x, i) -> (x, SSet.union i idx) :: acc)
        | TernaryIf _
        | BinOp _
        | PrefixOp _
        | PostfixOp _
        | IntNumeral _
        | RealNumeral _
        | FunApp _
        | CondDistApp _
        | GetLP
        | GetTarget
        | ArrayExpr _
        | RowVectorExpr _
        | Paren _ ->
          fold_untyped_expression deps acc expr
      end
    in
    deps []


  let untyped_expression stats _ = stats

  let rec untyped_statement_aux
      (curr_idx, sampled_under, sampled_over, resampled) stmt =
    begin match stmt.stmt_untyped with
    | Tilde { arg = e; _ } ->
        let vars = deps e in
        let t, f =
          List.partition_tf vars
            ~f:(fun x ->
                List.mem
                  ~equal:(fun (x,i) (y,j) -> x = y && SSet.equal i j)
                  sampled_over x)
        in
        let sampled_under, sampled_over, resampled =
          (vars @ sampled_under, f @ sampled_over, t @ resampled)
        in
        let sampled_under, sampled_over, resampled =
          begin match curr_idx with
          | None -> (sampled_under, sampled_over, resampled)
          | Some i ->
            let r =
              List.filter vars ~f:(fun (_, idx) -> not (SSet.mem idx i))
            in
            (sampled_under, sampled_over, r @ resampled)
          end
        in
        (curr_idx, sampled_under, sampled_over, resampled)
    | While (_, _) ->
      let _, sampled_under, sampled_over, resampled =
        fold_untyped_statement untyped_expression untyped_statement_aux
          (Some "", sampled_under, sampled_over, resampled) stmt
      in
      (curr_idx, sampled_under, sampled_over, resampled)
    | For { loop_variable = i; _} ->
      let _, sampled_under, sampled_over, resampled =
        fold_untyped_statement untyped_expression untyped_statement_aux
          (Some i.name, sampled_under, sampled_over, resampled) stmt
      in
      (curr_idx, sampled_under, sampled_over, resampled)
    | ForEach (i, _, _) ->
      let _, sampled_under, sampled_over, resampled =
        fold_untyped_statement untyped_expression untyped_statement_aux
          (Some i.name, sampled_under, sampled_over, resampled) stmt
      in
      (curr_idx, sampled_under, sampled_over, resampled)
    | IfThenElse (_, s1, None) ->
      let _, _, sampled_over, resampled =
        fold_untyped_statement untyped_expression untyped_statement_aux
          (curr_idx, [], sampled_over, resampled) s1
      in
      (curr_idx, sampled_under, sampled_over, resampled)
    | IfThenElse (_, s1, Some s2) ->
      let _, sampled_under1, sampled_over, resampled =
        fold_untyped_statement untyped_expression untyped_statement_aux
          (curr_idx, [], sampled_over, resampled) s1
      in
      let _, sampled_under2, sampled_over, resampled =
        fold_untyped_statement untyped_expression untyped_statement_aux
          (curr_idx, [], sampled_over, resampled) s2
      in
      let sampled_under =
        sampled_under @
        List.dedup_and_sort ~compare:compare (sampled_under1 @ sampled_under2)
      in
      (curr_idx, sampled_under, sampled_over, resampled)
    | Assignment _
    | NRFunApp (_, _)
    | TargetPE _
    | IncrementLogProb _
    | Break
    | Continue
    | Return _
    | ReturnVoid
    | Print _
    | Reject _
    | Skip
    | Block _
    | VarDecl _
    | FunDef _ ->
      fold_untyped_statement untyped_expression untyped_statement_aux
        (curr_idx, sampled_under, sampled_over, resampled) stmt
    end

  let get_parameters prog =
    let rec fold_stmt acc stmt =
      begin match stmt.stmt_untyped with
        | VarDecl { identifier = id; _ } -> SSet.add acc id.name
        | Tilde _
        | While _
        | For _
        | ForEach _
        | IfThenElse _
        | Assignment _
        | NRFunApp (_, _)
        | TargetPE _
        | IncrementLogProb _
        | Break
        | Continue
        | Return _
        | ReturnVoid
        | Print _
        | Reject _
        | Skip
        | Block _
        | FunDef _ ->
          fold_untyped_statement untyped_expression fold_stmt acc stmt
      end
    in
    let olfold f acc o =
      Option.fold o ~init:acc
        ~f:(fun acc x -> List.fold_left x ~init:acc ~f)
    in
    olfold fold_stmt SSet.empty prog.parametersblock

  let untyped_program stats prog =
    let _, sampled_under, sampled_over, resampled =
      fold_untyped_program untyped_statement_aux
      (None, stats.sampled_under, stats.sampled_over, stats.resampled) prog
    in
    let unsampled =
      SSet.filter (get_parameters prog)
        ~f:(fun x -> not (List.exists sampled_under ~f:(fun (y, _) -> x = y)))
    in
    { stats with unsampled; sampled_under; sampled_over; resampled }

end

module Parameters : sig
  val untyped_program : stats -> untyped_program -> stats
end = struct

  let untyped_expression stats _ = stats

  let rec untyped_statement stats stmt =
    begin match stmt.stmt_untyped with
    | VarDecl { transformation = t; _} ->
      begin match t with
      | Identity | Lower _ | Upper _ -> { stats with improper_prior = true }
      | LowerUpper _ -> stats
      | _ -> { stats with parameters_transformation = true }
      end
    | Assignment _
    | NRFunApp (_, _)
    | TargetPE _
    | IncrementLogProb _
    | Tilde _
    | Break
    | Continue
    | Return _
    | ReturnVoid
    | Print _
    | Reject _
    | Skip
    | IfThenElse (_, _, _)
    | While (_, _)
    | For _
    | ForEach (_, _, _)
    | Block _
    | FunDef _ ->
      fold_untyped_statement untyped_expression untyped_statement stats stmt
    end

  let untyped_program stats prog =
    let olfold f acc o =
      Option.fold o ~init:acc
        ~f:(fun acc x -> List.fold_left x ~init:acc ~f)
    in
    let stats =
      olfold untyped_statement stats prog.parametersblock
    in
    (* olfold Parameters.untyped_statement stats prog.transformedparametersblock *)
    stats

end


let stats_untyped_program prog =
  let stats =
    { target = 0; left_expr = 0; functions = SSet.empty;
      unsampled = SSet.empty;
      sampled_under = []; sampled_over = []; resampled = [];
      improper_prior = false; parameters_transformation = false; }
  in
  let stats = Target.untyped_program stats prog in
  let stats = Left_expr.untyped_program stats prog in
  let stats = Calls.untyped_program stats prog in
  let stats = Resampling.untyped_program stats prog in
  let stats = Parameters.untyped_program stats prog in
  stats

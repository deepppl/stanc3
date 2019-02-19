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
  ; sampled: (string * SSet.t) list
  ; resampled: (string * SSet.t) list
  ; improper_prior: bool
  ; parameters_transformation: bool }
[@@deriving yojson]

module Target : sig
  val untyped_statement : stats -> untyped_statement -> stats
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

end

module Left_expr : sig
  val untyped_statement : stats -> untyped_statement -> stats
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

end


module Calls : sig
  val untyped_statement : stats -> untyped_statement -> stats
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

end


module Resampling : sig
  val untyped_statement : stats -> untyped_statement -> stats
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

  let rec untyped_statement_aux (curr_idx, sampled, resampled) stmt =
    begin match stmt.stmt_untyped with
    | Tilde { arg = e; _ } ->
        let vars = deps e in
        let t, f =
          List.partition_tf vars
            ~f:(fun x ->
                List.mem
                  ~equal:(fun (x,i) (y,j) -> x = y && SSet.equal i j)
                  sampled x)
        in
        let sampled, resampled = (f @ sampled, t @ resampled) in
        let sampled, resampled =
          begin match curr_idx with
          | None -> (sampled, resampled)
          | Some i ->
            let r =
              List.filter vars ~f:(fun (_, idx) -> not (SSet.mem idx i))
            in
            (sampled, r @ resampled)
          end
        in
        (curr_idx, sampled, resampled)
    | While (_, _) ->
      let _, sampled, resampled =
        fold_untyped_statement untyped_expression untyped_statement_aux
          (Some "", sampled, resampled) stmt
      in
      (curr_idx, sampled, resampled)
    | For { loop_variable = i; _} ->
      let _, sampled, resampled =
        fold_untyped_statement untyped_expression untyped_statement_aux
          (Some i.name, sampled, resampled) stmt
      in
      (curr_idx, sampled, resampled)
    | ForEach (i, _, _) ->
      let _, sampled, resampled =
        fold_untyped_statement untyped_expression untyped_statement_aux
          (Some i.name, sampled, resampled) stmt
      in
      (curr_idx, sampled, resampled)
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
    | Block _
    | VarDecl _
    | FunDef _ ->
      fold_untyped_statement untyped_expression untyped_statement_aux
        (curr_idx, sampled, resampled) stmt
    end

  let untyped_statement stats stmt =
    let _, sampled, resampled =
      untyped_statement_aux (None, stats.sampled, stats.resampled) stmt
    in
    { stats with sampled; resampled }

end

module Parameters : sig
  val untyped_statement : stats -> untyped_statement -> stats
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

end


let stats_untyped_program prog =
  let stats =
    { target = 0; left_expr = 0; functions = SSet.empty;
      sampled = []; resampled = [];
      improper_prior = false; parameters_transformation = false; }
  in
  let stats =
    fold_untyped_program
      (fun acc stmt ->
         let acc = Target.untyped_statement acc stmt in
         let acc = Left_expr.untyped_statement acc stmt in
         let acc = Calls.untyped_statement acc stmt in
         let acc = Resampling.untyped_statement acc stmt in
         acc)
      stats
      prog
  in
  let stats =
    let olfold f acc o =
      Option.fold o ~init:acc
        ~f:(fun acc x -> List.fold_left x ~init:acc ~f)
    in
    let stats =
      olfold Parameters.untyped_statement stats prog.parametersblock
    in
    (* olfold Parameters.untyped_statement stats prog.transformedparametersblock *)
    stats
  in
  stats

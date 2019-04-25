open Mir

type t =
  | IdentifierIsKeyword of location_span * string
  | IdentifierIsModelName of location_span * string
  | IdentifierIsStanMathName of location_span * string
  | IdentifierInUse of location_span * string
  | IdentifierNotInScope of location_span * string
  | InvalidIndex of location_span * unsizedtype
  | TargetOutsideModelBlock of location_span
  | IllTypedIfReturnTypes of location_span * returntype * returntype
  | IllTypedTernaryIf of
      location_span * unsizedtype * unsizedtype * unsizedtype
  | IllTypedReturningFunction of location_span * string
  | IllTypedNonReturningFunction of location_span * string
  | IllTypedStanLibFunApp of location_span * string * unsizedtype list
  | IllTypedUserFunApp of
      location_span
      * string
      * (autodifftype * unsizedtype) list
      * returntype
      * unsizedtype list
  | IllTypedNotAFunction of location_span * string
  | IllTypedNotANRFunction of location_span * string
  | IllTypedNoSuchFunction of location_span * string
  | IllTypedNoSuchNRFunction of location_span * string
  | IllTypedBinOp of location_span * Ast.operator * unsizedtype * unsizedtype
  | IllTypedPrefixOp of location_span * Ast.operator * unsizedtype
  | IllTypedPostfixOp of location_span * Ast.operator * unsizedtype
  | FnMapRect of location_span * string
  | FnConditioning of location_span
  | FnTargetPlusEquals of location_span
  | FnRng of location_span

val to_exception : t -> 'a
val pretty_print_all_operator_signatures : string -> string

val operator_return_type :
  Ast.operator -> Ast.typed_expression list -> Mir.returntype option

val operator_return_type_from_string :
  string -> Ast.typed_expression list -> Mir.returntype option

"""
Lark Grammar for Module DSL

This module contains the EBNF grammar for parsing Module DSL source files.
The grammar supports indentation-sensitive syntax similar to Python.
"""

# The grammar string for Lark parser
GRAMMAR = r'''
// =============================================================================
// Top-level Program
// =============================================================================

start: _NL* (declaration _NL*)*

declaration: import_decl
           | module_decl
           | block_decl
           | model_decl
           | primitive_decl
           | recipe_decl

// =============================================================================
// Imports
// =============================================================================

import_decl: "import" module_path ["as" NAME] -> simple_import
           | "from" module_path "import" import_list -> from_import

module_path: NAME ("." NAME)* ["." VERSION]
VERSION: "v" INT

import_list: import_item ("," import_item)*
import_item: NAME ["as" NAME]

// =============================================================================
// Module Declaration
// =============================================================================

module_decl: ["abstract"] "module" NAME "(" [param_list] ")" ["extends" NAME] ":" _NL _INDENT module_body _DEDENT

module_body: [docstring] [let_section] [params_section] [forward_section] [backward_section]

// =============================================================================
// Block Declaration
// =============================================================================

block_decl: "block" NAME "(" [param_list] ")" ["extends" NAME] ":" _NL _INDENT block_body _DEDENT

block_body: [docstring] [let_section] [params_section] (pattern_section | ([forward_section] [backward_section]))

pattern_section: "pattern:" NAME _NL _INDENT pattern_body _DEDENT
pattern_body: ("sublayers:" _NL _INDENT sublayer_list _DEDENT)?
sublayer_list: (sublayer_item _NL)*
sublayer_item: "-" "(" NAME ("," NAME)* ")"

// =============================================================================
// Model Declaration
// =============================================================================

model_decl: "model" NAME "(" [param_list] ")" ":" _NL _INDENT model_body _DEDENT

model_body: [docstring] [let_section] [params_section] [forward_section] [backward_section] [hf_config_section] [hf_mapping_section] [hf_export_section]

// =============================================================================
// Primitive Declaration
// =============================================================================

primitive_decl: "primitive" NAME ":" _NL _INDENT primitive_body _DEDENT

primitive_body: [docstring] [primitive_params_section] primitive_forward primitive_backward [save_list] [impl_section] [invariants_section] [memory_section] [precision_section] [fusion_section]

primitive_params_section: "params:" _NL _INDENT (primitive_param _NL)* _DEDENT
primitive_param: NAME ":" type_annotation ["=" expression]

primitive_forward: "forward:" _NL _INDENT io_spec _DEDENT
primitive_backward: "backward:" _NL _INDENT (io_spec | backward_expr_list) _DEDENT

io_spec: "in:" io_type _NL "out:" io_type _NL
io_type: tensor_type
       | "(" named_tensor ("," named_tensor)* ")" -> named_tuple_type
       | named_tensor -> single_named_tensor

named_tensor: NAME ":" tensor_type

backward_expr_list: (backward_expr _NL)*
backward_expr: NAME "=" expression

impl_section: "impl:" _NL _INDENT impl_body _DEDENT
impl_body: ("forward:" kernel_ref _NL)? ("backward:" kernel_ref _NL)?
kernel_ref: module_path | "pointer_arithmetic" | "metadata_only"

invariants_section: "invariants:" _NL _INDENT (invariant_item _NL)* _DEDENT
invariant_item: "-" NAME ":" "[" NAME ("," NAME)* "]"

memory_section: "memory:" _NL _INDENT (memory_item _NL)* _DEDENT
memory_item: NAME ":" expression

precision_section: "precision:" _NL _INDENT (precision_item _NL)* _DEDENT
precision_item: NAME ":" "[" dtype ("," dtype)* "]"

fusion_section: "fusion:" _NL _INDENT fusion_body _DEDENT
fusion_body: ("patterns:" _NL _INDENT (fusion_pattern _NL)* _DEDENT)?
fusion_pattern: "-" "[" NAME ("," NAME)* "]" "->" NAME

// =============================================================================
// Recipe Declaration
// =============================================================================

recipe_decl: "recipe" NAME ":" _NL _INDENT recipe_body _DEDENT
recipe_body: (recipe_item _NL)*
recipe_item: NAME ":" expression

// =============================================================================
// Common Sections
// =============================================================================

docstring: DOCSTRING _NL

let_section: "let:" _NL _INDENT (let_binding _NL | constraint_section)* _DEDENT
let_binding: NAME "=" expression
constraint_section: "constraint:" _NL _INDENT (constraint_stmt _NL)* _DEDENT
constraint_stmt: expression "," STRING

params_section: "params:" _NL _INDENT (tensor_param _NL)* _DEDENT
tensor_param: NAME ":" tensor_type_or_module [condition_clause] annotation_list?
            | NAME ":" array_type annotation_list?
tensor_type_or_module: tensor_type | module_instantiation | "tied_to" "(" NAME ")"
condition_clause: "if" expression
module_instantiation: NAME "(" [arg_list] ")"
annotation_list: annotation+

forward_section: "forward:" _NL _INDENT forward_body _DEDENT
forward_body: [inputs_spec] [input_spec] [outputs_spec] [output_spec] graph_section [save_list] [recompute_list]

inputs_spec: "inputs:" _NL _INDENT (named_io _NL)* _DEDENT
outputs_spec: "outputs:" _NL _INDENT (named_io _NL)* _DEDENT
named_io: NAME ":" tensor_type

input_spec: "in:" io_type _NL
output_spec: "out:" io_type _NL

backward_section: "backward:" _NL _INDENT backward_body _DEDENT
backward_body: [gradient_inputs_section] [gradient_outputs_section] graph_section

gradient_inputs_section: "receives:" _NL _INDENT (gradient_decl _NL)* _DEDENT
gradient_outputs_section: "produces:" _NL _INDENT (gradient_decl _NL)* _DEDENT
gradient_decl: "d_" NAME ":" tensor_type

graph_section: "graph:" _NL _INDENT (graph_stmt _NL)* _DEDENT

save_list: "save:" tensor_list _NL
recompute_list: "recompute:" tensor_list _NL
tensor_list: "[" [NAME ("," NAME)*] "]"

// =============================================================================
// Graph Statements
// =============================================================================

graph_stmt: data_flow_stmt
          | conditional_stmt
          | recompute_block

data_flow_stmt: source "->" (operation "->")*  destination annotation_list?
source: tensor_ref
      | "(" tensor_ref ("," tensor_ref)* ")" -> tuple_source
      | call_expr -> call_source

tensor_ref: NAME -> simple_ref
          | "saved" "." NAME -> saved_ref
          | NAME "[" slice_list "]" -> indexed_ref

slice_list: slice_item ("," slice_item)*
slice_item: expression -> index_item
          | expression? ":" expression? -> slice_range
          | "..." -> ellipsis_item

destination: tensor_ref -> simple_dest
           | "(" tensor_ref ("," tensor_ref)* ")" -> tuple_dest
           | "_" -> discard_dest

operation: NAME "(" [arg_list] ")"

conditional_stmt: "if" expression ":" _NL _INDENT (graph_stmt _NL)* _DEDENT ["else:" _NL _INDENT (graph_stmt _NL)* _DEDENT]

recompute_block: "recompute:" _NL _INDENT (graph_stmt _NL)* _DEDENT

// =============================================================================
// HuggingFace Sections
// =============================================================================

hf_config_section: "hf_config:" _NL _INDENT hf_config_body _DEDENT
hf_config_body: (hf_config_item _NL)*
hf_config_item: NAME ":" (STRING | hf_param_mapping)
hf_param_mapping: _NL _INDENT (NAME ":" NAME _NL)* _DEDENT

hf_mapping_section: "hf_mapping:" _NL _INDENT (hf_weight_mapping _NL)* _DEDENT
hf_export_section: "hf_export:" _NL _INDENT (hf_weight_mapping _NL)* _DEDENT
hf_weight_mapping: weight_pattern ":" hf_weight_spec
weight_pattern: NAME ("[" "{" NAME "}" "]" "." NAME)*
              | NAME ("." NAME)*

hf_weight_spec: STRING -> direct_mapping
              | "fuse" "(" STRING ("," STRING)* ["," "dim" "=" INT] ")" -> fuse_mapping
              | "transform" "(" STRING "," "fn:" NAME ")" -> transform_mapping
              | "split" "(" split_spec ("," split_spec)* ["," "dim" "=" INT] ")" -> split_mapping
              | "tied_to" "(" NAME ")" -> tied_mapping
split_spec: STRING ":" "[" INT "," INT "]"

// =============================================================================
// Annotations
// =============================================================================

annotation: "@" NAME ["(" [annotation_args] ")"]
annotation_args: annotation_arg ("," annotation_arg)*
annotation_arg: expression -> positional_arg
              | NAME "=" expression -> keyword_arg

// =============================================================================
// Types
// =============================================================================

tensor_type: "[" shape_list ["," dtype] "]" ["?"]
shape_list: shape_dim ("," shape_dim)*
shape_dim: "*" -> variadic_dim
         | expression -> expr_dim

array_type: "[" expression "]" ("×" | "x") (NAME | module_instantiation) -> array_type_mul
          | "Array" "<" expression "," (NAME | module_instantiation) ">" -> array_type_generic

dtype: "bf16" | "fp32" | "fp16" | "fp8_e4m3" | "fp8_e5m2" | "fp4_e2m1" | "int8" | "int32"

type_annotation: "int" | "float" | "bool" | "string"
               | "enum" "(" NAME ("," NAME)* ")"

// =============================================================================
// Parameters
// =============================================================================

param_list: param ("," param)*
param: NAME [":" type_annotation] ["=" expression]

// =============================================================================
// Expressions
// =============================================================================

expression: ternary_expr

ternary_expr: or_expr ["if" or_expr "else" ternary_expr]

or_expr: and_expr ("or" and_expr)*

and_expr: not_expr ("and" not_expr)*

not_expr: "not" not_expr -> not_op
        | comparison

comparison: arith_expr (comp_op arith_expr)*
comp_op: "==" | "!=" | "<" | ">" | "<=" | ">="

arith_expr: term ((ADD_OP) term)*
ADD_OP: "+" | "-"

term: factor ((MUL_OP) factor)*
MUL_OP: "*" | "/" | "//" | "%"

factor: (UNARY_OP)? power
UNARY_OP: "+" | "-"

power: atom ["**" factor]

atom: "(" expression ")" -> paren_expr
    | "[" [expression ("," expression)*] "]" -> list_literal
    | "{" [dict_item ("," dict_item)*] "}" -> dict_literal
    | call_expr
    | attribute_expr
    | index_expr
    | NAME -> identifier
    | literal

dict_item: expression ":" expression

call_expr: NAME "(" [arg_list] ")"
arg_list: arg ("," arg)*
arg: expression -> positional_arg
   | NAME "=" expression -> keyword_arg

attribute_expr: NAME "." NAME ("." NAME)*

index_expr: NAME "[" expression "]"

literal: INT -> int_literal
       | FLOAT -> float_literal
       | STRING -> string_literal
       | "true" -> true_literal
       | "false" -> false_literal
       | "None" -> none_literal

// =============================================================================
// Tokens
// =============================================================================

NAME: /[a-zA-Z_][a-zA-Z0-9_]*/
INT: /[0-9]+/
FLOAT: /[0-9]+\.[0-9]+([eE][+-]?[0-9]+)?/
STRING: /"[^"]*"/ | /'[^']*'/
DOCSTRING: /\"{3}[^"]*\"{3}/

// Comments
COMMENT: /#[^\n]*/

// Whitespace handling (for indentation-sensitive parsing)
%import common.WS_INLINE
%ignore WS_INLINE
%ignore COMMENT

// Newline terminal (transformed by Indenter postlexer)
_NL: /(\r?\n[\t ]*)+/

// Declare INDENT/DEDENT tokens (handled by postlexer)
%declare _INDENT _DEDENT
'''


# Simplified grammar for initial parsing (without full indentation handling)
SIMPLE_GRAMMAR = r'''
// Simplified grammar for testing - uses explicit braces instead of indentation

start: (declaration)*

declaration: import_decl
           | module_decl
           | block_decl
           | model_decl
           | primitive_decl

// Imports
import_decl: "import" module_path ("as" NAME)?
           | "from" module_path "import" import_list

module_path: NAME ("." NAME)*

import_list: import_item ("," import_item)*
import_item: NAME ("as" NAME)?

// Module Declaration (simplified)
module_decl: "abstract"? "module" NAME "(" param_list? ")" ("extends" NAME)? "{" module_body "}"

module_body: docstring? let_section? params_section? forward_section? backward_section?

// Block Declaration (simplified)
block_decl: "block" NAME "(" param_list? ")" ("extends" NAME)? "{" block_body "}"

block_body: docstring? let_section? params_section? forward_section? backward_section?

// Model Declaration (simplified)
model_decl: "model" NAME "(" param_list? ")" "{" model_body "}"

model_body: docstring? let_section? params_section? forward_section? backward_section? hf_config_section? hf_mapping_section?

// Primitive Declaration (simplified)
primitive_decl: "primitive" NAME "{" primitive_body "}"

primitive_body: docstring? params_section? forward_section backward_section impl_section?

// Common Sections (simplified with braces)
docstring: DOCSTRING

let_section: "let" "{" (let_binding ";")* "}"
let_binding: NAME "=" expression

params_section: "params" "{" (tensor_param ";")* "}"
tensor_param: NAME ":" tensor_type annotation*

forward_section: "forward" "{" io_spec? graph_section save_clause? "}"
backward_section: "backward" "{" io_spec? graph_section "}"

io_spec: ("in" ":" io_type ";")? ("out" ":" io_type ";")?
io_type: tensor_type | "(" tensor_type ("," tensor_type)* ")"

graph_section: "graph" "{" (graph_stmt ";")* "}"
save_clause: "save" ":" tensor_list ";"

// Graph Statements
graph_stmt: source "->" (operation "->")*  destination annotation*

source: NAME
      | "(" NAME ("," NAME)* ")"
      | "saved" "." NAME

destination: NAME
           | "(" NAME ("," NAME)* ")"

operation: NAME "(" arg_list? ")"

// Implementation
impl_section: "impl" "{" ("forward" ":" NAME ";")? ("backward" ":" NAME ";")? "}"

// HuggingFace Sections
hf_config_section: "hf_config" "{" (hf_config_item ";")* "}"
hf_config_item: NAME ":" STRING

hf_mapping_section: "hf_mapping" "{" (hf_weight_mapping ";")* "}"
hf_weight_mapping: NAME ":" hf_weight_spec

hf_weight_spec: STRING
              | "fuse" "(" STRING ("," STRING)* ("," "dim" "=" INT)? ")"

// Annotations
annotation: "@" NAME ("(" annotation_args ")")?
annotation_args: annotation_arg ("," annotation_arg)*
annotation_arg: NAME "=" expression
              | expression

// Types
tensor_type: "[" shape_list ("," dtype)? "]" "?"?
shape_list: shape_dim ("," shape_dim)*
shape_dim: "*" | expression

array_type: "[" expression "]" "x" NAME

dtype: "bf16" | "fp32" | "fp16" | "fp8_e4m3" | "fp8_e5m2" | "fp4_e2m1" | "int8" | "int32"

// Parameters
param_list: param ("," param)*
param: NAME (":" NAME)? ("=" expression)?

// Tensor list
tensor_list: "[" (NAME ("," NAME)*)? "]"

// Expressions (simplified)
expression: or_expr

or_expr: and_expr ("or" and_expr)*

and_expr: comparison ("and" comparison)*

comparison: arith_expr (COMP_OP arith_expr)?
COMP_OP: "==" | "!=" | "<" | ">" | "<=" | ">="

arith_expr: term ((ADD_OP) term)*
ADD_OP: "+" | "-"

term: factor ((MUL_OP) factor)*
MUL_OP: "*" | "/" | "//" | "%"

factor: UNARY_OP? atom
UNARY_OP: "+" | "-"

atom: "(" expression ")"
    | NAME "(" arg_list? ")"  -> call
    | NAME "." NAME           -> attribute
    | NAME "[" expression "]" -> index
    | NAME                    -> identifier
    | INT                     -> integer
    | FLOAT                   -> float
    | STRING                  -> string
    | "true"                  -> true
    | "false"                 -> false
    | "None"                  -> none

arg_list: arg ("," arg)*
arg: NAME "=" expression -> kwarg
   | expression -> posarg

// Tokens
NAME: /[a-zA-Z_][a-zA-Z0-9_]*/
INT: /[0-9]+/
FLOAT: /[0-9]+\.[0-9]+([eE][+-]?[0-9]+)?/
STRING: /"[^"]*"/ | /'[^']*'/
DOCSTRING: /\"{3}[\s\S]*?\"{3}/

COMMENT: /#[^\n]*/
%import common.WS
%ignore WS
%ignore COMMENT
'''

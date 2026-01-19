"""
Parser for Module DSL

Uses Lark to parse DSL source into AST nodes.
Supports both indentation-sensitive and brace-based syntax.
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import re

from lark import Lark, Transformer, v_args, Token, Tree
from lark.indenter import Indenter

from .grammar import GRAMMAR, SIMPLE_GRAMMAR
from .ast_nodes import (
    ASTNode,
    Program,
    ModuleNode,
    BlockNode,
    ModelNode,
    PrimitiveNode,
    RecipeNode,
    ImportDecl,
    ImportItem,
    ParamDecl,
    TensorDecl,
    TensorType,
    TupleTypeNode,
    ArrayTypeNode,
    LetBinding,
    ConstraintDecl,
    ForwardBlock,
    BackwardBlock,
    GraphBody,
    GraphStatement,
    ConditionalGraph,
    RecomputeBlock,
    TensorRef,
    TupleRef,
    Operation,
    Annotation,
    HFConfigMapping,
    WeightMapping,
    HFMappingSection,
    ModuleInstantiation,
    # Expressions
    Literal,
    Identifier,
    BinaryOp,
    UnaryOp,
    CallExpr,
    IndexExpr,
    SliceExpr,
    AttributeExpr,
    TernaryExpr,
)
from .errors import DSLSyntaxError, SourceLocation


class DSLIndenter(Indenter):
    """Custom indenter for handling Python-like indentation."""

    NL_type = "_NL"
    OPEN_PAREN_types = ["LPAR", "LSQB", "LBRACE"]
    CLOSE_PAREN_types = ["RPAR", "RSQB", "RBRACE"]
    INDENT_type = "_INDENT"
    DEDENT_type = "_DEDENT"
    tab_len = 4


class DSLTransformer(Transformer):
    """Transforms Lark parse tree into typed AST nodes."""

    def __init__(self, source_file: Optional[str] = None):
        super().__init__()
        self.source_file = source_file

    def _loc(self, meta) -> Optional[SourceLocation]:
        """Create source location from Lark meta."""
        if meta:
            return SourceLocation(
                file=self.source_file,
                line=getattr(meta, "line", 0),
                column=getattr(meta, "column", 0),
                end_line=getattr(meta, "end_line", None),
                end_column=getattr(meta, "end_column", None),
            )
        return None

    # =========================================================================
    # Top-level
    # =========================================================================

    def start(self, items) -> Program:
        """Process top-level program."""
        imports = []
        modules = []
        blocks = []
        models = []
        primitives = []
        recipes = []

        for item in items:
            if item is None:
                continue
            if isinstance(item, ImportDecl):
                imports.append(item)
            elif isinstance(item, ModuleNode):
                modules.append(item)
            elif isinstance(item, BlockNode):
                blocks.append(item)
            elif isinstance(item, ModelNode):
                models.append(item)
            elif isinstance(item, PrimitiveNode):
                primitives.append(item)
            elif isinstance(item, RecipeNode):
                recipes.append(item)

        return Program(
            imports=imports,
            modules=modules,
            blocks=blocks,
            models=models,
            primitives=primitives,
            recipes=recipes,
            source_file=self.source_file,
        )

    def declaration(self, items):
        return items[0] if items else None

    # =========================================================================
    # Imports
    # =========================================================================

    def simple_import(self, items):
        path = items[0]
        alias = str(items[1]) if len(items) > 1 and items[1] else None
        return ImportDecl(module_path=path, alias=alias)

    def from_import(self, items):
        path = items[0]
        import_items = items[1] if len(items) > 1 else []
        return ImportDecl(module_path=path, items=import_items)

    def module_path(self, items):
        parts = [str(item) for item in items if not str(item).startswith("v")]
        version = None
        for item in items:
            if str(item).startswith("v"):
                version = str(item)
        return ".".join(parts) + (f".{version}" if version else "")

    def import_list(self, items):
        return list(items)

    def import_item(self, items):
        name = str(items[0])
        alias = str(items[1]) if len(items) > 1 else None
        return ImportItem(name=name, alias=alias)

    # =========================================================================
    # Module Declaration
    # =========================================================================

    @v_args(meta=True)
    def module_decl(self, meta, items):
        # Parse items based on structure
        is_abstract = False
        name = None
        params = []
        extends = None
        body = {}

        for item in items:
            if item is None:
                continue
            if isinstance(item, Token):
                token_val = str(item)
                if token_val == "abstract":
                    is_abstract = True
                elif item.type == "NAME":
                    if name is None:
                        name = token_val
                    else:
                        extends = token_val
            elif isinstance(item, list):
                # Could be param_list
                if all(isinstance(p, ParamDecl) for p in item):
                    params = item
            elif isinstance(item, dict):
                body.update(item)

        return ModuleNode(
            name=name or "Unknown",
            params=params,
            extends=extends,
            is_abstract=is_abstract,
            docstring=body.get("docstring"),
            let_bindings=body.get("let_bindings", []),
            constraints=body.get("constraints", []),
            param_decls=body.get("param_decls", []),
            forward=body.get("forward"),
            backward=body.get("backward"),
            location=self._loc(meta),
        )

    def module_body(self, items):
        result = {}
        for item in items:
            if item is None:
                continue
            if isinstance(item, str):
                result["docstring"] = item
            elif isinstance(item, dict):
                result.update(item)
            elif isinstance(item, ForwardBlock):
                result["forward"] = item
            elif isinstance(item, BackwardBlock):
                result["backward"] = item
        return result

    # =========================================================================
    # Block Declaration
    # =========================================================================

    @v_args(meta=True)
    def block_decl(self, meta, items):
        name = None
        params = []
        extends = None
        body = {}

        for item in items:
            if item is None:
                continue
            if isinstance(item, Token) and item.type == "NAME":
                if name is None:
                    name = str(item)
                else:
                    extends = str(item)
            elif isinstance(item, list) and all(isinstance(p, ParamDecl) for p in item):
                params = item
            elif isinstance(item, dict):
                body.update(item)

        return BlockNode(
            name=name or "Unknown",
            params=params,
            extends=extends,
            docstring=body.get("docstring"),
            let_bindings=body.get("let_bindings", []),
            constraints=body.get("constraints", []),
            param_decls=body.get("param_decls", []),
            pattern=body.get("pattern"),
            pattern_config=body.get("pattern_config", {}),
            forward=body.get("forward"),
            backward=body.get("backward"),
            location=self._loc(meta),
        )

    def block_body(self, items):
        result = {}
        for item in items:
            if item is None:
                continue
            if isinstance(item, dict):
                result.update(item)
            elif isinstance(item, ForwardBlock):
                result["forward"] = item
            elif isinstance(item, BackwardBlock):
                result["backward"] = item
        return result

    # =========================================================================
    # Model Declaration
    # =========================================================================

    @v_args(meta=True)
    def model_decl(self, meta, items):
        name = None
        params = []
        body = {}

        for item in items:
            if item is None:
                continue
            if isinstance(item, Token) and item.type == "NAME":
                if name is None:
                    name = str(item)
            elif isinstance(item, list) and all(isinstance(p, ParamDecl) for p in item):
                params = item
            elif isinstance(item, dict):
                body.update(item)

        return ModelNode(
            name=name or "Unknown",
            params=params,
            docstring=body.get("docstring"),
            let_bindings=body.get("let_bindings", []),
            constraints=body.get("constraints", []),
            param_decls=body.get("param_decls", []),
            forward=body.get("forward"),
            backward=body.get("backward"),
            hf_config=body.get("hf_config"),
            hf_mapping=body.get("hf_mapping"),
            hf_export=body.get("hf_export"),
            location=self._loc(meta),
        )

    def model_body(self, items):
        result = {}
        for item in items:
            if item is None:
                continue
            if isinstance(item, dict):
                result.update(item)
            elif isinstance(item, ForwardBlock):
                result["forward"] = item
            elif isinstance(item, BackwardBlock):
                result["backward"] = item
            elif isinstance(item, HFConfigMapping):
                result["hf_config"] = item
            elif isinstance(item, HFMappingSection):
                if "hf_mapping" not in result:
                    result["hf_mapping"] = item
                else:
                    result["hf_export"] = item
        return result

    # =========================================================================
    # Primitive Declaration
    # =========================================================================

    @v_args(meta=True)
    def primitive_decl(self, meta, items):
        name = None
        body = {}

        for item in items:
            if item is None:
                continue
            if isinstance(item, Token) and item.type == "NAME":
                name = str(item)
            elif isinstance(item, dict):
                body.update(item)

        return PrimitiveNode(
            name=name or "Unknown",
            docstring=body.get("docstring"),
            params=body.get("params", []),
            forward_in=body.get("forward_in"),
            forward_out=body.get("forward_out"),
            backward_in=body.get("backward_in"),
            backward_out=body.get("backward_out"),
            backward_exprs=body.get("backward_exprs", {}),
            forward_impl=body.get("forward_impl"),
            backward_impl=body.get("backward_impl"),
            save=body.get("save", []),
            invariants=body.get("invariants", {}),
            location=self._loc(meta),
        )

    def primitive_body(self, items):
        result = {}
        for item in items:
            if item is None:
                continue
            if isinstance(item, dict):
                result.update(item)
        return result

    # =========================================================================
    # Common Sections
    # =========================================================================

    def docstring(self, items):
        text = str(items[0])
        # Strip triple quotes
        return text[3:-3].strip()

    def let_section(self, items):
        bindings = []
        constraints = []
        for item in items:
            if isinstance(item, LetBinding):
                bindings.append(item)
            elif isinstance(item, ConstraintDecl):
                constraints.append(item)
            elif isinstance(item, dict):
                bindings.extend(item.get("bindings", []))
                constraints.extend(item.get("constraints", []))
        return {"let_bindings": bindings, "constraints": constraints}

    def let_binding(self, items):
        name = str(items[0])
        value = items[1]
        return LetBinding(name=name, value=value)

    def constraint_section(self, items):
        return {"constraints": list(items)}

    def constraint_stmt(self, items):
        condition = items[0]
        message = str(items[1]).strip('"\'')
        return ConstraintDecl(condition=condition, message=message)

    def params_section(self, items):
        return {"param_decls": list(items)}

    def tensor_param(self, items):
        name = str(items[0])
        type_spec = items[1]
        condition = None
        annotations = []

        for item in items[2:]:
            if isinstance(item, Annotation):
                annotations.append(item)
            elif item is not None:
                condition = item

        if isinstance(type_spec, TensorType):
            return TensorDecl(
                name=name,
                tensor_type=type_spec,
                condition=condition,
                annotations=annotations,
            )
        return TensorDecl(name=name, tensor_type=type_spec, annotations=annotations)

    def tensor_type_or_module(self, items):
        # Return the first item (TensorType, ModuleInstantiation, or tied_to)
        return items[0] if items else None

    def annotation_list(self, items):
        # Return list of annotations
        return list(items)

    # =========================================================================
    # Forward/Backward Sections
    # =========================================================================

    def forward_section(self, items):
        result = ForwardBlock()
        for item in items:
            if item is None:
                continue
            if isinstance(item, dict):
                if "inputs" in item:
                    result.inputs = item["inputs"]
                if "outputs" in item:
                    result.outputs = item["outputs"]
                if "input_type" in item:
                    result.input_type = item["input_type"]
                if "output_type" in item:
                    result.output_type = item["output_type"]
                if "graph" in item:
                    result.graph = item["graph"]
                if "save" in item:
                    result.save = item["save"]
                if "recompute" in item:
                    result.recompute = item["recompute"]
            elif isinstance(item, GraphBody):
                result.graph = item
        return result

    def forward_body(self, items):
        result = {}
        for item in items:
            if item is None:
                continue
            if isinstance(item, GraphBody):
                result["graph"] = item
            elif isinstance(item, dict):
                result.update(item)
        return result

    def input_spec(self, items):
        return {"input_type": items[0] if items else None}

    def output_spec(self, items):
        return {"output_type": items[0] if items else None}

    def io_type(self, items):
        # Return the tensor type or tuple of types
        if len(items) == 1:
            return items[0]
        return TupleTypeNode(elements=list(items))

    def backward_section(self, items):
        result = BackwardBlock()
        for item in items:
            if item is None:
                continue
            if isinstance(item, dict):
                result.gradient_inputs.update(item.get("gradient_inputs", {}))
                result.gradient_outputs.update(item.get("gradient_outputs", {}))
            elif isinstance(item, GraphBody):
                result.graph = item
        return result

    def backward_body(self, items):
        result = {"gradient_inputs": {}, "gradient_outputs": {}}
        for item in items:
            if item is None:
                continue
            if isinstance(item, GraphBody):
                result["graph"] = item
            elif isinstance(item, dict):
                result.update(item)
        return result

    def gradient_inputs_section(self, items):
        result = {}
        for item in items:
            if isinstance(item, tuple) and len(item) == 2:
                result[item[0]] = item[1]
        return {"gradient_inputs": result}

    def gradient_outputs_section(self, items):
        result = {}
        for item in items:
            if isinstance(item, tuple) and len(item) == 2:
                result[item[0]] = item[1]
        return {"gradient_outputs": result}

    def gradient_decl(self, items):
        # d_NAME: tensor_type -> ("d_NAME", tensor_type)
        name = "d_" + str(items[0])
        tensor_type = items[1]
        return (name, tensor_type)

    def graph_section(self, items):
        statements = [item for item in items if item is not None]
        return GraphBody(statements=statements)

    def save_list(self, items):
        return {"save": items[0] if items else []}

    def recompute_list(self, items):
        return {"recompute": items[0] if items else []}

    def tensor_list(self, items):
        return [str(item) for item in items if isinstance(item, Token)]

    # =========================================================================
    # Graph Statements
    # =========================================================================

    def graph_stmt(self, items):
        return items[0] if items else None

    @v_args(meta=True)
    def data_flow_stmt(self, meta, items):
        source = None
        operations = []
        dest = None
        annotations = []

        for item in items:
            if item is None:
                continue
            if isinstance(item, (TensorRef, TupleRef)):
                if source is None:
                    source = item
                else:
                    dest = item
            elif isinstance(item, Operation):
                operations.append(item)
            elif isinstance(item, Annotation):
                annotations.append(item)
            elif isinstance(item, str):
                if source is None:
                    source = item
                else:
                    dest = item

        return GraphStatement(
            source=source or "in",
            operations=operations,
            dest=dest or "out",
            annotations=annotations,
            location=self._loc(meta),
        )

    def source(self, items):
        return items[0] if items else None

    def tuple_source(self, items):
        return TupleRef(elements=list(items))

    def simple_ref(self, items):
        return TensorRef(name=str(items[0]))

    def saved_ref(self, items):
        return TensorRef(name=str(items[0]), is_saved=True)

    def indexed_ref(self, items):
        name = str(items[0])
        # Handle indexing later
        return TensorRef(name=name)

    def destination(self, items):
        return items[0] if items else None

    def simple_dest(self, items):
        return items[0]

    def tuple_dest(self, items):
        return TupleRef(elements=list(items))

    def discard_dest(self, items):
        return TensorRef(name="_")

    @v_args(meta=True)
    def operation(self, meta, items):
        name = str(items[0])
        args = []
        kwargs = {}

        if len(items) > 1 and items[1] is not None:
            for arg in items[1:]:
                if isinstance(arg, tuple):
                    kwargs[arg[0]] = arg[1]
                elif arg is not None:
                    args.append(arg)

        return Operation(
            name=name,
            args=args,
            kwargs=kwargs,
            location=self._loc(meta),
        )

    @v_args(meta=True)
    def conditional_stmt(self, meta, items):
        condition = items[0]
        true_branch = []
        false_branch = None

        # Parse branches
        collecting_false = False
        for item in items[1:]:
            if item is None:
                continue
            if isinstance(item, list):
                if not collecting_false:
                    true_branch = item
                    collecting_false = True
                else:
                    false_branch = item
            elif isinstance(item, (GraphStatement, ConditionalGraph, RecomputeBlock)):
                if not collecting_false:
                    true_branch.append(item)
                else:
                    if false_branch is None:
                        false_branch = []
                    false_branch.append(item)

        return ConditionalGraph(
            condition=condition,
            true_branch=true_branch,
            false_branch=false_branch,
            location=self._loc(meta),
        )

    @v_args(meta=True)
    def recompute_block(self, meta, items):
        statements = [item for item in items if isinstance(item, (GraphStatement, ConditionalGraph))]
        return RecomputeBlock(statements=statements, location=self._loc(meta))

    # =========================================================================
    # Annotations
    # =========================================================================

    @v_args(meta=True)
    def annotation(self, meta, items):
        name = str(items[0])
        args = []
        kwargs = {}

        for item in items[1:]:
            if item is None:
                continue
            if isinstance(item, tuple):
                kwargs[item[0]] = item[1]
            else:
                args.append(item)

        return Annotation(
            name=name,
            args=args,
            kwargs=kwargs,
            location=self._loc(meta),
        )

    def annotation_args(self, items):
        return list(items)

    def positional_arg(self, items):
        return items[0]

    def keyword_arg(self, items):
        return (str(items[0]), items[1])

    # =========================================================================
    # Types
    # =========================================================================

    @v_args(meta=True)
    def tensor_type(self, meta, items):
        dims = []
        dtype = None
        optional = False

        for item in items:
            if item is None:
                continue
            if isinstance(item, Token) and item.type == "NAME":
                # Could be dtype
                val = str(item)
                if val in ("bf16", "fp32", "fp16", "fp8_e4m3", "fp8_e5m2", "fp4_e2m1", "int8", "int32"):
                    dtype = val
                else:
                    dims.append(val)
            elif str(item) == "?":
                optional = True
            elif isinstance(item, list):
                dims = item
            elif isinstance(item, str):
                if item in ("bf16", "fp32", "fp16", "fp8_e4m3", "fp8_e5m2", "fp4_e2m1", "int8", "int32"):
                    dtype = item
                else:
                    dims.append(item)
            elif isinstance(item, int):
                dims.append(item)

        return TensorType(
            dims=dims,
            dtype=dtype,
            optional=optional,
            location=self._loc(meta),
        )

    def shape_list(self, items):
        return list(items)

    def variadic_dim(self, items):
        return "*"

    def expr_dim(self, items):
        item = items[0]
        if isinstance(item, Literal):
            return item.value
        elif isinstance(item, Identifier):
            return item.name
        return item

    def dtype(self, items):
        return str(items[0])

    def array_type_mul(self, items):
        size = items[0]
        element = items[1]
        return ArrayTypeNode(size=size, element_type=element)

    def array_type_generic(self, items):
        size = items[0]
        element = items[1]
        return ArrayTypeNode(size=size, element_type=element)

    # =========================================================================
    # Parameters
    # =========================================================================

    def param_list(self, items):
        return list(items)

    @v_args(meta=True)
    def param(self, meta, items):
        name = str(items[0])
        type_ann = None
        default = None

        for item in items[1:]:
            if item is None:
                continue
            if isinstance(item, Token) and item.type == "NAME":
                type_ann = str(item)
            elif isinstance(item, str) and item in ("int", "float", "bool", "string"):
                type_ann = item
            else:
                default = item

        return ParamDecl(
            name=name,
            type_annotation=type_ann,
            default=default,
            location=self._loc(meta),
        )

    # =========================================================================
    # Expressions
    # =========================================================================

    def expression(self, items):
        return items[0] if items else None

    def ternary_expr(self, items):
        # Grammar: or_expr ["if" or_expr "else" ternary_expr]
        # When no ternary, items[1] and items[2] are None
        if len(items) == 1 or items[1] is None:
            return items[0]
        # value if condition else else_value
        return TernaryExpr(
            condition=items[1],
            true_value=items[0],
            false_value=items[2],
        )

    def or_expr(self, items):
        if len(items) == 1:
            return items[0]
        result = items[0]
        for item in items[1:]:
            result = BinaryOp(left=result, op="or", right=item)
        return result

    def and_expr(self, items):
        if len(items) == 1:
            return items[0]
        result = items[0]
        for item in items[1:]:
            result = BinaryOp(left=result, op="and", right=item)
        return result

    def not_op(self, items):
        return UnaryOp(op="not", operand=items[0])

    def not_expr(self, items):
        # Pass through if single item, otherwise it's handled by not_op
        return items[0] if items else None

    def comparison(self, items):
        if len(items) == 1:
            return items[0]
        # Handle chained comparisons
        result = items[0]
        i = 1
        while i < len(items):
            op = str(items[i])
            right = items[i + 1]
            result = BinaryOp(left=result, op=op, right=right)
            i += 2
        return result

    def arith_expr(self, items):
        if len(items) == 1:
            return items[0]
        result = items[0]
        i = 1
        while i < len(items):
            op = str(items[i])
            right = items[i + 1]
            result = BinaryOp(left=result, op=op, right=right)
            i += 2
        return result

    def term(self, items):
        if len(items) == 1:
            return items[0]
        result = items[0]
        i = 1
        while i < len(items):
            op = str(items[i])
            right = items[i + 1]
            result = BinaryOp(left=result, op=op, right=right)
            i += 2
        return result

    def factor(self, items):
        if len(items) == 1:
            return items[0]
        # Unary operator
        op = str(items[0])
        operand = items[1]
        return UnaryOp(op=op, operand=operand)

    def power(self, items):
        # Grammar: atom ["**" factor]
        if len(items) == 1 or items[1] is None:
            return items[0]
        return BinaryOp(left=items[0], op="**", right=items[1])

    def atom(self, items):
        return items[0] if items else None

    def paren_expr(self, items):
        return items[0]

    def list_literal(self, items):
        return Literal(value=list(items))

    def dict_literal(self, items):
        result = {}
        for item in items:
            if isinstance(item, tuple) and len(item) == 2:
                result[item[0]] = item[1]
        return Literal(value=result)

    def dict_item(self, items):
        return (items[0], items[1])

    @v_args(meta=True)
    def call_expr(self, meta, items):
        name = str(items[0])
        args = []
        kwargs = {}

        for item in items[1:]:
            if item is None:
                continue
            if isinstance(item, tuple):
                kwargs[item[0]] = item[1]
            else:
                args.append(item)

        return CallExpr(
            func=name,
            args=args,
            kwargs=kwargs,
            location=self._loc(meta),
        )

    def call(self, items):
        return self.call_expr(None, items)

    def attribute_expr(self, items):
        parts = [str(item) for item in items]
        if len(parts) == 2:
            return AttributeExpr(
                base=Identifier(name=parts[0]),
                attr=parts[1],
            )
        # Chain of attributes
        result = Identifier(name=parts[0])
        for attr in parts[1:]:
            result = AttributeExpr(base=result, attr=attr)
        return result

    def attribute(self, items):
        return self.attribute_expr(items)

    def index_expr(self, items):
        name = str(items[0])
        index = items[1]
        return IndexExpr(
            base=Identifier(name=name),
            indices=[index],
        )

    def index(self, items):
        return self.index_expr(items)

    def identifier(self, items):
        return Identifier(name=str(items[0]))

    def int_literal(self, items):
        return Literal(value=int(str(items[0])))

    def integer(self, items):
        return self.int_literal(items)

    def float_literal(self, items):
        return Literal(value=float(str(items[0])))

    def float(self, items):
        return self.float_literal(items)

    def string_literal(self, items):
        s = str(items[0])
        # Strip quotes
        if s.startswith(('"""', "'''")):
            s = s[3:-3]
        elif s.startswith(('"', "'")):
            s = s[1:-1]
        return Literal(value=s)

    def string(self, items):
        return self.string_literal(items)

    def true_literal(self, items):
        return Literal(value=True)

    def true(self, items):
        return self.true_literal(items)

    def false_literal(self, items):
        return Literal(value=False)

    def false(self, items):
        return self.false_literal(items)

    def none_literal(self, items):
        return Literal(value=None)

    def none(self, items):
        return self.none_literal(items)

    def literal(self, items):
        return items[0] if items else None

    def arg_list(self, items):
        return list(items)

    def arg(self, items):
        return items[0]

    def posarg(self, items):
        return items[0]

    def kwarg(self, items):
        return (str(items[0]), items[1])


class ModuleDSLParser:
    """Main parser class for Module DSL.

    Supports both indentation-based and brace-based syntax.
    """

    def __init__(self, use_indentation: bool = True):
        """Initialize parser.

        Args:
            use_indentation: If True, use indentation-sensitive parsing (default True, Python-like)
        """
        self.use_indentation = use_indentation
        grammar = GRAMMAR if use_indentation else SIMPLE_GRAMMAR

        try:
            if use_indentation:
                self._parser = Lark(
                    grammar,
                    parser="lalr",
                    postlex=DSLIndenter(),
                    propagate_positions=True,
                )
            else:
                self._parser = Lark(
                    grammar,
                    parser="lalr",
                    propagate_positions=True,
                )
        except Exception as e:
            # Fallback to simple grammar if full grammar fails
            self._parser = Lark(
                SIMPLE_GRAMMAR,
                parser="lalr",
                propagate_positions=True,
            )
            self.use_indentation = False

    def parse(self, source: str, source_file: Optional[str] = None) -> Program:
        """Parse DSL source code into AST.

        Args:
            source: DSL source code string
            source_file: Optional source file path for error messages

        Returns:
            Program AST node

        Raises:
            DSLSyntaxError: If parsing fails
        """
        try:
            tree = self._parser.parse(source)
            transformer = DSLTransformer(source_file=source_file)
            return transformer.transform(tree)
        except Exception as e:
            raise DSLSyntaxError(
                message=str(e),
                hint="Check syntax and ensure proper formatting",
            ) from e

    def parse_file(self, path: Union[str, Path]) -> Program:
        """Parse a DSL file.

        Args:
            path: Path to DSL source file

        Returns:
            Program AST node
        """
        path = Path(path)
        source = path.read_text(encoding="utf-8")
        return self.parse(source, source_file=str(path))


# Convenience functions
def parse_source(source: str, source_file: Optional[str] = None) -> Program:
    """Parse DSL source code.

    Args:
        source: DSL source code string
        source_file: Optional file path for error messages

    Returns:
        Program AST node
    """
    parser = ModuleDSLParser(use_indentation=False)
    return parser.parse(source, source_file)


def parse_file(path: Union[str, Path]) -> Program:
    """Parse a DSL file.

    Args:
        path: Path to DSL source file

    Returns:
        Program AST node
    """
    parser = ModuleDSLParser(use_indentation=False)
    return parser.parse_file(path)

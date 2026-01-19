"""
DSL Compile Command

Compiles Module DSL files and outputs the compilation result.

Usage:
    surogate compile model.module
    surogate compile model.module --output ir
    surogate compile model.module --params d_model=4096 n_layers=32
    surogate compile model.module --validate-only
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

from surogate.dsl.types import ConcreteDim, SymbolicDim, VariadicDim, ComputedDim, Dtype, Shape
from surogate.dsl.ast_nodes import CallExpr, Identifier, Literal, WeightMapping

def prepare_command_parser(parser: argparse.ArgumentParser):
    """Prepare the compile command argument parser."""
    parser.add_argument(
        "file",
        type=str,
        help="Path to the .module file to compile",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        choices=["summary", "json", "ir", "ast"],
        default="summary",
        help="Output format (default: summary)",
    )
    parser.add_argument(
        "--out-file",
        type=str,
        default=None,
        help="Write output to file instead of stdout",
    )
    parser.add_argument(
        "--params", "-p",
        nargs="*",
        type=str,
        default=[],
        help="Parameter overrides in key=value format (e.g., d_model=4096)",
    )
    parser.add_argument(
        "--search-path", "-I",
        action="append",
        default=[],
        help="Additional search paths for imports (can be specified multiple times)",
    )
    parser.add_argument(
        "--validate-only", "-v",
        action="store_true",
        help="Only validate the source without full compilation",
    )
    parser.add_argument(
        "--module", "-m",
        type=str,
        default=None,
        help="Only output information for a specific module",
    )
    parser.add_argument(
        "--show-warnings", "-w",
        action="store_true",
        help="Show compilation warnings",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.set_defaults(func=run_compile)


def parse_params(param_list: List[str]) -> Dict[str, Any]:
    """Parse key=value parameter strings into a dictionary."""
    params = {}
    for param in param_list:
        if "=" not in param:
            print(f"Ignoring invalid parameter format: {param} (expected key=value)", file=sys.stderr)
            continue

        key, value = param.split("=", 1)
        key = key.strip()
        value = value.strip()

        # Try to parse as number
        try:
            if "." in value:
                params[key] = float(value)
            else:
                params[key] = int(value)
        except ValueError:
            # Check for boolean
            if value.lower() in ("true", "false"):
                params[key] = value.lower() == "true"
            else:
                # Keep as string
                params[key] = value

    return params


def _dtype_to_str(dtype: Any) -> Optional[str]:
    if dtype is None:
        return None
    if isinstance(dtype, Dtype):
        return dtype.value
    if hasattr(dtype, "value"):
        return str(dtype.value)
    return str(dtype)


def _shape_to_list(shape: Optional[Shape]) -> Optional[List[Any]]:
    if shape is None:
        return None
    dims = []
    for d in shape.dims:
        if isinstance(d, ConcreteDim):
            dims.append(d.value)
        elif isinstance(d, SymbolicDim):
            dims.append(d.name)
        elif isinstance(d, VariadicDim):
            dims.append("*")
        elif isinstance(d, ComputedDim):
            dims.append(str(d))
        else:
            dims.append(str(d))
    return dims


def _tensor_ref_to_dict(tref) -> Dict[str, Any]:
    return {
        "shape": _shape_to_list(getattr(tref, "shape", None)),
        "dtype": _dtype_to_str(getattr(tref, "dtype", None)),
        "is_param": getattr(tref, "is_param", False),
        "is_input": getattr(tref, "is_input", False),
        "is_output": getattr(tref, "is_output", False),
    }


def _serialize_attr(value: Any) -> Any:
    if isinstance(value, Dtype):
        return value.value
    if isinstance(value, (ConcreteDim, SymbolicDim, VariadicDim, ComputedDim)):
        return str(value)
    if isinstance(value, list):
        return [_serialize_attr(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_attr(v) for k, v in value.items()}
    try:
        # Expression nodes stringify nicely
        return str(value)
    except Exception:
        return value


def _expr_to_value(expr: Any) -> Any:
    if isinstance(expr, Literal):
        return expr.value
    if isinstance(expr, Identifier):
        return expr.name
    if isinstance(expr, CallExpr):
        return str(expr)
    return _serialize_attr(expr)


def _serialize_weight_spec(spec: Any) -> Any:
    if isinstance(spec, str):
        return spec
    if isinstance(spec, CallExpr):
        kind = spec.func
        if kind == "fuse":
            payload = {
                "type": "fuse",
                "sources": [_expr_to_value(arg) for arg in spec.args],
            }
            if "dim" in spec.kwargs:
                payload["dim"] = _expr_to_value(spec.kwargs["dim"])
            return payload
        if kind == "transform":
            payload = {
                "type": "transform",
                "source": _expr_to_value(spec.args[0]) if spec.args else None,
            }
            if "fn" in spec.kwargs:
                payload["fn"] = _expr_to_value(spec.kwargs["fn"])
            return payload
        if kind == "split":
            payload = {
                "type": "split",
                "targets": [_expr_to_value(arg) for arg in spec.args],
            }
            if "dim" in spec.kwargs:
                payload["dim"] = _expr_to_value(spec.kwargs["dim"])
            return payload
        if kind == "tied_to":
            return {
                "type": "tied_to",
                "source": _expr_to_value(spec.args[0]) if spec.args else None,
            }
        return {"type": kind, "expr": str(spec)}
    return _serialize_attr(spec)


def _serialize_weight_mapping(mapping: Any) -> Any:
    if isinstance(mapping, WeightMapping):
        payload = _serialize_weight_spec(mapping.external_spec)
        if mapping.optional:
            if isinstance(payload, str):
                payload = {"type": "direct", "source": payload}
            if isinstance(payload, dict):
                payload = dict(payload)
                payload["optional"] = True
        return payload
    return _serialize_weight_spec(mapping)


def format_module_ir(module_ir, indent: int = 0) -> str:
    """Format a ModuleIR for text output."""
    prefix = "  " * indent
    lines = []

    kind = "model" if module_ir.is_model else "block" if module_ir.is_block else "module"
    lines.append(f"{prefix}{kind} {module_ir.name}:")

    # Parameters
    if module_ir.params:
        lines.append(f"{prefix}  params:")
        if isinstance(module_ir.params, dict):
            param_items = module_ir.params.items()
        else:
            param_items = [(p.name, p) for p in module_ir.params]
        for name, param in param_items:
            dtype = getattr(param, "dtype", "unknown")
            shape = getattr(param, "shape", "?")
            lines.append(f"{prefix}    {name}: {shape} ({dtype})")

    # Forward graph
    if module_ir.forward_graph:
        graph = module_ir.forward_graph
        lines.append(f"{prefix}  forward: {len(graph.nodes)} ops")
        for node in graph.nodes[:10]:  # Show first 10 ops
            op_name = getattr(node, 'op_type', getattr(node, 'op', 'unknown'))
            lines.append(f"{prefix}    - {op_name}")
        if len(graph.nodes) > 10:
            lines.append(f"{prefix}    ... ({len(graph.nodes) - 10} more)")

    # Backward graph
    if module_ir.backward_graph:
        graph = module_ir.backward_graph
        lines.append(f"{prefix}  backward: {len(graph.nodes)} ops")

    # Save list
    if hasattr(module_ir, 'save_tensors') and module_ir.save_tensors:
        lines.append(f"{prefix}  save: {module_ir.save_tensors}")

    # Recompute list
    if hasattr(module_ir, 'recompute_tensors') and module_ir.recompute_tensors:
        lines.append(f"{prefix}  recompute: {module_ir.recompute_tensors}")

    return "\n".join(lines)


def format_graph_ir(graph_ir, name: str = "graph", indent: int = 0) -> str:
    """Format a GraphIR for detailed text output."""
    prefix = "  " * indent
    lines = [f"{prefix}{name}:"]

    # Inputs
    if hasattr(graph_ir, 'inputs') and graph_ir.inputs:
        lines.append(f"{prefix}  inputs:")
        for inp in graph_ir.inputs:
            lines.append(f"{prefix}    - {inp}")

    # Outputs
    if hasattr(graph_ir, 'outputs') and graph_ir.outputs:
        lines.append(f"{prefix}  outputs:")
        for out in graph_ir.outputs:
            lines.append(f"{prefix}    - {out}")

    # Operations
    lines.append(f"{prefix}  operations ({len(graph_ir.nodes)}):")
    for i, node in enumerate(graph_ir.nodes):
        op_type = getattr(node, 'op_type', getattr(node, 'op', 'unknown'))
        inputs = getattr(node, 'inputs', [])
        outputs = getattr(node, 'outputs', [])

        in_str = ", ".join(str(x) for x in inputs) if inputs else ""
        out_str = ", ".join(str(x) for x in outputs) if outputs else ""

        lines.append(f"{prefix}    [{i}] {out_str} = {op_type}({in_str})")

    return "\n".join(lines)


def module_ir_to_dict(module_ir) -> Dict[str, Any]:
    """Convert a ModuleIR to a JSON-serializable dictionary."""
    hf_config = getattr(module_ir, "hf_config_mapping", {})
    hf_mapping = getattr(module_ir, "hf_weight_mapping", {})
    hf_export = getattr(module_ir, "hf_export_mapping", {})

    hf_mapping_out = {
        name: _serialize_weight_mapping(spec) for name, spec in hf_mapping.items()
    } if isinstance(hf_mapping, dict) else hf_mapping

    hf_export_out = {
        name: _serialize_weight_mapping(spec) for name, spec in hf_export.items()
    } if isinstance(hf_export, dict) else hf_export

    result = {
        "name": module_ir.name,
        "kind": "model" if module_ir.is_model else "block" if module_ir.is_block else "module",
        "extends": module_ir.extends,
        "config": module_ir.config,
        "hf_config": hf_config,
        "hf_mapping": hf_mapping_out,
    }
    if hf_export_out:
        result["hf_export"] = hf_export_out

    # Parameters (use forward graph params if available for full shape info)
    result["params"] = {}
    if module_ir.forward_graph and module_ir.forward_graph.params:
        for name, param in module_ir.forward_graph.params.items():
            result["params"][name] = _tensor_ref_to_dict(param)
    elif module_ir.params:
        if isinstance(module_ir.params, dict):
            param_items = module_ir.params.items()
        else:
            param_items = [(p.name, p) for p in module_ir.params]
        for name, param in param_items:
            result["params"][name] = _tensor_ref_to_dict(param)

    # Forward graph
    if module_ir.forward_graph:
        result["forward"] = graph_ir_to_dict(module_ir.forward_graph)

    # Backward graph
    if module_ir.backward_graph:
        result["backward"] = graph_ir_to_dict(module_ir.backward_graph)

    # Metadata
    if hasattr(module_ir, 'save_tensors') and module_ir.save_tensors:
        result["save"] = module_ir.save_tensors
    if hasattr(module_ir, 'recompute_tensors') and module_ir.recompute_tensors:
        result["recompute"] = module_ir.recompute_tensors

    return result


def graph_ir_to_dict(graph_ir) -> Dict[str, Any]:
    """Convert a GraphIR to a JSON-serializable dictionary."""
    result = {
        "name": getattr(graph_ir, "name", None),
        "num_ops": len(graph_ir.nodes),
        "inputs": {name: _tensor_ref_to_dict(t) for name, t in getattr(graph_ir, "inputs", {}).items()},
        "outputs": {name: _tensor_ref_to_dict(t) for name, t in getattr(graph_ir, "outputs", {}).items()},
        "params": {name: _tensor_ref_to_dict(t) for name, t in getattr(graph_ir, "params", {}).items()},
        "intermediates": {name: _tensor_ref_to_dict(t) for name, t in getattr(graph_ir, "intermediates", {}).items()},
        "save": getattr(graph_ir, "save_list", []),
        "recompute": getattr(graph_ir, "recompute_list", []),
        "operations": [],
    }

    for node in graph_ir.nodes:
        op_dict = {
            "id": getattr(node, "id", None),
            "name": getattr(node, "name", None),
            "kernel_type": getattr(getattr(node, "kernel_type", None), "value", getattr(node, "kernel_type", None)),
            "inputs": [str(x) for x in getattr(node, "inputs", [])],
            "outputs": [str(x) for x in getattr(node, "outputs", [])],
        }

        # Include attributes if present
        if hasattr(node, 'attrs') and node.attrs:
            op_dict["attrs"] = {k: _serialize_attr(v) for k, v in node.attrs.items()}

        result["operations"].append(op_dict)

    return result


def run_compile(args: argparse.Namespace):
    """Run the compile command."""
    from surogate.dsl.compiler import (
        Compiler,
        CompilerOptions,
        compile_module,
        validate_source,
    )
    from surogate.dsl.errors import DSLError

    file_path = Path(args.file)

    # Check file exists
    if not file_path.exists():
        print(f"File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    # Check file extension
    if not file_path.suffix == ".module":
        print(f"File does not have .module extension: {args.file}", file=sys.stderr)

    # Parse parameters
    params = parse_params(args.params) if args.params else None
    if params and args.verbose:
        print(f"Parameter overrides: {params}")

    # Read source
    source = file_path.read_text()

    # Validate only mode
    if args.validate_only:
        print(f"Validating: {args.file}")
        is_valid, messages = validate_source(source)

        if is_valid:
            print("✓ Validation successful")
            if messages and args.show_warnings:
                for msg in messages:
                    print(msg)
            sys.exit(0)
        else:
            print("✗ Validation failed", file=sys.stderr)
            for msg in messages:
                print(f"  {msg}", file=sys.stderr)
            sys.exit(1)

    # Full compilation
    print(f"Compiling: {args.file}")

    # Set up compiler options
    options = CompilerOptions()
    options.search_paths = args.search_path.copy()

    # Add standard library path
    std_path = Path(__file__).parent.parent.parent / "std"
    if std_path.exists():
        options.search_paths.append(str(std_path))

    # Compile
    compiler = Compiler(options)

    try:
        result = compiler.compile_file(str(file_path), params)
    except Exception as e:
        print(f"Compilation failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Check for errors
    if not result.success:
        print("✗ Compilation failed", file=sys.stderr)
        for error in result.errors:
            print(f"  {error}", file=sys.stderr)
        sys.exit(1)

    print(f"✓ Compiled {len(result.modules)} module(s)")

    # Show warnings if requested
    if args.show_warnings and result.warnings.has_warnings():
        for warning in result.warnings:
            print(f"  {warning}")

    # Filter to specific module if requested
    modules = result.modules
    if args.module:
        modules = [m for m in modules if m.name == args.module]
        if not modules:
            print(f"Module not found: {args.module}", file=sys.stderr)
            print(f"Available modules: {[m.name for m in result.modules]}", file=sys.stderr)
            sys.exit(1)

    # Generate output
    output_text = ""

    if args.output == "summary":
        lines = [f"Compilation Result: {args.file}"]
        lines.append(f"Status: {'SUCCESS' if result.success else 'FAILED'}")
        lines.append(f"Modules: {len(modules)}")
        lines.append("")

        for module in modules:
            lines.append(format_module_ir(module))
            lines.append("")

        output_text = "\n".join(lines)

    elif args.output == "json":
        output_dict = {
            "source_file": str(file_path),
            "success": result.success,
            "modules": [module_ir_to_dict(m) for m in modules],
        }

        if result.warnings.has_warnings():
            output_dict["warnings"] = [str(w) for w in result.warnings]

        output_text = json.dumps(output_dict, indent=2)

    elif args.output == "ir":
        lines = [f"# Graph IR for {args.file}", ""]

        for module in modules:
            kind = "model" if module.is_model else "block" if module.is_block else "module"
            lines.append(f"# {kind} {module.name}")
            lines.append("")

            if module.forward_graph:
                lines.append(format_graph_ir(module.forward_graph, "forward"))
                lines.append("")

            if module.backward_graph:
                lines.append(format_graph_ir(module.backward_graph, "backward"))
                lines.append("")

            lines.append("-" * 60)
            lines.append("")

        output_text = "\n".join(lines)

    elif args.output == "ast":
        # For AST output, we need to parse again and show the AST
        from surogate.dsl.parser import ModuleDSLParser

        parser = ModuleDSLParser()
        try:
            program = parser.parse(source, str(file_path))

            # Simple AST dump
            lines = [f"# AST for {args.file}", ""]

            for decl in program.declarations:
                lines.append(f"{type(decl).__name__}: {getattr(decl, 'name', '?')}")

                # Show structure
                for attr in dir(decl):
                    if not attr.startswith('_') and attr != 'name':
                        val = getattr(decl, attr)
                        if val is not None and not callable(val):
                            lines.append(f"  {attr}: {type(val).__name__}")

                lines.append("")

            output_text = "\n".join(lines)
        except Exception as e:
            print(f"AST parsing failed: {e}", file=sys.stderr)
            sys.exit(1)

    # Output result
    if args.out_file:
        out_path = Path(args.out_file)
        out_path.write_text(output_text)
        print(f"Output written to: {args.out_file}")
    else:
        print(output_text)


def main():
    """Main entry point when run directly."""
    parser = argparse.ArgumentParser(
        description="Compile Module DSL files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m surogate.cli.dsl_compile model.module
  python -m surogate.cli.dsl_compile model.module --output json
  python -m surogate.cli.dsl_compile model.module --params d_model=4096
  python -m surogate.cli.dsl_compile model.module --validate-only
        """,
    )
    prepare_command_parser(parser)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

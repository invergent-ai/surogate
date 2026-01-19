"""
Module DSL Compiler

Main entry point for compiling Module DSL source code to Graph IR.

The compilation pipeline is:
1. Parse: Source text -> AST
2. Resolve: Import resolution, type checking, shape inference
3. Lower: AST -> Graph IR
4. (Optional) Schedule: Graph IR -> Schedule IR

Example usage:
    from surogate.dsl import compile_module, Compiler

    # Quick compilation
    result = compile_module("path/to/model.module")

    # Or with more control
    compiler = Compiler()
    result = compiler.compile_file("path/to/model.module")

    for module_ir in result.modules:
        print(f"Compiled: {module_ir.name}")
"""

from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging

from .parser import parse_source, parse_file, ModuleDSLParser
from .resolver import (
    ModuleResolver,
    ResolverContext,
    ResolvedModule,
    resolve_program,
)
from .lowering import (
    ModuleLowerer,
    lower_program,
    GraphBuilder,
)
from .ir import (
    ModuleIR,
    GraphIR,
    ScheduleIR,
    OpNode,
    TensorRef,
    CompilationContext,
)
from .errors import (
    DSLError,
    DSLSyntaxError,
    DSLTypeError,
    DSLShapeError,
    DSLResolutionError,
    WarningCollector,
    ErrorCode,
)
from .ast_nodes import Program


logger = logging.getLogger(__name__)


# =============================================================================
# Compilation Result
# =============================================================================


@dataclass
class CompilationResult:
    """Result of compiling Module DSL source."""

    # Compiled module IRs
    modules: List[ModuleIR] = field(default_factory=list)

    # Resolved module definitions (for reference)
    resolved: List[ResolvedModule] = field(default_factory=list)

    # Warnings generated during compilation
    warnings: WarningCollector = field(default_factory=WarningCollector)

    # Errors (if any, before raising)
    errors: List[DSLError] = field(default_factory=list)

    # Source file (if compiled from file)
    source_file: Optional[str] = None

    # Whether compilation succeeded
    success: bool = True

    def get_module(self, name: str) -> Optional[ModuleIR]:
        """Get a compiled module by name."""
        for module in self.modules:
            if module.name == name:
                return module
        return None

    def get_forward_graph(self, module_name: str) -> Optional[GraphIR]:
        """Get forward graph for a module."""
        module = self.get_module(module_name)
        if module:
            return module.forward_graph
        return None

    def get_backward_graph(self, module_name: str) -> Optional[GraphIR]:
        """Get backward graph for a module."""
        module = self.get_module(module_name)
        if module:
            return module.backward_graph
        return None

    def print_summary(self):
        """Print a summary of compilation results."""
        print(f"Compilation {'succeeded' if self.success else 'failed'}")
        print(f"  Modules: {len(self.modules)}")

        for module in self.modules:
            kind = "model" if module.is_model else "block" if module.is_block else "module"
            fwd_ops = len(module.forward_graph.nodes) if module.forward_graph else 0
            bwd_ops = len(module.backward_graph.nodes) if module.backward_graph else 0
            print(f"    {module.name} ({kind}): {fwd_ops} forward ops, {bwd_ops} backward ops")

        if self.warnings.has_warnings():
            print(f"  Warnings: {len(list(self.warnings))}")
            for warning in self.warnings:
                print(f"    {warning}")

        if self.errors:
            print(f"  Errors: {len(self.errors)}")
            for error in self.errors:
                print(f"    {error}")


# =============================================================================
# Compiler Options
# =============================================================================


@dataclass
class CompilerOptions:
    """Options for the compiler."""

    # Search paths for imports
    search_paths: List[str] = field(default_factory=list)

    # Whether to validate shapes
    validate_shapes: bool = True

    # Whether to infer types
    infer_types: bool = True

    # Whether to check constraints
    check_constraints: bool = True

    # Log level
    log_level: int = logging.WARNING

    # Target device (for scheduling hints)
    target_device: str = "cuda"

    # Optimization level (0=none, 1=basic, 2=full)
    optimization_level: int = 1


# =============================================================================
# Compiler Class
# =============================================================================


class Compiler:
    """Module DSL Compiler.

    Compiles Module DSL source to Graph IR for execution.
    """

    def __init__(self, options: Optional[CompilerOptions] = None):
        self.options = options or CompilerOptions()
        self.parser = ModuleDSLParser()
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging based on options."""
        logging.basicConfig(level=self.options.log_level)

    def compile_source(
        self,
        source: str,
        filename: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> CompilationResult:
        """Compile DSL source code.

        Args:
            source: DSL source code as a string
            filename: Optional filename for error messages
            params: Optional parameter values to override defaults

        Returns:
            CompilationResult with compiled modules
        """
        result = CompilationResult(source_file=filename)

        try:
            # Phase 1: Parse
            logger.info(f"Parsing source{f' from {filename}' if filename else ''}")
            program = self.parser.parse(source, filename)

            # Phase 2: Resolve
            logger.info("Resolving imports and types")
            resolved, warnings = resolve_program(
                program,
                search_paths=self.options.search_paths,
            )
            result.resolved = resolved
            result.warnings = warnings

            # Phase 3: Lower to IR
            logger.info("Lowering to Graph IR")
            module_irs = self._lower_with_params(program, params)
            result.modules = module_irs

            result.success = True

        except DSLError as e:
            logger.error(f"Compilation error: {e}")
            result.errors.append(e)
            result.success = False

        except Exception as e:
            logger.error(f"Unexpected error during compilation: {e}")
            result.errors.append(DSLError(
                ErrorCode.E001,
                f"Internal compiler error: {e}",
            ))
            result.success = False

        return result

    def compile_file(
        self,
        filepath: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> CompilationResult:
        """Compile a DSL file.

        Args:
            filepath: Path to the .module file
            params: Optional parameter values to override defaults

        Returns:
            CompilationResult with compiled modules
        """
        path = Path(filepath)

        if not path.exists():
            result = CompilationResult(source_file=filepath)
            result.errors.append(DSLError(
                ErrorCode.E001,
                f"File not found: {filepath}",
            ))
            result.success = False
            return result

        source = path.read_text()

        # Add file's directory and current working directory to search paths
        search_paths = self.options.search_paths.copy()
        existing_paths = [Path(p) for p in search_paths]
        if path.parent not in existing_paths:
            search_paths.append(str(path.parent))
            existing_paths.append(path.parent)
        cwd = Path.cwd()
        if cwd not in existing_paths:
            search_paths.append(str(cwd))

        old_paths = self.options.search_paths
        self.options.search_paths = search_paths

        try:
            result = self.compile_source(source, str(path), params)
        finally:
            self.options.search_paths = old_paths

        return result

    def compile_program(
        self,
        program: Program,
        params: Optional[Dict[str, Any]] = None,
    ) -> CompilationResult:
        """Compile a parsed program.

        Args:
            program: Parsed Program AST
            params: Optional parameter values to override defaults

        Returns:
            CompilationResult with compiled modules
        """
        result = CompilationResult()

        try:
            # Phase 2: Resolve
            logger.info("Resolving imports and types")
            resolved, warnings = resolve_program(
                program,
                search_paths=self.options.search_paths,
            )
            result.resolved = resolved
            result.warnings = warnings

            # Phase 3: Lower to IR
            logger.info("Lowering to Graph IR")
            module_irs = self._lower_with_params(program, params)
            result.modules = module_irs

            result.success = True

        except DSLError as e:
            logger.error(f"Compilation error: {e}")
            result.errors.append(e)
            result.success = False

        return result

    def _lower_with_params(
        self,
        program: Program,
        params: Optional[Dict[str, Any]],
    ) -> List[ModuleIR]:
        """Lower program to IR with optional parameter overrides."""
        warnings = WarningCollector()
        lowerer = ModuleLowerer(warnings)

        # Apply parameter overrides
        if params:
            for key, value in params.items():
                lowerer.ctx.module_params[key] = value

        return lower_program(program, warnings)


# =============================================================================
# Convenience Functions
# =============================================================================


def compile_module(
    source_or_path: str,
    params: Optional[Dict[str, Any]] = None,
    search_paths: Optional[List[str]] = None,
) -> CompilationResult:
    """Compile a Module DSL source or file.

    This is the main entry point for compilation.

    Args:
        source_or_path: Either DSL source code or path to a .module file
        params: Optional parameter values to override defaults
        search_paths: Optional search paths for imports

    Returns:
        CompilationResult with compiled modules

    Example:
        # Compile from file
        result = compile_module("models/llama.module")

        # Compile from source
        result = compile_module('''
            module Linear(C: int, K: int):
                param W: [K, C]

                forward(in: [*, C]) -> [*, K]:
                    in -> matmul(W) -> out

                backward:
                    d_out: [*, K]
                    produces d_in: [*, C], d_W: [K, C]

                    d_out -> matmul(W, transpose=true) -> d_in
                    in.T -> matmul(d_out) -> d_W
        ''')

        # With parameter overrides
        result = compile_module("models/llama.module", params={
            "d_model": 4096,
            "n_layers": 32,
        })
    """
    options = CompilerOptions()
    if search_paths:
        options.search_paths = search_paths

    compiler = Compiler(options)

    # Determine if source or file
    if source_or_path.endswith(".module") or Path(source_or_path).exists():
        return compiler.compile_file(source_or_path, params)
    else:
        return compiler.compile_source(source_or_path, params=params)


def compile_and_lower(
    source: str,
    module_name: str,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[GraphIR], Optional[GraphIR]]:
    """Compile source and return forward/backward graphs for a specific module.

    Args:
        source: DSL source code
        module_name: Name of module to extract
        params: Optional parameter overrides

    Returns:
        Tuple of (forward_graph, backward_graph), either may be None
    """
    result = compile_module(source, params)

    if not result.success:
        raise result.errors[0] if result.errors else DSLError(
            ErrorCode.E001,
            "Compilation failed",
        )

    module = result.get_module(module_name)
    if module is None:
        raise DSLError(
            ErrorCode.E002,
            f"Module '{module_name}' not found in compiled result",
        )

    return module.forward_graph, module.backward_graph


def validate_source(source: str) -> Tuple[bool, List[str]]:
    """Validate DSL source without full compilation.

    Args:
        source: DSL source code to validate

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    try:
        parser = ModuleDSLParser()
        program = parser.parse(source)

        # Try resolution
        resolved, warnings = resolve_program(program)

        # Collect warnings as messages
        messages = [str(w) for w in warnings]

        return True, messages

    except DSLError as e:
        return False, [str(e)]

    except Exception as e:
        return False, [f"Validation error: {e}"]


# =============================================================================
# Module Registry
# =============================================================================


class ModuleRegistry:
    """Registry for compiled modules.

    Maintains a cache of compiled modules for reuse.
    """

    def __init__(self):
        self._cache: Dict[str, ModuleIR] = {}
        self._compiler = Compiler()

    def register(self, source: str, name: Optional[str] = None) -> ModuleIR:
        """Compile and register a module.

        Args:
            source: DSL source code
            name: Optional name override

        Returns:
            Compiled ModuleIR
        """
        result = self._compiler.compile_source(source)

        if not result.success:
            raise result.errors[0]

        for module in result.modules:
            module_name = name or module.name
            self._cache[module_name] = module

        if result.modules:
            return result.modules[0]

        raise DSLError(ErrorCode.E001, "No modules found in source")

    def register_file(self, filepath: str) -> List[ModuleIR]:
        """Compile and register modules from a file.

        Args:
            filepath: Path to .module file

        Returns:
            List of compiled ModuleIRs
        """
        result = self._compiler.compile_file(filepath)

        if not result.success:
            raise result.errors[0]

        for module in result.modules:
            self._cache[module.name] = module

        return result.modules

    def get(self, name: str) -> Optional[ModuleIR]:
        """Get a registered module by name."""
        return self._cache.get(name)

    def list(self) -> List[str]:
        """List all registered module names."""
        return list(self._cache.keys())

    def clear(self):
        """Clear the registry."""
        self._cache.clear()


# Global registry instance
_global_registry = ModuleRegistry()


def get_registry() -> ModuleRegistry:
    """Get the global module registry."""
    return _global_registry

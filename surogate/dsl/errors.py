"""
DSL Error definitions and error codes.

Error codes follow the specification:
- E001-E027: Compilation errors
- W001-W005: Warnings
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class ErrorCode(str, Enum):
    """Compilation error codes as defined in the DSL specification."""

    # Syntax and basic errors
    E001 = "E001"  # Syntax error
    E002 = "E002"  # Undefined identifier
    E003 = "E003"  # Type mismatch
    E004 = "E004"  # Shape mismatch

    # Gradient errors
    E005 = "E005"  # Missing required gradient
    E006 = "E006"  # Saved tensor not available in backward
    E007 = "E007"  # Circular dependency in graph

    # Annotation and parameter errors
    E008 = "E008"  # Invalid annotation
    E009 = "E009"  # Duplicate parameter name
    E010 = "E010"  # Invalid weight mapping
    E011 = "E011"  # Incompatible inheritance
    E012 = "E012"  # Missing required parameter

    # Operation errors
    E013 = "E013"  # Invalid fusion pattern
    E014 = "E014"  # Unsupported primitive
    E015 = "E015"  # Invalid dtype for operation

    # Graph and control flow errors
    E016 = "E016"  # if guard must be ConstExpr
    E017 = "E017"  # Tensor redefinition in same scope (SSA violation)

    # Gradient validation errors
    E018 = "E018"  # Gradient shape mismatch
    E019 = "E019"  # Circular gradient dependency
    E020 = "E020"  # Gradient dtype incompatible

    # Memory and recompute errors
    E021 = "E021"  # Recompute tensor not derivable from saved tensors
    E022 = "E022"  # Circular recompute dependency

    # Import errors
    E023 = "E023"  # Import name conflict
    E024 = "E024"  # Import alias shadows existing name

    # Pipeline errors
    E025 = "E025"  # Checkpoint spans pipeline stages

    # Constraint errors
    E026 = "E026"  # Invalid constraint expression
    E027 = "E027"  # Constraint violation


class WarningCode(str, Enum):
    """Compilation warning codes."""

    W001 = "W001"  # User definition shadows primitive
    W002 = "W002"  # Local definition shadows import
    W003 = "W003"  # Suboptimal auto-derived backward
    W004 = "W004"  # Unused saved tensor
    W005 = "W005"  # Implicit dtype narrowing
    W006 = "W006"  # Unresolved string dimension in IR shape
    W007 = "W007"  # Graph tensor has no matching activation slot


@dataclass
class SourceLocation:
    """Source location for error reporting."""

    file: Optional[str]
    line: int
    column: int
    end_line: Optional[int] = None
    end_column: Optional[int] = None

    def __str__(self) -> str:
        if self.file:
            return f"{self.file}:{self.line}:{self.column}"
        return f"line {self.line}, column {self.column}"


class DSLError(Exception):
    """Base exception for all DSL errors."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        location: Optional[SourceLocation] = None,
        hint: Optional[str] = None,
    ):
        self.code = code
        self.message = message
        self.location = location
        self.hint = hint
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        parts = [f"[{self.code.value}]"]
        if self.location:
            parts.append(f" at {self.location}:")
        parts.append(f" {self.message}")
        if self.hint:
            parts.append(f"\n  hint: {self.hint}")
        return "".join(parts)


class DSLSyntaxError(DSLError):
    """Syntax error in DSL source."""

    def __init__(
        self,
        message: str,
        location: Optional[SourceLocation] = None,
        hint: Optional[str] = None,
    ):
        super().__init__(ErrorCode.E001, message, location, hint)


class DSLTypeError(DSLError):
    """Type mismatch error."""

    def __init__(
        self,
        message: str,
        location: Optional[SourceLocation] = None,
        hint: Optional[str] = None,
    ):
        super().__init__(ErrorCode.E003, message, location, hint)


class DSLShapeError(DSLError):
    """Shape mismatch error."""

    def __init__(
        self,
        message: str,
        location: Optional[SourceLocation] = None,
        hint: Optional[str] = None,
    ):
        super().__init__(ErrorCode.E004, message, location, hint)


class DSLResolutionError(DSLError):
    """Resolution error (undefined identifier, import error, etc.)."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        location: Optional[SourceLocation] = None,
        hint: Optional[str] = None,
    ):
        super().__init__(code, message, location, hint)


class DSLUndefinedError(DSLResolutionError):
    """Undefined identifier error."""

    def __init__(
        self,
        name: str,
        location: Optional[SourceLocation] = None,
        hint: Optional[str] = None,
    ):
        super().__init__(
            ErrorCode.E002,
            f"undefined identifier '{name}'",
            location,
            hint,
        )


class DSLGradientError(DSLError):
    """Gradient-related error."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        location: Optional[SourceLocation] = None,
        hint: Optional[str] = None,
    ):
        super().__init__(code, message, location, hint)


class DSLConstraintError(DSLError):
    """Constraint violation error."""

    def __init__(
        self,
        constraint_expr: str,
        error_message: str,
        location: Optional[SourceLocation] = None,
    ):
        super().__init__(
            ErrorCode.E027,
            f"constraint '{constraint_expr}' failed: {error_message}",
            location,
        )


@dataclass
class DSLWarning:
    """Warning message from compilation."""

    code: WarningCode
    message: str
    location: Optional[SourceLocation] = None

    def __str__(self) -> str:
        loc_str = f" at {self.location}" if self.location else ""
        return f"[{self.code.value}]{loc_str}: {self.message}"


class WarningCollector:
    """Collects warnings during compilation."""

    def __init__(self):
        self.warnings: list[DSLWarning] = []

    def warn(
        self,
        code: WarningCode,
        message: str,
        location: Optional[SourceLocation] = None,
    ):
        self.warnings.append(DSLWarning(code, message, location))

    def clear(self):
        self.warnings.clear()

    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def __iter__(self):
        return iter(self.warnings)

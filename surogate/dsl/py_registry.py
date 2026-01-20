"""
Registry for Python DSL Modules and Primitives

Provides a central registry for all modules, blocks, models, and primitives
defined using the Python DSL decorators.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .specs import ModuleSpec, BlockSpec, ModelSpec, PrimitiveSpec


class Registry:
    """Central registry for DSL definitions."""

    def __init__(self):
        self._modules: dict[str, ModuleSpec] = {}
        self._blocks: dict[str, BlockSpec] = {}
        self._models: dict[str, ModelSpec] = {}
        self._primitives: dict[str, PrimitiveSpec] = {}

    # =========================================================================
    # Registration
    # =========================================================================

    def register_module(self, spec: ModuleSpec) -> None:
        """Register a module specification."""
        self._modules[spec.name] = spec

    def register_block(self, spec: BlockSpec) -> None:
        """Register a block specification."""
        self._blocks[spec.name] = spec

    def register_model(self, spec: ModelSpec) -> None:
        """Register a model specification."""
        self._models[spec.name] = spec

    def register_primitive(self, spec: PrimitiveSpec) -> None:
        """Register a primitive specification."""
        self._primitives[spec.name] = spec

    # =========================================================================
    # Lookup
    # =========================================================================

    def get_module(self, name: str) -> ModuleSpec | None:
        """Get a module by name."""
        return self._modules.get(name)

    def get_block(self, name: str) -> BlockSpec | None:
        """Get a block by name."""
        return self._blocks.get(name)

    def get_model(self, name: str) -> ModelSpec | None:
        """Get a model by name."""
        return self._models.get(name)

    def get_primitive(self, name: str) -> PrimitiveSpec | None:
        """Get a primitive by name."""
        return self._primitives.get(name)

    def get_any(self, name: str) -> ModuleSpec | BlockSpec | ModelSpec | PrimitiveSpec | None:
        """Get any definition by name (checks all registries)."""
        return (
            self._modules.get(name)
            or self._blocks.get(name)
            or self._models.get(name)
            or self._primitives.get(name)
        )

    # =========================================================================
    # Listing
    # =========================================================================

    def list_modules(self) -> list[str]:
        """List all registered module names."""
        return list(self._modules.keys())

    def list_blocks(self) -> list[str]:
        """List all registered block names."""
        return list(self._blocks.keys())

    def list_models(self) -> list[str]:
        """List all registered model names."""
        return list(self._models.keys())

    def list_primitives(self) -> list[str]:
        """List all registered primitive names."""
        return list(self._primitives.keys())

    def list_all(self) -> dict[str, list[str]]:
        """List all registered definitions by category."""
        return {
            "modules": self.list_modules(),
            "blocks": self.list_blocks(),
            "models": self.list_models(),
            "primitives": self.list_primitives(),
        }

    # =========================================================================
    # Utilities
    # =========================================================================

    def clear(self) -> None:
        """Clear all registrations."""
        self._modules.clear()
        self._blocks.clear()
        self._models.clear()
        self._primitives.clear()

    def __contains__(self, name: str) -> bool:
        """Check if a name is registered."""
        return self.get_any(name) is not None

    def __len__(self) -> int:
        """Total number of registered definitions."""
        return (
            len(self._modules)
            + len(self._blocks)
            + len(self._models)
            + len(self._primitives)
        )


# Global registry instance
registry = Registry()


# Convenience functions
def get_module(name: str) -> ModuleSpec | None:
    """Get a module from the global registry."""
    return registry.get_module(name)


def get_block(name: str) -> BlockSpec | None:
    """Get a block from the global registry."""
    return registry.get_block(name)


def get_model(name: str) -> ModelSpec | None:
    """Get a model from the global registry."""
    return registry.get_model(name)


def get_primitive(name: str) -> PrimitiveSpec | None:
    """Get a primitive from the global registry."""
    return registry.get_primitive(name)


def list_modules() -> list[str]:
    """List all registered modules."""
    return registry.list_modules()


def list_blocks() -> list[str]:
    """List all registered blocks."""
    return registry.list_blocks()


def list_models() -> list[str]:
    """List all registered models."""
    return registry.list_models()


def list_primitives() -> list[str]:
    """List all registered primitives."""
    return registry.list_primitives()

"""What a checkpoint says about its own quantisation, and whether its tensors agree.

A `quantization_config` states more than the format: it names the modules the quantiser
left alone (`ignore`, or `modules_to_not_convert`), and it may ask for things the serving
path does not implement at all. Reading only the format and ignoring the rest is how a
checkpoint gets served as something other than what it is — the failure that hid F16
re-quantisation for weeks behind a perplexity number that looked fine.

Two halves, and they are deliberately not the same half:

- **Refuse what cannot be honoured.** A declared KV-cache scheme, a sparsity structure or a
  transform is either implemented or refused. Serving the checkpoint anyway, silently, is the
  one outcome ruled out.
- **Cross-check what is claimed against what is there.** The declaration is a claim, not
  ground truth: a weight is quantised if and only if a scale sits beside it, and that is
  observable. Two published NVFP4 exports of the same 27B declare `ignore` lists of 2 and 303
  entries respectively while quantising exactly the same tensors, so the claim is a document
  of varying quality and the tensors are the fact. What matters is that a disagreement is
  *reported* rather than absorbed.

This module reads; it decides nothing about formats. What a converter does with a
disagreement is the converter's business.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

#: Declared fields the serving path has no implementation for at all. Each is refused when it
#: asks for something rather than being present and empty, which is how every published
#: checkpoint on this machine carries them.
_UNHONOURED = (
    ("sparsity_config", "a sparsity structure; the engine stores dense weights"),
    ("transform_config", "a weight transform (rotation, permutation) applied at load"),
)

#: What the engine's `auto` KV policy resolves to, which is the one thing a declared
#: `kv_cache_scheme` can already be honoured by without a flag. Kept here rather than
#: imported because it is the C++ planner's rule (`resolve_kv_storage`) restated for the
#: converter to check against, and a drift between the two should fail a test, not a serve.
def auto_kv_dtype(gdn_layers: int) -> str:
    """BF16 where every layer is attention, e4m3 where linear-attention layers carry it."""
    return "bf16" if gdn_layers == 0 else "fp8"

#: `.weight` companions that mark a quantised linear. `weight_packed` is compressed-tensors'
#: name for the codes themselves; the rest are scales beside an unpacked `weight`.
_QUANTISED_SUFFIXES = ("weight_scale", "weight_scale_inv", "weight_scale_2", "weight_packed",
                       "weight_global_scale")

_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")

#: A text-stack layer, and only that: `mtp.layers.0` and `model.visual.blocks.0` number their
#: own sub-stacks from zero, so a bare `layers.(\d+)` search would fold a draft head's single
#: layer onto text layer 0 and report an exception that is not one.
_TEXT_LAYER = re.compile(r"^(?:model\.)?(?:language_model\.)?layers\.(\d+)\.")


class QuantScopeError(ValueError):
    """A checkpoint asks for something the serving path does not implement."""


@dataclass(frozen=True)
class DeclaredScope:
    """What `quantization_config` says. `patterns` are the entries of `ignore` or
    `modules_to_not_convert`, verbatim: compressed-tensors allows a `re:` prefix and shell
    globs, so they are matched rather than compared."""

    method: str = ""
    patterns: tuple[str, ...] = ()
    #: Fields present and non-empty that name something unimplemented, as (field, what).
    unhonoured: tuple[tuple[str, str], ...] = ()

    def ignores(self, module: str) -> bool:
        """Whether the declaration says this module was left alone."""
        return any(_matches(pattern, module) for pattern in self.patterns)


@dataclass(frozen=True)
class ObservedScope:
    """What the checkpoint's tensor names show. A module is quantised exactly when a scale
    or a packed-code companion sits beside its weight."""

    quantised: frozenset[str] = frozenset()
    plain: frozenset[str] = frozenset()

    @property
    def modules(self) -> frozenset[str]:
        return self.quantised | self.plain

    def layers_of(self, suffix: str, *, quantised: bool) -> tuple[int, ...]:
        """The **text-stack** layer numbers whose `suffix` module is (or is not) quantised.

        This is what replaces a hand-measured table of exception layers: the same question,
        asked of the file rather than of a note about the file.
        """
        source = self.quantised if quantised else self.plain
        out = set()
        for module in source:
            match = _TEXT_LAYER.match(module)
            if match and module.endswith(suffix):
                out.add(int(match.group(1)))
        return tuple(sorted(out))


@dataclass(frozen=True)
class ScopeDisagreement:
    """Where the declaration and the tensors differ."""

    declared_but_quantised: tuple[str, ...] = ()
    undeclared_but_plain: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.declared_but_quantised or self.undeclared_but_plain)

    def describe(self, limit: int = 6) -> str:
        parts = []
        if self.declared_but_quantised:
            names = self.declared_but_quantised
            parts.append(f"{len(names)} named by `ignore` but quantised in the file "
                         f"({', '.join(names[:limit])}{' ...' if len(names) > limit else ''})")
        if self.undeclared_but_plain:
            names = self.undeclared_but_plain
            parts.append(f"{len(names)} quantised by the declaration but stored plain "
                         f"({', '.join(names[:limit])}{' ...' if len(names) > limit else ''})")
        return "; ".join(parts)


def declared_scope(config: Mapping[str, Any]) -> DeclaredScope:
    """The quantisation scope a `config.json` states. An unquantised checkpoint states none."""
    quantization = config.get("quantization_config")
    if not isinstance(quantization, Mapping):
        return DeclaredScope()
    patterns = quantization.get("ignore") or quantization.get("modules_to_not_convert") or []
    if isinstance(patterns, str):
        patterns = [patterns]
    unhonoured = tuple(
        (field_name, what) for field_name, what in _UNHONOURED if quantization.get(field_name)
    )
    return DeclaredScope(
        method=str(quantization.get("quant_method", "")),
        patterns=tuple(str(p) for p in patterns),
        unhonoured=unhonoured,
    )


def kv_cache_request(config: Mapping[str, Any]) -> str | None:
    """The cache dtype a `kv_cache_scheme` asks for: `"bf16"`, `"fp8"`, or None if it asks
    for nothing. Raises for a scheme the engine has no cache for.

    An integer cache is refused rather than approximated. It is a different numeric object
    from an e4m3 one, and this engine does not serve one.
    """
    quantization = config.get("quantization_config")
    scheme = quantization.get("kv_cache_scheme") if isinstance(quantization, Mapping) else None
    if not isinstance(scheme, Mapping) or not scheme:
        return None
    kind = str(scheme.get("type", "")).lower()
    bits = int(scheme.get("num_bits", 0) or 0)
    if kind in ("int", "integer", "uint"):
        raise QuantScopeError(
            f"the checkpoint asks for a {bits}-bit integer KV cache, which this engine does "
            f"not serve. Drop `kv_cache_scheme` to accept the engine's own cache dtype."
        )
    if kind == "float" and bits == 8:
        return "fp8"
    if kind == "float" and bits == 16:
        return "bf16"
    raise QuantScopeError(
        f"the checkpoint asks for a KV cache of type {kind!r} at {bits} bits, which names no "
        f"cache dtype this engine has (`bf16` or `fp8`)."
    )


def require_honourable(config: Mapping[str, Any], *, gdn_layers: int,
                       what: str = "this checkpoint") -> DeclaredScope:
    """The declared scope, refusing anything the serving path cannot honour.

    Silence is the failure mode being ruled out: a checkpoint that asks for an FP8 KV cache
    and is served with a BF16 one is not the model the publisher shipped, and nothing in the
    output says so. But a request the engine's own default already satisfies is honoured, not
    refused — `gdn_layers` is what decides that, exactly as the planner decides it.
    """
    scope = declared_scope(config)
    if scope.unhonoured:
        asks = "; ".join(f"`{name}` — {what_it_is}" for name, what_it_is in scope.unhonoured)
        raise QuantScopeError(
            f"{what} declares {asks}. The serving path does not implement it, and serving the "
            f"checkpoint as though it had not been asked for would be a different model. "
            f"Implement it or drop the field from the config."
        )
    wanted = kv_cache_request(config)
    if wanted is not None and wanted != auto_kv_dtype(gdn_layers):
        raise QuantScopeError(
            f"{what} asks for a {wanted} KV cache; on this geometry the engine's `auto` policy "
            f"resolves to {auto_kv_dtype(gdn_layers)}, so converting it would serve a cache the "
            f"checkpoint did not ask for. Serve it with `--kv-cache-dtype {wanted}`, which "
            f"honours the request, or drop `kv_cache_scheme` to accept the default."
        )
    return scope


def observed_scope(names: Iterable[str]) -> ObservedScope:
    """Which modules the checkpoint's tensor names show as quantised.

    Ground truth, and cheap: it reads the shard index, not the weights.
    """
    quantised: set[str] = set()
    plain: set[str] = set()
    for name in names:
        for suffix in _QUANTISED_SUFFIXES:
            if name.endswith("." + suffix):
                quantised.add(name[: -len(suffix) - 1])
                break
        else:
            if name.endswith(".weight"):
                plain.add(name[: -len(".weight")])
    return ObservedScope(frozenset(quantised), frozenset(plain - quantised))


def disagreement(declared: DeclaredScope, observed: ObservedScope) -> ScopeDisagreement:
    """Where the two differ, over the modules that carry a weight at all.

    Only linear-shaped modules are compared: a norm or an embedding is never quantised by
    these exports and naming one in `ignore` says nothing.
    """
    if not declared.patterns:
        return ScopeDisagreement()
    named_but_quantised = tuple(sorted(m for m in observed.quantised if declared.ignores(m)))
    unnamed_but_plain = tuple(sorted(
        m for m in observed.plain if not declared.ignores(m) and _looks_projected(m)
    ))
    return ScopeDisagreement(named_but_quantised, unnamed_but_plain)


def _looks_projected(module: str) -> bool:
    """Whether a module is the kind these exports quantise. Norms, biases and embeddings are
    stored plain by every profile, so their absence from `ignore` is not a disagreement."""
    leaf = module.rsplit(".", 1)[-1]
    if "norm" in leaf or leaf in ("embed_tokens", "bias"):
        return False
    return leaf.endswith("_proj") or leaf in ("lm_head", "gate", "qkv", "fc1", "fc2")


def _matches(pattern: str, module: str) -> bool:
    """compressed-tensors' matching: a `re:` prefix is a regular expression, a trailing `*`
    is a prefix match, anything else is the module path or a suffix of it."""
    if pattern.startswith("re:"):
        try:
            return re.search(pattern[3:], module) is not None
        except re.error:
            return False
    if pattern.endswith("*"):
        return module.startswith(pattern[:-1])
    return module == pattern or module.endswith("." + pattern)


def report(config: Mapping[str, Any], names: Sequence[str], *, what: str = "checkpoint") -> str:
    """One line for the conversion log: what was declared, what was found, and whether the
    two agree. Returns the empty string for an unquantised checkpoint."""
    declared = declared_scope(config)
    if not declared.method and not declared.patterns:
        return ""
    observed = observed_scope(names)
    differ = disagreement(declared, observed)
    line = (f"quantisation scope: {what} declares `{declared.method or 'unnamed'}` with "
            f"{len(declared.patterns)} module(s) left alone; the tensors show "
            f"{len(observed.quantised)} quantised, {len(observed.plain)} plain")
    return f"{line} -- DISAGREES: {differ.describe()}" if differ else f"{line} -- agrees"

"""surogate package init.

Installs an import-time shim that redirects dstack's `pydantic` imports
to the `pydantic.v1` namespace shipped with pydantic v2. dstack 0.20.x
is written against pydantic v1 (1.10.x) and we run pydantic v2 for the
rest of the codebase.

The shim is implemented as a `MetaPathFinder` that intercepts loading
of any module under the `dstack` package and rewrites the source text:

    from pydantic        →  from pydantic.v1
    from pydantic.X      →  from pydantic.v1.X
    import pydantic      →  import pydantic.v1 as pydantic
    import pydantic.X    →  import pydantic.v1.X as pydantic_X  (rare)

A spec-only finder is not enough because `from pydantic import Foo`
resolves `Foo` as an attribute of the cached `pydantic` module object,
which is the same v2 module surogate uses elsewhere. Rewriting source
binds dstack's symbols directly to v1 at import time.

Must run before any `import dstack` — placing it here guarantees that,
since importing any `surogate.*` submodule executes this file first.
"""

import importlib.abc
import importlib.machinery
import importlib.util
import os
import re
import sys


_PYDANTIC_FROM_RE = re.compile(r"^(?P<i>[ \t]*)from[ \t]+pydantic(?P<sub>(?:\.[A-Za-z_][A-Za-z0-9_.]*)?)[ \t]+import\b", re.MULTILINE)
_PYDANTIC_IMPORT_RE = re.compile(r"^(?P<i>[ \t]*)import[ \t]+pydantic(?P<sub>(?:\.[A-Za-z_][A-Za-z0-9_.]*)?)\b(?P<rest>[^\n]*)$", re.MULTILINE)


def _rewrite_source(text: str) -> str:
    text = _PYDANTIC_FROM_RE.sub(lambda m: f"{m.group('i')}from pydantic.v1{m.group('sub')} import ", text)

    def _rewrite_import(m: re.Match) -> str:
        indent = m.group("i")
        sub = m.group("sub")
        rest = m.group("rest")
        # `import pydantic` / `import pydantic.x` — preserve `as` alias if
        # present, otherwise add one so the bound name matches the original.
        if " as " in rest:
            return f"{indent}import pydantic.v1{sub}{rest}"
        bound = ("pydantic" + sub).replace(".", "_")
        original_bound = ("pydantic" + sub).split(".", 1)[0] if not sub else "pydantic" + sub
        return f"{indent}import pydantic.v1{sub} as {original_bound.replace('.', '_')}{rest}"

    text = _PYDANTIC_IMPORT_RE.sub(_rewrite_import, text)
    return text


class _DstackRewritingLoader(importlib.machinery.SourceFileLoader):
    """SourceFileLoader that rewrites `pydantic` → `pydantic.v1` on the way in.

    Subclassing the stdlib loader preserves resource-reader support
    (importlib.resources / pkg_resources) that Alembic and others rely on,
    and keeps bytecode caching disabled for rewritten files so stale pyc
    files from a previous non-rewriting run never win.
    """

    def get_data(self, path):  # type: ignore[override]
        data = super().get_data(path)
        if not path.endswith(".py"):
            return data
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return data
        return _rewrite_source(text).encode("utf-8")

    def set_data(self, *args, **kwargs):  # type: ignore[override]
        # Refuse to write .pyc — rewritten source must not be cached under
        # the unmodified source's hash, and we don't want the cache to shadow
        # future rewriter changes.
        return None

    def get_code(self, fullname):  # type: ignore[override]
        # Force source compilation on every import. SourceFileLoader.get_code
        # otherwise prefers any existing .pyc, which would bypass our rewrite
        # (stale .pyc from a pre-rewriter run, or a .pyc shipped in a wheel).
        source = self.get_source(fullname)
        if source is None:
            return None
        return compile(source, self.get_filename(fullname), "exec",
                       dont_inherit=True, optimize=-1)


class _DstackImportRewriter(importlib.abc.MetaPathFinder):
    """Find dstack modules and load them via the rewriting loader."""

    # Packages whose source must be rewritten because they were written
    # against pydantic v1 and we run pydantic v2 in this process.
    _PREFIXES = ("dstack", "pydantic_duality")

    def find_spec(self, name, path, target=None):
        if not any(name == p or name.startswith(p + ".") for p in self._PREFIXES):
            return None

        # Locate the module on disk via the regular import machinery, but
        # skip our own finder to avoid recursion.
        for finder in sys.meta_path:
            if finder is self:
                continue
            try:
                spec = finder.find_spec(name, path, target)
            except (ImportError, AttributeError):
                spec = None
            if spec is not None:
                break
        else:
            return None

        origin = getattr(spec, "origin", None)
        if not origin or not origin.endswith(".py") or not os.path.isfile(origin):
            # Namespace package, extension module, or missing source —
            # nothing to rewrite, let the normal loader handle it.
            return spec

        loader = _DstackRewritingLoader(name, origin)
        new_spec = importlib.util.spec_from_file_location(
            name,
            origin,
            loader=loader,
            submodule_search_locations=list(spec.submodule_search_locations)
            if spec.submodule_search_locations is not None
            else None,
        )
        return new_spec


sys.meta_path.insert(0, _DstackImportRewriter())

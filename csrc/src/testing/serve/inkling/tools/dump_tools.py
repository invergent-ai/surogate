# Dump the surogates harness's builtin tool schemas (OpenAI format, as ToolRegistry.get_schemas exports them).
import json, sys, importlib, pkgutil, asyncio, os
# SUROGATES_REPO: a checkout of the surogates harness (required).
if not os.environ.get("SUROGATES_REPO"):
    sys.exit("dump_tools.py: set SUROGATES_REPO to a checkout of the surogates harness")
sys.path.insert(0, os.environ["SUROGATES_REPO"])
from surogates.tools.registry import ToolRegistry
import surogates.tools.builtin as B
reg = ToolRegistry()
ok, bad = [], []
for m in pkgutil.iter_modules(B.__path__):
    try:
        mod = importlib.import_module("surogates.tools.builtin." + m.name)
        if hasattr(mod, "register"):
            mod.register(reg); ok.append(m.name)
    except Exception as e:
        bad.append((m.name, repr(e)[:160]))
schemas = reg.get_schemas()
json.dump(schemas, open(sys.argv[1], "w"), ensure_ascii=False)
print(len(schemas), "schemas;", "ok", len(ok), "bad", bad[:20], file=sys.stderr)

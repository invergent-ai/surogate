# agent-EN-f2: dump the surogates harness's builtin tool schemas (OpenAI format, as ToolRegistry.get_schemas exports them).
import json, sys, importlib, pkgutil, asyncio
sys.path.insert(0, "/mnt/data/work/repos/surogates")
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

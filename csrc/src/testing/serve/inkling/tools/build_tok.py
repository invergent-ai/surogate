# agent-EN-f2: tokenizer corpus: every distinct text the renderer encodes in a sample of the real bodies (message texts,
# reasoning, tool results, tool-call JSON, tool declarations), plus the edge texts. -> {"case","text"} per line.
import json, sys
seen, out = set(), []
def add(t):
    if isinstance(t, str) and t and t not in seen:
        seen.add(t); out.append({"case": f"tok-{len(out):05d}", "text": t})
for path in sys.argv[2:]:
    for i, line in enumerate(open(path)):
        if i % 3:
            continue
        body = json.loads(line)["body"]
        for m in body.get("messages") or []:
            c = m.get("content")
            if isinstance(c, str): add(c)
            elif isinstance(c, list):
                for p in c:
                    if isinstance(p, dict): add(p.get("text"))
            add(m.get("reasoning_content")); add(m.get("reasoning"))
            for tc in m.get("tool_calls") or []:
                add((tc.get("function") or {}).get("arguments")); add((tc.get("function") or {}).get("name"))
        add(json.dumps(body.get("tools") or [], separators=(",", ":"), ensure_ascii=False))
extra = ["", " ", "  ", "\n", "\r\n", "\t", " \n ", "á", "é", "I'm", "I'M", "'s", "x's", "12345678", "٣٤٥٦",
         "\U0001F600\U0001F601", "‍", "﻿", "　", " x", "x ", "foo  bar", "foo \n\n bar", "  \n\n", "/a/b/c\n/d",
         "a//b", "--> <!-- -->", "क्ष", "각", "ÅB", "  ", "\x00\x01\x7f\x85"]
for t in extra: add(t)
with open(sys.argv[1], "w") as f:
    for it in out: f.write(json.dumps(it, ensure_ascii=False) + "\n")
print(len(out), "texts", file=sys.stderr)

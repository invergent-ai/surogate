#!/usr/bin/env bash
# minja implements Jinja2's `defined` test but not its `undefined` one, and real chat templates
# use both -- Qwen3.5/3.6/3.8's opens with `{%- if enable_thinking is undefined or ... %}` and
# fails to render with "Unknown type for 'is' operator: undefined". The test is the negation of
# `defined`, which minja already evaluates as "not null", so this adds the one missing line.
# Idempotent: FetchContent re-runs the patch step whenever it re-populates the source.
set -eu
header="include/minja/minja.hpp"
[ -f "$header" ] || { echo "patch_minja: $header not found in $(pwd)" >&2; exit 1; }
if grep -q 'name == "undefined"' "$header"; then exit 0; fi
sed -i '/if (name == "defined") return !l.is_null();/a\              if (name == "undefined") return l.is_null();' "$header"
grep -q 'name == "undefined"' "$header" || { echo "patch_minja: anchor not found" >&2; exit 1; }

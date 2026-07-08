"""Applies the paper-exact few-shot swap: EXAMPLE 3 (story) and EXAMPLE 4 (spanning-tree code)
replaced by the two Countdown examples; Examples 1-2 (Medreason, DeepMath) and the rest of the
module untouched. Run ONLY after the step-90 eval row has landed.
Usage: apply_fewshot_swap.py [path-to-env-file]   (default: the live env file)"""
import importlib.util
import shutil
import sys

sys.path.insert(0, "/home/densemax/work/flavius/surogate/ultra")
sys.path.insert(0, "/home/densemax/work/flavius/surogate")

ENV = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/densemax/work/flavius/surogate/environments/fugu-ultra-pilot/fugu_ultra_pilot.py"

NEW_34 = '''EXAMPLE 3:
Question: Using the numbers [3, 7, 25, 50] and the operations +, -, *, / with each number used at most once, write an arithmetic expression that evaluates exactly to 475. Provide the final expression in <answer> </answer> tags.
Assistant Response: This is a search problem over a small space, so a long pipeline adds little. One strong model can search for a valid expression, and a second model can independently verify the arithmetic and the number-usage constraint before formatting the answer.
model_id = [0, 3]
subtasks = ["Find an arithmetic expression that uses each of the numbers 3, 7, 25 and 50 at most once and evaluates exactly to 475. Search systematically, for example by building large products first and adjusting with the remaining numbers. Show your search in <idea> </idea> tags.", "Verify that the proposed expression evaluates exactly to 475 and uses each allowed number at most once. Fix it if it is wrong, then provide the final expression in <answer> </answer> tags."]
access_list = [[], ["all"]]

EXAMPLE 4:
Question: Using the numbers [2, 5, 8, 9, 75] and the operations +, -, *, / with each number used at most once, write an arithmetic expression that evaluates exactly to 632. Provide the final expression in <answer> </answer> tags.
Assistant Response: The target is far from any single product, so I will run two searchers independently so they cannot anchor on each other's partial attempts, have a third model compare their candidates, a fourth model re-check only the chosen expression's arithmetic in isolation, and a final model format the result.
model_id = [2, 0, 1, 3, 2]
subtasks = ["Search for an arithmetic expression using each of the numbers 2, 5, 8, 9, 75 at most once that evaluates exactly to 632. Consider anchoring on multiples of 75 or 8 and adjusting with the smaller numbers. Show your search in <idea> </idea> tags.", "Search for an arithmetic expression using each of the numbers 2, 5, 8, 9, 75 at most once that evaluates exactly to 632. Consider anchoring on multiples of 75 or 8 and adjusting with the smaller numbers. Show your search in <idea> </idea> tags.", "Compare the two candidate expressions, check which ones satisfy the target and the usage constraint, and choose the best verified candidate.", "Re-compute the chosen expression step by step in isolation and confirm it equals exactly 632 and respects the usage constraint.", "Provide the final expression in <answer> </answer> tags according to the question's formatting instructions."]
access_list = [[], [], ["all"], [3], ["all"]]'''

src = open(ENV).read()
start = src.index("EXAMPLE 3:")
end = src.index("'''", start)  # closing quotes of _FEWSHOT_EXAMPLES; keep them and the rest
old_block = src[start:end]
assert "graph" in old_block, "unexpected EXAMPLE 4 content - aborting"
assert "model_id = [3]" in old_block, "unexpected EXAMPLE 3 content - aborting"

shutil.copy(ENV, ENV + ".pre_ood_swap")
open(ENV, "w").write(src[:start] + NEW_34 + src[end:])
print(f"swapped {len(old_block)} -> {len(NEW_34)} chars (backup: {ENV}.pre_ood_swap)")

spec = importlib.util.spec_from_file_location("fpe_swapped", ENV)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sysmsg = mod._system_prompt(5)
assert sysmsg.count("EXAMPLE") == 4, f"expected 4 examples, got {sysmsg.count('EXAMPLE')}"
assert "475" in sysmsg and "632" in sysmsg, "Countdown examples missing"
assert "spanning" not in sysmsg.lower() and "graph" not in sysmsg.lower(), "old code example survived"
assert "intestinal muscle" in sysmsg, "Example 1 (Medreason) damaged"
assert "limit" in sysmsg.lower(), "Example 2 (DeepMath) damaged"
print("VALIDATION OK: 4 examples; Countdown x2 in; story+code out; Medreason+DeepMath intact")

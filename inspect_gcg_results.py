"""Inspect GCG attack results across all models.

Auto-discovers all .pt files in gcg_results/ and prints a summary table.
Also saves the table to gcg_results/summary.txt.
"""
import glob
import os
import sys
import torch

RESULTS_DIR = os.environ.get("GCG_RESULTS_DIR", "gcg_results")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "summary.txt")

files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.pt")))

if not files:
    print(f"No .pt files found in {RESULTS_DIR}/")
    sys.exit(0)

lines = []
def out(s=""):
    print(s)
    lines.append(s)

header = f"{'Model':<25} {'Total':>6} {'Succ':>6} {'Fail':>6} {'ASR':>8} {'Avg Steps':>10}"
out(header)
out("-" * len(header))

for path in files:
    fname = os.path.basename(path)
    name = fname.replace(".pt", "")

    data = torch.load(path, weights_only=False)
    if not data:
        out(f"{name:<25} {'(empty)':>6}")
        continue

    total = len(data)
    succeeded = sum(1 for e in data if e.get("succeeded", False))
    failed = total - succeeded
    asr = succeeded / total if total > 0 else 0
    avg_steps = sum(e.get("steps", 0) for e in data) / total if total > 0 else 0

    out(f"{name:<25} {total:>6} {succeeded:>6} {failed:>6} {asr:>7.1%} {avg_steps:>10.1f}")

    succ_examples = [e for e in data if e.get("succeeded", False)]
    fail_examples = [e for e in data if not e.get("succeeded", False)]

    if succ_examples:
        e = succ_examples[0]
        suffix = e.get("suffix_text", "")[:60]
        text = (e.get("input_text", ""))[:60]
        out(f"  Success: \"{text}...\" + \"{suffix}...\" ({e.get('steps', '?')} steps)")

    if fail_examples:
        e = fail_examples[0]
        text = (e.get("input_text", ""))[:60]
        out(f"  Fail:    \"{text}...\" ({e.get('steps', '?')} steps)")

out()

with open(OUTPUT_FILE, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Summary saved to {OUTPUT_FILE}")

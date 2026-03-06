"""Inspect GCG attack results across all models."""
import os
import torch

RESULTS_DIR = "gcg_results"
FILES = [
    "bert_harm.pt",
    "bert_obfus.pt",
    "roberta_harm.pt",
    "roberta_obfus.pt",
    "deberta_harm.pt",
    "deberta_obfus.pt",
]

print(f"{'Model':<20} {'Total':>6} {'Succ':>6} {'Fail':>6} {'ASR':>8} {'Avg Steps':>10}")
print("-" * 60)

for fname in FILES:
    path = os.path.join(RESULTS_DIR, fname)
    name = fname.replace(".pt", "")

    if not os.path.exists(path):
        print(f"{name:<20} {'(missing)':>6}")
        continue

    data = torch.load(path, weights_only=False)
    if not data:
        print(f"{name:<20} {'(empty)':>6}")
        continue

    total = len(data)
    succeeded = sum(1 for e in data if e.get("succeeded", False))
    failed = total - succeeded
    asr = succeeded / total if total > 0 else 0
    avg_steps = sum(e.get("steps", 0) for e in data) / total if total > 0 else 0

    print(f"{name:<20} {total:>6} {succeeded:>6} {failed:>6} {asr:>7.1%} {avg_steps:>10.1f}")

    # Show a few succeeded/failed examples
    succ_examples = [e for e in data if e.get("succeeded", False)]
    fail_examples = [e for e in data if not e.get("succeeded", False)]

    if succ_examples:
        e = succ_examples[0]
        suffix = e.get("suffix_text", "")[:60]
        text = (e.get("input_text", ""))[:60]
        print(f"  Example success: \"{text}...\" + \"{suffix}...\" ({e.get('steps', '?')} steps)")

    if fail_examples:
        e = fail_examples[0]
        text = (e.get("input_text", ""))[:60]
        print(f"  Example fail:    \"{text}...\" ({e.get('steps', '?')} steps)")

print()

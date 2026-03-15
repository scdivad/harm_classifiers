"""Inspect all .pt files in gcg_results/ — show format, counts, and key fields.

Helps distinguish old runs from new ones by showing modification time,
schema (keys), restart/step counts, and example index ranges.
"""
import glob
import os
import torch
from datetime import datetime

RESULTS_DIR = os.environ.get("GCG_RESULTS_DIR", "gcg_results")
pt_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.pt")))

if not pt_files:
    print(f"No .pt files found in {RESULTS_DIR}/")
    exit()

for path in pt_files:
    fname = os.path.basename(path)
    mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
    size_kb = os.path.getsize(path) / 1024

    print(f"\n{'='*70}")
    print(f"{fname}  ({size_kb:.0f} KB, modified {mtime})")
    print(f"{'='*70}")

    try:
        data = torch.load(path, weights_only=False, map_location="cpu")
    except Exception as e:
        print(f"  FAILED TO LOAD: {e}")
        continue

    if isinstance(data, list):
        print(f"  Type: list of {len(data)} entries")
        if len(data) == 0:
            print("  (empty)")
            continue

        first = data[0]
        if isinstance(first, dict):
            print(f"  Keys: {list(first.keys())}")

            # Succeeded / failed
            n_succ = sum(1 for e in data if e.get("succeeded", False))
            n_fail = len(data) - n_succ
            print(f"  Succeeded: {n_succ}/{len(data)}  (ASR={n_succ/len(data):.4f})")
            print(f"  Failed:    {n_fail}/{len(data)}  (Robust acc={n_fail/len(data):.4f})")

            # Restart info
            if "restart" in first:
                restarts = [e.get("restart", 0) for e in data]
                print(f"  Restarts: min={min(restarts)}, max={max(restarts)}, mean={sum(restarts)/len(restarts):.1f}")

            # Steps info
            if "steps" in first:
                steps = [e.get("steps", 0) for e in data]
                print(f"  Steps: min={min(steps)}, max={max(steps)}, mean={sum(steps)/len(steps):.1f}")

            # Example index range
            if "example_idx" in first:
                idxs = [e["example_idx"] for e in data]
                print(f"  Example indices: {min(idxs)}..{max(idxs)} ({len(set(idxs))} unique out of {len(idxs)} entries)")

            # Other identifying fields
            for key in ["model_type", "status", "dataset_idx"]:
                if key in first:
                    vals = set(e.get(key) for e in data)
                    print(f"  {key}: {vals}")

            # Sample entry
            print(f"\n  Sample entry (first):")
            for k, v in first.items():
                if isinstance(v, torch.Tensor):
                    print(f"    {k}: Tensor shape={v.shape}, dtype={v.dtype}")
                elif isinstance(v, str) and len(v) > 80:
                    print(f"    {k}: '{v[:80]}...'")
                else:
                    print(f"    {k}: {v}")
        else:
            print(f"  Entry type: {type(first)}")

    elif isinstance(data, dict):
        print(f"  Type: dict with keys {list(data.keys())}")
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                print(f"    {k}: Tensor shape={v.shape}, dtype={v.dtype}")
            elif isinstance(v, list):
                print(f"    {k}: list of {len(v)} items")
            else:
                print(f"    {k}: {type(v).__name__} = {v}")
    else:
        print(f"  Type: {type(data)}")

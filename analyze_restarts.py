"""Analyze GCG attack results with random restarts.

Computes cumulative ASR at each restart level:
  - Restart 0: ASR from first attempt only
  - Restart ≤1: ASR from first two attempts
  - ...
  - Restart ≤N: ASR from all attempts (final ASR)

Usage:
    python analyze_restarts.py --pattern "gcg_results/*_r5.pt" --num_restarts 5
"""

import argparse
import glob
import os
import torch


def analyze_file(path, num_restarts):
    """Analyze a single results file. Returns (model_name, total, per_restart_counts)."""
    data = torch.load(path, weights_only=False)
    model_name = os.path.splitext(os.path.basename(path))[0]

    total = len(data)
    if total == 0:
        return model_name, 0, []

    # Count successes at each restart level
    # succeeded_at[r] = number of examples that succeeded exactly on restart r
    succeeded_at = [0] * num_restarts
    for entry in data:
        if entry.get("succeeded", False):
            r = entry.get("restart", 0)
            if r < num_restarts:
                succeeded_at[r] += 1

    return model_name, total, succeeded_at


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=str, default="gcg_results/*_r5.pt",
                        help="Glob pattern for result files")
    parser.add_argument("--num_restarts", type=int, default=5)
    args = parser.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"No files found matching '{args.pattern}'")
        return

    print(f"{'Model':<30s}", end="")
    print(f"  {'Total':>5s}", end="")
    for r in range(args.num_restarts):
        print(f"  {'≤' + str(r):>6s}", end="")
    print(f"  {'Δ(0→max)':>8s}")
    print("-" * (30 + 7 + 8 * args.num_restarts + 10))

    all_results = []
    for path in files:
        model_name, total, succeeded_at = analyze_file(path, args.num_restarts)
        if total == 0:
            print(f"{model_name:<30s}  {'(empty)':>5s}")
            continue
        all_results.append((model_name, total, succeeded_at))

        cumulative = 0
        cum_rates = []
        for r in range(args.num_restarts):
            cumulative += succeeded_at[r]
            cum_rates.append(cumulative / total * 100)

        delta = cum_rates[-1] - cum_rates[0] if cum_rates else 0

        print(f"{model_name:<30s}", end="")
        print(f"  {total:>5d}", end="")
        for rate in cum_rates:
            print(f"  {rate:>5.1f}%", end="")
        print(f"  {'+' if delta >= 0 else ''}{delta:>5.1f}%")

    # Summary across all models
    if len(all_results) > 1:
        print("-" * (30 + 7 + 8 * args.num_restarts + 10))
        total_all = sum(t for _, t, _ in all_results)
        merged_at = [0] * args.num_restarts
        for _, _, sa in all_results:
            for r in range(args.num_restarts):
                merged_at[r] += sa[r]
        cumulative = 0
        cum_rates = []
        for r in range(args.num_restarts):
            cumulative += merged_at[r]
            cum_rates.append(cumulative / total_all * 100)
        delta = cum_rates[-1] - cum_rates[0]

        print(f"{'AVERAGE':<30s}", end="")
        print(f"  {total_all:>5d}", end="")
        for rate in cum_rates:
            print(f"  {rate:>5.1f}%", end="")
        print(f"  {'+' if delta >= 0 else ''}{delta:>5.1f}%")

    # Save to file
    out_path = os.path.join(os.path.dirname(args.pattern.replace("*", "")), "restart_analysis.txt")
    if out_path.startswith("/"):
        pass
    else:
        out_path = "gcg_results/restart_analysis.txt"

    with open(out_path, "w") as f:
        for model_name, total, succeeded_at in all_results:
            cumulative = 0
            rates = []
            for r in range(args.num_restarts):
                cumulative += succeeded_at[r]
                rates.append(cumulative / total * 100)
            f.write(f"{model_name}\t{total}\t" + "\t".join(f"{r:.1f}" for r in rates) + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

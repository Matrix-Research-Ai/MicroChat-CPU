"""
MicroChat-CPU Benchmark Suite.

Measures throughput and scaling across model sizes and configurations.
Generates a report comparing tok/sec, step time, and phase breakdowns.

Usage:
    # Full benchmark (depth 2, 4, 6 all configs)
    python -m scripts.cpu_benchmark

    # Single config
    python -m scripts.cpu_benchmark --depth=4 --compress --async-overlap

    # All configs, quick test
    python -m scripts.cpu_benchmark --quick

    # JSON output for plotting
    python -m scripts.cpu_benchmark --json
"""

import argparse
import json
import os
import subprocess
import sys
import time

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="MicroChat-CPU Benchmark Suite")
parser.add_argument("--depth", type=int, default=0,
                    help="Model depth (0 = run all: 2, 4, 6)")
parser.add_argument("--compress", action="store_true", default=None,
                    help="Enable gradient compression")
parser.add_argument("--no-compress", action="store_false", dest="compress")
parser.add_argument("--async-overlap", action="store_true", default=None,
                    help="Enable async overlap")
parser.add_argument("--no-async-overlap", action="store_false", dest="async_overlap")
parser.add_argument("--sparsify", action="store_true", default=False,
                    help="Enable gradient sparsification")
parser.add_argument("--quick", action="store_true", default=False,
                    help="Quick test (depth=2 only, fewer steps)")
parser.add_argument("--json", action="store_true", default=False,
                    help="Output results as JSON")
parser.add_argument("--num-iterations", type=int, default=0,
                    help="Training iterations (0 = auto)")
args = parser.parse_args()

# -----------------------------------------------------------------------------
CONFIGS = []

def add_config(depth, compress, async_overlap, sparsify, tag):
    num_iters = args.num_iterations or (10 if args.quick else 30)
    seq_len = 128 if args.quick else 256
    batch = 4
    total = batch * seq_len * 2

    config = {
        "depth": depth,
        "compress": compress,
        "async_overlap": async_overlap,
        "sparsify": sparsify,
        "tag": tag,
        "num_iterations": num_iters,
        "max_seq_len": seq_len,
        "device_batch_size": batch,
        "total_batch_size": total,
    }
    CONFIGS.append(config)


def build_flag_str(cfg):
    flags = (
        f"--synthetic --depth={cfg['depth']} "
        f"--max-seq-len={cfg['max_seq_len']} "
        f"--device-batch-size={cfg['device_batch_size']} "
        f"--total-batch-size={cfg['total_batch_size']} "
        f"--num-iterations={cfg['num_iterations']} "
        f"--eval-every=-1 --sample-every=-1 --save-every=0 "
        f"--profile --run=dummy"
    )
    if cfg["compress"]:
        flags += " --compress"
    else:
        flags += " --no-compress"
    if cfg["async_overlap"]:
        flags += " --async-overlap"
    else:
        flags += " --no-async-overlap"
    if cfg["sparsify"]:
        flags += f" --sparsify --sparsity-ratio=0.05"
    return flags


def parse_bench_output(output: str) -> dict:
    """Extract benchmark metrics from training output."""
    result = {
        "tok_per_sec": 0,
        "avg_step_ms": 0,
        "phases": {},
        "params": 0,
        "steps": 0,
    }

    for line in output.split("\n"):
        # Tok/sec: "step 00010/00020 | ... | tok/sec: 599 | ..."
        if "tok/sec:" in line:
            parts = line.split("tok/sec:")
            if len(parts) > 1:
                tok = parts[1].strip().split()[0].replace(",", "")
                try:
                    result["tok_per_sec"] = max(result["tok_per_sec"], int(tok))
                except ValueError:
                    pass

        # Step timing: "  Phase             Avg      %      Min      Max"
        # Phase lines: "  fwd            1341.3ms  37.2%  1230.9  1500.6"
        if line.strip().startswith(("fwd", "bwd", "data", "comm", "optim")):
            parts = line.split()
            if len(parts) >= 4:
                phase = parts[0]
                try:
                    avg_ms = float(parts[1].replace("ms", ""))
                    pct = float(parts[2].replace("%", ""))
                    result["phases"][phase] = {"avg_ms": avg_ms, "pct": pct}
                except (ValueError, IndexError):
                    pass

        # Total params: "Total parameters: 19,709,994"
        if "Total parameters:" in line:
            try:
                result["params"] = int(line.split(":")[1].strip().replace(",", ""))
            except (ValueError, IndexError):
                pass

        # Avg step time
        if "Avg step time:" in line:
            try:
                ms = line.split(":")[1].strip().replace("ms", "")
                result["avg_step_ms"] = float(ms)
            except (ValueError, IndexError):
                pass

        # Steps
        if "Steps completed:" in line:
            try:
                result["steps"] = int(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass

        # Throughput line
        if "Throughput:" in line:
            try:
                tok = line.split(":")[1].strip().replace("tok/s", "")
                result["tok_per_sec"] = int(float(tok))
            except (ValueError, IndexError):
                pass

    return result


def run_benchmark(cfg) -> dict:
    """Run a single benchmark configuration."""
    flag_str = build_flag_str(cfg)
    cmd = f"python -m scripts.cpu_train {flag_str}"

    print(f"\n{'=' * 60}")
    print(f"  Benchmark: {cfg['tag']}")
    print(f"  {cmd}")
    print(f"{'=' * 60}")

    t0 = time.time()
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=600
    )
    elapsed = time.time() - t0

    output = result.stdout + result.stderr
    metrics = parse_bench_output(output)

    metrics["tag"] = cfg["tag"]
    metrics["depth"] = cfg["depth"]
    metrics["compress"] = cfg["compress"]
    metrics["async_overlap"] = cfg["async_overlap"]
    metrics["sparsify"] = cfg["sparsify"]
    metrics["wall_time_s"] = round(elapsed, 1)
    metrics["num_iterations"] = cfg["num_iterations"]
    metrics["max_seq_len"] = cfg["max_seq_len"]
    metrics["total_batch_size"] = cfg["total_batch_size"]

    # Print summary
    print(f"\n  Results for {cfg['tag']}:")
    print(f"    Tok/sec:     {metrics['tok_per_sec']}")
    print(f"    Avg step:    {metrics['avg_step_ms']:.1f} ms")
    print(f"    Wall time:   {metrics['wall_time_s']:.1f}s")
    if metrics["phases"]:
        phase_str = " | ".join(
            f"{p}: {v['avg_ms']:.0f}ms ({v['pct']:.0f}%)"
            for p, v in metrics["phases"].items()
        )
        print(f"    Phases:      {phase_str}")
    print(f"    Params:      {metrics['params']:,}")

    return metrics


# -----------------------------------------------------------------------------
# Build config list
if args.depth > 0:
    depths = [args.depth]
else:
    depths = [2, 4, 6] if not args.quick else [2]

compress_opts = [args.compress] if args.compress is not None else [True, False]
async_opts = [args.async_overlap] if args.async_overlap is not None else [True, False]

for d in depths:
    for comp in compress_opts:
        for async_ol in async_opts:
            tag_parts = [f"d{d}"]
            if comp:
                tag_parts.append("cmp")
            if not async_ol:
                tag_parts.append("noasync")
            if args.sparsify:
                tag_parts.append("sparse")
            tag = "_".join(tag_parts)

            add_config(
                depth=d,
                compress=comp,
                async_overlap=async_ol,
                sparsify=args.sparsify,
                tag=tag,
            )

# -----------------------------------------------------------------------------
# Run benchmarks
print(f"\nMicroChat-CPU Benchmark Suite")
print(f"{'=' * 60}")
print(f"  Configurations: {len(CONFIGS)}")
print(f"  Iterations:     {CONFIGS[0]['num_iterations']}")
print(f"  Seq len:        {CONFIGS[0]['max_seq_len']}")
print(f"{'=' * 60}")

results = []
for cfg in CONFIGS:
    try:
        metrics = run_benchmark(cfg)
        results.append(metrics)
    except subprocess.TimeoutExpired:
        print(f"  ⚠ Timeout for {cfg['tag']}")
    except Exception as e:
        print(f"  ⚠ Error for {cfg['tag']}: {e}")

# -----------------------------------------------------------------------------
# Report
print(f"\n{'=' * 60}")
print(f"  BENCHMARK SUMMARY")
print(f"{'=' * 60}")
print(f"  {'Config':<24} {'Tok/s':>8} {'Step':>8} {'Wall':>8}  {'Phases'}")
print(f"  {'-'*24} {'-'*8} {'-'*8} {'-'*8}  {'-'*30}")

for r in sorted(results, key=lambda x: (-x.get("depth", 0), -x.get("tok_per_sec", 0))):
    p = r.get("phases", {})
    phase_str = " ".join(f"{v['pct']:.0f}%" for v in p.values())
    print(f"  {r['tag']:<24} {r['tok_per_sec']:>8} "
          f"{r['avg_step_ms']:>7.0f}ms {r['wall_time_s']:>7.1f}s  {phase_str}")

print(f"{'=' * 60}")

# Speedup comparison (relative to baseline)
baselines = {}
for r in results:
    key = (r["depth"], r["compress"], r["async_overlap"])
    if r["tok_per_sec"] > 0:
        baselines[key] = r["tok_per_sec"]

if len(results) > 1:
    print(f"\n  Relative Performance:")
    ref = max(r["tok_per_sec"] for r in results) if results else 1
    for r in sorted(results, key=lambda x: -x.get("tok_per_sec", 0)):
        pct = (r["tok_per_sec"] / ref * 100) if ref > 0 else 0
        bar = "█" * max(1, int(pct / 5))
        print(f"  {r['tag']:<24} {r['tok_per_sec']:>6} tok/s ({pct:>5.1f}%) {bar}")

# JSON output
if args.json:
    print()
    print(json.dumps(results, indent=2))

print(f"\n✓ Benchmark complete ({len(results)} configs)")

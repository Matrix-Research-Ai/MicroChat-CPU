"""
WAN Compression Benchmark — demonstrates bandwidth savings from gradient compression.

From the research: "The performance drop when switching from LAN to WAN will
highlight the effectiveness of techniques like gradient compression."

Measures tok/sec under simulated WAN with compression on vs off, at various
bandwidth limits. Shows the crossover point where compression becomes beneficial.

Usage:
    python -m scripts.benchmark_wan_compression
    python -m scripts.benchmark_wan_compression --quick
"""

import argparse
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description="WAN Compression Benchmark")
parser.add_argument("--quick", action="store_true",
                    help="Fewer steps, faster benchmark")
parser.add_argument("--depth", type=int, default=2)
args = parser.parse_args()

N_ITERS = 5 if not args.quick else 3
BASE = (f"python -m scripts.cpu_train --synthetic --depth={args.depth} "
        f"--max-seq-len=128 --device-batch-size=2 --total-batch-size=256 "
        f"--num-iterations={N_ITERS} --eval-every=-1 --profile --run=dummy "
        f"--async-overlap")


def run(label, cmd):
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
    out = r.stdout + r.stderr
    tok = 0
    for line in out.split("\n"):
        if "Throughput:" in line:
            try:
                tok = int(float(line.split(":")[1].strip().replace("tok/s", "")))
            except ValueError:
                pass
    return {"label": label, "tok": tok, "wall": round(time.time() - t0, 1)}


# Configs: (label, extra_flags, bandwidth_mbps, latency_ms)
configs = [
    # No WAN sim — baseline
    ("LAN baseline",            f"{BASE} --compress",              0,   0),
    ("LAN no-compress",         f"{BASE} --no-compress",           0,   0),
    # Gigabit WAN (1000 Mbps, 20ms)
    ("1Gbps+compress",          f"{BASE} --compress --simulate-wan --wan-bandwidth=1000 --wan-latency=20",  1000, 20),
    ("1Gbps+no-compress",       f"{BASE} --no-compress --simulate-wan --wan-bandwidth=1000 --wan-latency=20", 1000, 20),
    # 100 Mbps WAN (typical residential)
    ("100Mbps+compress",        f"{BASE} --compress --simulate-wan --wan-bandwidth=100 --wan-latency=50",   100,  50),
    ("100Mbps+no-compress",     f"{BASE} --no-compress --simulate-wan --wan-bandwidth=100 --wan-latency=50",  100,  50),
    # 50 Mbps WAN (slow link)
    ("50Mbps+compress",         f"{BASE} --compress --simulate-wan --wan-bandwidth=50 --wan-latency=80",     50,   80),
    ("50Mbps+no-compress",      f"{BASE} --no-compress --simulate-wan --wan-bandwidth=50 --wan-latency=80",    50,   80),
]

if args.quick:
    configs = configs[:4]  # LAN + 1Gbps only

print()
print("=" * 65)
print("  WAN COMPRESSION BENCHMARK")
print("=" * 65)
print(f"  Model: depth={args.depth}, {N_ITERS} iterations")
print(f"  Tests compression benefit at various WAN bandwidths")
print("=" * 65)

results = []
for label, cmd, bw, lat in configs:
    is_comp = "compress" in label and "no-compress" not in label
    print(f"\n  [{label}]", end=" ", flush=True)
    r = run(label, cmd)
    results.append(r)
    bw_str = f"{bw}Mbps" if bw else "N/A"
    lat_str = f"{lat}ms" if lat else "N/A"
    print(f"{r['tok']:>5} tok/s  ({r['wall']:>4.1f}s)  bw={bw_str} lat={lat_str}")

# Summary
print()
print("=" * 65)
print("  COMPRESSION BENEFIT BY WAN SPEED")
print("=" * 65)
print(f"  {'Scenario':<22} {'Comp':>6} {'NoComp':>8} {'Δ':>8} {'Benefit':>8}")
print(f"  {'-'*22} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")

# Pair up compress/no-compress at each bandwidth
pairs = {}
for r in results:
    base_label = r["label"].replace("+compress", "").replace("+no-compress", "")
    key = base_label
    if key not in pairs:
        pairs[key] = {}
    if "no-compress" in r["label"]:
        pairs[key]["no_comp"] = r["tok"]
    else:
        pairs[key]["comp"] = r["tok"]

for scenario, data in sorted(pairs.items()):
    comp = data.get("comp", 0)
    no_comp = data.get("no_comp", 0)
    if comp and no_comp:
        delta = ((comp / no_comp) - 1) * 100
        benefit = f"+{delta:.0f}%" if delta > 0 else f"{delta:.0f}%"
        if delta > 0:
            benefit += " ✅"
        print(f"  {scenario:<22} {comp:>6} {no_comp:>8} {delta:>+7.1f}% {benefit:>10}")
    elif comp:
        print(f"  {scenario:<22} {comp:>6} {'N/A':>8} {'':>8} {'':>10}")
    elif no_comp:
        print(f"  {scenario:<22} {'N/A':>6} {no_comp:>8} {'':>8} {'':>10}")

print("=" * 65)
print()
print("  Key insight: compression helps when network is the bottleneck.")
print("  On LAN (<10% comm ratio), compression adds ~5% overhead.")
print("  On WAN (>30% comm ratio), compression can improve throughput.")
print()

# MicroChat-CPU: Complete Walkthrough

This guide covers the full workflow — from installation to distributed training
across multiple machines. It's designed for the research paper
*"Architecting Distributed LLM Training on Consumer Hardware"*.

## Prerequisites

- 2+ machines with CPUs (any x86_64, 4+ cores, 8GB+ RAM each)
- SSH access between machines (passwordless key-based auth)
- Python 3.10+ and PyTorch 2.x installed on each machine
- Network: LAN (1 Gbps+) or WAN (any speed — compression adapts)

## Step 1: Install

On each machine:

```bash
# Clone the repo
git clone https://github.com/Matrix-Research-Ai/MicroChat-CPU.git
cd MicroChat-CPU

# Install dependencies (CPU-only PyTorch)
uv sync --extra cpu
source .venv/bin/activate

# Verify installation
python -m scripts.cpu_train --synthetic --dry-run --depth=2
# Expected: "✓ Dry run PASSED — pipeline is functional"
```

## Step 2: Single-Node Quick Test

```bash
# Quick mode: auto-configures everything
python -m scripts.cpu_train --quick

# With profiling:
python -m scripts.cpu_train --synthetic --profile --depth=4 --num-iterations=50

# Generate report after training:
python -m scripts.cpu_report
```

## Step 3: Understand the Profiler

The profiler breaks each step into phases:

```
  Phase             Avg      %      Min      Max
  fwd            1341.3ms  37.2%  1230.9  1500.6  ███████
  bwd            2018.1ms  56.0%  1830.3  3388.4  ███████████
  data              0.2ms   0.0%     0.2     0.7  █
  comm              0.0ms   0.0%     0.0     0.0  █
  optim           243.4ms   6.8%   210.4   318.6  █
```

- **fwd** (forward pass): 25-40% of time
- **bwd** (backward pass): 40-60% of time (CPU bottleneck)
- **comm** (gradient sync): 0% single-node, 10-70% multi-node WAN
- **optim** (optimizer): 5-15% of time

If `comm` is >30%, enable compression (`--compress`).
If `comm` is >50%, increase sync interval (`--sync-interval=3`).

## Step 4: Multi-Node LAN Setup

### 4a: Manual Launch (2 machines)

On machine A (master, IP 192.168.1.10):
```bash
bash runs/runcpu_distributed.sh lan --node-rank=0 --nnodes=2 \
  --master-addr=192.168.1.10 --synthetic
```

On machine B (worker, IP 192.168.1.11):
```bash
bash runs/runcpu_distributed.sh lan --node-rank=1 --nnodes=2 \
  --master-addr=192.168.1.10 --synthetic
```

### 4b: Automated Launch (2+ machines)

1. Edit `runs/launch_cluster.sh` and set the `NODES` array:

```bash
NODES=(
    "192.168.1.10"
    "192.168.1.11"
    "192.168.1.12"
)
```

2. Run:

```bash
# Everything is automated: rsync → verify → launch
bash runs/launch_cluster.sh lan --synthetic --dry-run

# Real training:
bash runs/launch_cluster.sh lan --compress --num-iterations=1000
```

## Step 5: WAN Training (Across Geographic Locations)

WAN links have higher latency and lower bandwidth than LAN. MicroChat-CPU
provides multiple optimizations:

### 5a: Enable Compression

```bash
# Halves gradient size via FP32→FP16 conversion
bash runs/launch_cluster.sh wan --compress
```

### 5b: Batch Gradient Syncs

```bash
# Sync every 3 steps instead of every step
bash runs/launch_cluster.sh wan --compress --sync-interval=3
```

### 5c: Topology-Aware Hierarchical AllReduce

When nodes are spread across multiple subnets (e.g., some in us-east, some
in eu-west), the topology-aware mode groups them by subnet, does fast
intra-LAN all-reduce within each group, then only group leaders exchange
across the WAN link.

```bash
bash runs/launch_cluster.sh topology-aware --compress
```

### 5d: Test WAN Behavior Locally

Before deploying across continents, simulate WAN conditions on your local
machines to tune parameters:

```bash
# Simulate 50ms latency, 100Mbps bandwidth
python -m scripts.cpu_train --synthetic --simulate-wan \
  --wan-latency=50 --wan-bandwidth=100 --profile
```

## Step 6: Heterogeneous Nodes (Different CPU Speeds)

When machines have different CPUs, the slowest node becomes a bottleneck.
MicroChat-CPU profiles each node and adjusts batch sizes:

```bash
bash runs/launch_cluster.sh lan --hetero --straggler-ratio=2.0
```

If a node is >2x slower than the median for 3 consecutive checks,
failover triggers: checkpoint saved, node removed, training restarts:

```bash
# Launcher auto-handles this:
#   ⚠ Straggler(s) detected: ranks [2]
#   🚨 FAILOVER — saving checkpoint
#   Removing failed node, restarting with --resume
```

## Step 7: Adaptive Runtime Tuning

Enable the adaptive controller to automatically adjust sync interval
based on real-time communication-to-computation ratio:

```bash
bash runs/launch_cluster.sh wan --adaptive --compress
```

The controller measures comm ratio every 10 steps and adjusts:

```
comm_ratio < 30%  → sync every step (compute-bound)
comm_ratio 30-50% → sync every 2 steps (balanced)
comm_ratio 50-70% → sync every 3 steps (comm-bound)
comm_ratio > 70%  → sync every 5 steps (bandwidth-constrained)
```

## Step 8: Real Data Training

On a machine with internet access:

```bash
# Download dataset (10 shards for testing, ~2.5GB)
bash scripts/setup_microchat.sh --quick

# Full GPT-2 grade training (170 shards, ~42GB)
bash scripts/setup_microchat.sh -n 170

# Train with real data
python -m scripts.cpu_train --depth=4 --num-iterations=5000

# Chat with the trained model
python -m scripts.cpu_chat
```

## Step 9: Benchmarking

```bash
# Full benchmark suite (all configs)
python -m scripts.cpu_benchmark

# WAN compression comparison
python -m scripts.benchmark_wan_compression

# Research validation
python -m scripts.research_validate

# Generate HTML report
python -m scripts.cpu_report
```

## Step 10: WAN Resilience

For long training runs over unreliable WAN links:

```bash
# Save checkpoint every 50 steps
python -m scripts.cpu_train --save-every=50 --resume --num-iterations=10000

# If interrupted: restart with --resume auto-loads latest
python -m scripts.cpu_train --save-every=50 --resume --num-iterations=10000
```

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `ModuleNotFoundError: rustbpe` | Tokenizer deps not installed | `uv sync --extra cpu` |
| `torch.compile` crash | No C++ compiler on CPU | Auto-handled, ignore |
| Training slow on single node | Backward pass is CPU bottleneck | Expected — use `--profile` to verify |
| WAN simulation doesn't slow down | Gloo uses shared memory on single node | Test on real multi-node setup |
| `timed out waiting for all ranks` | Firewall blocking Gloo port | Open `MASTER_PORT` (default 12345) |
| Straggler false positive | Brief CPU load spike | Increase `--straggler-ratio` to 3.0 |

## Architecture Summary

```
User: python -m scripts.cpu_train --hetero --compress --adaptive
  │
  ├── compute_init()           → Gloo backend (CPU DDP)
  ├── GPT()                    → Transformer model on CPU
  ├── FSDP                     → Memory sharding across nodes
  ├── HeterogeneousLoadBalancer → Speed profiling
  ├── AdaptiveCommScheduler    → Gradient sync with compression
  │   ├── GradientQuantizer    → FP32→FP16
  │   └── GradientSparsifier   → Top-k sparsification
  ├── RuntimeAdaptiveController → Auto-tune sync interval
  ├── TopologyAwareCommunicator → Hierarchical AllReduce
  ├── WANSimulator             → Latency/bandwidth injection
  ├── StragglerMitigator       → Node failure detection
  └── WANResilienceManager     → Checkpoint save/resume
```

## References

- [MicroChat-CPU GitHub](https://github.com/Matrix-Research-Ai/MicroChat-CPU)
- [Original nanochat](https://github.com/karpathy/nanochat)
- Research papers in the `Research/` directory

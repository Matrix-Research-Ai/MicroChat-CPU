# MicroChat-CPU

**Distributed CPU-only training of LLMs** — a fork of [karpathy/nanochat](https://github.com/karpathy/nanochat) adapted for multi-node CPU clusters connected via LAN or WAN.

Train transformer language models across any machines with CPUs — no GPU required. Uses PyTorch's Gloo backend for distributed communication, FSDP for memory efficiency, and a suite of network optimizations for heterogeneous environments.

## Quick Start

```bash
# Install
uv sync --extra cpu
source .venv/bin/activate

# Single-node smoke test (2 steps, synthetic data, no download)
python -m scripts.cpu_train --synthetic --dry-run --depth=2

# Full 50-step training on synthetic data
bash runs/runcpu_distributed.sh single --synthetic

# Profile where time goes
python -m scripts.cpu_train --synthetic --profile --depth=4 --num-iterations=10
```

## Multi-Node Training

### LAN (two machines on same network)

On node 0 (master):
```bash
bash runs/runcpu_distributed.sh lan --node-rank=0 --nnodes=2 --master-addr=192.168.1.10
```

On node 1:
```bash
bash runs/runcpu_distributed.sh lan --node-rank=1 --nnodes=2 --master-addr=192.168.1.10
```

### WAN (across geographic locations)

```bash
bash runs/runcpu_distributed.sh wan --node-rank=0 --nnodes=2 --master-addr=<PUBLIC_IP>
```

### Auto-detect LAN groups by subnet

```bash
bash runs/runcpu_distributed.sh topology-aware --node-rank=0 --nnodes=4 --master-addr=10.0.0.1
```

### Heterogeneous nodes (different CPU speeds)

```bash
bash runs/runcpu_distributed.sh lan --hetero --node-rank=0 --nnodes=3 --master-addr=192.168.1.10
```

## What's Different from nanochat

| Feature | nanochat (original) | MicroChat-CPU |
|---------|-------------------|---------------|
| Hardware | 8×H100 GPU (NVIDIA DGX) | Any CPU nodes (multi-machine) |
| Distributed backend | NCCL (CUDA only) | **Gloo** (CPU, any network) |
| Memory optimization | — | **FSDP**: shard model across nodes |
| Gradient compression | — | **FP32→FP16** (2× bandwidth savings) |
| Sparsification | — | **Top-k** (send only top 1% gradients) |
| WAN optimization | — | Batched sync every N steps |
| Network awareness | — | LAN/WAN tier selection |
| Topology-aware AllReduce | — | Hierarchical: intra-LAN fast → cross-WAN minimal |
| Network benchmark | — | Auto-detect bandwidth, recommend settings |
| Async overlap | — | Gradient sync runs while CPU computes |
| Heterogeneous nodes | — | Profile node speeds, auto-adjust batch sizes |

## Architecture

```
scripts/cpu_train.py         → Main training script (FSDP + Gloo + all optimizations)
nanochat/cpu_distributed.py   → 8 classes across 6 optimization phases
nanochat/common.py            → Modified compute_init: Gloo backend for CPU
runs/runcpu_distributed.sh    → Launch wrapper for single/LAN/WAN/topology-aware modes
runs/benchmark_network.sh     → Standalone network bandwidth benchmark
```

### 6 Optimization Phases

**Phase 1 — Gradient Quantization:** FP32→FP16 all-reduce halves network payload. Enabled with `--compress`.

**Phase 2 — Top-k Sparsification:** Transmits only the largest-magnitude gradients (default 1%), accumulates error locally. Enabled with `--sparsify --sparsity-ratio=0.01`.

**Phase 3 — Adaptive Scheduling:** Batches AllReduce every N steps instead of every step. Set `--sync-interval=N` (higher for WAN).

**Phase 4 — Topology-Aware Hierarchical AllReduce:** Nodes auto-discover LAN groups by IP subnet. Intra-group syncs run over fast LAN; only group leaders exchange across WAN. Enable with `--topology-aware`.

**Phase 5 — Network Benchmark:** Measures real inter-node bandwidth using PyTorch Gloo all-reduce before training. Auto-recommends compression, sync interval, and sparsification. Run with `--benchmark`.

**Phase 6 — Heterogeneous Load Balancing:** Profiles each node's compute speed, assigns proportional batch sizes, detects stragglers at runtime. Enable with `--hetero`.

### Async Overlap

Gradient all-reduce runs asynchronously — after `loss.backward()`, Gloo starts transferring gradients immediately while the CPU computes the learning rate schedule. Only `optimizer.step()` blocks on sync completion.

```
Without:  backward → [idle] → all-reduce → [idle] → LR calc → optim.step()
With:     backward → [all-reduce starts] → LR calc (overlapped) → wait → optim.step()
```

### Step Timing Profiler

Break down each step into 6 phases to identify bottlenecks:

```bash
python -m scripts.cpu_train --synthetic --profile --depth=4 --num-iterations=20
```

Example output:
```
  Phase             Avg      %      Min      Max
  fwd             184.7ms  25.2%   171.7   197.6  █████
  bwd             264.9ms  36.1%   236.6   293.1  ███████
  data              0.1ms   0.0%     0.1     0.1  █
  comm              0.0ms   0.0%     0.0     0.0  █
  optim           284.4ms  38.7%   264.2   304.7  ███████
```

## CLI Reference

```
python -m scripts.cpu_train [flags]

  --depth N           Transformer depth (default: 4, smaller for CPU)
  --max-seq-len N     Context length (default: 512, smaller for CPU memory)
  --device-batch-size N  Per-node batch size (default: 1)
  --total-batch-size N   Global batch size in tokens (default: 512)
  --num-iterations N     Training steps (default: 50)

  --compress          FP32→FP16 gradient compression (default: on)
  --no-compress       Disable gradient compression
  --sparsify          Enable top-k gradient sparsification
  --sparsity-ratio F  Fraction of gradients to keep (default: 0.01)
  --sync-interval N   Sync every N steps (default: 1, higher for WAN)
  --network-tier      lan|wan (default: lan)

  --topology-aware    Enable hierarchical AllReduce by subnet
  --benchmark         Run bandwidth benchmark before training
  --async-overlap     Async gradient sync (default: on)
  --no-async-overlap  Disable async overlap
  --hetero            Enable heterogeneous node profiling
  --straggler-ratio F Straggler threshold (default: 2.0)

  --synthetic         Use synthetic random data (no download needed)
  --dry-run           Run 2 steps, print diagnostics, exit
  --profile           Enable per-phase step timing
  --profile-every N   Print per-step timing every N steps
```

## Requirements

- Python 3.10+
- PyTorch 2.x (CPU build: `uv sync --extra cpu`)
- CPU with ≥4 cores, ≥8GB RAM per node
- SSH access between nodes for torchrun
- Network: LAN (≥1 Gbps) or WAN (any — compression adapts)

## Research

The optimization techniques in this fork are documented in the accompanying research papers in `Research/`:

- **Accelerating Distributed LLM Training on CPUs: A Hybrid Parallelism Approach**
- **An Adaptive Framework for Distributed CPU Training: Minimizing Latency in Unreliable WAN Environments**
- **Architecting Distributed LLM Training on Consumer Hardware: A Guide to CPU-Only Forks**

## Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) for the original [nanochat](https://github.com/karpathy/nanochat) — the minimal full-stack ChatGPT clone
- The nanochat community for the speedrun leaderboard and ongoing improvements
- PyTorch team for DDP, FSDP, and the Gloo distributed backend

## License

MIT — same as the original nanochat.

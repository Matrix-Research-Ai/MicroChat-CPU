# MicroChat-CPU — Distributed CPU-Only LLM Training

A fork of [karpathy/nanochat](https://github.com/karpathy/nanochat) modified for **distributed CPU-only training across LAN and WAN networks**, based on the research analysis in this directory.

**GitHub:** https://github.com/Matrix-Research-Ai/MicroChat-CPU

## What's Different

| Feature | Original nanochat | This fork |
|---------|------------------|-----------|
| Hardware | 8×H100 GPU (NVIDIA DGX) | Any CPU nodes (multi-machine) |
| Distributed backend | NCCL (CUDA only) | Gloo (CPU, any network) |
| Memory optimization | — | FSDP: shard model across nodes |
| Gradient compression | — | FP32→FP16 (2× bandwidth savings) |
| Sparsification | — | Top-k (send only top 1% gradients) |
| WAN optimization | — | Batched sync every N steps |
| Network awareness | — | LAN/WAN tier selection |
| Topology-aware AllReduce | — | Hierarchical: intra-LAN fast → cross-WAN minimal |
| Network benchmark | — | Auto-detect bandwidth, recommend settings |
| Async overlap | — | Gradient sync runs while CPU computes LR/metrics |
| Heterogeneous nodes | — | Profile node speeds, auto-adjust batch sizes, detect stragglers |

## Quick Start

```bash
uv sync --extra cpu
source .venv/bin/activate
```

### Single-node test (runs on your machine)

```bash
bash runs/runcpu_distributed.sh single
```

### Multi-node LAN (two machines on same network)

On node 0 (master):
```bash
bash runs/runcpu_distributed.sh lan --node-rank=0 --nnodes=2 --master-addr=192.168.1.10
```

On node 1 (worker):
```bash
bash runs/runcpu_distributed.sh lan --node-rank=1 --nnodes=2 --master-addr=192.168.1.10
```

### Multi-node WAN (across geographic locations)

On master (public IP accessible by workers):
```bash
bash runs/runcpu_distributed.sh wan --node-rank=0 --nnodes=2 --master-addr=<PUBLIC_IP>
```

### Topology-aware mode (auto-detect LAN groups)

```bash
# Run on each node — auto-groups by subnet
bash runs/runcpu_distributed.sh topology-aware --node-rank=0 --nnodes=4 --master-addr=10.0.0.1
```

### Benchmark network first, then auto-configure

```bash
bash runs/runcpu_distributed.sh benchmark --node-rank=0 --nnodes=2 --master-addr=192.168.1.10
```

### Heterogeneous mode (different CPU speeds across nodes)

```bash
# Auto-profiles each node's speed, assigns proportional batch sizes
bash runs/runcpu_distributed.sh lan --hetero --node-rank=0 --nnodes=3 --master-addr=192.168.1.10

# Adjust straggler sensitivity (default: 2.0 = 2x slower is a straggler)
bash runs/runcpu_distributed.sh wan --hetero --straggler-ratio=3.0 --node-rank=0 --nnodes=3 --master-addr=10.0.0.1
```

## How It Works

### 5-Phase Communication Optimization (from the research)

**Phase 1** — Gradient Quantization (FP32→FP16): Halves network payload before each AllReduce. Enabled with `--compress`.

**Phase 2** — Top-k Sparsification: Only transmits the largest-magnitude gradients (default 1%), accumulates error locally. Enabled with `--sparsify --sparsity-ratio=0.01`.

**Phase 3** — Adaptive Scheduling: Batches AllReduce every N steps instead of every step. Set `--sync-interval=N` (higher for WAN).

**Phase 4** — Topology-Aware Hierarchical AllReduce: Nodes auto-discover LAN groups by IP subnet. Intra-group syncs run over fast LAN; only group leaders exchange across WAN. For G groups of S nodes, each node sends O(S-1) locally + O(G-1) over WAN instead of O(G*S-1). Enable with `--topology-aware`.

**Phase 5** — Network Benchmark: Measures real inter-node bandwidth using PyTorch Gloo all-reduce before training starts. Auto-recommends compression, sync interval, and sparsification settings. Run standalone with `runs/benchmark_network.sh` or in training with `--benchmark`.

### Async Overlap (default on)

Gradient all-reduce runs **asynchronously** — after `loss.backward()`, the Gloo backend starts transferring gradients between nodes immediately while the CPU computes the learning rate schedule and preps metrics. Only `optimizer.step()` blocks on sync completion.

```
Without overlap:  backward → [idle] → all-reduce → [idle] → LR calc → optimizer.step()
With overlap:     backward → [all-reduce starts] → LR calc (overlapped) → wait → optimizer.step()
```

### LAN vs WAN Configuration

| Setting | LAN | WAN |
|---------|-----|-----|
| `--sync-interval` | 1 (every step) | 3–5 (batched) |
| `--compress` | Enabled | Enabled |
| `--sparsify` | Optional | Recommended |
| `--network-tier` | lan | wan |

## Architecture

```
scripts/cpu_train.py         → Main training script (FSDP + Gloo)
nanochat/cpu_distributed.py   → Compression, sparsification, scheduler
nanochat/common.py            → Modified compute_init: Gloo backend for CPU
runs/runcpu_distributed.sh    → Launch wrapper for single/LAN/WAN modes
```

## Requirements

- CPU with ≥4 cores, ≥8GB RAM (per node)
- PyTorch 2.x (CPU build: `uv sync --extra cpu`)
- SSH access between nodes for torchrun
- Network: LAN (≥1 Gbps) or WAN (any) — compression adapts automatically

## Related Research

See these papers in this directory for the full analysis:

- `Architecting Distributed LLM Training on Consumer Hardware.pdf`
- `Accelerating Distributed LLM Training on CPUs - A Hybrid Parallelism Approach.pdf`
- `An Adaptive Framework for Distributed CPU Training.pdf`

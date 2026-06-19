#!/bin/bash
# Network bandwidth benchmark for distributed CPU training.
#
# Measures inter-node bandwidth using PyTorch Gloo backend.
#
# Usage:
#   # Single node (baseline):
#   bash runs/benchmark_network.sh
#
#   # Multi-node:
#   # On node 0:
#   bash runs/benchmark_network.sh --node-rank=0 --nnodes=2 --master-addr=192.168.1.10
#   # On node 1:
#   bash runs/benchmark_network.sh --node-rank=1 --nnodes=2 --master-addr=192.168.1.10

set -e

NODE_RANK=0
NNODES=1
MASTER_ADDR="127.0.0.1"
MASTER_PORT=12345
PAYLOAD_MB=8  # Size of test tensor in MB

while [[ $# -gt 0 ]]; do
    case "$1" in
        --node-rank=*) NODE_RANK="${1#*=}"; shift ;;
        --node-rank) NODE_RANK="$2"; shift 2 ;;
        --master-addr=*) MASTER_ADDR="${1#*=}"; shift ;;
        --master-addr) MASTER_ADDR="$2"; shift 2 ;;
        --master-port=*) MASTER_PORT="${1#*=}"; shift ;;
        --master-port) MASTER_PORT="$2"; shift 2 ;;
        --nnodes=*) NNODES="${1#*=}"; shift ;;
        --nnodes) NNODES="$2"; shift 2 ;;
        --payload-mb=*) PAYLOAD_MB="${1#*=}"; shift ;;
        --payload-mb) PAYLOAD_MB="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

PAYLOAD_BYTES=$((PAYLOAD_MB * 1024 * 1024))

echo "================================================"
echo "Network Bandwidth Benchmark"
echo "================================================"
echo "Nodes:    $NNODES"
echo "Payload:  ${PAYLOAD_MB}MB per tensor"
echo "Master:   $MASTER_ADDR:$MASTER_PORT"
echo "Node:     $NODE_RANK"
echo "================================================"

if [ "$NNODES" -eq 1 ]; then
    python3 -c "
import torch
import torch.distributed as dist
import os

os.environ['MASTER_ADDR'] = '$MASTER_ADDR'
os.environ['MASTER_PORT'] = '$MASTER_PORT'
os.environ['RANK'] = '$NODE_RANK'
os.environ['WORLD_SIZE'] = '$NNODES'
os.environ['LOCAL_RANK'] = '$NODE_RANK'

dist.init_process_group(backend='gloo')
rank = dist.get_rank()
ws = dist.get_world_size()

import time
num_elements = $PAYLOAD_BYTES // 4
tensor = torch.ones(num_elements, dtype=torch.float32)

# Warmup
for _ in range(3):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

# Measure
dist.barrier()
t0 = time.perf_counter()
for _ in range(10):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
dist.barrier()
t1 = time.perf_counter()

total_bytes = $PAYLOAD_BYTES * 10 * 2
elapsed = t1 - t0
bw = total_bytes / elapsed / (1024*1024)

# Latency
t2 = time.perf_counter()
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
dist.barrier()
t3 = time.perf_counter()
lat = (t3 - t2) * 1000

if rank == 0:
    print(f'Bandwidth: {bw:.1f} MB/s')
    print(f'Latency:   {lat:.2f} ms')
    if bw < 50:
        print('Tier: WAN (slow) — use --sparsify --sync-interval=5')
    elif bw < 500:
        print('Tier: WAN (moderate) — use --compress --sync-interval=3')
    elif bw < 2000:
        print('Tier: LAN (good) — standard settings OK')
    else:
        print('Tier: LAN (excellent) — standard settings OK')

dist.destroy_process_group()
" 2>&1
else
    torchrun \
        --nproc_per_node=1 \
        --nnodes=$NNODES \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        -m scripts.cpu_train \
        --depth=4 --max-seq-len=64 --device-batch-size=1 \
        --eval-every=-1 --core-metric-every=-1 \
        --num-iterations=1 --total-batch-size=64 \
        --benchmark --device-type=cpu
fi

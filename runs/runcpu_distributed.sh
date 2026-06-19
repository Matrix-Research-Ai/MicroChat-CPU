#!/bin/bash
# Distributed CPU training launcher for nanochat fork.
#
# This script launches multi-node CPU training across LAN/WAN using torchrun.
#
# Usage:
#
#   1) Single-node (quick test):
#      bash runs/runcpu_distributed.sh single
#
#   2) Multi-node LAN (run on each node):
#      # On node 0 (master):
#      bash runs/runcpu_distributed.sh lan --node-rank=0 --master-addr=192.168.1.10 --master-port=12345
#      # On node 1 (worker):
#      bash runs/runcpu_distributed.sh lan --node-rank=1 --master-addr=192.168.1.10 --master-port=12345
#
#   3) Multi-node WAN (more compression, less frequent syncs):
#      # On node 0 (master):
#      bash runs/runcpu_distributed.sh wan --node-rank=0 --master-addr=<PUBLIC_IP> --master-port=12345
#      # On node 1 (worker):
#      bash runs/runcpu_distributed.sh wan --node-rank=1 --master-addr=<PUBLIC_IP> --master-port=12345
#
#   4) Aggressive sparsification for very slow WAN links:
#      bash runs/runcpu_distributed.sh sparse-wan --node-rank=0 --master-addr=<PUBLIC_IP> --master-port=12345

set -e

MODE="${1:-single}"  # single, lan, wan, sparse-wan, topology-aware, benchmark
shift || true

# ---------------------------------------------------------------------------
# Parse optional flags
NODE_RANK=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT=12345
NNODES=1

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
        --no-async-overlap) EXTRA_ARGS="$EXTRA_ARGS --no-async-overlap"; shift ;;
        --hetero) EXTRA_ARGS="$EXTRA_ARGS --hetero"; shift ;;
        --profile) EXTRA_ARGS="$EXTRA_ARGS --profile"; shift ;;
        --profile-every=*) EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
        --synthetic) EXTRA_ARGS="$EXTRA_ARGS --synthetic"; shift ;;
        --dry-run) EXTRA_ARGS="$EXTRA_ARGS --dry-run"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Common training args — small model for CPU feasibility
COMMON_ARGS="--depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --device-type=cpu --eval-every=10 --num-iterations=50 --total-batch-size=512"

# Optional flags
EXTRA_ARGS=""

# ---------------------------------------------------------------------------
echo "================================================"
echo "nanochat fork — Distributed CPU Training"
echo "================================================"
echo "Mode:         $MODE"
echo "Node rank:    $NODE_RANK"
echo "Master addr:  $MASTER_ADDR:$MASTER_PORT"
echo "Num nodes:    $NNODES"
echo "================================================"

case "$MODE" in
    single)
        echo "Starting single-node CPU training..."
        python -m scripts.cpu_train $COMMON_ARGS --compress --network-tier=lan $EXTRA_ARGS
        ;;

    lan)
        echo "Starting multi-node LAN training..."
        torchrun \
            --nproc_per_node=1 \
            --nnodes=$NNODES \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            -m scripts.cpu_train $COMMON_ARGS \
            --compress \
            --sync-interval=1 \
            --network-tier=lan $EXTRA_ARGS
        ;;

    wan)
        echo "Starting multi-node WAN training (compression enabled, batched sync)..."
        torchrun \
            --nproc_per_node=1 \
            --nnodes=$NNODES \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            -m scripts.cpu_train $COMMON_ARGS \
            --compress \
            --sync-interval=3 \
            --network-tier=wan $EXTRA_ARGS
        ;;

    sparse-wan)
        echo "Starting multi-node WAN training (sparsification + batched sync)..."
        torchrun \
            --nproc_per_node=1 \
            --nnodes=$NNODES \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            -m scripts.cpu_train $COMMON_ARGS \
            --sparsify \
            --sparsity-ratio=0.01 \
            --sync-interval=5 \
            --network-tier=wan $EXTRA_ARGS
        ;;

    topology-aware)
        echo "Starting hierarchical AllReduce (intra-LAN fast, cross-WAN minimal)..."
        torchrun \
            --nproc_per_node=1 \
            --nnodes=$NNODES \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            -m scripts.cpu_train $COMMON_ARGS \
            --compress \
            --topology-aware $EXTRA_ARGS
        ;;

    benchmark)
        echo "Running network bandwidth benchmark..."
        bash runs/benchmark_network.sh \
            --node-rank=$NODE_RANK \
            --nnodes=$NNODES \
            --master-addr=$MASTER_ADDR \
            --master-port=$MASTER_PORT
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Usage: bash $0 {single|lan|wan|sparse-wan|topology-aware|benchmark} [options]"
        exit 1
        ;;
esac

echo "Done."

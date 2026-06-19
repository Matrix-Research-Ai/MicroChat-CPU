#!/bin/bash
# ===========================================================================
# MicroChat-CPU Cluster Launcher
# ===========================================================================
# Automates distributed training across multiple machines via SSH.
#
# Usage:
#   1) Edit the NODES array below with your machine IPs/hostnames
#   2) Ensure passwordless SSH works between all nodes
#   3) Run:
#        bash runs/launch_cluster.sh [mode] [extra flags]
#
# Modes:
#   lan             Multi-node LAN (sync every step)
#   wan             Multi-node WAN (compression + batched sync)
#   topology-aware  Hierarchical AllReduce (auto-detect subnets)
#   benchmark       Run network bandwidth benchmark only
#
# Examples:
#   bash runs/launch_cluster.sh lan --synthetic --dry-run
#   bash runs/launch_cluster.sh wan --hetero --num-iterations=500
#   bash runs/launch_cluster.sh benchmark
# ===========================================================================

set -euo pipefail

# ===========================================================================
# CONFIGURATION — EDIT THESE
# ===========================================================================

# List of node hostnames or IPs. The first node is the master.
NODES=(
    "192.168.1.10"
    "192.168.1.11"
    "192.168.1.12"
)

# SSH user (must have key-based auth on all nodes)
SSH_USER="${SSH_USER:-root}"

# Path to MicroChat-CPU code on remote nodes (auto-rsync'd if different)
REMOTE_DIR="${REMOTE_DIR:-/root/MicroChat-CPU}"

# Local repo path (auto-detected from git root)
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# SSH options
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

# Master port for torchrun
MASTER_PORT="${MASTER_PORT:-12345}"

# ===========================================================================
# Parse mode and extra args
# ===========================================================================
MODE="${1:-lan}"
shift || true
EXTRA_ARGS="$@"

# ===========================================================================
# Help
# ===========================================================================
if [ "$MODE" = "help" ] || [ "$MODE" = "--help" ] || [ "$MODE" = "-h" ]; then
    head -40 "$0"
    exit 0
fi

# ===========================================================================
# Validate
# ===========================================================================
NNODES=${#NODES[@]}
if [ $NNODES -lt 1 ]; then
    echo "ERROR: No nodes configured. Edit NODES array in $0"
    exit 1
fi

MASTER_ADDR="${NODES[0]}"

echo "============================================"
echo " MicroChat-CPU Cluster Launcher"
echo "============================================"
echo " Mode:          $MODE"
echo " Nodes:         $NNODES (${NODES[*]})"
echo " Master:        $MASTER_ADDR:$MASTER_PORT"
echo " Remote dir:    $REMOTE_DIR"
echo " Local dir:     $LOCAL_DIR"
echo " Extra args:    $EXTRA_ARGS"
echo "============================================"

# ===========================================================================
# Step 1: Rsync code to all remote nodes
# ===========================================================================
echo ""
echo ">>> Step 1/4: Syncing code to remote nodes..."

for NODE in "${NODES[@]}"; do
    echo "  Syncing to $NODE..."
    ssh $SSH_OPTS "${SSH_USER}@${NODE}" "mkdir -p $REMOTE_DIR" 2>/dev/null || true
    rsync -avz --delete --exclude='.git' --exclude='__pycache__' --exclude='.venv' \
        --exclude='*.pyc' --exclude='.cache' \
        -e "ssh $SSH_OPTS" \
        "$LOCAL_DIR/" "${SSH_USER}@${NODE}:$REMOTE_DIR/" 2>&1 | tail -5
    echo "  ✓ $NODE synced"
done

# ===========================================================================
# Step 2: Verify connectivity and Python/PyTorch on all nodes
# ===========================================================================
echo ""
echo ">>> Step 2/4: Verifying environment on remote nodes..."

for NODE in "${NODES[@]}"; do
    PYTHON_OK=$(ssh $SSH_OPTS "${SSH_USER}@${NODE}" \
        "cd $REMOTE_DIR && python3 -c 'import torch; print(f\"OK torch={torch.__version__} gloo={torch.distributed.is_gloo_available()}\")'" 2>/dev/null || echo "FAIL")
    echo "  $NODE: $PYTHON_OK"
    if [[ "$PYTHON_OK" == FAIL ]]; then
        echo "  ⚠ PyTorch not found on $NODE — install with: uv sync --extra cpu"
    fi
done

# ===========================================================================
# Step 3: Check passwordless SSH between nodes
# ===========================================================================
echo ""
echo ">>> Step 3/4: Testing SSH between nodes..."

TEST_NODE="${NODES[1]:-${NODES[0]}}"
if [ $NNODES -ge 2 ]; then
    # Test that master can SSH to itself and to the first worker
    SELF_TEST=$(ssh $SSH_OPTS "${SSH_USER}@${MASTER_ADDR}" \
        "hostname" 2>/dev/null || echo "FAIL")
    echo "  Master ($MASTER_ADDR): hostname=$SELF_TEST"

    WORKER_TEST=$(ssh $SSH_OPTS "${SSH_USER}@${TEST_NODE}" \
        "hostname" 2>/dev/null || echo "FAIL")
    echo "  Worker ($TEST_NODE): hostname=$WORKER_TEST"
fi

# ===========================================================================
# Step 4: Launch training on remote nodes
# ===========================================================================
echo ""
echo ">>> Step 4/4: Launching distributed training..."
echo ""

# Build mode-specific flags
case "$MODE" in
    lan)
        MODE_FLAGS="--compress --sync-interval=1 --network-tier=lan"
        ;;
    wan)
        MODE_FLAGS="--compress --sync-interval=3 --network-tier=wan"
        ;;
    topology-aware)
        MODE_FLAGS="--compress --topology-aware"
        ;;
    benchmark)
        echo "Running network benchmark across cluster..."
        for NODE in "${NODES[@]}"; do
            RANK=$(echo "${NODES[@]}" | tr ' ' '\n' | grep -n "$NODE" | cut -d: -f1)
            RANK=$((RANK - 1))
            CMD="cd $REMOTE_DIR && bash runs/benchmark_network.sh \
                --node-rank=$RANK --nnodes=$NNODES \
                --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT"
            echo "  Launching rank $RANK on $NODE..."
            ssh $SSH_OPTS "${SSH_USER}@${NODE}" "$CMD" &
        done
        wait
        echo ""
        echo "✓ Benchmark complete"
        exit 0
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: bash $0 {lan|wan|topology-aware|benchmark}"
        exit 1
        ;;
esac

# Launch torchrun on each node
PID_LIST=""
for NODE in "${NODES[@]}"; do
    RANK=$(echo "${NODES[@]}" | tr ' ' '\n' | grep -n "$NODE" | cut -d: -f1)
    RANK=$((RANK - 1))

    CMD="cd $REMOTE_DIR && torchrun \
        --nproc_per_node=1 \
        --nnodes=$NNODES \
        --node_rank=$RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        -m scripts.cpu_train \
        --depth=4 --max-seq-len=512 --device-batch-size=1 \
        --eval-every=10 --num-iterations=50 --total-batch-size=512 \
        $MODE_FLAGS $EXTRA_ARGS \
        2>&1"

    echo "  Launching rank $RANK on $NODE..."
    echo "  $CMD" | head -1

    # SSH in background and prefix output with node name
    ssh $SSH_OPTS "${SSH_USER}@${NODE}" "$CMD" | \
        while IFS= read -r line; do echo "[$NODE] $line"; done &
    PID=$!
    PID_LIST="$PID_LIST $PID"
done

echo ""
echo "============================================"
echo " Launched $NNODES nodes. PIDs: $PID_LIST"
echo " Training running in background."
echo " Use 'kill $PID_LIST' to stop all nodes."
echo "============================================"

# Wait for all background processes
FAILED=0
STOPPED_NODES=""
for PID in $PID_LIST; do
    wait $PID || {
        EXIT_CODE=$?
        FAILED=$((FAILED + 1))
        if [ $EXIT_CODE -eq 42 ]; then
            echo "  ⚠ Node exited with code 42 (straggler failover)"
            STOPPED_NODES="$STOPPED_NODES $PID"
        fi
    }
done

echo ""
if [ $FAILED -eq 0 ]; then
    echo "✓ All nodes completed successfully"
elif [ -n "$STOPPED_NODES" ]; then
    echo "⚠ $FAILED node(s) failed due to straggler detection"

    # Remove failed nodes and restart
    if [ $NNODES -gt 1 ]; then
        # Remove the last failed node from NODES list
        LAST_FAILED_IP=""
        for NODE in "${NODES[@]}"; do
            ssh $SSH_OPTS "${SSH_USER}@${NODE}" "exit 42" 2>/dev/null && LAST_FAILED_IP="$NODE"
        done

        if [ -n "$LAST_FAILED_IP" ]; then
            echo "  Removing failed node: $LAST_FAILED_IP"
            NEW_NODES=()
            for NODE in "${NODES[@]}"; do
                if [ "$NODE" != "$LAST_FAILED_IP" ]; then
                    NEW_NODES+=("$NODE")
                fi
            done
            NODES=("${NEW_NODES[@]}")
            NNODES=${#NODES[@]}
            MASTER_ADDR="${NODES[0]}"

            echo "  Restarting with $NNODES nodes: ${NODES[*]}"
            echo "  Using --resume to continue from last checkpoint"
            echo ""

            # Re-launch with resume
            bash "$0" "$MODE" --resume $EXTRA_ARGS
            exit $?
        fi
    fi
else
    echo "⚠ $FAILED node(s) failed with unknown errors"
fi

# Cleanup: kill any remaining remote processes
for NODE in "${NODES[@]}"; do
    ssh $SSH_OPTS "${SSH_USER}@${NODE}" \
        "pkill -f 'torchrun.*scripts.cpu_train' 2>/dev/null || true" &
done
wait
echo "Done."

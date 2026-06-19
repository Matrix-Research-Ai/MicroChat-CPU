#!/bin/bash
# ===========================================================================
# MicroChat-CPU Setup — prepares environment for real training
# ===========================================================================
# Run this on a machine with internet access (e.g., RunPod GPU instance)
# to download the dataset and train the tokenizer.
#
# Usage:
#   bash scripts/setup_microchat.sh [options]
#
# Options:
#   --data-only       Only download dataset, skip tokenizer training
#   --tokenizer-only  Only train tokenizer, skip dataset download
#   -n N              Download N dataset shards (default: 10 for test, 170 for full GPT-2)
#   --quick           Download minimal data for testing (3 shards)
#   --full            Download full dataset (all 6542 shards, ~400GB)
#   -y                Skip confirmation prompts
# ===========================================================================

set -euo pipefail

# Config
NUM_SHARDS=10
DOWNLOAD_DATA=true
TRAIN_TOKENIZER=true
SKIP_CONFIRM=false

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-only) TRAIN_TOKENIZER=false; shift ;;
        --tokenizer-only) DOWNLOAD_DATA=false; shift ;;
        --quick) NUM_SHARDS=3; shift ;;
        --full) NUM_SHARDS=-1; shift ;;
        -n) NUM_SHARDS="$2"; shift 2 ;;
        -y) SKIP_CONFIRM=true; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo "  MicroChat-CPU Setup"
echo "============================================"
echo "  Dataset shards: $NUM_SHARDS"
echo "  Download:       $DOWNLOAD_DATA"
echo "  Train tok:      $TRAIN_TOKENIZER"
echo "============================================"
echo ""

# Check Python/PyTorch
echo ">>> Checking environment..."
python3 -c "
import torch
print(f'  PyTorch {torch.__version__}')
print(f'  Gloo available: {torch.distributed.is_gloo_available()}')
print(f'  CPU cores: {torch.get_num_threads()}')
print(f'  RAM: None (psutil not checked)')
"

# Check disk space
ROOT_DIR="$(dirname "$0")/.."
REQUIRED_GB=$(( NUM_SHARDS > 0 ? NUM_SHARDS * 250 / 1000 : 400 ))
echo "  Estimated dataset size: ${REQUIRED_GB}GB"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Download dataset
# ---------------------------------------------------------------------------
if [ "$DOWNLOAD_DATA" = true ]; then
    echo "============================================"
    echo "  Step 1: Download ClimbMix dataset"
    echo "============================================"
    echo "  Target: ~/.cache/nanochat/base_data_climbmix/"
    echo "  Shards: $NUM_SHARDS"
    echo ""

    if [ "$SKIP_CONFIRM" = false ]; then
        echo -n "  Continue? [Y/n]: "
        read -r REPLY
        if [[ "$REPLY" =~ ^[Nn] ]]; then
            echo "  Skipping dataset download."
        else
            python3 -m nanochat.dataset -n "$NUM_SHARDS" -w 4
        fi
    else
        python3 -m nanochat.dataset -n "$NUM_SHARDS" -w 4
    fi
    echo ""
fi

# ---------------------------------------------------------------------------
# Step 2: Train tokenizer
# ---------------------------------------------------------------------------
if [ "$TRAIN_TOKENIZER" = true ]; then
    echo "============================================"
    echo "  Step 2: Train BPE tokenizer"
    echo "============================================"
    echo "  This trains a BPE tokenizer on the downloaded data."
    echo ""

    if [ "$SKIP_CONFIRM" = false ]; then
        echo -n "  Continue? [Y/n]: "
        read -r REPLY
        if [[ "$REPLY" =~ ^[Nn] ]]; then
            echo "  Skipping tokenizer training."
        else
            python3 -m scripts.tok_train
        fi
    else
        python3 -m scripts.tok_train
    fi
    echo ""
fi

# ---------------------------------------------------------------------------
# Step 3: Verify
# ---------------------------------------------------------------------------
echo "============================================"
echo "  Step 3: Verification"
echo "============================================"
echo ""

python3 -c "
from nanochat.common import get_base_dir
import os

base = get_base_dir()
data_dir = os.path.join(base, 'base_data_climbmix')
tok_dir = os.path.join(base, 'tokenizer')

print(f'  Base dir:       {base}')
print(f'  Data dir:       {data_dir}')
print(f'  Data exists:    {os.path.exists(data_dir)}')
print(f'  Tokenizer dir:  {tok_dir}')
print(f'  Tokenizer ex:   {os.path.exists(tok_dir)}')

if os.path.exists(data_dir):
    files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    total_gb = sum(os.path.getsize(os.path.join(data_dir, f)) for f in files) / (1024**3)
    print(f'  Parquet files:  {len(files)}')
    print(f'  Total size:     {total_gb:.1f} GB')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "  You can now train with real data:"
echo ""
echo "    Single-node:"
echo "      python -m scripts.cpu_train --depth=4 --num-iterations=1000"
echo ""
echo "    Distributed (LAN):"
echo "      bash runs/runcpu_distributed.sh lan --node-rank=0 --nnodes=2"
echo ""
echo "    Distributed (WAN):"
echo "      bash runs/runcpu_distributed.sh wan --node-rank=0 --nnodes=2"
echo ""
echo "    With all optimizations:"
echo "      bash runs/launch_cluster.sh wan --compress --async-overlap --adaptive"
echo ""
echo "============================================"

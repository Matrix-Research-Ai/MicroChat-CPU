"""
CPU Distributed Training for nanochat fork.

Distributed CPU training across LAN/WAN using:
  - PyTorch DDP + Gloo backend for inter-node communication
  - FSDP (Fully Sharded Data Parallelism) for memory efficiency
  - Gradient compression (FP32→FP16) to halve network payload
  - Optional top-k sparsification for aggressive bandwidth reduction
  - Adaptive communication scheduler (batch syncs for WAN)

Usage:
  # Single-node CPU training:
  python -m scripts.cpu_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20

  # Multi-node CPU training (LAN/WAN):
  torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=192.168.1.10 --master_port=12345 -m scripts.cpu_train --depth=4 --sync-interval=3 --compress

  # With sparsification (aggressive bandwidth reduction for WAN):
  torchrun --nproc_per_node=1 --nnodes=3 --node_rank=0 --master_addr=10.0.0.1 --master_port=12345 -m scripts.cpu_train --depth=4 --sparsify --sparsity-ratio=0.01 --sync-interval=5
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import time
import math
import argparse
from dataclasses import asdict
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn as nn

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.cpu_distributed import AdaptiveCommScheduler, TopologyAwareCommunicator, NetworkBenchmark, HeterogeneousLoadBalancer, StragglerMitigator, SyntheticDataLoader, StepProfiler, WANResilienceManager

# Tokenizer import — deferred because it requires rustbpe (only needed for non-synthetic)
_tokenizer_imported = False
_tokenizer_module = None
def _lazy_tokenizer():
    global _tokenizer_imported, _tokenizer_module
    if not _tokenizer_imported:
        from nanochat import tokenizer as _tokenizer_module
        _tokenizer_imported = True
    return _tokenizer_module

print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="CPU Distributed Pretraining (nanochat fork)")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="cpu", help="cpu (default for this fork)")
# Model architecture
parser.add_argument("--depth", type=int, default=4, help="depth of the Transformer model (smaller for CPU)")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=512, help="max context length (smaller for CPU memory)")
parser.add_argument("--window-pattern", type=str, default="L", help="sliding window pattern (L=full context for CPU)")
# Training horizon
parser.add_argument("--num-iterations", type=int, default=50, help="number of optimization steps")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=float, default=-1, help="calculate num_iterations from data:param ratio (-1 = disable)")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=1, help="per-device batch size (small for CPU memory)")
parser.add_argument("--total-batch-size", type=int, default=512, help="total batch size in tokens")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding params (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.008, help="learning rate for unembedding params (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix params (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars")
parser.add_argument("--weight-decay", type=float, default=0.28, help="weight decay")
parser.add_argument("--warmup-steps", type=int, default=5, help="number of steps for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.5, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.05, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
# Distributed communication
parser.add_argument("--compress", action="store_true", default=True, help="enable FP32->FP16 gradient compression (default: on)")
parser.add_argument("--no-compress", action="store_false", dest="compress", help="disable gradient compression")
parser.add_argument("--sparsify", action="store_true", default=False, help="enable top-k gradient sparsification")
parser.add_argument("--sparsity-ratio", type=float, default=0.01, help="fraction of gradients to keep when sparsifying")
parser.add_argument("--sync-interval", type=int, default=1, help="sync gradients every N steps (1 = every step, 3 = every 3 steps for WAN)")
parser.add_argument("--network-tier", type=str, default="lan", choices=["lan", "wan"], help="network tier: lan (fast) or wan (slow, more compression)")
parser.add_argument("--topology-aware", action="store_true", default=False, help="enable hierarchical AllReduce: group nodes by subnet, intra-LAN fast then cross-WAN minimal")
parser.add_argument("--benchmark", action="store_true", default=False, help="run network bandwidth benchmark before training, then auto-configure")
parser.add_argument("--async-overlap", action="store_true", default=True, help="enable async gradient sync overlapped with computation (default: on)")
parser.add_argument("--no-async-overlap", action="store_false", dest="async_overlap", help="disable async overlap, use sync all-reduce")
# Heterogeneous hardware
parser.add_argument("--hetero", action="store_true", default=False, help="enable heterogeneous node support: profile node speeds, auto-adjust batch sizes, detect stragglers")
parser.add_argument("--profile-steps", type=int, default=5, help="number of profiling steps for heterogeneity detection")
parser.add_argument("--straggler-ratio", type=float, default=2.0, help="node is a straggler if Nx slower than median (2.0 = 2x)")
# Testing
parser.add_argument("--synthetic", action="store_true", default=False, help="use synthetic random data instead of real dataset (no download needed)")
parser.add_argument("--dry-run", action="store_true", default=False, help="run 2 steps, print diagnostics, then exit")
# Profiling
parser.add_argument("--profile", action="store_true", default=False, help="enable per-phase step timing breakdown")
parser.add_argument("--profile-every", type=int, default=0, help="print per-step timing every N steps (0 = only at end)")
# Resilience
parser.add_argument("--resume", action="store_true", default=False, help="resume from latest checkpoint if available")
# Evaluation
parser.add_argument("--eval-every", type=int, default=10, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=512, help="number of tokens to evaluate val loss on")
parser.add_argument("--sample-every", type=int, default=-1, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
args = parser.parse_args()
user_config = vars(args).copy()
# -----------------------------------------------------------------------------
# Compute init — uses Gloo backend for CPU distributed via our modified common.py

device_type = args.device_type  # "cpu" by default for this fork
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
get_max_memory = lambda: 0  # CPU doesn't have CUDA max_memory tracking
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
print0(f"World size: {ddp_world_size} (distributed across {ddp_world_size} node(s))")

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else None
if not use_dummy_wandb:
    import wandb
    wandb_run = wandb.init(project="nanochat-cpu", name=args.run, config=user_config)
else:
    wandb_run = DummyWandb()

# -----------------------------------------------------------------------------
# Tokenizer (optional in synthetic mode)
vocab_size = 50304  # default GPT-2 vocab size
if args.synthetic:
    tokenizer = None
    token_bytes = 1.0
    print0(f"Vocab size: {vocab_size:,} (synthetic mode)")
else:
    try:
        tok = _lazy_tokenizer()
        tokenizer = tok.get_tokenizer()
        token_bytes = tok.get_token_bytes(device=device)
        vocab_size = tokenizer.get_vocab_size()
        print0(f"Vocab size: {vocab_size:,}")
    except ImportError as e:
        print0(f"Warning: tokenizer import failed ({e}), using default vocab_size={vocab_size}")
        print0("Hint: run 'uv sync --extra cpu' to install dependencies")

# -----------------------------------------------------------------------------
# Initialize the Model
base_dim = args.depth * args.aspect_ratio
model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
num_heads = model_dim // args.head_dim
config = GPTConfig(
    sequence_len=args.max_seq_len, vocab_size=vocab_size,
    n_layer=args.depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
    window_pattern=args.window_pattern,
)

with torch.device("meta"):
    model_meta = GPT(config)

model_config = model_meta.config
model_config_kwargs = asdict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
print0(f"Total parameters: {sum(p.numel() for p in model_meta.parameters()):,}")

# Move model to CPU and init weights
model = model_meta.to_empty(device=device)
model.init_weights()

# -----------------------------------------------------------------------------
# FSDP: Fully Sharded Data Parallel for memory efficiency across nodes
# Wrap the model with FSDP to shard parameters across all workers
if ddp and ddp_world_size > 1:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    from torch.distributed.fsdp import ShardingStrategy

    # Auto-wrap: each transformer block gets its own FSDP unit
    auto_wrap_policy = size_based_auto_wrap_policy(
        min_num_params=model_dim * 4  # wrap blocks with >4*dim params
    )

    orig_model = model  # keep reference to unwrapped model for eval/inference
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=None,  # CPU — no device_id needed
    )
    print0(f"✓ FSDP enabled: model sharded across {ddp_world_size} node(s)")
else:
    orig_model = model
    print0("Single-node mode: FSDP not needed")

# -----------------------------------------------------------------------------
# Resuming from checkpoint
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}"
checkpoint_dir = os.path.join(base_dir, "cpu_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    if isinstance(model, FSDP):
        model.load_state_dict(model_data)
    else:
        model.load_state_dict(model_data, strict=True, assign=True)
    del model_data

# -----------------------------------------------------------------------------
# Calculate training horizon
num_scaling_params = model.num_scaling_params() if hasattr(model, 'num_scaling_params') else None

if args.target_param_data_ratio > 0:
    target_tokens = int(args.target_param_data_ratio * num_scaling_params)
    num_iterations = target_tokens // args.total_batch_size
elif args.target_flops > 0:
    num_flops_per_token = model.estimate_flops() if hasattr(model, 'estimate_flops') else 1
    num_iterations = round(args.target_flops / (num_flops_per_token * args.total_batch_size))
else:
    num_iterations = args.num_iterations

print0(f"Training for {num_iterations} iterations")

# -----------------------------------------------------------------------------
# Learning rate schedule
def get_lr_multiplier(it):
    warmup_iters = args.warmup_steps
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

# -----------------------------------------------------------------------------
# Initialize the Optimizer
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    scalar_lr=args.scalar_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)

if resuming:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

# -----------------------------------------------------------------------------
# Initialize distributed communication

# Option 1: Run bandwidth benchmark first
if args.benchmark and ddp:
    print0("Running network bandwidth benchmark...")
    bench = NetworkBenchmark(payload_bytes=8 * 1024 * 1024)
    bench_result = bench.run()
    # Auto-configure from benchmark
    args.sync_interval = bench_result["recommended_sync_interval"]
    args.sparsify = bench_result["recommended_sparsify"]
    args.network_tier = bench_result["tier"]
    print0(f"Auto-configured: sync_interval={args.sync_interval}, sparsify={args.sparsify}, tier={args.network_tier}")

# Option 2: Topology-aware hierarchical AllReduce
if args.topology_aware and ddp and ddp_world_size > 1:
    comm = TopologyAwareCommunicator(
        subnet_prefix_len=24,
        compression_enabled=args.compress,
        sparsification_enabled=args.sparsify,
        sparsification_ratio=args.sparsity_ratio,
    )

    # Discover topology across all ranks
    topology = comm.discover_topology()
    if master_process:
        print0("Network topology discovered:")
        print0(f"  Nodes: {topology['nodes']}, Groups: {topology['groups']}")
        print0(f"  Local group: {topology['group_id']} ({topology['group_size']} nodes)")
        print0(f"  This node: {'LEADER' if topology['is_leader'] else 'member'}")

    # Create process groups for hierarchical communication
    comm.create_process_groups()
    all_params = [p for p in model.parameters() if p.requires_grad]
    comm.register_params(all_params)
    print0(f"✓ Topology-aware hierarchical AllReduce enabled")
    print0(f"  Topology: {comm.summary()['topology']}")
else:
    # Standard flat AllReduce with compression
    comm = AdaptiveCommScheduler(
        sync_interval=args.sync_interval,
        network_tier=args.network_tier,
        compression_enabled=args.compress,
        sparsification_enabled=args.sparsify,
        sparsification_ratio=args.sparsity_ratio,
        world_size=ddp_world_size,
        warmup_steps=args.warmup_steps,
    )
    all_params = [p for p in model.parameters() if p.requires_grad]
    comm.register_params(all_params)

    print0("Communication config:")
    for key, val in comm.summary().items():
        print0(f"  {key}: {val}")

# -----------------------------------------------------------------------------
# Step profiler
profiler = StepProfiler(
    enabled=args.profile or args.dry_run,
    print_every=args.profile_every,
)
if args.profile and master_process:
    print0("✓ Per-phase step profiling enabled")

# -----------------------------------------------------------------------------
# WAN resilience: periodic checkpoints + auto-resume
resilience = WANResilienceManager(
    checkpoint_dir=checkpoint_dir,
    save_every=args.save_every,
    master_process=master_process,
    rank=ddp_rank,
)
if args.save_every > 0:
    resilience.register_signal_handlers()
    print0(f"✓ WAN resilience enabled: checkpoint every {args.save_every} steps")

# Auto-resume from latest checkpoint
if args.resume and resilience.has_checkpoint():
    resume_step = resilience.get_resume_step()
    if args.resume_from_step == -1:
        args.resume_from_step = resume_step
        if master_process:
            print0(f"✓ Auto-resuming from step {resume_step}")
    else:
        # Try finding the requested step, fall back to latest
        print0(f"  --resume-from-step={args.resume_from_step} set, checking...")

# -----------------------------------------------------------------------------
# Heterogeneous load balancer (profile node speeds, detect stragglers)
hetero_balancer = None
straggler_watch = None
if args.hetero and ddp and ddp_world_size > 1:
    hetero_balancer = HeterogeneousLoadBalancer(
        profile_steps=args.profile_steps,
        straggler_ratio=args.straggler_ratio,
        adapt_batch=True,
        world_size=ddp_world_size,
        rank=ddp_rank,
        base_batch_size=args.device_batch_size,
    )
    straggler_watch = StragglerMitigator(
        window_size=10,
        straggler_ratio=args.straggler_ratio,
        rank=ddp_rank,
        world_size=ddp_world_size,
    )
    if master_process:
        print0(f"✓ Heterogeneous mode enabled ({ddp_world_size} nodes, {args.profile_steps} profile steps)")

    # Override device batch size with adjusted value after profiling
    if hetero_balancer.adjusted_batch_size != args.device_batch_size:
        args.device_batch_size = hetero_balancer.adjusted_batch_size
        print0(f"  Adjusted device batch size: {args.device_batch_size}")

# -----------------------------------------------------------------------------
# DataLoaders (real or synthetic)
if args.synthetic:
    train_loader = SyntheticDataLoader(
        vocab_size=vocab_size, B=args.device_batch_size,
        T=args.max_seq_len, device=device, num_batches=1000,
    )
    build_val_loader = lambda: SyntheticDataLoader(
        vocab_size=vocab_size, B=args.device_batch_size,
        T=args.max_seq_len, device=device, num_batches=10,
    )
    if master_process:
        print0("✓ Using synthetic data (no dataset download needed)")
else:
    train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer, args.device_batch_size, args.max_seq_len,
        split="train", device=device,
    )
    build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, args.device_batch_size, args.max_seq_len,
        split="val", device=device,
    )
x, y, dataloader_state_dict = next(train_loader)

# Dry-run: override to minimal settings
if args.dry_run:
    args.num_iterations = 2
    args.eval_every = -1
    args.sample_every = -1
    args.save_every = -1
    if master_process:
        print0("✓ Dry-run mode: 2 steps only, no eval/save")

# -----------------------------------------------------------------------------
# Training loop
step = 0
val_bpb = None
min_val_bpb = float("inf")
smooth_train_loss = 0
total_training_time = 0

# Figure out the needed gradient accumulation micro-steps to reach the desired total batch size per step
effective_device_batch = hetero_balancer.adjusted_batch_size if hetero_balancer else args.device_batch_size
tokens_per_fwdbwd = effective_device_batch * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert args.total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd

print0(f"Tokens / micro-batch / rank: {tokens_per_fwdbwd:,}")
print0(f"Total batch size {args.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")
print0(f"Starting training...")

while True:
    last_step = step >= num_iterations
    flops_so_far = 0  # not tracked on CPU

    # --- Evaluation ---
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        # For FSDP, gather full model on all ranks for eval
        if isinstance(model, FSDP):
            with FSDP.summon_full_params(model):
                val_loader = build_val_loader()
                eval_steps = max(1, args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size))
                val_bpb = evaluate_bpb(orig_model, val_loader, eval_steps, token_bytes)
        else:
            val_loader = build_val_loader()
            eval_steps = max(1, args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size))
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    if last_step:
        break

    # --- Training step with async overlap ---
    profiler.start()
    for micro_step in range(grad_accum_steps):
        loss = model(x, y)
        train_loss = loss.detach()
        profiler.mark('fwd')
        loss = loss / grad_accum_steps
        loss.backward()
        profiler.mark('bwd')
        x, y, dataloader_state_dict = next(train_loader)  # prefetch next batch
        profiler.mark('data')

    # Fire off async gradient sync (all-reduce runs in the background
    # over Gloo while we do compute-independent work below)
    if args.async_overlap and hasattr(comm, 'start_async_sync'):
        async_handle = comm.start_async_sync()
    else:
        # Sync mode: fall through to traditional sync
        async_handle = None

    # ── Overlap window: work that doesn't need synced gradients ──
    # Compute LR schedule, momentum, weight decay for this step
    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm

    # ── End overlap window: wait for gradients / sync if not async ──
    if async_handle is not None:
        comm.wait_for_sync(async_handle)
    elif isinstance(comm, TopologyAwareCommunicator):
        comm.sync_gradients(do_average=True)
    else:
        comm.on_optimizer_step(model)
    profiler.mark('comm')

    optimizer.step()
    profiler.mark('optim')
    model.zero_grad(set_to_none=True)

    profiler.end()
    dt = profiler.last_step_time

    # --- Heterogeneous profiling & straggler detection ---
    if hetero_balancer is not None:
        hetero_balancer.record_step_time(dt)
    if straggler_watch is not None and step > 0 and step % 10 == 0:
        straggler_watch.record(dt)
        sg = straggler_watch.check_stragglers()
        if sg and master_process:
            print0(f"⚠ Straggler ranks: {sg} — consider --straggler-ratio adjustment")

    # --- Logging ---
    train_loss_f = train_loss.item()
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(args.total_batch_size / dt) if dt > 0 else 0

    if step > 5:
        total_training_time += dt

    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | "
           f"lrm: {lrm:.2f} | dt: {dt*1000:.2f}ms | tok/sec: {tok_per_sec:,} | "
           f"total: {total_training_time/60:.2f}m")

    if step % 10 == 0:
        wandb_run.log({
            "step": step,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
        })

    step += 1

    # --- Periodic checkpoint save (WAN resilience) ---
    resilience.maybe_save(
        step=step,
        model=orig_model,
        optimizer=optimizer,
        metadata={
            "model_config": model_config_kwargs,
            "user_config": user_config,
            "device_batch_size": effective_device_batch,
            "max_seq_len": args.max_seq_len,
            "total_batch_size": args.total_batch_size,
            "val_bpb": val_bpb,
            "min_val_bpb": min_val_bpb,
            "smooth_train_loss": smooth_train_loss,
            "total_training_time": total_training_time,
        },
        dataloader_state_dict=dataloader_state_dict,
    )

    # GC: freeze after first step to reduce overhead on CPU
    if step == 1:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif step % 5000 == 0:
        gc.collect()

# --- Cleanup ---
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

# Profiling report
if master_process and (args.profile or args.dry_run):
    avg_tok = total_training_time / max(step, 1) if total_training_time > 0 else 0
    profiler.print_report(toks_per_sec=args.total_batch_size / max(dt, 1e-6) if dt > 0 else None)

# Dry-run diagnostics
if args.dry_run and master_process:
    print0("=" * 50)
    print0("DRY RUN DIAGNOSTICS")
    print0("=" * 50)
    print0(f"  Steps completed: {step}")
    print0(f"  Model depth:     {args.depth}")
    print0(f"  Model params:    {sum(p.numel() for p in orig_model.parameters()):,}")
    print0(f"  World size:      {ddp_world_size}")
    print0(f"  Batch size:      {args.total_batch_size} tokens")
    print0(f"  Grad accum:      {grad_accum_steps}")
    print0(f"  Device batch:    {effective_device_batch}")
    print0(f"  Total time:      {total_training_time:.2f}s")
    if hetero_balancer:
        print0(f"  Hetero profile:  {hetero_balancer.summary()}")
    if isinstance(comm, TopologyAwareCommunicator):
        print0(f"  Topology:        {comm.summary()['topology']}")
    elif hasattr(comm, 'summary'):
        print0(f"  Comm config:     {comm.summary()}")
    print0("✓ Dry run PASSED — pipeline is functional")
    print0("=" * 50)

# Save final checkpoint
if master_process:
    save_checkpoint(
        checkpoint_dir,
        step,
        orig_model.state_dict(),
        optimizer.state_dict(),
        {
            "step": step,
            "val_bpb": val_bpb,
            "model_config": model_config_kwargs,
            "user_config": user_config,
            "device_batch_size": args.device_batch_size,
            "max_seq_len": args.max_seq_len,
            "total_batch_size": args.total_batch_size,
            "dataloader_state_dict": dataloader_state_dict,
        },
        rank=ddp_rank,
    )

wandb_run.finish()
compute_cleanup()
print0("✓ Training complete.")

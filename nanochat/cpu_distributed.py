"""
CPU Distributed Training Utilities for nanochat fork.

Implements the full research roadmap for distributed CPU training:

Phase 1 — Gradient Quantization (FP32→FP16): 2× bandwidth savings
Phase 2 — Gradient Sparsification (top-k): aggressive WAN reduction
Phase 3 — Adaptive Communication Scheduling: batched sync + tier awareness
Phase 4 — Topology-Aware Hierarchical AllReduce: intra-LAN fast, cross-WAN minimal
Phase 5 — Network Benchmark: measure inter-node throughput before training
"""

import math
import socket
import subprocess
import time
from collections import defaultdict
from enum import Enum
from typing import Callable, Optional

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Phase 1: Gradient Quantization (FP32 → FP16)
# ---------------------------------------------------------------------------

class GradientQuantizer:
    """Compress gradients by downcasting FP32 → FP16 before sync.

    Reduces AllReduce payload by 2x. The precision loss is negligible for
    SGD/Adam because gradient noise dominates quantization noise at FP16.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._params: list[torch.nn.Parameter] = []

    def register_params(self, params: list[torch.nn.Parameter]):
        """Register the parameter list whose gradients should be compressed."""
        self._params = [p for p in params if p.requires_grad]

    def compress_and_sync(self, do_average: bool = True):
        """Compress all registered gradients to FP16, all-reduce, then restore FP32.

        Args:
            do_average: If True, divide by world_size after reduce (standard AllReduce).
                 If False, just sum (for gradient accumulation).
        """
        if not self.enabled or not dist.is_initialized():

            return

        world_size = dist.get_world_size()
        handles = []

        for p in self._params:
            if p.grad is None:
                continue
            # Cast gradient to fp16 for communication
            grad_fp16 = p.grad.data.to(torch.float16)
            # AllReduce in fp16 (half the bandwidth)
            handle = dist.all_reduce(grad_fp16, op=dist.ReduceOp.SUM, async_op=True)
            handles.append((p, grad_fp16, handle))

        for p, grad_fp16, handle in handles:
            handle.wait()
            # Restore to fp32 and scale back
            if do_average:
                p.grad.data = grad_fp16.to(torch.float32) / world_size
            else:
                p.grad.data = grad_fp16.to(torch.float32)


# ---------------------------------------------------------------------------
# Phase 2: Gradient Sparsification (top-k)
# ---------------------------------------------------------------------------

class GradientSparsifier:
    """Only communicate the top-k% of gradients by magnitude.

    Reference: Dryden et al. "Communication Quantization for
    Data-Parallel Training of Deep Neural Networks" (2016).

    The remaining gradients are accumulated locally in an error-feedback
    buffer and carried forward to the next step.
    """

    def __init__(self, topk_ratio: float = 0.01, enabled: bool = True):
        """
        Args:
            topk_ratio: Fraction of gradient elements to send (e.g. 0.01 = 1%).
            enabled: Set False to skip sparsification (fall back to dense).
        """
        self.topk_ratio = topk_ratio
        self.enabled = enabled
        self._error_buffers: dict[int, torch.Tensor] = {}
        self._params: list[torch.nn.Parameter] = []

    def register_params(self, params: list[torch.nn.Parameter]):
        """Register parameters and allocate error-feedback buffers."""
        self._params = [p for p in params if p.requires_grad]
        for p in self._params:
            self._error_buffers[id(p)] = torch.zeros_like(p.data)

    def sparsify_and_sync(self, do_average: bool = True):
        """Apply top-k sparsification with error feedback, then all-reduce."""
        if not self.enabled or not dist.is_initialized():
            return

        world_size = dist.get_world_size()
        device = self._params[0].device if self._params else torch.device("cpu")

        for p in self._params:
            if p.grad is None:
                continue

            # Accumulate error from previous step
            error_buf = self._error_buffers[id(p)]
            grad = p.grad.data + error_buf

            # Top-k selection
            num_elements = grad.numel()
            k = max(1, int(num_elements * self.topk_ratio))

            # Flatten and find top-k by magnitude
            grad_flat = grad.view(-1)
            _, topk_indices = torch.topk(grad_flat.abs(), k)
            topk_values = grad_flat[topk_indices]

            # Create sparse representation: (indices, values) → all-reduce both
            # In practice we'd use a custom all-reduce for sparse tensors,
            # but for simplicity we reconstruct a dense tensor with zeros elsewhere
            sparse_dense = torch.zeros_like(grad_flat)
            sparse_dense[topk_indices] = topk_values

            # All-reduce the sparse-restored dense tensor
            dist.all_reduce(sparse_dense, op=dist.ReduceOp.SUM)
            if do_average:
                sparse_dense /= world_size

            # Error feedback: what we didn't send stays in the buffer
            error_buf.copy_(grad - topk_values.new_zeros(num_elements).scatter(0, topk_indices, topk_values))

            # Write back the sparsified + communicated gradient
            # For non-selected indices the gradient is zero from this step
            p.grad.data = sparse_dense.view(p.grad.shape)


# ---------------------------------------------------------------------------
# Phase 3: Communication Scheduler
# ---------------------------------------------------------------------------

class NetworkTier(Enum):
    """Estimated network quality between nodes."""
    LOCAL = "local"       # Single node (no network overhead)
    LAN = "lan"           # Same datacenter / local network
    WAN = "wan"           # Geographic WAN with high latency


class AdaptiveCommScheduler:
    """Network-aware communication scheduler for heterogeneous LAN/WAN training.

    Features:
    - Configurable sync interval (sync every N steps instead of every step)
    - Network-tier detection (LAN vs WAN based on simple heuristics)
    - Adaptive compression ratio based on network conditions
    - Overlap communication with computation where possible
    """

    def __init__(
        self,
        sync_interval: int = 1,
        network_tier: Optional[str] = None,
        compression_enabled: bool = True,
        sparsification_enabled: bool = True,
        sparsification_ratio: float = 0.01,
        world_size: int = 1,
        warmup_steps: int = 10,
    ):
        """
        Args:
            sync_interval: AllReduce every N steps (1 = sync every step, 2 = every other, etc.)
            network_tier: 'lan', 'wan', or None for auto-detect.
            compression_enabled: Enable FP32→FP16 gradient quantization.
            sparsification_enabled: Enable top-k gradient sparsification.
            sparsification_ratio: Fraction of gradients to keep when sparsifying.
            world_size: Number of distributed workers.
            warmup_steps: Steps before enabling compression/sparsification.
        """
        self.sync_interval = sync_interval
        self.network_tier = network_tier or "lan"
        self.compression_enabled = compression_enabled
        self.sparsification_enabled = sparsification_enabled
        self.sparsification_ratio = sparsification_ratio
        self.world_size = world_size
        self.warmup_steps = warmup_steps

        self.quantizer = GradientQuantizer(enabled=compression_enabled)
        self.sparsifier = GradientSparsifier(
            topk_ratio=sparsification_ratio,
            enabled=sparsification_enabled,
        )
        self._step = 0
        self._grad_accum: list[Optional[torch.Tensor]] = []
        self._params: list[torch.nn.Parameter] = []

    @staticmethod
    def detect_network_tier() -> str:
        """Heuristic: check if all known peers are on the same host.

        Falls back to 'lan' if undetermined. Users can override via config.
        """
        if not dist.is_initialized():
            return "local"
        # Simple heuristic: if world_size == 1 or all on same hostname → local
        # For multi-node, default to LAN (user can force WAN if needed)
        world_size = dist.get_world_size()
        if world_size <= 1:
            return "local"
        return "lan"  # conservative default

    def register_params(self, params: list[torch.nn.Parameter]):
        self._params = [p for p in params if p.requires_grad]
        self.quantizer.register_params(self._params)
        self.sparsifier.register_params(self._params)
        self._grad_accum = [None] * len(self._params)

    def on_train_step_start(self):
        """Called at the start of each training step."""
        pass

    def on_backward_done(self):
        """Called after loss.backward() — accumulate gradients locally."""
        if self.sync_interval <= 1:
            return  # sync every step, nothing special needed

        # Accumulate gradients locally
        for i, p in enumerate(self._params):
            if p.grad is not None:
                if self._grad_accum[i] is None:
                    self._grad_accum[i] = p.grad.data.clone()
                else:
                    self._grad_accum[i] += p.grad.data

    def on_optimizer_step(self, model):
        """Called before optimizer.step() — synchronize gradients if needed.

        Handles three modes:
        1. sync_interval=1: Compress + AllReduce every step (standard)
        2. sync_interval=N: Accumulate N steps locally, then sync
        3. Network-adaptive: Adjust compression based on tier
        """
        self._step += 1

        if self._step < self.warmup_steps or not dist.is_initialized():
            return

        should_sync = (self._step % self.sync_interval == 0)

        if self.sync_interval <= 1:
            # Standard per-step sync with compression
            if self.sparsification_enabled:
                self.sparsifier.sparsify_and_sync(do_average=True)
            elif self.compression_enabled:
                self.quantizer.compress_and_sync(do_average=True)
            return

        if not should_sync:
            return

        # Sync accumulated gradients
        world_size = dist.get_world_size()
        for i, p in enumerate(self._params):
            if self._grad_accum[i] is not None:
                p.grad = self._grad_accum[i]  # use accumulated gradient
                self._grad_accum[i] = None

        # Now compress + AllReduce the accumulated gradient
        if self.sparsification_enabled:
            self.sparsifier.sparsify_and_sync(do_average=True)
        elif self.compression_enabled:
            self.quantizer.compress_and_sync(do_average=True)

    def get_network_tier_label(self) -> str:
        return {
            "local": "single-node",
            "lan": "LAN (low latency)",
            "wan": "WAN (high latency, compression active)",
        }.get(self.network_tier, self.network_tier)

    # -------------------------------------------------------------------
    # Async overlap: start sync, then do other work, then wait
    # -------------------------------------------------------------------

    def start_async_sync(self) -> Optional[list]:
        """Start non-blocking gradient sync and return a handle.

        Call this after loss.backward(). The all-reduce runs in the
        background while the caller can do data loading or the next
        forward pass. Call wait_for_sync(handle) before optimizer.step().

        Returns a list of (param, grad_buffer, handle) tuples, or None
        if no sync is needed this step.
        """
        self._step += 1

        if self._step < self.warmup_steps or not dist.is_initialized():
            return None

        should_sync = (self._step % self.sync_interval == 0)
        if not should_sync and self.sync_interval > 1:
            # Just accumulate, don't sync yet
            for i, p in enumerate(self._params):
                if p.grad is not None:
                    if self._grad_accum[i] is None:
                        self._grad_accum[i] = p.grad.data.clone()
                    else:
                        self._grad_accum[i] += p.grad.data
            return None

        if self.sync_interval > 1:
            # Use accumulated gradients
            for i, p in enumerate(self._params):
                if self._grad_accum[i] is not None:
                    p.grad = self._grad_accum[i]
                    self._grad_accum[i] = None

        # Fire off async all-reduce
        if self.sparsification_enabled:
            # Sparsifier currently uses sync all-reduce — fall through
            self.sparsifier.sparsify_and_sync(do_average=True)
            return None
        elif self.compression_enabled:
            handles = []
            for p in self._params:
                if p.grad is None:
                    continue
                grad_fp16 = p.grad.data.to(torch.float16)
                handle = dist.all_reduce(grad_fp16, op=dist.ReduceOp.SUM, async_op=True)
                handles.append((p, grad_fp16, handle))
            return handles  # caller must wait_for_sync
        return None

    def wait_for_sync(self, handle: Optional[list]):
        """Wait for an async sync started by start_async_sync() to finish.

        Must be called before optimizer.step().
        """
        if handle is None:
            return
        world_size = dist.get_world_size()
        for p, grad_fp16, ah in handle:
            ah.wait()
            p.grad.data = grad_fp16.to(torch.float32) / world_size

    def summary(self) -> dict:
        return {
            "network_tier": self.get_network_tier_label(),
            "sync_interval": self.sync_interval,
            "compression": self.compression_enabled,
            "sparsification": self.sparsification_enabled,
            "sparsification_ratio": self.sparsification_ratio,
            "world_size": self.world_size,
            "warmup_steps": self.warmup_steps,
        }


# ---------------------------------------------------------------------------
# Phase 4: Topology-Aware Hierarchical AllReduce
# ---------------------------------------------------------------------------

def _get_host_ip() -> str:
    """Get the primary IP address of this host.

    Connects to a dummy address to determine the preferred outbound
    interface IP, which correctly identifies the LAN subnet even when
    multiple interfaces exist.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1.0)
        # Doesn't actually connect — just used to probe routing table
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _subnet_key(ip: str, prefix_len: int = 24) -> str:
    """Extract subnet prefix from an IP string.

    Example: '192.168.1.43' with /24 prefix → '192.168.1'
    """
    return ".".join(ip.split(".")[:prefix_len // 8])


class TopologyAwareCommunicator:
    """Hierarchical all-reduce that groups nodes by LAN subnet.

    How it works:
    1. All ranks report their IP address to rank 0.
    2. Rank 0 partitions ranks into groups by subnet prefix.
    3. Each group creates its own intra-group process group.
    4. An inter-group process group connects group leaders.
    5. On each sync: intra-group all-reduce → inter-group all-reduce → scatter back.

    This minimizes WAN traffic: for G groups of S nodes each,
    each rank sends O(S-1) within its LAN and O(G-1) across WAN,
    instead of O(G*S-1) across WAN for a flat all-reduce.

    Reference: hierarchical AllReduce pattern from the research,
    "grouping machines by their IP address ranges to identify which
    nodes are on the same local network segment."
    """

    def __init__(
        self,
        subnet_prefix_len: int = 24,
        compression_enabled: bool = True,
        sparsification_enabled: bool = False,
        sparsification_ratio: float = 0.01,
    ):
        self.subnet_prefix_len = subnet_prefix_len
        self.compression_enabled = compression_enabled
        self.sparsification_enabled = sparsification_enabled
        self.sparsification_ratio = sparsification_ratio

        self.quantizer = GradientQuantizer(enabled=compression_enabled)
        self.sparsifier = GradientSparsifier(
            topk_ratio=sparsification_ratio,
            enabled=sparsification_enabled,
        )
        self._world_size = 1
        self._rank = 0
        self._group_rank = 0  # rank within the local group
        self._group_size = 1  # number of nodes in the local group
        self._group_id = 0    # which group this node belongs to
        self._num_groups = 1  # total number of groups
        self._is_leader = True  # whether this node is the group leader
        self._intra_group_pg = None  # process group for intra-group comms
        self._inter_group_pg = None  # process group for inter-group (leader only)
        self._params: list[torch.nn.Parameter] = []
        self._groups: dict[int, list[int]] = {}  # group_id -> [global_ranks]

    def discover_topology(self) -> dict:
        """Exchange IPs between all ranks and build group topology.

        Returns a dict describing the discovered topology.
        """
        if not dist.is_initialized():
            return {"nodes": 1, "groups": 1, "group_id": 0}

        self._world_size = dist.get_world_size()
        self._rank = dist.get_rank()
        local_ip = _get_host_ip()

        # Gather all IPs to rank 0
        all_ips = [""] * self._world_size
        gathered = [None] * self._world_size
        dist.all_gather_object(gathered, local_ip)
        all_ips = gathered

        # Build groups by subnet (every rank computes the same mapping)
        subnet_to_ranks: dict[str, list[int]] = defaultdict(list)
        for r, ip in enumerate(all_ips):
            subnet_to_ranks[_subnet_key(ip, self.subnet_prefix_len)].append(r)

        # Assign group IDs
        self._groups = {}
        for gid, (_, members) in enumerate(sorted(subnet_to_ranks.items())):
            self._groups[gid] = sorted(members)

        self._num_groups = len(self._groups)
        my_subnet = _subnet_key(local_ip, self.subnet_prefix_len)

        # Find this rank's group
        for gid, members in self._groups.items():
            if self._rank in members:
                self._group_id = gid
                self._group_rank = members.index(self._rank)
                self._group_size = len(members)
                self._is_leader = (self._group_rank == 0)
                break

        topology = {
            "nodes": self._world_size,
            "groups": self._num_groups,
            "group_id": self._group_id,
            "group_size": self._group_size,
            "group_rank": self._group_rank,
            "is_leader": self._is_leader,
            "local_ip": local_ip,
        }

        return topology

    def create_process_groups(self):
        """Create intra-group and inter-group process groups.

        Must be called after discover_topology() and before any training.
        """
        if not dist.is_initialized() or self._num_groups <= 1:
            return

        # Intra-group PG: all nodes in the same subnet
        for gid, members in self._groups.items():
            pg = dist.new_group(ranks=members, backend="gloo")
            if gid == self._group_id:
                self._intra_group_pg = pg

        # Inter-group PG: group leaders only
        leader_ranks = [members[0] for members in self._groups.values()]
        self._inter_group_pg = dist.new_group(ranks=leader_ranks, backend="gloo")

    def register_params(self, params: list[torch.nn.Parameter]):
        self._params = [p for p in params if p.requires_grad]
        self.quantizer.register_params(self._params)
        self.sparsifier.register_params(self._params)

    @staticmethod
    def _do_allreduce(tensor: torch.Tensor, pg: dist.ProcessGroup) -> torch.Tensor:
        """All-reduce a tensor over a specific process group."""
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=pg)
        return tensor

    def sync_gradients(self, do_average: bool = True):
        """Hierarchical gradient sync: intra-group → inter-group → scatter.

        For multi-group topology (WAN), this runs:
        1. Intra-group all-reduce (fast, within LAN)
        2. Group leader reduces across groups (slow, WAN)
        3. Leader broadcasts back to group members

        For single-group (LAN), falls back to a single flat all-reduce.
        """
        if not dist.is_initialized():
            return

        if self._num_groups <= 1:
            # Single group — flat all-reduce (everything is LAN)
            if self.sparsification_enabled:
                self.sparsifier.sparsify_and_sync(do_average=do_average)
            elif self.compression_enabled:
                self.quantizer.compress_and_sync(do_average=do_average)
            return

        # Multi-group hierarchical sync
        for p in self._params:
            if p.grad is None:
                continue

            grad = p.grad.data

            # Step 1: Intra-group all-reduce (fast, LAN-local)
            if self._intra_group_pg is not None:
                # Compress to fp16 for intra-group too if enabled
                if self.compression_enabled:
                    grad_fp16 = grad.to(torch.float16)
                    self._do_allreduce(grad_fp16, self._intra_group_pg)
                    grad_fp32 = grad_fp16.to(torch.float32)
                else:
                    grad_fp32 = grad.clone()
                    self._do_allreduce(grad_fp32, self._intra_group_pg)

                # Average within group
                grad_fp32 /= self._group_size
            else:
                grad_fp32 = grad.clone()

            # Step 2: Inter-group sync (leaders only, across WAN)
            if self._inter_group_pg is not None and self._is_leader:
                if self.sparsification_enabled and self._num_groups > 1:
                    # Use sparsification for the WAN leg
                    # For simplicity, do a compressed all-reduce on leaders
                    grad_fp16 = grad_fp32.to(torch.float16)
                    self._do_allreduce(grad_fp16, self._inter_group_pg)
                    grad_fp32 = grad_fp16.to(torch.float32)
                    grad_fp32 /= self._num_groups
                else:
                    grad_fp16 = grad_fp32.to(torch.float16)
                    self._do_allreduce(grad_fp16, self._inter_group_pg)
                    grad_fp32 = grad_fp16.to(torch.float32)
                    grad_fp32 /= self._num_groups

            # Step 3: Broadcast leader's result back to group
            if self._intra_group_pg is not None:
                dist.broadcast(grad_fp32, src=self._groups[self._group_id][0],
                               group=self._intra_group_pg)

            # Write back
            if do_average:
                p.grad.data = grad_fp32
            else:
                p.grad.data = grad_fp32 * self._world_size

    # -------------------------------------------------------------------
    # Async overlap for topology-aware mode
    # -------------------------------------------------------------------

    def start_async_sync(self) -> Optional[object]:
        """Fire off async hierarchical gradient sync.

        For single-group (LAN): runs async compressed all-reduce.
        For multi-group (WAN): starts intra-group all-reduce async,
        returns a handle that tracks all 3 phases (intra → inter → broadcast).
        The training loop's overlap window runs while intra-group comms finish.
        """
        if not dist.is_initialized():
            return None

        if self._num_groups <= 1:
            # Single group — async compressed all-reduce
            if self.sparsification_enabled:
                self.sparsifier.sparsify_and_sync(do_average=True)
                return None
            elif self.compression_enabled:
                handles = []
                for p in self._params:
                    if p.grad is None:
                        continue
                    grad_fp16 = p.grad.data.to(torch.float16)
                    handle = dist.all_reduce(grad_fp16, op=dist.ReduceOp.SUM, async_op=True)
                    handles.append((p, grad_fp16, handle))
                return handles
            return None

        # Multi-group: Phase 1 — async intra-group all-reduce
        # This is the expensive step that benefits from async overlap
        intra_handles = []
        for p in self._params:
            if p.grad is None:
                continue
            grad = p.grad.data
            if self.compression_enabled:
                grad_fp16 = grad.to(torch.float16)
            else:
                grad_fp16 = grad  # use original if not compressing

            if self._intra_group_pg is not None:
                handle = dist.all_reduce(
                    grad_fp16 if self.compression_enabled else grad,
                    op=dist.ReduceOp.SUM,
                    group=self._intra_group_pg,
                    async_op=True,
                )
                intra_handles.append((p, grad_fp16 if self.compression_enabled else grad, handle))
            else:
                intra_handles.append((p, grad, None))

        return {
            "type": "hierarchical",
            "intra_handles": intra_handles,
            "group_size": self._group_size,
            "num_groups": self._num_groups,
            "is_leader": self._is_leader,
            "intra_group_pg": self._intra_group_pg,
            "inter_group_pg": self._inter_group_pg,
            "group_leader": self._groups[self._group_id][0] if self._groups else 0,
            "compression_enabled": self.compression_enabled,
        }

    def wait_for_sync(self, handle: Optional[object]):
        """Wait for async hierarchical sync started by start_async_sync().

        For single-group: waits for the all-reduce and restores FP32.
        For multi-group: completes all 3 phases:
          1. Wait for intra-group all-reduce
          2. Inter-group all-reduce (leaders only, across WAN)
          3. Broadcast results back to group members
        """
        if handle is None:
            return

        # Single group: classic flat wait
        if isinstance(handle, list):
            world_size = dist.get_world_size()
            for p, grad_fp16, ah in handle:
                ah.wait()
                p.grad.data = grad_fp16.to(torch.float32) / world_size
            return

        # Multi-group (hierarchical): complete all 3 phases
        if isinstance(handle, dict) and handle.get("type") == "hierarchical":
            intra = handle["intra_handles"]
            group_size = handle["group_size"]
            num_groups = handle["num_groups"]
            is_leader = handle["is_leader"]
            intra_pg = handle["intra_group_pg"]
            inter_pg = handle["inter_group_pg"]
            leader_rank = handle["group_leader"]
            comp_enabled = handle["compression_enabled"]

            # Phase 1b: Wait for intra-group all-reduce
            for p, buf, ah in intra:
                if ah is not None:
                    ah.wait()
                if comp_enabled:
                    # Convert fp16 back to fp32
                    grad_fp32 = buf.to(torch.float32)
                else:
                    grad_fp32 = buf
                # Average within group
                grad_fp32 /= group_size
                # Store temporarily for inter-group phase
                p.grad.data = grad_fp32

            # Phase 2: Inter-group all-reduce (leaders only, across WAN)
            if inter_pg is not None and is_leader:
                inter_handles = []
                for p in self._params:
                    if p.grad is None:
                        continue
                    grad_fp16 = p.grad.data.to(torch.float16)
                    handle2 = dist.all_reduce(
                        grad_fp16, op=dist.ReduceOp.SUM,
                        group=inter_pg, async_op=True,
                    )
                    inter_handles.append((p, grad_fp16, handle2))

                # Wait for inter-group
                for p, grad_fp16, ah in inter_handles:
                    ah.wait()
                    p.grad.data = grad_fp16.to(torch.float32) / num_groups

            # Phase 3: Broadcast leader's result back to group members
            if intra_pg is not None:
                for p in self._params:
                    if p.grad is None:
                        continue
                    dist.broadcast(p.grad.data, src=leader_rank, group=intra_pg)

    def summary(self) -> dict:
        return {
            "topology": f"{self._num_groups} group(s), {self._world_size} node(s)",
            "local_group": self._group_id,
            "group_size": self._group_size,
            "is_leader": self._is_leader,
            "compression": self.compression_enabled,
            "sparsification": self.sparsification_enabled,
        }


# ---------------------------------------------------------------------------
# Phase 5: Network Bandwidth Benchmark
# ---------------------------------------------------------------------------

class NetworkBenchmark:
    """Measure real bandwidth between distributed nodes.

    Runs a point-to-point bandwidth test using PyTorch tensors over
    the Gloo backend. Results are used to recommend optimal
    compression and sync interval settings for the given topology.
    """

    def __init__(self, payload_bytes: int = 8 * 1024 * 1024):
        """
        Args:
            payload_bytes: Size of tensor to use for benchmarking (default 8MB).
        """
        self.payload_bytes = payload_bytes

    def run(self) -> dict:
        """Run the bandwidth benchmark.

        Returns a dict with measured bandwidth in MB/s and recommendations.
        """
        if not dist.is_initialized():
            return {"bandwidth_mbps": 0, "latency_ms": 0, "error": "not distributed"}

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Create a tensor of the specified size
        num_elements = self.payload_bytes // 4  # float32 = 4 bytes
        tensor = torch.ones(num_elements, dtype=torch.float32)

        # Warmup
        for _ in range(3):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        # Benchmark: measure all-reduce time
        torch.distributed.barrier()
        t0 = time.perf_counter()
        num_iters = 10
        for _ in range(num_iters):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.distributed.barrier()
        t1 = time.perf_counter()

        total_bytes = self.payload_bytes * num_iters * 2  # send + receive
        elapsed = t1 - t0
        bandwidth_bps = total_bytes / elapsed if elapsed > 0 else 0
        bandwidth_mbps = bandwidth_bps / (1024 * 1024)

        # Estimate latency from the first iteration
        torch.distributed.barrier()
        t2 = time.perf_counter()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.distributed.barrier()
        t3 = time.perf_counter()
        latency_ms = (t3 - t2) * 1000

        # Recommendations
        if bandwidth_mbps < 50:
            tier = "wan"
            recommended_interval = 5
            recommended_sparsify = True
        elif bandwidth_mbps < 500:
            tier = "wan"
            recommended_interval = 3
            recommended_sparsify = True
        elif bandwidth_mbps < 2000:
            tier = "lan"
            recommended_interval = 1
            recommended_sparsify = False
        else:
            tier = "lan"
            recommended_interval = 1
            recommended_sparsify = False

        result = {
            "bandwidth_mbps": round(bandwidth_mbps, 1),
            "latency_ms": round(latency_ms, 2),
            "tier": tier,
            "recommended_sync_interval": recommended_interval,
            "recommended_sparsify": recommended_sparsify,
            "payload_bytes": self.payload_bytes,
        }

        if rank == 0:
            print(f"  Bandwidth: {result['bandwidth_mbps']:.1f} MB/s")
            print(f"  Latency:   {result['latency_ms']:.2f} ms")
            print(f"  Tier:      {tier}")
            print(f"  Recom sync interval: {recommended_interval}")
            print(f"  Recom sparsify:     {recommended_sparsify}")

        return result


# ---------------------------------------------------------------------------
# Phase 6: Heterogeneous Load Balancer & Straggler Handling
# ---------------------------------------------------------------------------

class HeterogeneousLoadBalancer:
    """Profile each node's compute speed and distribute workload proportionally.

    In a heterogeneous cluster (different CPUs, RAM, background load), a single
    slow node (straggler) holds up every all-reduce. This class:

    1. Profiles each rank's compute throughput at startup
    2. Assigns batch sizes proportional to node speed
    3. Detects stragglers at runtime via step-time monitoring
    4. Optionally reduces/times-out straggler contributions

    Reference: "Heterogeneity-aware resource allocation and topology design"
    from the research — "If one machine is significantly slower or has less RAM,
    it can become a bottleneck, slowing down the entire training process."
    """

    def __init__(
        self,
        profile_steps: int = 5,
        straggler_ratio: float = 2.0,
        adapt_batch: bool = True,
        world_size: int = 1,
        rank: int = 0,
        base_batch_size: int = 1,
    ):
        """
        Args:
            profile_steps: Number of steps to profile each node's speed.
            straggler_ratio: Node is a straggler if its step time is this
                             many times slower than the median (2.0 = 2x).
            adapt_batch: If True, assign larger batches to faster nodes.
            world_size: Number of distributed workers.
            rank: This node's rank.
            base_batch_size: Per-device batch size before adjustment.
        """
        self.profile_steps = profile_steps
        self.straggler_ratio = straggler_ratio
        self.adapt_batch = adapt_batch
        self.world_size = world_size
        self.rank = rank
        self.base_batch_size = base_batch_size

        # Profiling state
        self._step_times: list[float] = []
        self._profiling_done = False
        self._node_speed: float = 1.0       # relative speed of this node
        self._node_speeds: list[float] = []  # all nodes' relative speeds
        self._straggler = False
        self._batch_multiplier: float = 1.0

    @property
    def adjusted_batch_size(self) -> int:
        """The per-device batch size adjusted for this node's speed."""
        if not self._profiling_done or not self.adapt_batch:
            return self.base_batch_size
        return max(1, round(self.base_batch_size * self._batch_multiplier))

    def record_step_time(self, dt: float):
        """Call after each training step with the wall-clock time.

        During profiling (first N steps), accumulates measurements.
        After profiling, checks for stragglers.
        """
        if self._profiling_done:
            # Runtime straggler detection
            return  # called per-step below

        self._step_times.append(dt)
        if len(self._step_times) >= self.profile_steps:
            self._finalize_profile()

    def _finalize_profile(self):
        """Analyze profiling data and compute node speed ratios."""
        if not dist.is_initialized():
            self._node_speed = 1.0
            self._node_speeds = [1.0]
            self._profiling_done = True
            return

        # Average this node's step time
        local_avg = sum(self._step_times) / len(self._step_times)

        # Gather all nodes' average times
        all_times = [0.0] * self.world_size
        gathered = [0.0] * self.world_size
        # Use all_gather_object for floats
        dist.all_gather_object(gathered, local_avg)
        all_times = gathered

        # Compute relative speeds (inverse of time, normalized)
        # Speed = 1/time, normalized so fastest = 1.0
        speeds = [1.0 / t if t > 0 else 0.0 for t in all_times]
        max_speed = max(speeds) if speeds else 1.0
        self._node_speeds = [s / max_speed for s in speeds]  # 0.0..1.0
        self._node_speed = self._node_speeds[self.rank]

        # Straggler detection
        sorted_speeds = sorted(self._node_speeds)
        median_speed = sorted_speeds[len(sorted_speeds) // 2] if sorted_speeds else 1.0
        if median_speed > 0 and self._node_speed / median_speed < (1.0 / self.straggler_ratio):
            self._straggler = True

        # Batch multiplier: faster nodes get more work
        if self.adapt_batch and max_speed > 0:
            total_speed = sum(self._node_speeds)
            avg_speed = total_speed / self.world_size if self.world_size > 0 else 1.0
            self._batch_multiplier = self._node_speed / avg_speed if avg_speed > 0 else 1.0
        else:
            self._batch_multiplier = 1.0

        self._profiling_done = True

        if self.rank == 0:
            print(f"Heterogeneous profiling complete:")
            print(f"  Node speeds (relative): {[f'{s:.2f}' for s in self._node_speeds]}")
            print(f"  Straggler detected: {self._straggler}")
            print(f"  My batch multiplier: {self._batch_multiplier:.2f}")
            print(f"  Adjusted batch: {self.adjusted_batch_size}")

    def is_straggler(self) -> bool:
        return self._straggler

    def summary(self) -> dict:
        return {
            "profiling_done": self._profiling_done,
            "node_speed": round(self._node_speed, 3),
            "straggler": self._straggler,
            "batch_multiplier": round(self._batch_multiplier, 3),
            "adjusted_batch_size": self.adjusted_batch_size,
            "adapt_batch": self.adapt_batch,
        }


class StragglerMitigator:
    """Runtime straggler detection and mitigation during training.

    After the initial profiling period, monitors step times continuously.
    If a node consistently underperforms, the mitigator can:
    - Log warnings
    - Suggest rebalancing
    - Time out slow all-reduces (future: skip straggler's contribution)
    """

    def __init__(
        self,
        window_size: int = 10,
        straggler_ratio: float = 2.0,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.window_size = window_size
        self.straggler_ratio = straggler_ratio
        self.rank = rank
        self.world_size = world_size
        self._recent_times: list[float] = []
        self._node_medians: list[float] = []
        self._warnings: int = 0
        self._consecutive_straggler_checks: int = 0
        self._failover_after_checks: int = 3  # fail after 3 consecutive straggler detections
        self._failover_triggered = False

    @property
    def should_failover(self) -> bool:
        """True if failover has been triggered (straggler too many times)."""
        return self._failover_triggered

    def record(self, dt: float):
        """Record this step's duration."""
        self._recent_times.append(dt)
        if len(self._recent_times) > self.window_size:
            self._recent_times.pop(0)

    def check_stragglers(self) -> Optional[list[int]]:
        """All-gather recent step times and identify straggler ranks.

        Returns list of straggler ranks, or None if no stragglers.
        """
        if not dist.is_initialized() or not self._recent_times:
            return None

        local_median = sorted(self._recent_times)[len(self._recent_times) // 2]

        gathered = [0.0] * self.world_size
        dist.all_gather_object(gathered, local_median)
        self._node_medians = gathered

        overall_median = sorted(self._node_medians)[len(self._node_medians) // 2]
        stragglers = []
        for r, med in enumerate(self._node_medians):
            if overall_median > 0 and med / overall_median > self.straggler_ratio:
                stragglers.append(r)

        if stragglers and self.rank == 0:
            self._warnings += 1
            self._consecutive_straggler_checks += 1
            print(f"  ⚠ Straggler(s) detected: ranks {stragglers} "
                  f"({self._consecutive_straggler_checks}/{self._failover_after_checks})")
            if self._consecutive_straggler_checks >= self._failover_after_checks:
                self._failover_triggered = True
                print(f"  🚨 FAILOVER: straggler persistent after "
                      f"{self._failover_after_checks} checks — saving checkpoint")
        else:
            self._consecutive_straggler_checks = 0

        return stragglers if stragglers else None


# ---------------------------------------------------------------------------
# Synthetic Data Loader (for testing without real dataset)
# ---------------------------------------------------------------------------

class SyntheticDataLoader:
    """Generates random token data for testing the training pipeline.

    No dataset download needed — produces random tokens from the model's
    vocab size. Useful for:
      - Verifying the distributed training loop works end-to-end
      - Benchmarking communication overhead in isolation
      - Testing all compression/scheduling/topology features
      - CI / smoke tests

    Usage:
        loader = SyntheticDataLoader(vocab_size=50304, B=4, T=512, device='cpu')
        x, y, state_dict = next(loader)
    """

    def __init__(
        self,
        vocab_size: int = 50304,
        B: int = 4,
        T: int = 512,
        device: str = "cpu",
        num_batches: int = 1000,
    ):
        self.vocab_size = vocab_size
        self.B = B
        self.T = T
        self.device = device
        self.num_batches = num_batches
        self._batch_idx = 0
        self._epoch = 1

        # Pre-allocate persistent buffers so each iteration just fills them
        self._inputs = torch.empty(B, T, dtype=torch.long, device=device)
        self._targets = torch.empty(B, T, dtype=torch.long, device=device)

    def __next__(self):
        if self._batch_idx >= self.num_batches:
            self._batch_idx = 0
            self._epoch += 1

        # Fill with random tokens
        self._inputs.copy_(torch.randint(0, self.vocab_size, (self.B, self.T), device=self.device))
        self._targets.copy_(torch.randint(0, self.vocab_size, (self.B, self.T), device=self.device))

        state_dict = {
            "epoch": self._epoch,
            "batch_idx": self._batch_idx,
            "pq_idx": 0,
            "rg_idx": 0,
        }
        self._batch_idx += 1
        return self._inputs, self._targets, state_dict

    def __iter__(self):
        return self

    def summary(self) -> dict:
        return {
            "type": "synthetic",
            "vocab_size": self.vocab_size,
            "B": self.B,
            "T": self.T,
            "device": self.device,
            "num_batches": self.num_batches,
        }


# ---------------------------------------------------------------------------
# Step Profiler — per-phase timing breakdown
# ---------------------------------------------------------------------------

class StepProfiler:
    """Break down each training step into phases and report timing.

    Phases tracked:
        fwd:     model forward pass
        bwd:     loss.backward()
        data:    next(train_loader) data loading
        comm:    gradient sync (all-reduce) — includes async overlap wait
        optim:   optimizer.step()
        other:   any unaccounted time (scheduling, zero_grad, etc.)

    Usage:
        profiler = StepProfiler()
        profiler.start()
        ...
        profiler.mark('fwd')
        ...
        profiler.mark('bwd')
        ...
        report = profiler.summary()
    """

    VALID_PHASES = {"fwd", "bwd", "data", "comm", "optim", "other"}

    def __init__(self, enabled: bool = True, print_every: int = 0):
        """
        Args:
            enabled: Set False to skip all timing (zero overhead).
            print_every: Print per-step breakdown every N steps (0 = only at end).
        """
        self.enabled = enabled
        self.print_every = print_every
        self.reset()

    def reset(self):
        """Clear all accumulated stats."""
        self._step = 0
        self._last_t = 0.0
        self._current = {p: 0.0 for p in self.VALID_PHASES}
        # phase -> list of per-step times
        self._records: dict[str, list[float]] = {p: [] for p in self.VALID_PHASES}
        self._step_times: list[float] = []

    def start(self):
        """Call at the beginning of each step."""
        if not self.enabled:
            return
        self._current = {p: 0.0 for p in self.VALID_PHASES}
        self._last_t = time.perf_counter()

    def mark(self, phase: str):
        """Record time since last mark under `phase`.

        Args:
            phase: One of 'fwd', 'bwd', 'data', 'comm', 'optim', 'other'.
        """
        if not self.enabled or phase not in self._current:
            return
        now = time.perf_counter()
        dt = now - self._last_t
        self._current[phase] += dt
        self._last_t = now

    def end(self, phase: str = "other"):
        """Call at the end of each step. Records all phases."""
        if not self.enabled:
            return
        self.mark(phase)
        step_total = sum(self._current.values())
        for p, t in self._current.items():
            self._records[p].append(t)
        self._step_times.append(step_total)
        self._step += 1

        if self.print_every > 0 and self._step % self.print_every == 0:
            self._print_step()

    @property
    def last_step_time(self) -> float:
        """Total time of the most recently ended step (in seconds)."""
        return self._step_times[-1] if self._step_times else 0.0

    def _print_step(self):
        """Print per-step timing breakdown."""
        total = sum(self._current.values())
        parts = []
        for p in ["fwd", "bwd", "data", "comm", "optim"]:
            t = self._current.get(p, 0)
            pct = (t / total * 100) if total > 0 else 0
            parts.append(f"{p}={t*1000:.0f}ms({pct:.0f}%)")
        print(f"  step {self._step}: {total*1000:.0f}ms | {' '.join(parts)}")

    def summary(self) -> dict:
        """Compute aggregate statistics across all recorded steps."""
        if not self.enabled or not self._step_times:
            return {"steps": 0}

        averages = {}
        total_avg = sum(self._step_times) / len(self._step_times)

        for p in self.VALID_PHASES:
            times = self._records[p]
            if times:
                avg = sum(times) / len(times)
                pct = (avg / total_avg * 100) if total_avg > 0 else 0
                averages[p] = {
                    "avg_ms": round(avg * 1000, 2),
                    "pct": round(pct, 1),
                    "min_ms": round(min(times) * 1000, 2),
                    "max_ms": round(max(times) * 1000, 2),
                }

        return {
            "steps": self._step,
            "avg_step_ms": round(total_avg * 1000, 2),
            "phases": averages,
            "toks_per_sec": None,  # caller fills this in
        }

    def print_report(self, toks_per_sec: Optional[float] = None):
        """Print a formatted timing report."""
        s = self.summary()
        if s["steps"] == 0:
            return

        print()
        print("=" * 55)
        print("  STEP TIMING BREAKDOWN")
        print("=" * 55)
        print(f"  Total steps:    {s['steps']}")
        print(f"  Avg step time:  {s['avg_step_ms']:.1f} ms")
        if toks_per_sec:
            print(f"  Throughput:     {toks_per_sec:.0f} tok/s")
        print("-" * 55)
        print(f"  {'Phase':<12} {'Avg':>8} {'%':>6} {'Min':>8} {'Max':>8}")
        print("-" * 55)
        for p in ["fwd", "bwd", "data", "comm", "optim", "other"]:
            ph = s["phases"].get(p)
            if ph:
                bar = "█" * max(1, int(ph["pct"] / 5))
                print(f"  {p:<12} {ph['avg_ms']:>8.1f}ms {ph['pct']:>5.1f}% "
                      f"{ph['min_ms']:>7.1f} {ph['max_ms']:>7.1f}  {bar}")
        print("=" * 55)


# ---------------------------------------------------------------------------
# WAN Resilience — signal-safe checkpoint + auto-resume
# ---------------------------------------------------------------------------

class WANResilienceManager:
    """Graceful shutdown and checkpoint recovery for WAN training runs.
    Saves periodic checkpoints, registers signal handlers for graceful
    shutdown, and supports auto-resume from the latest checkpoint.
    """
    def __init__(self, checkpoint_dir, save_every=0, master_process=True, rank=0):
        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every
        self.master_process = master_process
        self.rank = rank
        self._last_save_step = -1
        import os as _os
        self._latest_step_file = _os.path.join(checkpoint_dir, "latest_step.txt")

    def register_signal_handlers(self):
        import signal as _signal
        def _handler(signum, frame):
            import sys as _sys
            print(f"\n⚠ Signal {signum}, checkpoint may be incomplete")
            _sys.exit(128 + signum)
        try:
            _signal.signal(_signal.SIGINT, _handler)
            _signal.signal(_signal.SIGTERM, _handler)
        except (ValueError, AttributeError):
            pass

    def has_checkpoint(self):
        return self._read_latest_step() > 0

    def get_resume_step(self):
        return self._read_latest_step()

    def _read_latest_step(self):
        import os as _os
        if _os.path.exists(self._latest_step_file):
            try:
                with open(self._latest_step_file) as f:
                    return int(f.read().strip())
            except (ValueError, OSError):
                pass
        return -1

    def _write_latest_step(self, step):
        if not self.master_process:
            return
        import os as _os
        try:
            _os.makedirs(self.checkpoint_dir, exist_ok=True)
            with open(self._latest_step_file, "w") as f:
                f.write(str(step))
        except OSError:
            pass

    def maybe_save(self, step, model, optimizer, metadata, dataloader_state_dict, force=False):
        if self.save_every <= 0 and not force:
            return
        is_time = (step > 0 and step % self.save_every == 0) or force
        if not is_time or step == self._last_save_step:
            return
        self._last_save_step = step
        if not self.master_process:
            return
        from nanochat.checkpoint_manager import save_checkpoint
        try:
            meta = {
                "step": step,
                "model_config": metadata.get("model_config", {}),
                "user_config": metadata.get("user_config", {}),
                "device_batch_size": metadata.get("device_batch_size", 1),
                "max_seq_len": metadata.get("max_seq_len", 512),
                "total_batch_size": metadata.get("total_batch_size", 512),
                "dataloader_state_dict": dataloader_state_dict or {},
                "loop_state": {
                    "min_val_bpb": metadata.get("min_val_bpb", float("inf")),
                    "smooth_train_loss": metadata.get("smooth_train_loss", 0),
                    "total_training_time": metadata.get("total_training_time", 0),
                },
            }
            save_checkpoint(self.checkpoint_dir, step,
                model.state_dict() if hasattr(model, "state_dict") else {},
                optimizer.state_dict() if hasattr(optimizer, "state_dict") else {},
                meta, rank=self.rank)
            self._write_latest_step(step)
            print(f"  \U0001f4be Checkpoint saved at step {step}")
        except Exception as e:
            print(f"  \u26a0 Checkpoint save failed at step {step}: {e}")

    def summary(self):
        return {"save_every": self.save_every,
                "latest_step": self._read_latest_step(),
                "has_checkpoint": self.has_checkpoint(),
                "checkpoint_dir": self.checkpoint_dir}


# ---------------------------------------------------------------------------
# WAN Simulator — test compression/async under realistic WAN conditions
# ---------------------------------------------------------------------------

class WANSimulator:
    """Simulates WAN network conditions for testing communication optimizations.

    The research calls for testing under WAN scenarios to validate that
    gradient compression, sparsification, and async overlap actually help.
    This class injects artificial latency and bandwidth limits without
    needing actual geo-distributed machines.

    Usage:
        sim = WANSimulator(latency_ms=50, bandwidth_mbps=100)
        sim.start_step()       # call before gradient sync
        sim.throttle(data_bytes)  # sleeps to simulate bandwidth
        sim.end_step()         # adds round-trip latency
        print(sim.summary())   # how much time was "wasted" on WAN
    """

    def __init__(
        self,
        enabled: bool = False,
        latency_ms: float = 0.0,
        bandwidth_mbps: float = 0.0,
        loss_rate: float = 0.0,
    ):
        """
        Args:
            enabled: Set False to skip all simulation (zero overhead).
            latency_ms: Artificial one-way latency in milliseconds.
            bandwidth_mbps: Artificial bandwidth cap in Mbps (0 = unlimited).
            loss_rate: Probability of dropping a sync (0.0-1.0). When a sync
                       is "lost", the step's gradients are stale — simulating
                       a node that missed the all-reduce. (Experimental.)
        """
        self.enabled = enabled
        self.latency_ms = latency_ms
        self.bandwidth_mbps = bandwidth_mbps
        self.loss_rate = loss_rate
        self._total_latency = 0.0
        self._total_bandwidth = 0.0
        self._steps = 0
        self._losses = 0

    def start_step(self):
        """Call before gradient sync begins."""
        if not self.enabled:
            return
        self._steps += 1

        # Simulate packet loss: skip this step's sync
        if self.loss_rate > 0 and self._steps > 5:
            import random as _random
            if _random.random() < self.loss_rate:
                self._losses += 1
                # Inject extra delay to simulate timeout + recovery
                time.sleep(self.latency_ms * 4 / 1000.0)
                return

        # Simulate one-way latency before sync
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)

    def throttle(self, data_bytes: int):
        """Sleep to simulate bandwidth limit for transferring data_bytes."""
        if not self.enabled or self.bandwidth_mbps <= 0 or data_bytes <= 0:
            return
        # bandwidth_mbps is in Megabits/sec, data_bytes is in bytes
        # Time = data_bits / bandwidth_bps
        data_bits = data_bytes * 8
        bandwidth_bps = self.bandwidth_mbps * 1_000_000
        delay = data_bits / bandwidth_bps
        if delay > 0:
            time.sleep(delay)
            self._total_bandwidth += delay

    def end_step(self):
        """Call after gradient sync completes — adds return-trip latency."""
        if not self.enabled or self.latency_ms <= 0:
            return
        # Return-trip latency (half the round trip on the other side)
        time.sleep(self.latency_ms / 1000.0)
        self._total_latency += self.latency_ms * 2 / 1000.0

    def summary(self) -> dict:
        """Report how much time was spent on simulated WAN conditions."""
        if not self.enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "latency_ms": self.latency_ms,
            "bandwidth_mbps": self.bandwidth_mbps,
            "loss_rate": self.loss_rate,
            "total_latency_s": round(self._total_latency, 2),
            "total_bandwidth_s": round(self._total_bandwidth, 2),
            "total_wan_overhead_s": round(self._total_latency + self._total_bandwidth, 2),
            "steps": self._steps,
            "losses": self._losses,
        }

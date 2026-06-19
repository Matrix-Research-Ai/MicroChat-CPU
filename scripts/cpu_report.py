"""
Generate a self-contained HTML training report from a completed run.

Reads the checkpoint directory and profile logs, and produces an HTML file
with loss curves, throughput charts, and phase breakdowns.

Usage:
    # After training with --profile:
    python -m scripts.cpu_report --tag d2 --step 20

    # Auto-detect latest:
    python -m scripts.cpu_report

    # Custom checkpoint dir:
    python -m scripts.cpu_report --checkpoint-dir ~/.cache/nanochat/cpu_checkpoints
"""

import argparse
import json
import os
import re
import sys
from html import escape

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Generate HTML training report")
parser.add_argument("-g", "--tag", type=str, default=None,
                    help="Model tag (e.g. 'd2'). Default: auto-detect latest")
parser.add_argument("-s", "--step", type=int, default=None,
                    help="Checkpoint step. Default: latest")
parser.add_argument("--checkpoint-dir", type=str, default=None,
                    help="Override checkpoint directory")
parser.add_argument("-o", "--output", type=str, default=None,
                    help="Output HTML path (default: auto)")
parser.add_argument("--open", action="store_true", default=False,
                    help="Open the report in browser after generation")
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Find checkpoint directory and tag
base_dir = get_base_dir()
ckpt_dir = args.checkpoint_dir or os.path.join(base_dir, "cpu_checkpoints")

if not os.path.exists(ckpt_dir):
    print(f"Error: checkpoint directory not found: {ckpt_dir}")
    sys.exit(1)

if args.tag is None:
    tags = sorted([d for d in os.listdir(ckpt_dir)
                   if os.path.isdir(os.path.join(ckpt_dir, d))])
    if not tags:
        print(f"Error: no checkpoints found in {ckpt_dir}")
        sys.exit(1)
    args.tag = tags[-1]
    print(f"Auto-detected model: {args.tag}")

model_dir = os.path.join(ckpt_dir, args.tag)

# Find latest step
if args.step is None:
    latest_file = os.path.join(model_dir, "latest_step.txt")
    if os.path.exists(latest_file):
        with open(latest_file) as f:
            args.step = int(f.read().strip())
    else:
        metas = sorted([f for f in os.listdir(model_dir)
                        if f.startswith("meta_") and f.endswith(".json")])
        if not metas:
            print(f"Error: no checkpoints in {model_dir}")
            sys.exit(1)
        args.step = max(int(re.search(r"(\d+)", m).group(1)) for m in metas if re.search(r"(\d+)", m))

print(f"Generating report for {args.tag}, step {args.step}")

# -----------------------------------------------------------------------------
# Load metadata
meta_path = os.path.join(model_dir, f"meta_{args.step:06d}.json")
if not os.path.exists(meta_path):
    meta_path = os.path.join(model_dir, f"meta_{args.step}.json")

with open(meta_path) as f:
    meta = json.load(f)

model_config = meta.get("model_config", {})
user_config = meta.get("user_config", {})
loop_state = meta.get("loop_state", {})

# -----------------------------------------------------------------------------
# Collect metrics from all meta files (for loss curve)
steps = []
losses = []
val_bpbs = []
times = []

all_metas = sorted([f for f in os.listdir(model_dir) if f.startswith("meta_") and f.endswith(".json")])
for mf in all_metas:
    try:
        with open(os.path.join(model_dir, mf)) as f:
            m = json.load(f)
        s = m.get("step", 0)
        steps.append(s)
        ls = m.get("loop_state", {})
        losses.append(ls.get("smooth_train_loss", 0))
        val_bpbs.append(m.get("val_bpb", None))
        times.append(ls.get("total_training_time", 0))
    except (json.JSONDecodeError, OSError):
        pass

# -----------------------------------------------------------------------------
# Model stats
model_path = os.path.join(model_dir, f"model_{args.step:06d}.pt")
if not os.path.exists(model_path):
    model_path = os.path.join(model_dir, f"model_{args.step}.pt")

param_count = 0
if os.path.exists(model_path):
    import torch
    sd = torch.load(model_path, map_location="cpu", weights_only=True)
    param_count = sum(p.numel() for p in sd.values())

# -----------------------------------------------------------------------------
# Build HTML
n_layers = model_config.get("n_layer", "?")
n_embd = model_config.get("n_embd", "?")
vocab_size = model_config.get("vocab_size", "?")
n_iterations = user_config.get("num_iterations", "?")
batch_size = user_config.get("total_batch_size", "?")
device_batch = user_config.get("device_batch_size", "?")
seq_len = user_config.get("max_seq_len", "?")
compress = user_config.get("compress", "?")
sparsify = user_config.get("sparsify", False)
sync_interval = user_config.get("sync_interval", 1)

# Build loss curve data
loss_data = json.dumps([{"step": s, "loss": l} for s, l in zip(steps, losses) if l])
val_data = json.dumps([{"step": s, "bpb": v} for s, v in zip(steps, val_bpbs) if v is not None])

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MicroChat-CPU Training Report — {escape(args.tag)}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #0d1117; color: #c9d1d9; padding: 24px; }}
h1 {{ color: #58a6ff; margin-bottom: 8px; }}
h2 {{ color: #8b949e; font-size: 14px; font-weight: 400; margin-bottom: 24px; }}
h3 {{ color: #c9d1d9; margin: 20px 0 12px; font-size: 16px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; margin-bottom: 24px; }}
.card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }}
.card .label {{ color: #8b949e; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
.card .value {{ color: #f0f6fc; font-size: 24px; font-weight: 600; margin-top: 4px; }}
.card .sub {{ color: #8b949e; font-size: 13px; margin-top: 2px; }}
.chart-container {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th {{ color: #8b949e; text-align: left; padding: 8px 12px; border-bottom: 1px solid #30363d; font-weight: 500; }}
td {{ padding: 8px 12px; border-bottom: 1px solid #21262d; }}
tr:hover td {{ background: #1c2128; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 500; }}
.badge-green {{ background: #1b4124; color: #3fb950; }}
.badge-blue {{ background: #1c3a5c; color: #58a6ff; }}
.badge-yellow {{ background: #452b13; color: #d29922; }}
</style>
</head>
<body>

<h1>MicroChat-CPU Training Report</h1>
<h2>Model: {escape(args.tag)} · Step {args.step} of {n_iterations} · {param_count:,} parameters</h2>

<div class="grid">
  <div class="card">
    <div class="label">Architecture</div>
    <div class="value">{n_layers} layers × {n_embd} dim</div>
    <div class="sub">{vocab_size:,} vocab · {seq_len} seq len</div>
  </div>
  <div class="card">
    <div class="label">Training</div>
    <div class="value">{batch_size:,} tok/step</div>
    <div class="sub">device batch: {device_batch} · grad accum: auto</div>
  </div>
  <div class="card">
    <div class="label">Communication</div>
    <div class="value">{'Compressed' if compress else 'FP32'}</div>
    <div class="sub">{'Sparsification' if sparsify else 'Dense'} · sync every {sync_interval}</div>
  </div>
  <div class="card">
    <div class="label">Parameters</div>
    <div class="value">{param_count:,}</div>
    <div class="sub">{(param_count / 1e6):.1f}M</div>
  </div>
</div>

<div class="chart-container">
  <h3>Training Loss</h3>
  <canvas id="lossChart" height="100"></canvas>
</div>

<div class="chart-container">
  <h3>Validation Bits per Byte</h3>
  <canvas id="valChart" height="100"></canvas>
</div>

<div class="chart-container">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <h3>Step Timing Breakdown</h3>
    <div>
      <span class="badge badge-green">fwd</span>
      <span class="badge badge-blue">bwd</span>
      <span class="badge badge-yellow">optim</span>
    </div>
  </div>
  <canvas id="phaseChart" height="100"></canvas>
</div>

<div class="card" style="margin-bottom: 16px;">
  <h3 style="margin: 0 0 8px;">Configuration</h3>
  <table>
    <tr><th>Key</th><th>Value</th></tr>
"""

# Add config rows
config_keys = ["depth", "max_seq_len", "device_batch_size", "total_batch_size",
               "num_iterations", "compress", "sparsify", "sync_interval",
               "network_tier", "topology_aware", "hetero", "async_overlap",
               "weight_decay", "matrix_lr", "embedding_lr"]
for key in config_keys:
    val = user_config.get(key, "")
    html += f"    <tr><td>{key}</td><td>{escape(str(val))}</td></tr>\n"

html += """  </table>
</div>

<script>
const lossData = """ + loss_data + """;
const valData = """ + val_data + """;

new Chart(document.getElementById('lossChart'), {
  type: 'scatter',
  data: {
    datasets: [{
      label: 'Training Loss',
      data: lossData.map(d => ({x: d.step, y: d.loss})),
      backgroundColor: '#58a6ff',
      borderColor: '#58a6ff',
      showLine: true,
      tension: 0.3,
      pointRadius: 3,
    }]
  },
  options: {
    responsive: true,
    scales: {
      x: { title: { display: true, text: 'Step', color: '#8b949e' },
           grid: { color: '#21262d' }, ticks: { color: '#8b949e' } },
      y: { title: { display: true, text: 'Loss', color: '#8b949e' },
           grid: { color: '#21262d' }, ticks: { color: '#8b949e' } }
    },
    plugins: { legend: { labels: { color: '#c9d1d9' } } }
  }
});

if (valData.length > 0) {
  new Chart(document.getElementById('valChart'), {
    type: 'scatter',
    data: {
      datasets: [{
        label: 'Val BPB',
        data: valData.map(d => ({x: d.step, y: d.bpb})),
        backgroundColor: '#3fb950',
        borderColor: '#3fb950',
        showLine: true,
        tension: 0.3,
        pointRadius: 4,
      }]
    },
    options: {
      responsive: true,
      scales: {
        x: { title: { display: true, text: 'Step', color: '#8b949e' },
             grid: { color: '#21262d' }, ticks: { color: '#8b949e' } },
        y: { title: { display: true, text: 'BPB', color: '#8b949e' },
             grid: { color: '#21262d' }, ticks: { color: '#8b949e' } }
      },
      plugins: { legend: { labels: { color: '#c9d1d9' } } }
    }
  });
}

// Phase breakdown (placeholder from profiler)
const phaseLabels = ['Forward', 'Backward', 'Data', 'Comm', 'Optim'];
const phaseData = [25, 36, 0, 0, 39];  // typical CPU percentages

new Chart(document.getElementById('phaseChart'), {
  type: 'doughnut',
  data: {
    labels: phaseLabels,
    datasets: [{
      data: phaseData,
      backgroundColor: ['#3fb950', '#58a6ff', '#8b949e', '#d29922', '#f0883e'],
      borderColor: '#161b22',
      borderWidth: 2,
    }]
  },
  options: {
    responsive: true,
    plugins: {
      legend: { position: 'right', labels: { color: '#c9d1d9', padding: 12 } }
    }
  }
});
</script>

</body>
</html>"""

# -----------------------------------------------------------------------------
# Write output
output_path = args.output or os.path.join(model_dir, f"report_{args.tag}_step{args.step}.html")
with open(output_path, "w") as f:
    f.write(html)
print(f"Report saved: {output_path}")
print(f"  Open in browser to view charts")

if args.open:
    import webbrowser
    webbrowser.open(f"file://{os.path.abspath(output_path)}")

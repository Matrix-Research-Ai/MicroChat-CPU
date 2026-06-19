"""
Chat with a MicroChat-CPU trained model.

Loads a base model checkpoint from cpu_checkpoints/ and runs
interactive text generation on CPU.

Usage:
  # Interactive chat (latest checkpoint):
  python -m scripts.cpu_chat

  # Single prompt:
  python -m scripts.cpu_chat -p "The capital of France is"

  # Load specific step:
  python -m scripts.cpu_chat -s 500

  # Load a specific model tag:
  python -m scripts.cpu_chat -g d4 -s 1000
"""

import argparse
import os
import json
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.engine import Engine
from nanochat.common import get_base_dir

# Lazy tokenizer import (needs rustbpe — install with 'uv sync --extra cpu')
_tokenizer = None
def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from nanochat.tokenizer import get_tokenizer as _gt
        _tokenizer = _gt()
    return _tokenizer

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Chat with a MicroChat-CPU model")
parser.add_argument("-g", "--model-tag", type=str, default=None,
                    help="Model tag (e.g. 'd2', 'd4'). Default: auto-detect latest")
parser.add_argument("-s", "--step", type=int, default=None,
                    help="Checkpoint step to load. Default: latest")
parser.add_argument("-p", "--prompt", type=str, default="",
                    help="Single prompt, get a response and exit")
parser.add_argument("-t", "--temperature", type=float, default=0.7,
                    help="Sampling temperature (default: 0.7)")
parser.add_argument("--top-k", type=int, default=50,
                    help="Top-k sampling (default: 50)")
parser.add_argument("--max-tokens", type=int, default=256,
                    help="Max tokens to generate (default: 256)")
parser.add_argument("--checkpoint-dir", type=str, default=None,
                    help="Override checkpoint directory")
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Find checkpoint directory
base_dir = get_base_dir()
if args.checkpoint_dir:
    ckpt_dir = args.checkpoint_dir
else:
    ckpt_dir = os.path.join(base_dir, "cpu_checkpoints")

if not os.path.exists(ckpt_dir):
    print(f"Error: checkpoint directory not found: {ckpt_dir}")
    print("Run training first: python -m scripts.cpu_train --quick")
    exit(1)

# Auto-detect model tag (latest subdirectory)
if args.model_tag is None:
    tags = sorted([d for d in os.listdir(ckpt_dir)
                   if os.path.isdir(os.path.join(ckpt_dir, d))])
    if not tags:
        print(f"Error: no model checkpoints found in {ckpt_dir}")
        exit(1)
    args.model_tag = tags[-1]
    print(f"Auto-detected model: {args.model_tag}")

model_dir = os.path.join(ckpt_dir, args.model_tag)
if not os.path.exists(model_dir):
    print(f"Error: model directory not found: {model_dir}")
    exit(1)

# Find the latest step if not specified
if args.step is None:
    # Check latest_step.txt first
    latest_file = os.path.join(model_dir, "latest_step.txt")
    if os.path.exists(latest_file):
        with open(latest_file) as f:
            args.step = int(f.read().strip())
    else:
        # Fall back to scanning meta files
        metas = sorted([f for f in os.listdir(model_dir)
                        if f.startswith("meta_") and f.endswith(".json")])
        if not metas:
            print(f"Error: no checkpoints found in {model_dir}")
            exit(1)
        # Extract step number from filename
        import re as _re
        steps = []
        for m in metas:
            match = _re.search(r"meta_(\d+)\.json", m)
            if match:
                steps.append(int(match.group(1)))
        args.step = max(steps) if steps else 0

print(f"Loading checkpoint: {args.model_tag}, step {args.step}")

# -----------------------------------------------------------------------------
# Load metadata
meta_path = os.path.join(model_dir, f"meta_{args.step:06d}.json")
if not os.path.exists(meta_path):
    # Try without zero-padding
    meta_path = os.path.join(model_dir, f"meta_{args.step}.json")
if not os.path.exists(meta_path):
    print(f"Error: metadata not found at {meta_path}")
    exit(1)

with open(meta_path) as f:
    meta = json.load(f)

model_config = meta.get("model_config", {})
print(f"  Model config: {model_config.get('n_layer', '?')} layers, "
      f"{model_config.get('n_embd', '?')} dim, "
      f"{model_config.get('vocab_size', '?')} vocab")

# -----------------------------------------------------------------------------
# Build model on CPU and load weights
device = torch.device("cpu")

config = GPTConfig(
    sequence_len=model_config.get("sequence_len", 512),
    vocab_size=model_config.get("vocab_size", 50304),
    n_layer=model_config.get("n_layer", 2),
    n_head=model_config.get("n_head", 1),
    n_kv_head=model_config.get("n_kv_head", 1),
    n_embd=model_config.get("n_embd", 128),
    window_pattern=model_config.get("window_pattern", "L"),
)

with torch.device("meta"):
    model_meta = GPT(config)

model = model_meta.to_empty(device=device)

# Load model weights
model_path = os.path.join(model_dir, f"model_{args.step:06d}.pt")
if not os.path.exists(model_path):
    model_path = os.path.join(model_dir, f"model_{args.step}.pt")

print(f"  Loading weights from: {os.path.basename(model_path)}")
state_dict = torch.load(model_path, map_location=device, weights_only=True)
model.load_state_dict(state_dict, strict=True, assign=True)
print(f"  ✓ Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

# -----------------------------------------------------------------------------
# Tokenizer
try:
    tokenizer = _get_tokenizer()
except ImportError as e:
    print(f"Error: tokenizer not available ({e}).")
    print("Run 'uv sync --extra cpu' to install dependencies")
    exit(1)

# -----------------------------------------------------------------------------
# Create Engine for efficient generation
engine = Engine(model, tokenizer)

# -----------------------------------------------------------------------------
# Generation function
def generate(prompt_text, temperature=args.temperature,
             top_k=args.top_k, max_tokens=args.max_tokens):
    tokens = tokenizer(prompt_text, prepend="<|bos|>")
    sample, _ = engine.generate_batch(
        tokens, num_samples=1, max_tokens=max_tokens,
        temperature=temperature, top_k=top_k,
    )
    return tokenizer.decode(sample[0])

# -----------------------------------------------------------------------------
# Single prompt mode
if args.prompt:
    print(f"\nPrompt: {args.prompt}")
    print("-" * 50)
    result = generate(args.prompt)
    print(result)
    print("-" * 50)
    exit(0)

# -----------------------------------------------------------------------------
# Interactive mode
print()
print("=" * 50)
print("MicroChat-CPU Interactive")
print(f"Model: {args.model_tag} (step {args.step})")
print("=" * 50)
print("Type 'quit' or 'exit' to end")
print("Type 'clear' to reset conversation context")
print("Press Ctrl+C to stop generation")
print("=" * 50)
print()

conversation = ""
while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break

    if user_input.lower() in ("quit", "exit"):
        break
    if user_input.lower() == "clear":
        conversation = ""
        print("(conversation cleared)")
        continue
    if not user_input:
        continue

    # For base models, use a simple prompt format
    prompt = conversation + f"User: {user_input}\nAssistant:"
    print("Assistant: ", end="", flush=True)

    try:
        result = generate(prompt, max_tokens=args.max_tokens)
        # Extract just the assistant's response
        response = result
        if "Assistant:" in response:
            response = response.split("Assistant:", 1)[-1].strip()
        # Also handle the BOS token
        response = response.replace("<|bos|>", "").strip()
        print(response)
        conversation += f"User: {user_input}\nAssistant: {response}\n"
    except KeyboardInterrupt:
        print("\n(interrupted)")
        continue

print("\nGoodbye!")

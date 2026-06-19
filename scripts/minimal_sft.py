"""Minimal SFT training — encyclopedia dataset only, no internet needed."""
import os, json, sys, torch, gc, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes
from nanochat.common import get_base_dir, print0

device = torch.device("cpu")
base_dir = get_base_dir()

# Load checkpoint
ckpt_dir = os.path.join(base_dir, "base_checkpoints", "d6")
model_data, _, meta = load_checkpoint(ckpt_dir, step=200, device=device, load_optimizer=False)
cfg = GPTConfig(**meta["model_config"])
model = GPT(cfg)
model.load_state_dict(model_data)
model.train()
print0("Model loaded")

# Load tokenizer
tok = RustBPETokenizer.from_directory(os.path.join(base_dir, "tokenizer"))
tokenizer = tok
print0("Tokenizer loaded")

# Load SFT data
sft_path = r"C:\Users\Admin\epub2dataset_data\encyclopedia_sft.jsonl"
with open(sft_path, encoding="utf-8") as f:
    raw = [json.loads(l) for l in f if l.strip()]
    # Wrap in {"messages": ...} format for render_conversation
    conversations = [{"messages": c} if isinstance(c, list) else c for c in raw]
print0(f"Loaded {len(conversations)} conversations")

# Optimizer
optimizer = model.setup_optimizer(unembedding_lr=0.008, embedding_lr=0.3, matrix_lr=0.02, weight_decay=0.0)
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * 0.8

# Training loop
max_seq_len = 822
bos = tokenizer.get_bos_token_id()
num_iters = 20
step = 0
for epoch in range(5):
    import random
    indices = list(range(len(conversations)))
    random.shuffle(indices)
    for idx in indices:
        if step >= num_iters:
            break
        conv = conversations[idx]
        ids, mask = tokenizer.render_conversation(conv)
        if len(ids) > max_seq_len + 1:
            ids = ids[:max_seq_len + 1]
            mask = mask[:max_seq_len + 1]
        # Pad if too short
        if len(ids) < max_seq_len + 1:
            pad_len = max_seq_len + 1 - len(ids)
            ids = ids + [bos] * pad_len
            mask = mask + [0] * pad_len
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
        m = torch.tensor([mask[1:]], dtype=torch.int8, device=device)
        y[m == 0] = -1  # mask non-assistant tokens
        
        loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        step += 1
        if step % 5 == 0 or step == num_iters:
            print0(f"Step {step}/{num_iters} | loss: {loss.item():.4f}")

# Save checkpoint
out_dir = os.path.join(base_dir, "chatsft_checkpoints", "d6")
os.makedirs(out_dir, exist_ok=True)
save_checkpoint(out_dir, step, model.state_dict(), optimizer.state_dict(), {
    "step": step, "model_config": meta["model_config"]
}, rank=0)
print0(f"Saved to {out_dir}")

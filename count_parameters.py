#%%

import torch
from argparse import Namespace
from multipartite_utils import _load_transformer

ckpt_path = "outputs/checkpoints/multipartite_003/checkpoint_step_6000_best.pt"
raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
state_dict = raw.get("state_dict") or raw.get("model_state_dict")
if state_dict is None:
    raise RuntimeError(f"No state dict in {ckpt_path}")

vocab_size = state_dict["embed.W_E"].shape[0]  # 432 here
args = Namespace(
    d_model=128,
    n_heads=4,
    n_layers=3,
    n_ctx=16,
    d_vocab=vocab_size,
    act_fn="relu",
    d_head=32,
    model_ckpt=ckpt_path,
)

model, _ = _load_transformer(args, device="cpu", vocab_size=vocab_size)
trainable = sum(p.numel() for p in model.parameters())
buffers = sum(b.numel() for b in model.buffers())
print(f"Trainable parameters: {trainable:,}")
print(f"Buffer entries: {buffers:,}")
print(f"Entries in state_dict: {trainable + buffers:,}")


#%%
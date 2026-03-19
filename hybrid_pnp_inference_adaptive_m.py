import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from attn import (
    replace_attention_layers_dropin_og,
    landmark_pool,
    moore_penrose_iter_pinv
)

# ----------------------------
# Settings
# ----------------------------
IMAGE_PATH   = "test4.jpg"
TOP_K        = 5
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
M            = 64       # PnP landmark threshold
THRESHOLD    = 64       # if rank < THRESHOLD → use PnP for that head

# ----------------------------
# ImageNet label helper
# ----------------------------
def get_top_k(output, k=5):
    probs = torch.softmax(output, dim=1)
    top_probs, top_indices = torch.topk(probs, k)
    return top_probs.cpu().numpy()[0], top_indices.cpu().numpy()[0]

def print_predictions(label, top_probs, top_indices, info):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    for i in range(TOP_K):
        class_idx   = int(top_indices[i])
        description = info.index_to_description(class_idx)
        probability = float(top_probs[i]) * 100
        print(f"  {i+1}. {description}")
        print(f"     → Probability: {probability:.2f}%")

# ----------------------------
# Load image + transform
# ----------------------------
base_model = timm.create_model('vit_base_patch16_224', pretrained=True)
base_model.eval()
config    = resolve_data_config({}, model=base_model)
transform = create_transform(**config)

img          = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(DEVICE)

from timm.data import ImageNetInfo, infer_imagenet_subset
subset = infer_imagenet_subset(base_model)
info   = ImageNetInfo(subset)

# ============================================================
# STEP 1: Run softmax dropin on single image → 12×12 rank matrix
# ============================================================
print("\n>>> STEP 1: Computing 12x12 rank matrix from softmax dropin...")

softmax_model = timm.create_model('vit_base_patch16_224', pretrained=True)
softmax_model.eval().to(DEVICE)
softmax_model = replace_attention_layers_dropin_og(softmax_model, layers_to_replace=list(range(12)))

with torch.no_grad():
    _ = softmax_model(input_tensor)

# Build 12x12 rank matrix
rank_matrix = np.zeros((12, 12), dtype=int)  # [layer, head]

for i, block in enumerate(softmax_model.blocks):
    attn_probs = block.attn.attention_probs  # (1, 12, 197, 197)
    for h in range(attn_probs.shape[1]):
        A      = attn_probs[0, h]  # (197, 197)
        S      = torch.linalg.svdvals(A.float())
        S      = S / S.sum()
        cumsum = torch.cumsum(S, dim=0)
        rank_matrix[i][h] = (cumsum < 0.95).sum().item() + 1

print("\n12x12 Rank Matrix (Layer x Head) at 95% Energy:")
print(f"{'Layer':<6} " + " ".join([f"H{h:<5}" for h in range(12)]))
print("-" * 85)
for i in range(12):
    row = f"L{i:<5} " + " ".join([f"{rank_matrix[i][h]:<6}" for h in range(12)])
    print(row)

# ============================================================
# STEP 2: Softmax model inference → top5
# ============================================================
print("\n>>> STEP 2: Softmax model inference...")

with torch.no_grad():
    softmax_output = softmax_model(input_tensor)

softmax_probs, softmax_indices = get_top_k(softmax_output)
print_predictions("Softmax Model - Top 5 Predictions", softmax_probs, softmax_indices, info)

# ============================================================
# STEP 3: Build Hybrid Attention Module
# ============================================================

class HybridHeadAttention(nn.Module):
    """
    Per-head hybrid attention with adaptive M:
    - If rank[head] < THRESHOLD → use PnP Nystrom with M = rank[head] landmarks
    - If rank[head] >= THRESHOLD → use original softmax for that head
    """
    def __init__(self, dim, num_heads, head_ranks, pinv_iters=6):
        super().__init__()
        self.num_heads  = num_heads
        self.head_dim   = dim // num_heads
        self.scale      = self.head_dim ** -0.5
        self.pinv_iters = pinv_iters
        self.head_ranks = head_ranks  # list of 12 rank values for this layer

        self.qkv  = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (B, H, N, Dh)
        q = q * self.scale

        head_outputs = []

        for h in range(self.num_heads):
            q_h = q[:, h:h+1, :, :]  # (B, 1, N, Dh)
            k_h = k[:, h:h+1, :, :]
            v_h = v[:, h:h+1, :, :]

            rank = self.head_ranks[h]

            if rank < THRESHOLD:
                # --- PnP Nystrom for this head, M = rank of this head ---
                num_landmarks = max(1, rank)  # use head's own rank as M
                q_m = landmark_pool(q_h, num_landmarks)
                k_m = landmark_pool(k_h, num_landmarks)

                S_A = q_h @ k_m.transpose(-2, -1)
                S_C = q_m @ k_h.transpose(-2, -1)
                S_B = q_m @ k_m.transpose(-2, -1)

                expA = torch.exp(S_A - S_A.amax(dim=-1, keepdim=True))
                expC = torch.exp(S_C - S_C.amax(dim=-1, keepdim=True))
                expB = torch.exp((S_B - S_C.amax(dim=-1, keepdim=True)).clamp(min=-88.0))

                v_aug      = torch.cat([v_h, torch.ones_like(v_h[..., :1])], dim=-1)
                pseudo_inv = moore_penrose_iter_pinv(expB, self.pinv_iters)
                prod       = expA @ pseudo_inv @ (expC @ v_aug)
                out_h      = prod[..., :-1] / prod[..., -1:].clamp(min=1e-8)  # (B, 1, N, Dh)

            else:
                # --- Original softmax for this head ---
                scores    = q_h @ k_h.transpose(-2, -1)  # (B, 1, N, N)
                attn      = scores.softmax(dim=-1)
                out_h     = attn @ v_h                    # (B, 1, N, Dh)

            head_outputs.append(out_h)

        # Concatenate all heads: (B, H, N, Dh) → (B, N, C)
        context_layer = torch.cat(head_outputs, dim=1)  # (B, H, N, Dh)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        return self.proj(context_layer)


# ============================================================
# STEP 4: Build hybrid model and run inference
# ============================================================
print("\n>>> STEP 3: Building hybrid model...")

hybrid_model = timm.create_model('vit_base_patch16_224', pretrained=True)
hybrid_model.eval().to(DEVICE)

# Print which heads use PnP vs softmax
print("\nHead assignment (P=PnP, S=Softmax):")
print(f"{'Layer':<6} " + " ".join([f"H{h:<4}" for h in range(12)]))
print("-" * 75)

for i, block in enumerate(hybrid_model.blocks):
    old_attn    = block.attn
    dim         = old_attn.qkv.weight.shape[1]
    num_heads   = old_attn.num_heads
    head_ranks  = [int(rank_matrix[i][h]) for h in range(num_heads)]

    new_attn = HybridHeadAttention(
        dim=dim,
        num_heads=num_heads,
        head_ranks=head_ranks,
    )
    # Copy pretrained weights
    new_attn.qkv.weight.data  = old_attn.qkv.weight.data.clone()
    new_attn.qkv.bias.data    = old_attn.qkv.bias.data.clone()
    new_attn.proj.weight.data = old_attn.proj.weight.data.clone()
    new_attn.proj.bias.data   = old_attn.proj.bias.data.clone()
    block.attn = new_attn

    # Print assignment row — show M value for PnP heads
    assignment = [f"P({rank_matrix[i][h]})" if rank_matrix[i][h] < THRESHOLD else "S" for h in range(12)]
    pnp_count  = sum(1 for h in range(12) if rank_matrix[i][h] < THRESHOLD)
    row = f"L{i:<5} " + " ".join([f"{a:<8}" for a in assignment]) + f"  ({pnp_count}/12 PnP)"
    print(row)

print("\n>>> STEP 4: Hybrid model inference...")

with torch.no_grad():
    hybrid_output = hybrid_model(input_tensor)

hybrid_probs, hybrid_indices = get_top_k(hybrid_output)
print_predictions("Hybrid Adaptive-M Model - Top 5 Predictions", hybrid_probs, hybrid_indices, info)

# ============================================================
# Summary comparison
# ============================================================
print("\n" + "="*50)
print("  SUMMARY COMPARISON")
print("="*50)
print(f"\n{'Rank':<6} {'Softmax':^30} {'Hybrid Adaptive-M':^30}")
print("-"*66)
for i in range(TOP_K):
    s_desc = info.index_to_description(int(softmax_indices[i]))[:25]
    h_desc = info.index_to_description(int(hybrid_indices[i]))[:25]
    s_prob = float(softmax_probs[i]) * 100
    h_prob = float(hybrid_probs[i]) * 100
    print(f"{i+1:<6} {s_desc+f' ({s_prob:.1f}%)':^30} {h_desc+f' ({h_prob:.1f}%)':^30}")

print("\n>>> Done!")

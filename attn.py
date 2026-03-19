import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional
from einops import rearrange
from transformers.models.vit.modeling_vit import ViTSelfAttention


# ---------------------------------------------
# Helper: find closest square factors of N
# ---------------------------------------------
def closest_square_factors(N):
    s = int(math.isqrt(N))
    for i in range(s, 0, -1):
        if N % i == 0:
            return i, N // i


# ---------------------------------------------
# Helper: landmark pooling (keeps CLS token)
# ---------------------------------------------
def landmark_pool(q, num_landmarks):
    B, H, N, Dh = q.shape
    w1, w2 = closest_square_factors(N - 1)  # exclude CLS
    h, w = closest_square_factors(num_landmarks)

    q_cls  = q[:, :, :1, :]   # (B, H, 1,   Dh)
    q_rest = q[:, :, 1:, :]   # (B, H, N-1, Dh)

    q_m = F.adaptive_avg_pool2d(
        q_rest.reshape(B * H, w1, w2, Dh).permute(0, 3, 1, 2),
        output_size=(h, w)
    ).permute(0, 2, 3, 1).reshape(B, H, num_landmarks, Dh)

    return torch.cat([q_cls, q_m], dim=2)  # (B, H, 1+num_landmarks, Dh)


# ---------------------------------------------
# Helper: iterative Moore-Penrose pseudo-inverse
# ---------------------------------------------
def moore_penrose_iter_pinv(x, iters=6):
    abs_x = torch.abs(x)
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row) + 1e-15)

    I = torch.eye(x.shape[-1], device=x.device)
    I = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))
    return z


# ---------------------------------------------
# PnP Nystrom Attention (Stable)
# ---------------------------------------------
class PnPNystromAttention_Sir_Stable(ViTSelfAttention):
    def __init__(self, config, num_landmarks=64, pinv_iters=6):
        super().__init__(config)
        self.num_landmarks = num_landmarks
        self.pinv_iters = pinv_iters

    def forward(
        self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden_states.shape[0]
        new_shape = batch_size, -1, self.num_attention_heads, self.attention_head_size

        k = self.key(hidden_states).view(*new_shape).transpose(1, 2)
        v = self.value(hidden_states).view(*new_shape).transpose(1, 2)
        q = self.query(hidden_states).view(*new_shape).transpose(1, 2) * self.scaling

        q_m = landmark_pool(q, self.num_landmarks)
        k_m = landmark_pool(k, self.num_landmarks)

        S_A = q  @ k_m.transpose(-2, -1)
        S_C = q_m @ k.transpose(-2, -1)
        S_B = q_m @ k_m.transpose(-2, -1)

        expA = torch.exp(S_A - S_A.amax(dim=-1, keepdim=True))
        expC = torch.exp(S_C - S_C.amax(dim=-1, keepdim=True))
        expB = torch.exp((S_B - S_C.amax(dim=-1, keepdim=True)).clamp(min=-88.0))

        v_aug = torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)
        pseudo_inv = moore_penrose_iter_pinv(expB, self.pinv_iters)

        prod = expA @ pseudo_inv @ (expC @ v_aug)
        context_layer = prod[..., :-1] / prod[..., -1:].clamp(min=1e-8)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.all_head_size)
        return context_layer, None


# ---------------------------------------------
# Layer replacement utility
# ---------------------------------------------
def replace_attention_layers(model, layers_to_replace, num_landmarks=64, pinv_iters=6):
    """
    Replace specific ViT self-attention layers with PnPNystromAttention_Sir_Stable.
    Supports both timm (model.blocks) and HuggingFace (model.vit.encoder.layer) ViT models.

    Args:
        model:              timm or HuggingFace ViT model
        layers_to_replace:  list of layer indices to replace, e.g. [6, 7, 8, 9, 10, 11]
        num_landmarks:      number of landmarks for Nystrom approximation
        pinv_iters:         iterations for Moore-Penrose pseudo-inverse

    Returns:
        model with replaced layers
    """
    # Detect model type
    is_timm = hasattr(model, 'blocks')
    is_hf   = hasattr(model, 'vit')

    if is_timm:
        # timm ViT: model.blocks[i].attn
        for i, block in enumerate(model.blocks):
            if i in layers_to_replace:
                old_attn = block.attn

                # Build a minimal config-like object timm attn has no HF config
                # Instead, directly build PnP using timm attn dimensions
                dim       = old_attn.qkv.weight.shape[1]  # head_dim * num_heads
                num_heads = old_attn.num_heads

                new_attn = TimmPnPNystromAttention(
                    dim=dim,
                    num_heads=num_heads,
                    num_landmarks=num_landmarks,
                    pinv_iters=pinv_iters,
                )
                # Copy pretrained QKV and proj weights
                new_attn.qkv.weight.data  = old_attn.qkv.weight.data.clone()
                new_attn.qkv.bias.data    = old_attn.qkv.bias.data.clone()
                new_attn.proj.weight.data = old_attn.proj.weight.data.clone()
                new_attn.proj.bias.data   = old_attn.proj.bias.data.clone()
                block.attn = new_attn
                print(f"  Layer {i:2d} ? TimmPnPNystromAttention")

    elif is_hf:
        # HuggingFace ViT: model.vit.encoder.layer[i].attention.attention
        for i, layer in enumerate(model.vit.encoder.layer):
            if i in layers_to_replace:
                old_attn = layer.attention.attention
                new_attn = PnPNystromAttention_Sir_Stable(
                    config=model.config,
                    num_landmarks=num_landmarks,
                    pinv_iters=pinv_iters,
                )
                new_attn.query.weight.data = old_attn.query.weight.data.clone()
                new_attn.query.bias.data   = old_attn.query.bias.data.clone()
                new_attn.key.weight.data   = old_attn.key.weight.data.clone()
                new_attn.key.bias.data     = old_attn.key.bias.data.clone()
                new_attn.value.weight.data = old_attn.value.weight.data.clone()
                new_attn.value.bias.data   = old_attn.value.bias.data.clone()
                layer.attention.attention  = new_attn
                print(f"  Layer {i:2d} ? PnPNystromAttention_Sir_Stable")
    else:
        raise ValueError("Model is neither a timm nor HuggingFace ViT - cannot replace layers.")

    return model


# ---------------------------------------------
# Timm-compatible PnP Nystrom Attention
# ---------------------------------------------
class TimmPnPNystromAttention(nn.Module):
    def __init__(self, dim, num_heads, num_landmarks=64, pinv_iters=6):
        super().__init__()
        self.num_heads    = num_heads
        self.head_dim     = dim // num_heads
        self.scale        = self.head_dim ** -0.5
        self.num_landmarks = num_landmarks
        self.pinv_iters   = pinv_iters

        self.qkv  = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (B, H, N, Dh)
        q = q * self.scale

        q_m = landmark_pool(q, self.num_landmarks)
        k_m = landmark_pool(k, self.num_landmarks)

        S_A = q  @ k_m.transpose(-2, -1)
        S_C = q_m @ k.transpose(-2, -1)
        S_B = q_m @ k_m.transpose(-2, -1)

        expA = torch.exp(S_A - S_A.amax(dim=-1, keepdim=True))
        expC = torch.exp(S_C - S_C.amax(dim=-1, keepdim=True))
        expB = torch.exp((S_B - S_C.amax(dim=-1, keepdim=True)).clamp(min=-88.0))

        v_aug     = torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)
        pseudo_inv = moore_penrose_iter_pinv(expB, self.pinv_iters)

        prod          = expA @ pseudo_inv @ (expC @ v_aug)
        context_layer = prod[..., :-1] / prod[..., -1:].clamp(min=1e-8)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        return self.proj(context_layer)
        
# ---------------------------------------------
# Timm-compatible Original Attention
# ---------------------------------------------
'''
class TimmAttentionDropinOG(nn.Module):
    """
    Timm-specific drop-in of OriginalAttention logic.
    Mirrors OriginalAttention (HF) but uses timm's fused qkv weight format.
    Sanity check - should give same accuracy as baseline.
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv  = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query_layer, key_layer, value_layer = qkv.unbind(0)
        query_layer = query_layer * self.scale

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-2, -1))
        attention_probs  = attention_scores.softmax(dim=-1)
        context_layer    = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        return self.proj(context_layer)
'''

class TimmAttentionDropinOG(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv  = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attention_probs = None  # store here

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query_layer, key_layer, value_layer = qkv.unbind(0)
        query_layer = query_layer * self.scale

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-2, -1))
        attention_probs  = attention_scores.softmax(dim=-1)
        
        #find rank here itself
        self.attention_probs = attention_probs.detach()  # save it!
        #self.attention_exp = torch.exp(attention_scores).detach()  # ADD THIS - exp only  #rank for this
        #attn_exp = torch.exp(attention_scores)
        #self.attention_exp = (attn_exp / attn_exp.sum(dim=-1, keepdim=True)).detach() #rank for this
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        return self.proj(context_layer)

def replace_attention_layers_dropin_og(model, layers_to_replace):
    """
    Replace specific ViT layers with TimmAttentionDropinOG.
    Mirrors OriginalAttention logic in timm format.
    """
    if not hasattr(model, 'blocks'):
        raise ValueError("Only timm models supported.")

    for i, block in enumerate(model.blocks):
        if i in layers_to_replace:
            old_attn = block.attn
            dim       = old_attn.qkv.weight.shape[1]
            num_heads = old_attn.num_heads

            new_attn = TimmAttentionDropinOG(dim=dim, num_heads=num_heads)
            new_attn.qkv.weight.data  = old_attn.qkv.weight.data.clone()
            new_attn.qkv.bias.data    = old_attn.qkv.bias.data.clone()
            new_attn.proj.weight.data = old_attn.proj.weight.data.clone()
            new_attn.proj.bias.data   = old_attn.proj.bias.data.clone()
            block.attn = new_attn
            print(f"  Layer {i:2d} -> TimmAttentionDropinOG")

    return model
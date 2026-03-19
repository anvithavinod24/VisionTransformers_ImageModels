import torch
import timm
import numpy as np

model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.eval()

print(f"{'Layer':<8} {'Matrix':<12} {'Full Rank':<12} {'Eff Rank (99%)':<16} {'Eff Rank (95%)'}")
print("-" * 65)

for i, block in enumerate(model.blocks):
    # Get QKV weight matrix
    W = block.attn.qkv.weight.data  # shape: (3*dim, dim)
    
    # Split into Q, K, V
    dim = W.shape[1]
    W_q, W_k, W_v = W[:dim], W[dim:2*dim], W[2*dim:]
    
    for name, mat in [('Q', W_q), ('K', W_k), ('V', W_v)]:
        S = torch.linalg.svdvals(mat.float())
        S = S / S.sum()
        cumsum = torch.cumsum(S, dim=0)
        rank_99 = (cumsum < 0.99).sum().item() + 1
        rank_95 = (cumsum < 0.95).sum().item() + 1
        print(f"L{i:<7} {name:<12} {dim:<12} {rank_99:<16} {rank_95}")

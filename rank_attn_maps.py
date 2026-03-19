import torch
import timm
import numpy as np
from attn import replace_attention_layers_dropin_og
from timm.data import create_dataset, create_loader, resolve_data_config



model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.eval()
model = model.cuda()  # ADD THIS

# Replace all layers with dropin so we can capture attention maps
model = replace_attention_layers_dropin_og(model, layers_to_replace=list(range(12)))

# Load real ImageNet validation images
data_config = resolve_data_config({}, model=model)
dataset = create_dataset(root='/home/guest/Anvitha/datasets', name='', split='val')
loader = create_loader(dataset, input_size=data_config['input_size'], batch_size=8)

# Accumulate ranks across multiple batches
num_batches = 10
layer_ranks_99 = {i: [] for i in range(12)}
layer_ranks_95 = {i: [] for i in range(12)}

for batch_idx, (images, _) in enumerate(loader):
    if batch_idx >= num_batches:
        break

    images = images.cuda()  # ADD THIS   
     
    with torch.no_grad():
        _ = model(images)

    # Extract attention maps from each layer
    for i, block in enumerate(model.blocks):
        attn_probs = block.attn.attention_probs  # (B, H, N, N)
        for h in range(attn_probs.shape[1]):
            for b in range(attn_probs.shape[0]):
                A = attn_probs[b, h]  # (N, N)
                S = torch.linalg.svdvals(A.float())
                S = S / S.sum()
                cumsum = torch.cumsum(S, dim=0)
                layer_ranks_99[i].append((cumsum < 0.99).sum().item() + 1)
                layer_ranks_95[i].append((cumsum < 0.95).sum().item() + 1)

# Print results
print(f"{'Layer':<8} {'Avg Rank (99%)':<18} {'Avg Rank (95%)'}")
print("-" * 45)
for i in range(12):
    print(f"L{i:<7} {np.mean(layer_ranks_99[i]):<18.1f} {np.mean(layer_ranks_95[i]):.1f}")
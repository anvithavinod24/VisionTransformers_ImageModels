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
layer_head_ranks_99 = {i: {h: [] for h in range(12)} for i in range(12)}
layer_head_ranks_95 = {i: {h: [] for h in range(12)} for i in range(12)}

for batch_idx, (images, _) in enumerate(loader):
    if batch_idx >= num_batches:
        break

    images = images.cuda()  # ADD THIS   
     
    with torch.no_grad():
        _ = model(images)

    # Extract attention maps from each layer
    for i, block in enumerate(model.blocks):
        attn_probs = block.attn.attention_probs  # (B, H, N, N)
        #attn_probs = block.attn.attention_exp
        for h in range(attn_probs.shape[1]):
            for b in range(attn_probs.shape[0]):
                A = attn_probs[b, h]  # (N, N)
                S = torch.linalg.svdvals(A.float())
                S = S / S.sum()
                cumsum = torch.cumsum(S, dim=0)
                layer_head_ranks_99[i][h].append((cumsum < 0.99).sum().item() + 1)
                layer_head_ranks_95[i][h].append((cumsum < 0.95).sum().item() + 1)

print(f"Batch {batch_idx} done")


# Print results
print("Effective Rank at 99% Energy - Per Head")
print(f"{'Layer':<6} " + " ".join([f"H{h:<6}" for h in range(12)]) + " Mean")
print("-" * 105)
for i in range(12):
    head_means = [np.mean(layer_head_ranks_99[i][h]) for h in range(12)]
    row = f"L{i:<5} " + " ".join([f"{v:<7.1f}" for v in head_means])
    row += f"  {np.mean(head_means):.1f}"
    print(row)

print("\nEffective Rank at 95% Energy - Per Head")
print(f"{'Layer':<6} " + " ".join([f"H{h:<6}" for h in range(12)]) + " Mean")
print("-" * 105)
for i in range(12):
    head_means = [np.mean(layer_head_ranks_95[i][h]) for h in range(12)]
    row = f"L{i:<5} " + " ".join([f"{v:<7.1f}" for v in head_means])
    row += f"  {np.mean(head_means):.1f}"
    print(row)

  
print("\nStd Dev of Head Ranks at 95% Energy - Per Layer")
print(f"{'Layer':<6} {'Std Dev':<12} {'Min Head':<12} {'Max Head':<12} {'Range'}")
print("-" * 55)
for i in range(12):
    head_means = [np.mean(layer_head_ranks_95[i][h]) for h in range(12)]
    print(f"L{i:<5} {np.std(head_means):<12.1f} {np.min(head_means):<12.1f} {np.max(head_means):<12.1f} {np.max(head_means)-np.min(head_means):.1f}")


# -- NEW: Per image rank stability analysis --
selected_layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
image_ranks = {i: [] for i in selected_layers}

for batch_idx, (images, _) in enumerate(loader):
    if batch_idx >= num_batches:
        break
    images = images.cuda()
    with torch.no_grad():
        _ = model(images)

    for i in selected_layers:
        block = model.blocks[i]
        attn_probs = block.attn.attention_probs
        #attn_probs = block.attn.attention_exp
        for b in range(attn_probs.shape[0]):
            head_ranks = []
            for h in range(attn_probs.shape[1]):
                A = attn_probs[b, h]
                S = torch.linalg.svdvals(A.float())
                S = S / S.sum()
                cumsum = torch.cumsum(S, dim=0)
                head_ranks.append((cumsum < 0.95).sum().item() + 1)
            image_ranks[i].append(np.mean(head_ranks))

print(f"Batch {batch_idx} done")
print("\nRank Stability Across Images (95% Energy, Avg across heads)")
print(f"{'Layer':<6} {'Mean':<10} {'Std Dev':<10} {'Min':<10} {'Max':<10} {'CV(%)'}")
print("-" * 55)
for i in selected_layers:
    ranks = image_ranks[i]
    mean = np.mean(ranks)
    std  = np.std(ranks)
    cv   = (std / mean) * 100
    print(f"L{i:<5} {mean:<10.1f} {std:<10.1f} {np.min(ranks):<10.1f} {np.max(ranks):<10.1f} {cv:.1f}%")
    
    # Per image - std dev across heads
print("\nHead Rank Std Dev Stability Across Images (95% Energy)")
print(f"{'Layer':<6} {'Mean SD':<12} {'SD of SD':<12} {'Min SD':<12} {'Max SD'}")
print("-" * 55)

image_head_stds = {i: [] for i in selected_layers}

for batch_idx, (images, _) in enumerate(loader):
    if batch_idx >= num_batches:
        break
    images = images.cuda()
    with torch.no_grad():
        _ = model(images)

    for i in selected_layers:
        block = model.blocks[i]
        attn_probs = block.attn.attention_probs
        #attn_probs = block.attn.attention_exp
        for b in range(attn_probs.shape[0]):
            head_ranks = []
            for h in range(attn_probs.shape[1]):
                A = attn_probs[b, h]
                S = torch.linalg.svdvals(A.float())
                S = S / S.sum()
                cumsum = torch.cumsum(S, dim=0)
                head_ranks.append((cumsum < 0.95).sum().item() + 1)
            image_head_stds[i].append(np.std(head_ranks))  # std across heads for this image
print(f"Batch {batch_idx} done")

for i in selected_layers:
    stds = image_head_stds[i]
    print(f"L{i:<5} {np.mean(stds):<12.1f} {np.std(stds):<12.1f} {np.min(stds):<12.1f} {np.max(stds):.1f}")


print("\nStd Dev of Rank at 95% Energy - Per Head (across 80 images)")
print(f"{'Layer':<6} " + " ".join([f"H{h:<6}" for h in range(12)]) + " Mean")
print("-" * 105)
for i in range(12):
    head_stds = [np.std(layer_head_ranks_95[i][h]) for h in range(12)]
    row = f"L{i:<5} " + " ".join([f"{v:<7.1f}" for v in head_stds])
    row += f"  {np.mean(head_stds):.1f}"
    print(row)
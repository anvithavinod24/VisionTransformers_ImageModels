import torch
import timm
from PIL import Image
import numpy as np
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# ----------------------------
# Settings
# ----------------------------
IMAGE_PATH = "test.jpg"   # <-- change this
TOP_K = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Load Model
# ----------------------------
model = timm.create_model(
    "vit_base_patch16_224",
    pretrained=True
)
model.eval()
model.to(DEVICE)

# ----------------------------
# Create Transform
# ----------------------------
config = resolve_data_config({}, model=model)
transform = create_transform(**config)

# ----------------------------
# Load and Preprocess Image
# ----------------------------
img = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(DEVICE)

# ----------------------------
# Inference
# ----------------------------
with torch.no_grad():
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)

# ----------------------------
# Get Top-K
# ----------------------------
top_probs, top_indices = torch.topk(probs, TOP_K)

top_probs = top_probs.cpu().numpy()[0]
top_indices = top_indices.cpu().numpy()[0]

# ----------------------------
# Load ImageNet Labels
# ----------------------------
from timm.data import ImageNetInfo
from timm.data import infer_imagenet_subset

subset = infer_imagenet_subset(model)
info = ImageNetInfo(subset)

print("\n🔍 Top Predictions:\n")

for i in range(TOP_K):
    class_idx = int(top_indices[i])
    class_name = info.index_to_label_name(class_idx)
    description = info.index_to_description(class_idx)
    probability = float(top_probs[i]) * 100

    print(f"{i+1}. {description} ({class_name})")
    print(f"   → Probability: {probability:.2f}%\n")

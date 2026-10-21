# -*- coding: utf-8 -*-
import os
import clip
import torch
from torchvision.datasets import CIFAR100
print(clip.available_models())

TEXT_FEATURES_PATH = "cifar100_text_features.pt"
def load_or_compute_text_features(model, dataset_classes, device):
    """加载预计算的文本特征，如果不存在则重新计算并保存"""
    if os.path.exists(TEXT_FEATURES_PATH):
        print(f"加载缓存的文本特征: {TEXT_FEATURES_PATH}")
        return torch.load(TEXT_FEATURES_PATH)

    print("文本特征缓存不存在，开始计算...")
    # Prepare the inputs
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in dataset_classes]).to(device)

    # Calculate features
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)

    # norm
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # save
    torch.save(text_features, TEXT_FEATURES_PATH)
    print(f"文本特征已保存至: {TEXT_FEATURES_PATH}")

    return text_features
# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)


# load text features
text_features = load_or_compute_text_features(model, cifar100.classes, device)
# Prepare the inputs
image, class_id = cifar100[3637]
image_input = preprocess(image).unsqueeze(0).to(device)
# text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    # text_features = model.encode_text(text_inputs)

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
# text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(5)

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")
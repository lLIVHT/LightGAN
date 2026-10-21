
import os
import clip
import torch
import numpy as np
import torch
from torch import nn
import torchvision.transforms as transforms
import cv2 as cv2
import PIL.Image as Image
import torch.nn.functional as F

pos = "A high-quality photo with natural color, sharp details, and no degradation"
neg = "An underexposed or overexposed image with poor clarity, noise, or compression artifacts"
# Load the model
img = Image.open("E:\\lowlight_dataset\\NIQE\\VV\\testA\\P1010157.jpg").convert('RGB')
img1 = Image.open("D:\\File\\lowlight\\python\\hap_run\\ablation\\lr_net1_11_gammavgg_relu4_3_SELU_2DA_2fea_ls_1g1d_mirror1_20241121\\VV\\images\\P1010157_fake_B.png").convert('RGB')
img2 = Image.open("E:\\results\\URetinex\\VV\\P1010157.jpg").convert('RGB')
# img = Image.open("E:\\lowlight_dataset\\VE-LOL-L\\VE-LOL-L-Cap-Full\\VE-LOL-L-Cap-Low_test\\testA\\low00780.png").convert('RGB')
# img = Image.open("E:\\lowlight_dataset\\VE-LOL-L\\VE-LOL-L-Cap-Full\\VE-LOL-L-Cap-Normal_test\\normal00780.png").convert('RGB')
# img1 = Image.open("D:\\File\\lowlight\\python\\hap_run\\ablation\\lr_net1_11_gammavgg_relu4_3_SELU_2DA_2fea_ls_1g1d_mirror1_20241121\\VE-LOL-L-Cap-Low_test\\images\\low00780_fake_B.png").convert('RGB')
# img2 = Image.open("E:\\results\\retinexformer\\ve_lol\\low00780.png").convert('RGB')
tran = transforms.Compose([
    transforms.ToTensor(),]
)
image = tran(img)
image1 = tran(img1)
image2 = tran(img2)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)
# Prepare the inputs
clip_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                         std=(0.26862954, 0.26130258, 0.27577711))
])
image_input = clip_transform(image).unsqueeze(0).to(device)
image_input1 = clip_transform(image1).unsqueeze(0).to(device)
image_input2 = clip_transform(image2).unsqueeze(0).to(device)
# image_input = preprocess(image).unsqueeze(0).to(device)
# text_pos = clip.tokenize([pos]).to(device)
# text_neg = clip.tokenize([neg]).to(device)
# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    image_features1 = model.encode_image(image_input1)
    image_features2 = model.encode_image(image_input2)
    # text_pos = model.encode_text(text_pos)
    # text_neg = model.encode_text(text_neg)

image_features /= image_features.norm(dim=-1, keepdim=True)
image_features1 /= image_features1.norm(dim=-1, keepdim=True)
image_features2 /= image_features2.norm(dim=-1, keepdim=True)
# text_pos /= text_pos.norm(dim=-1, keepdim=True)
# text_neg /= text_neg.norm(dim=-1, keepdim=True)
sim1 = F.cosine_similarity(image_features, image_features1, dim=-1) #0.9409  #0.8374
sim2 = F.cosine_similarity(image_features, image_features2, dim=-1) #0.9595  #0.8525
print(sim1, sim2)
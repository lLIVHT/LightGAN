# -*- coding: utf-8 -*-
import os
import clip
# import faiss
from PIL import Image
from tqdm import tqdm
import torch

from sklearn.cluster import KMeans


def run_kmeans(features, n_clusters, use_gpu=False, n_iter=20, verbose=True):
    """
    使用scikit-learn或PyTorch运行K-means聚类

    参数:
        features: 输入特征张量或数组
        n_clusters: 聚类中心数量
        use_gpu: 是否使用GPU加速(仅PyTorch版本有效)
        n_iter: 最大迭代次数
        verbose: 是否打印详细信息
    """
    # 检查输入类型
    if isinstance(features, torch.Tensor):
        if use_gpu:
            # 使用PyTorch GPU版本
            device = torch.device("cuda")
            features_gpu = features.to(device)

            # 随机初始化中心
            indices = torch.randperm(features_gpu.size(0), device=device)[:n_clusters]
            centroids = features_gpu[indices].clone()

            for i in range(n_iter):
                # 计算距离
                distances = torch.cdist(features_gpu, centroids, p=2)
                labels = torch.argmin(distances, dim=1)

                # 更新中心
                new_centroids = torch.zeros_like(centroids)
                for j in range(n_clusters):
                    mask = (labels == j)
                    if mask.sum() > 0:
                        new_centroids[j] = features_gpu[mask].mean(0)
                    else:
                        # 处理空聚类
                        new_centroids[j] = features_gpu[torch.randint(0, features_gpu.size(0), (1,), device=device)]

                # 检查收敛
                if torch.allclose(centroids, new_centroids, atol=1e-6):
                    if verbose:
                        print(f"Converged at iteration {i + 1}/{n_iter}")
                    break

                centroids = new_centroids

                if verbose and (i + 1) % 5 == 0:
                    print(f"Iteration {i + 1}/{n_iter} complete")

            return centroids.cpu()
        else:
            # 转换为numpy数组
            features_np = features.cpu().numpy()
    else:
        features_np = features

    # 使用scikit-learn CPU版本
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=10,
        max_iter=n_iter,
        verbose=verbose,
        random_state=42
    )

    kmeans.fit(features_np)
    return torch.from_numpy(kmeans.cluster_centers_).float()
def build_clip_memory_bank(image_dir, output_path='clip_memory.pt', n_clusters=100, device='cuda'):
    # 1. 加载 CLIP 模型
    model, preprocess = clip.load("ViT-B/32", device=device)

    # 2. 加载图像 & 提取特征
    all_features = []
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('jpg', 'png'))]
    print(f"Processing {len(image_files)} images...")

    for img_path in tqdm(image_files):
        try:
            img = Image.open(img_path).convert('RGB')
            image_input = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.encode_image(image_input)
            # feat /= feat.norm(dim=-1, keepdim=True)
            all_features.append(feat.cpu())
        except:
            print(f"Failed: {img_path}")
            continue

    all_features = torch.cat(all_features, dim=0).numpy()  # [N, 512]
    print(f"Total features: {all_features.shape}")

    # 3. KMeans 聚类去冗余，保留语义中心
    print(f"Clustering into {n_clusters} centers with Faiss...")
    d = all_features.shape[1]
    # kmeans = faiss.Kmeans(d, n_clusters, niter=20, verbose=True, gpu=torch.cuda.is_available())
    # kmeans.train(all_features)
    # memory_bank = torch.from_numpy(kmeans.centroids).float()
    memory_bank = run_kmeans(all_features, n_clusters, use_gpu=torch.cuda.is_available())

    # 4. 保存为 .pt 文件
    torch.save(memory_bank, output_path)
    print(f"Memory bank saved to {output_path}, shape: {memory_bank.shape}")

build_clip_memory_bank(
    image_dir='normal',       # 输入图像文件夹
    output_path='clip_memory.pt',            # 输出 memory bank
    n_clusters=100,                          # 可调节语义压缩程度
    device='cuda'                            # CPU / CUDA
)
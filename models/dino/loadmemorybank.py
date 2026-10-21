import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoImageProcessor
from matplotlib.colors import ListedColormap
from performer_pytorch import SelfAttention, CrossAttention
from skimage import filters
import imagehash
from collections import Counter

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


# --------------------------
# 图像预处理工具
# --------------------------
def image_clarity_score(img_path):
    """计算图像清晰度（拉普拉斯方差）"""
    try:
        with Image.open(img_path).convert('L') as img:
            img_np = np.array(img)
            laplacian = filters.laplacian(img_np, ksize=3)
            return np.var(laplacian)
    except Exception as e:
        print(f"计算清晰度时出错，路径: {img_path}, 错误: {str(e)}")
        return 0.0


def filter_low_quality_images(image_paths, min_clarity=50):
    """过滤低清晰度图像"""
    if not image_paths:
        return []

    high_quality = []
    for path in image_paths:
        score = image_clarity_score(path)
        if score >= min_clarity:
            high_quality.append(path)
    print(f"清晰度筛选：{len(image_paths)} → {len(high_quality)} 张（阈值{min_clarity}）")
    return high_quality


def dhash(image, hash_size=8):
    """计算图像的dHash值用于去重"""
    image = ImageOps.grayscale(image)
    image = image.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(image)

    hash_bits = []
    for row in range(hash_size):
        for col in range(hash_size):
            hash_bits.append(pixels[row, col] < pixels[row, col + 1])

    return imagehash.hex_to_hash(''.join(['1' if b else '0' for b in hash_bits]))


def remove_duplicate_images(image_paths, hash_threshold=5):
    """使用dHash去除重复图像"""
    if not image_paths:
        return []

    hash_dict = {}
    unique_paths = []

    for path in tqdm(image_paths, desc="去重处理"):
        try:
            with Image.open(path).convert('RGB') as img:
                img_hash = dhash(img)

                duplicate = False
                for existing_hash in hash_dict.values():
                    if img_hash - existing_hash < hash_threshold:
                        duplicate = True
                        break

                if not duplicate:
                    hash_dict[path] = img_hash
                    unique_paths.append(path)
        except Exception as e:
            print(f"处理图像哈希时出错 {path}: {e}")
            continue

    print(f"去重完成：{len(image_paths)} → {len(unique_paths)} 张图像")
    return unique_paths


# --------------------------
# 特征优化与聚类工具
# --------------------------
def optimize_features(features, method='l2_norm'):
    """特征优化处理"""
    if method == 'l2_norm':
        return features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    elif method == 'standard':
        scaler = StandardScaler()
        return scaler.fit_transform(features)
    else:
        return features


def select_significant_patches(patch_features, top_k=10000):
    """筛选显著的patch特征（基于特征强度）"""
    if len(patch_features) == 0:
        return np.array([]), np.array([])

    patch_norms = np.linalg.norm(patch_features, axis=1)
    top_indices = np.argsort(-patch_norms)[:top_k]
    significant_patches = patch_features[top_indices]
    print(f"从{len(patch_features)}个patch中筛选出{len(significant_patches)}个显著特征")
    return significant_patches, top_indices  # 同时返回索引


def find_optimal_clusters(features, min_clusters=100, max_clusters=150, step=5):
    """寻找100-150之间的最优聚类数量"""
    if len(features) < min_clusters:
        raise ValueError(f"特征数量({len(features)})少于最小聚类数量({min_clusters})")

    best_k = min_clusters
    best_score = -np.inf
    wcss = []
    cluster_range = range(min_clusters, max_clusters + 1, step)

    print(f"在{min_clusters}-{max_clusters}范围内寻找最优聚类数量...")
    for k in tqdm(cluster_range, desc="评估聚类数量"):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(features)
        wcss.append(kmeans.inertia_)

        # 使用Calinski-Harabasz分数评估（数值越大越好）
        score = calinski_harabasz_score(features, labels)
        if score > best_score:
            best_score = score
            best_k = k

    # 绘制肘部图辅助验证
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, wcss, 'bo-')
    plt.xlabel('聚类数量')
    plt.ylabel('WCSS')
    plt.title(f'肘部法则 (最优聚类数: {best_k})')
    plt.grid(True)
    plt.savefig('cluster_evaluation_elbow.png', dpi=300)
    plt.close()

    print(f"最优聚类数量: {best_k} (Calinski-Harabasz分数: {best_score:.2f})")
    return best_k


# --------------------------
# Patch+语义特征融合Memory Bank模型
# --------------------------
class PatchSemanticMemoryBank(nn.Module):
    def __init__(self, proj_dim=128, memory_path="models/optimized_memory_bank.pt",
                 top_k_clusters=50):
        super().__init__()
        self.proj_dim = proj_dim
        self.top_k_clusters = top_k_clusters  # 选择最相似的50个聚类

        if not os.path.exists(memory_path):
            raise FileNotFoundError(f"Memory bank文件不存在: {memory_path}")

        # 加载memory bank数据
        bank_data = torch.load(memory_path, map_location='cpu', weights_only=True)
        self.cluster_centers = bank_data['cluster_centers']  # patch聚类中心
        self.cluster_patches = bank_data['cluster_patches']  # 每个类的代表patch
        self.cluster_semantics = bank_data['cluster_semantics']  # 每个类的语义特征
        self.num_clusters = len(self.cluster_centers)
        self.original_patch_dim = self.cluster_centers.shape[1]
        self.semantic_dim = self.cluster_semantics.shape[1]

        # 特征投影层（统一维度）
        self.patch_proj = nn.Linear(self.original_patch_dim, proj_dim, bias=False)
        self.semantic_proj = nn.Linear(self.semantic_dim, proj_dim, bias=False)

        # 注意力机制
        self.cross_attn = CrossAttention(
            dim=proj_dim, heads=4, dim_head=proj_dim // 4, causal=False
        )
        self.self_attn = SelfAttention(
            dim=proj_dim, heads=4, dim_head=proj_dim // 4, causal=False
        )

        # 融合控制因子
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.gamma_proj = nn.Sequential(
            nn.Conv2d(proj_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, x_semantic=None):
        """
        输入特征图和语义特征，返回融合后的特征图
        x: 形状为[B, C, H, W]的特征图（C=proj_dim）
        x_semantic: 可选，输入的语义特征[B, S]
        """
        if x.size(1) != self.proj_dim:
            raise ValueError(f"输入特征通道数{x.size(1)}与投影维度{self.proj_dim}不匹配")

        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # 处理语义特征
        if x_semantic is None:
            global_feat = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)  # [B, C]
            x_semantic = self.semantic_proj(global_feat)  # 投影到proj_dim
        else:
            x_semantic = self.semantic_proj(x_semantic.to(x.device))  # [B, proj_dim]

        # 归一化特征
        global_feat_norm = F.normalize(x_semantic, dim=-1)  # [B, proj_dim]

        # 1. 计算综合相似度
        projected_centers = self.patch_proj(self.cluster_centers.to(x.device))  # [K, proj_dim]
        projected_semantics = self.semantic_proj(self.cluster_semantics.to(x.device))  # [K, proj_dim]

        # 计算两种相似度（加权求和）
        patch_sim = F.cosine_similarity(
            global_feat_norm.unsqueeze(1),
            projected_centers.unsqueeze(0),
            dim=-1
        )  # [B, K]

        semantic_sim = F.cosine_similarity(
            global_feat_norm.unsqueeze(1),
            projected_semantics.unsqueeze(0),
            dim=-1
        )  # [B, K]

        combined_sim = 0.5 * patch_sim + 0.5 * semantic_sim  # [B, K]

        # 选择最相似的top_k个聚类
        _, top_cluster_indices = torch.topk(combined_sim, k=min(self.top_k_clusters, self.num_clusters), dim=1)

        # 2. 收集选中聚类的patch特征和语义特征（不进行填充）
        has_any_features = False
        selected_features = []
        for b in range(B):
            clusters = top_cluster_indices[b].tolist()
            candidate_patches = []
            candidate_semantics = []
            seen_semantics = set()  # 用于跟踪已添加的语义特征（避免重复）

            for cluster_idx in clusters:
                if 0 <= cluster_idx < self.num_clusters:
                    # 收集patch特征
                    patch_feat = self.cluster_patches[cluster_idx].to(x.device)  # [1, D]
                    projected_patch = self.patch_proj(patch_feat)  # [1, proj_dim]
                    candidate_patches.append(projected_patch)

                    # 收集语义特征（去重）
                    sem_feat = self.cluster_semantics[cluster_idx].to(x.device)
                    sem_hash = hash(tuple(sem_feat.cpu().numpy().tolist()))  # 生成语义特征的哈希值
                    if sem_hash not in seen_semantics:
                        seen_semantics.add(sem_hash)
                        projected_sem = self.semantic_proj(sem_feat)  # [proj_dim]
                        candidate_semantics.append(projected_sem.unsqueeze(0))  # [1, proj_dim]

            # 合并特征（不填充，使用实际数量）
            if candidate_patches or candidate_semantics:
                has_any_features = True
                all_features = candidate_patches + candidate_semantics
                selected_features.append(torch.cat(all_features, dim=0))
            else:
                selected_features.append(None)  # 标记为无特征

        # 3. 计算互注意力（仅当有特征时）
        if has_any_features:
            # 处理每个样本的特征（可能长度不同）
            cross_attn_outputs = []
            for b in range(B):
                features = selected_features[b]
                if features is None:
                    # 当前样本无特征，输出零向量
                    cross_attn_outputs.append(torch.zeros_like(x_flat[b]))
                    continue

                # 对单个样本计算互注意力
                attn_out = self.cross_attn(
                    x_flat[b].unsqueeze(0),  # [1, H*W, C]
                    context=features.unsqueeze(0)  # [1, F, C]
                )
                cross_attn_outputs.append(attn_out.squeeze(0))  # [H*W, C]

            # 合并所有样本的结果
            cross_attn_out = torch.stack(cross_attn_outputs, dim=0)  # [B, H*W, C]
            cross_attn_out = cross_attn_out.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]
        else:
            # 完全没有特征，不进行互注意力计算
            cross_attn_out = torch.zeros_like(x)

        # 4. 计算自注意力
        self_attn_out = self.self_attn(x_flat).transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]

        # 5. 注意力融合（仅当有特征时才融合互注意力结果）
        if has_any_features:
            gamma_map = self.gamma_proj(cross_attn_out)  # [B, 1, H, W]
            fused_out = gamma_map * cross_attn_out + (1 - gamma_map) * self_attn_out
        else:
            # 没有任何特征，直接使用自注意力结果
            fused_out = self_attn_out

        # 6. 残差连接
        return x + self.gamma * fused_out


# --------------------------
# Memory Bank生成工具
# --------------------------
def generate_optimized_memory_bank(image_dir,
                                   save_path="models/optimized_memory_bank.pt",
                                   vis_dir="optimized_vis",
                                   model_name="facebook/dinov2-small",
                                   enable_clarity_filter=True,
                                   clarity_threshold=50,
                                   hash_threshold=5,
                                   feature_optim_method='l2_norm',
                                   min_clusters=100,
                                   max_clusters=150,
                                   top_significant_patches=10000,
                                   pca_components=None):
    """生成100-150个最优聚类的memory bank，每个类保存1个代表patch和对应的语义特征"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载DINOv2模型
    try:
        model = AutoModel.from_pretrained(model_name).to(device)
        processor = AutoImageProcessor.from_pretrained(model_name)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"加载模型失败: {str(e)}")

    # 收集图像路径
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_paths = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(image_extensions)
    ]

    if not image_paths:
        raise ValueError(f"在{image_dir}中未找到任何图像文件")
    print(f"初始找到{len(image_paths)}张图像")

    # 图像预处理
    clear_paths = image_paths
    # if enable_clarity_filter:
    #     clear_paths = filter_low_quality_images(image_paths, min_clarity=clarity_threshold)
    #     if not clear_paths:
    #         clear_paths = image_paths
    #         print("警告：清晰度筛选后无有效图像，使用原始图像列表")

    unique_paths = remove_duplicate_images(clear_paths, hash_threshold)
    if not unique_paths:
        unique_paths = clear_paths
        print("警告：去重后无有效图像，使用筛选后的图像列表")
    print(f"有效图像数量: {len(unique_paths)}")

    # 提取patch特征和对应的语义特征
    all_patch_features = []
    all_semantic_features = []
    patch_to_image_mapping = []  # 记录每个patch属于哪个图像索引
    print("开始提取patch特征和语义特征...")
    for img_idx, img_path in enumerate(tqdm(unique_paths, desc="提取特征")):
        try:
            with Image.open(img_path).convert('RGB') as img:
                inputs = processor(images=img, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                # patch特征（排除CLS token）
                patch_feats = outputs.last_hidden_state[:, 1:, :].cpu().numpy()  # [1, N_patches, D]
                # 语义特征（CLS token）
                semantic_feat = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [1, D]

                all_patch_features.append(patch_feats[0])
                all_semantic_features.append(semantic_feat[0])
                # 记录每个patch对应的图像索引
                patch_to_image_mapping.extend([img_idx] * patch_feats.shape[1])

        except Exception as e:
            print(f"处理图像{img_path}时出错，已跳过: {str(e)}")
            continue

    if not all_patch_features or not all_semantic_features:
        raise RuntimeError("未成功提取任何特征，请检查图像路径和格式")

    # 合并所有patch特征和语义特征
    all_patches = np.concatenate(all_patch_features, axis=0)  # [total_patches, D]
    all_semantics = np.array(all_semantic_features)  # [N_images, D]
    original_dim = all_patches.shape[1]
    semantic_dim = all_semantics.shape[1]
    print(
        f"提取到{len(all_patches)}个patch特征（维度: {original_dim}）和{len(all_semantics)}个语义特征（维度: {semantic_dim}）")

    # 筛选显著的patch特征
    significant_patches, top_indices = select_significant_patches(all_patches, top_k=top_significant_patches)
    if len(significant_patches) < min_clusters:
        raise ValueError(
            f"筛选后的patch数量({len(significant_patches)})少于最小聚类数量({min_clusters})，请增加top_significant_patches参数")

    # 特征优化
    optimized_patches = optimize_features(significant_patches, method=feature_optim_method)
    print(f"特征优化完成，优化后维度: {optimized_patches.shape[1]}")

    # 可选的PCA降维
    pca = None
    if pca_components is not None and pca_components < optimized_patches.shape[1]:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=pca_components, random_state=42)
        optimized_patches = pca.fit_transform(optimized_patches)
        explained_variance = np.sum(pca.explained_variance_ratio_) * 100
        print(f"PCA降维: {original_dim} → {pca_components}维，保留信息量: {explained_variance:.2f}%")

    # 确定100-150之间的最优聚类数量
    optimal_clusters = find_optimal_clusters(
        optimized_patches,
        min_clusters=min_clusters,
        max_clusters=max_clusters
    )

    # 执行最优聚类
    print(f"使用最优聚类数量{optimal_clusters}进行聚类...")
    kmeans = KMeans(n_clusters=optimal_clusters, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(optimized_patches)
    cluster_centers = kmeans.cluster_centers_

    # 评估最终聚类效果
    if optimal_clusters > 1:
        ch_score = calinski_harabasz_score(optimized_patches, cluster_labels)
        print(f"最终聚类Calinski-Harabasz分数: {ch_score:.2f} (数值越大越好)")
        if optimal_clusters <= 20:
            silhouette = silhouette_score(optimized_patches, cluster_labels)
            print(f"轮廓系数: {silhouette:.4f} (接近1最好)")

    # 统计聚类分布
    cluster_counts = Counter(cluster_labels)
    print(f"聚类分布: {cluster_counts.most_common(5)} (前5个最大的簇)")

    # 为每个聚类选择最显著的1个patch特征作为代表，并关联对应的语义特征
    cluster_patches = {}
    cluster_semantics = []
    cluster_centers_valid = []
    significant_patches_tensor = torch.from_numpy(significant_patches)

    for cluster_id in range(optimal_clusters):
        mask = (cluster_labels == cluster_id)
        if not np.any(mask):
            # 空聚类处理
            cluster_patches[cluster_id] = torch.zeros(1, original_dim)
            cluster_semantics.append(np.zeros(semantic_dim))
            cluster_centers_valid.append(np.zeros(original_dim))
            print(f"警告：聚类 {cluster_id} 为空，已用零向量填充")
            continue

        # 找到该聚类中最显著的patch
        cluster_patches_subset = significant_patches[mask]
        patch_norms = np.linalg.norm(cluster_patches_subset, axis=1)
        most_significant_idx = np.argmax(patch_norms)
        representative_patch = cluster_patches_subset[most_significant_idx:most_significant_idx + 1]

        # 找到该patch对应的图像索引
        significant_patch_original_idx = top_indices[np.where(mask)[0][most_significant_idx]]
        img_idx = patch_to_image_mapping[significant_patch_original_idx]
        representative_semantic = all_semantics[img_idx]

        # 保存代表patch和语义特征
        cluster_patches[cluster_id] = torch.from_numpy(representative_patch)
        cluster_semantics.append(representative_semantic)

        # 聚类中心转换回原始维度
        center = cluster_centers[cluster_id]
        if pca is not None and pca_components < original_dim:
            center = pca.inverse_transform([center])[0]
        cluster_centers_valid.append(center)

        if cluster_id % 20 == 0:  # 每20个聚类打印一次进度
            print(f"已处理 {cluster_id + 1}/{optimal_clusters} 个聚类")

    # 转换为张量
    cluster_centers_tensor = torch.from_numpy(np.array(cluster_centers_valid)).float()
    cluster_semantics_tensor = torch.from_numpy(np.array(cluster_semantics)).float()

    # 可视化聚类结果
    print("生成聚类可视化结果...")
    try:
        from sklearn.manifold import TSNE
        centers_2d = TSNE(
            n_components=2,
            perplexity=min(30, optimal_clusters - 1),
            random_state=42,
            n_iter=2000
        ).fit_transform(cluster_centers_valid)

        plt.figure(figsize=(12, 10))
        cmap = ListedColormap(plt.cm.tab20(np.linspace(0, 1, min(20, optimal_clusters))))

        scatter = plt.scatter(
            centers_2d[:, 0], centers_2d[:, 1],
            c=range(optimal_clusters), cmap=cmap,
            alpha=0.8, s=100, edgecolors='black'
        )

        plt.title(f'Patch特征聚类中心分布（{optimal_clusters}个聚类）')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{vis_dir}/patch_cluster_centers.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"聚类中心可视化失败: {str(e)}")

    # 保存memory bank
    try:
        torch.save({
            'cluster_centers': cluster_centers_tensor,
            'cluster_patches': cluster_patches,
            'cluster_semantics': cluster_semantics_tensor,
            'num_clusters': optimal_clusters,
            'model_name': model_name,
            'original_patch_dim': original_dim,
            'semantic_dim': semantic_dim,
            'total_patches_used': len(significant_patches)
        }, save_path)
        print(f"优化后的Memory bank已成功保存到: {save_path}")
        print(f"包含{optimal_clusters}个聚类中心，每个聚类包含1个代表patch特征和对应的语义特征")
    except Exception as e:
        raise RuntimeError(f"保存memory bank失败: {str(e)}")


# --------------------------
# 测试函数（修复返回值问题）
# --------------------------
def memory_bank_generation():
    """测试memory bank生成功能（使用assert而非return）"""
    try:
        # 创建临时图像目录并生成测试图像
        test_dir = "test_images"
        os.makedirs(test_dir, exist_ok=True)

        # 生成20张测试图像
        for i in range(20):
            img = Image.new('RGB', (224, 224), color=(i * 10, i * 5, 255 - i * 10))
            img.save(os.path.join(test_dir, f"test_{i}.png"))

        # 生成memory bank（缩小范围以便快速测试）
        generate_optimized_memory_bank(
            image_dir=test_dir,
            save_path="models/test_optimized_memory_bank.pt",
            vis_dir="test_optimized_vis",
            min_clusters=10,  # 测试时缩小范围
            max_clusters=15,
            top_significant_patches=2000
        )

        # 验证文件是否生成
        assert os.path.exists("models/test_optimized_memory_bank.pt"), "Memory bank文件未生成"
        print("Memory bank生成测试通过")
    except Exception as e:
        print(f"Memory bank生成测试失败: {str(e)}")
        assert False, f"测试失败: {str(e)}"


def model_forward():
    """测试模型前向传播功能（使用assert而非return）"""
    try:
        # 检查是否存在测试用memory bank
        if not os.path.exists("models/optimized_memory_bank.pt"):
            print("未找到测试用memory bank，先运行生成测试")
            memory_bank_generation()

        # 初始化模型
        model = PatchSemanticMemoryBank(
            proj_dim=128,
            memory_path="models/optimized_memory_bank.pt",
            top_k_clusters=50
        )

        # 创建测试输入
        test_input = torch.randn(2, 128, 20, 20)  # [B, C, H, W]
        test_semantic = torch.randn(2, 384)  # DINOv2-small的语义特征维度

        # 前向传播
        output = model(test_input, test_semantic)

        # 检查输出形状
        assert output.shape == test_input.shape, f"输出形状错误: {output.shape} vs {test_input.shape}"
        print("模型前向传播测试通过")
    except Exception as e:
        print(f"模型前向传播测试失败: {str(e)}")
        assert False, f"测试失败: {str(e)}"


# --------------------------
# 使用示例
# --------------------------
if __name__ == "__main__":
    # 运行测试
    # print("开始运行测试...")
    # memory_bank_generation()


    # 实际使用示例（注释掉测试后启用）
    # generate_optimized_memory_bank(
    #     image_dir="normal",  # 替换为你的图像文件夹路径
    #     save_path="models/optimized_memory_bank.pt",
    #     vis_dir="optimized_vis",
    #     model_name="facebook/dinov2-small",
    #     min_clusters=100,
    #     max_clusters=150,
    #     top_significant_patches=10000
    # )

    # model = PatchSemanticMemoryBank(
    #     proj_dim=128,
    #     memory_path="models/optimized_memory_bank.pt",
    #     top_k_clusters=50
    # )
    model_forward()
    print("所有测试通过！")
import os
import logging
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import imagehash
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.decomposition import IncrementalPCA
from torchvision import models, transforms
from matplotlib import pyplot as plt
import pickle
import gc
import shutil

# --------------------------
# 配置与初始化
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('workflow.log'), logging.StreamHandler()]
)

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False


# --------------------------
# 1. 图像级别去重与筛选
# --------------------------
def image_level_deduplication(image_dir, hash_threshold=5, max_images=1000):
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"图像目录不存在：{image_dir}")

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    image_paths = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(image_extensions)
    ]

    if not image_paths:
        raise ValueError(f"图像目录中未找到有效图像：{image_dir}")

    hash_map = {}
    unique_paths = []

    for path in tqdm(image_paths, desc="图像级别去重与筛选"):
        try:
            with Image.open(path) as img:
                img = img.convert('RGB')
                img_hash = imagehash.dhash(img, hash_size=16)

                if not any(abs(img_hash - h) < hash_threshold for h in hash_map.values()):
                    hash_map[path] = img_hash
                    unique_paths.append(path)

                if len(unique_paths) >= max_images:
                    logging.info(f"已达最大图像数 {max_images}，停止去重")
                    break

        except (UnidentifiedImageError, OSError) as e:
            logging.warning(f"跳过无效图像：{path}，错误：{str(e)}")

    if not unique_paths:
        raise RuntimeError("去重后无有效图像保留")

    logging.info(f"图像去重筛选完成：原始{len(image_paths)}张 → 保留{len(unique_paths)}张")
    return unique_paths


# --------------------------
# 2. 特征提取与处理
# --------------------------
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(
            self.resnet.conv1, self.resnet.bn1, self.resnet.relu,
            self.resnet.maxpool, self.resnet.layer1,
            self.resnet.layer2, self.resnet.layer3
        )
        for param in list(self.features.parameters())[:-10]:
            param.requires_grad = False
        self.eval()

    def forward(self, x):
        return self.features(x)  # 输出[B, 1024, 20, 20]


class FeatureProcessor:
    def __init__(self, model, device, pca=None):
        self.model = model
        self.device = device
        self.pca = pca
        self.preprocess = transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(320),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_full_feature(self, img_path):
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    feat_map = self.model(img_tensor).squeeze(0).cpu().numpy()

                full_feat = feat_map.transpose(1, 2, 0).reshape(-1, 1024)  # 400×1024
                return full_feat

        except Exception as e:
            logging.warning(f"提取特征失败 {img_path}：{str(e)}")
            return None

    def process(self, img_path):
        full_feat = self.extract_full_feature(img_path)
        if full_feat is None:
            return None

        if self.pca is not None:
            try:
                return self.pca.transform(full_feat)
            except Exception as e:
                logging.error(f"PCA降维失败：{str(e)}")
                return None

        return full_feat


# --------------------------
# 3. 分批次PCA训练（特征筛选）
# --------------------------
class PCAFeatureReducer:
    def __init__(self, min_dim=128, variance_threshold=0.8):
        self.min_dim = min_dim
        self.variance_threshold = variance_threshold
        self.pca = None
        self.best_dim = None
        self._temp_pca = None

    def fit_in_batches(self, image_paths, processor, batch_size=50000):
        batch_size = max(batch_size, 400)

        # 第一阶段：计算方差解释率
        logging.info("第一阶段：计算方差解释率以确定最佳维度")
        self._temp_pca = IncrementalPCA(n_components=1024, batch_size=batch_size)

        current_batch = []
        total_features = 0

        for path in tqdm(image_paths, desc="计算方差解释率"):
            feat = processor.extract_full_feature(path)
            if feat is None:
                continue

            current_batch.append(feat)
            total_features += len(feat)

            if total_features >= batch_size:
                batch_array = np.concatenate(current_batch, axis=0)
                self._temp_pca.partial_fit(batch_array)
                del current_batch, batch_array
                gc.collect()
                current_batch = []
                total_features = 0

        if current_batch:
            batch_array = np.concatenate(current_batch, axis=0)
            self._temp_pca.partial_fit(batch_array)
            del current_batch, batch_array
            gc.collect()

        # 确定最佳维度
        cum_var = np.cumsum(self._temp_pca.explained_variance_ratio_)
        var达标维度 = np.argmax(cum_var >= self.variance_threshold) + 1
        self.best_dim = max(var达标维度, self.min_dim)
        self.best_dim = min(self.best_dim, 1024)
        logging.info(f"根据方差解释率确定最佳维度：{self.best_dim}（保留{cum_var[var达标维度 - 1]:.4f}方差）")

        # 第二阶段：训练IncrementalPCA
        logging.info(f"第二阶段：按最佳维度{self.best_dim}训练IncrementalPCA")
        self.pca = IncrementalPCA(n_components=self.best_dim, batch_size=batch_size)
        current_batch = []
        total_features = 0

        for path in tqdm(image_paths, desc="训练IncrementalPCA"):
            feat = processor.extract_full_feature(path)
            if feat is None:
                continue

            current_batch.append(feat)
            total_features += len(feat)

            if total_features >= batch_size:
                batch_array = np.concatenate(current_batch, axis=0)
                self.pca.partial_fit(batch_array)
                del current_batch, batch_array
                gc.collect()
                current_batch = []
                total_features = 0

        if current_batch:
            batch_array = np.concatenate(current_batch, axis=0)
            self.pca.partial_fit(batch_array)
            del current_batch, batch_array
            gc.collect()

        retained_var = np.sum(self.pca.explained_variance_ratio_)
        logging.info(f"分批次PCA训练完成：1024 → {self.best_dim}维，保留方差：{retained_var:.4f}")
        return self

    def transform(self, features):
        if self.pca is None:
            raise RuntimeError("PCA模型未训练，请先调用fit_in_batches方法")
        return self.pca.transform(features)

    def save(self, path):
        if self.pca is None:
            raise RuntimeError("PCA模型未训练，无法保存")
        with open(path, 'wb') as f:
            pickle.dump({
                'pca': self.pca,
                'best_dim': self.best_dim,
                'min_dim': self.min_dim,
                'variance_threshold': self.variance_threshold
            }, f)

    @classmethod
    def load(cls, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"PCA模型文件不存在：{path}")
        with open(path, 'rb') as f:
            data = pickle.load(f)
        reducer = cls(
            min_dim=data['min_dim'],
            variance_threshold=data['variance_threshold']
        )
        reducer.pca = data['pca']
        reducer.best_dim = data['best_dim']
        return reducer


# --------------------------
# 4. 聚类（100-150类）
# --------------------------
def cluster_images(all_features, num_clusters=125):
    if len(all_features) < num_clusters:
        raise ValueError(f"特征数量（{len(all_features)}）少于聚类数（{num_clusters}）")

    # 特征展平与拼接
    flattened_feats = []
    for i, feat in enumerate(all_features):
        flat_feat = feat.flatten().reshape(1, -1)
        flattened_feats.append(flat_feat)

        if (i + 1) % 100 == 0:
            batch_array = np.concatenate(flattened_feats, axis=0)
            flattened_feats = [batch_array]
            gc.collect()

    # 处理剩余样本
    if len(flattened_feats) > 1:
        flattened_feats = np.concatenate(flattened_feats, axis=0)
    else:
        flattened_feats = flattened_feats[0]

    # L2归一化
    norms = np.linalg.norm(flattened_feats, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    feats_norm = flattened_feats / norms

    # 聚类
    kmeans = KMeans(
        n_clusters=num_clusters,
        n_init=10,
        random_state=42
    )
    labels = kmeans.fit_predict(feats_norm)

    # 计算轮廓系数
    sil_score = silhouette_score(feats_norm, labels) if num_clusters > 1 else 1.0
    logging.info(f"图像级聚类完成：{len(all_features)}张图像 → {num_clusters}簇，轮廓系数：{sil_score:.4f}")
    return labels, kmeans.cluster_centers_, flattened_feats


# --------------------------
# 5. 可视化聚类结果
# --------------------------
def visualize_clusters(flattened_feats, labels, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    try:
        if len(flattened_feats) > 1000:
            idx = np.random.choice(len(flattened_feats), 1000, replace=False)
            flattened_feats = flattened_feats[idx]
            labels = labels[idx]

        tsne = TSNE(
            n_components=2,
            perplexity=50,
            random_state=42,
            n_iter=2000
        )
        tsne_results = tsne.fit_transform(flattened_feats)

        plt.figure(figsize=(15, 12))
        scatter = plt.scatter(
            tsne_results[:, 0],
            tsne_results[:, 1],
            c=labels,
            cmap='tab20',
            s=20,
            alpha=0.8
        )
        plt.colorbar(scatter, label='簇ID')
        plt.title(f'图像特征聚类分布（共{len(np.unique(labels))}个簇）')
        plt.axis('off')
        save_path = os.path.join(save_dir, 'cluster_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"聚类可视化结果保存至：{save_path}")

    except Exception as e:
        logging.error(f"可视化失败：{str(e)}")


# --------------------------
# 6. 构建记忆库（Tensor格式保存）
# --------------------------
def build_memory_bank(
        image_dir,
        save_path="image_memory_bank.pt",
        vis_dir="cluster_visualization",
        pca_path="pca_model.pkl",
        device=None,
        num_clusters=125,  # 100-150之间
        max_images=1000,
        pca_batch_size=50000
):
    # 限制聚类数在100-150之间
    num_clusters = max(100, min(num_clusters, 150))
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(
        f"开始构建记忆库（Tensor格式）：\n"
        f"设备={device}，聚类数={num_clusters}，最大图像数={max_images}"
    )

    # 1. 图像去重与筛选
    unique_image_paths = image_level_deduplication(
        image_dir,
        max_images=max_images
    )

    # 2. 初始化模型和处理器
    try:
        model = FeatureExtractor().to(device)
        processor = FeatureProcessor(model, device, pca=None)
        logging.info("模型和处理器初始化成功")
    except Exception as e:
        raise RuntimeError(f"模型加载失败：{str(e)}")

    # 3. 训练PCA（特征筛选）
    pca = PCAFeatureReducer(min_dim=128)
    pca.fit_in_batches(
        image_paths=unique_image_paths,
        processor=processor,
        batch_size=pca_batch_size
    )
    pca.save(pca_path)
    logging.info(f"PCA模型保存至：{pca_path}")

    # 4. 提取降维特征
    processor.pca = pca
    all_features = []
    for path in tqdm(unique_image_paths, desc="提取降维特征"):
        feat = processor.process(path)
        if feat is not None:
            all_features.append(feat)
            if len(all_features) % 100 == 0:
                gc.collect()

    if not all_features:
        raise RuntimeError("未提取到有效降维特征")
    D = all_features[0].shape[1]
    logging.info(f"降维特征提取完成：{len(all_features)}张图像，特征形状400×{D}")

    # 5. 聚类并获取中心
    labels, centers, flattened_feats = cluster_images(all_features, num_clusters=num_clusters)

    # 6. 构建记忆库（聚类中心转为Tensor格式）
    memory_bank = {
        "num_clusters": num_clusters,
        # 核心：将numpy数组转为Tensor并保存
        "cluster_centers": torch.tensor(centers, dtype=torch.float32),
        "feature_dim": D,
        "flattened_feature_dim": flattened_feats.shape[1],
        "pca_path": pca_path
    }

    # 7. 可视化聚类结果
    visualize_clusters(flattened_feats, labels, vis_dir)

    # 8. 保存记忆库（包含Tensor格式数据）
    try:
        torch.save(memory_bank, save_path)
        logging.info(f"记忆库保存成功：{save_path}（聚类中心为Tensor格式）")
    except Exception as e:
        raise RuntimeError(f"记忆库保存失败：{str(e)}")

    return save_path


# --------------------------
# 测试代码
# --------------------------
# def test_workflow():
#     test_dir = "test_images"
#     os.makedirs(test_dir, exist_ok=True)
#
#     # 生成10张测试图像
#     for i in range(10):
#         img = Image.new('RGB', (320, 320), color=(i * 25, 100, 200))
#         img.save(os.path.join(test_dir, f"test_{i}.png"))
#
#     try:
#         save_path = build_memory_bank(
#             image_dir=test_dir,
#             num_clusters=100,  # 测试用100类
#             max_images=10,
#             save_path="test_memory_bank.pt",
#             vis_dir="test_vis"
#         )
#
#         # 验证保存的Tensor格式
#         memory_bank = torch.load(save_path)
#         assert isinstance(memory_bank["cluster_centers"], torch.Tensor), \
#             "聚类中心未保存为Tensor格式"
#         assert memory_bank["num_clusters"] == 100, \
#             f"聚类数错误：实际{memory_bank['num_clusters']}，预期100"
#
#         logging.info("测试成功：记忆库中聚类中心为Tensor格式")
#         return True
#     except Exception as e:
#         logging.error(f"测试失败：{str(e)}")
#         return False


if __name__ == "__main__":
    # 实际运行时使用
    build_memory_bank(
        image_dir="normal",
        num_clusters=150,  # 100-150之间
        max_images=1000
    )

    # 测试时取消注释
    # test_workflow()

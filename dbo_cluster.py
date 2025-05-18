# dbo_cluster.py
import torch
from dbo_net.model import DBONet
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import torch.nn.functional as F
from dbo_loss import reconstruction_loss, sparsity_loss, view_alignment_loss
from sklearn.decomposition import PCA


class DBOClusterer:
    def __init__(self, n_clusters=2, device='cpu', n_epochs=10, lr=1e-3):
        self.n_clusters = n_clusters
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.n_epochs = n_epochs
        self.lr = lr

    def build_model(self, view_dims, n_blocks=2):
        self.model = DBONet(
            nfeats=view_dims,
            n_view=len(view_dims),
            n_clusters=self.n_clusters,
            blocks=n_blocks,
            device=self.device
        ).to(self.device)

    def train(self, features):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.n_epochs):
            optimizer.zero_grad()

            Z_list = self.model.encode_each_view(features)  # 获取每个视图的 Z
            Z_concat = self.model(features)  # 多视图融合后的 Z（用于聚类）

            loss = 0
            # 逐视图损失
            for i in range(len(features)):
                Xi = features[i]
                Zi = Z_list[i]
                Ui = self.model.U[i]

                loss += reconstruction_loss(Xi, Ui, Zi)  # 显式重构项
                loss += sparsity_loss(Zi, l1_weight=1e-3)  # 稀疏约束

            # 多视图对齐损失
            loss += view_alignment_loss(Z_list)

            loss.backward()
            optimizer.step()

    def cluster(self, peer_views):
        n_peers = len(peer_views)
        n_views = len(peer_views[0]) if n_peers > 0 else 0
        features = []
        MAX_FEATURE_DIM = 512

        # print("[Debug] ▶️ 开始构造特征视图，n_peers =", n_peers, ", n_views =", n_views)

        for v in range(n_views):
            view_data_raw = [peer_views[i][v] for i in range(n_peers)]

            # 检查每个 peer 的该视图是否为合法 array
            for i, vdata in enumerate(view_data_raw):
                if vdata is None:
                    raise ValueError(f"[Error] Peer {i} 的视图 {v} 为 None，请检查 participant_update 返回值")
                if not hasattr(vdata, 'shape'):
                    raise ValueError(f"[Error] Peer {i} 的视图 {v} 非 array，类型为 {type(vdata)}")

            try:
                view_data = np.stack([x.flatten() for x in view_data_raw], axis=0)  # [n_peers, feature_dim]
            except Exception as e:
                raise RuntimeError(f"[Error] stacking 第 {v} 个视图失败: {e}")

            # print(f"[Debug] ✔️ 视图 {v} shape: {view_data.shape}")
            if view_data.shape[1] > MAX_FEATURE_DIM:
                from sklearn.decomposition import PCA
                max_pca_dim = min(MAX_FEATURE_DIM, view_data.shape[0])
                # print(f"[Debug] ⚠️ View {v} 维度太大 ({view_data.shape[1]}), 执行 PCA 降维至 {max_pca_dim}")
                pca = PCA(n_components=max_pca_dim)
                view_data = pca.fit_transform(view_data)

            view_data = StandardScaler().fit_transform(view_data)
            view_tensor = torch.tensor(view_data, dtype=torch.float32, device=self.device)
            features.append(view_tensor)

        # print("[Debug] ✅ 所有视图构建完成")
        # for idx, f in enumerate(features):
            # print(f" - features[{idx}].shape = {f.shape}")

        if self.model is None:
            view_dims = [f.shape[1] for f in features]
            # print("[Debug] 🚀 构建模型，view_dims =", view_dims)
            assert all(d > 0 for d in view_dims), "🚨 有视图维度为0，U 初始化将失败"
            self.build_model(view_dims)

        # === Step 1: 训练模型，优化 latent 表征 ===
        self.train(features)

        # === Step 2: 聚类得到标签 ===
        self.model.eval()
        with torch.no_grad():
            Z = self.model(features)
            pred = self.model.get_cluster_labels(Z)

        return pred.cpu().numpy().tolist()


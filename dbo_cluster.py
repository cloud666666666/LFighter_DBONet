import torch
from dbo_net.model import DBONet
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
from dbo_loss import reconstruction_loss, sparsity_loss, view_alignment_loss
import torch.nn.functional as F


class DBOClusterer:
    def __init__(self, n_clusters=2, device='cpu', n_epochs=10, lr=1e-3,
                 pca_dims=[None, 512, 128]):  # 每个视图降维维度（None 表示不降维）
        self.n_clusters = n_clusters
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.n_epochs = n_epochs
        self.lr = lr
        self.pca_dims = pca_dims
        self.view_weights = None  # 可学习的视图权重

    def build_model(self, view_dims, n_blocks=2):
        self.model = DBONet(
            nfeats=view_dims,
            n_view=len(view_dims),
            n_clusters=self.n_clusters,
            blocks=n_blocks,
            device=self.device
        ).to(self.device)
        
        # 初始化可学习的视图权重，设置为 0.7、0.2、0.1 的分布
        initial_weights = torch.tensor([0.7, 0.2, 0.1], device=self.device)
        
        self.view_weights = torch.nn.Parameter(
            initial_weights,
            requires_grad=True
        )

    def train(self, features):
        self.model.train()
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.view_weights, 'lr': self.lr * 0.1}  # 权重学习率稍小
        ], lr=self.lr)

        for epoch in range(self.n_epochs):
            optimizer.zero_grad()

            Z_list = self.model.encode_each_view(features)
            
            # 使用 softmax 确保权重和为 1
            normalized_weights = F.softmax(self.view_weights, dim=0)
            Z_concat = self.model.fuse_views(Z_list, weights=normalized_weights)

            loss = 0
            for i in range(len(features)):
                Xi = features[i]
                Zi = Z_list[i]
                Ui = self.model.U[i]
                loss += reconstruction_loss(Xi, Ui, Zi)
                loss += sparsity_loss(Zi, l1_weight=1e-3)

            loss += view_alignment_loss(Z_list)
            loss += self.model.compute_implicit_loss(Z_concat, features) * 1.0
            loss.backward()
            optimizer.step()

        self.Z_final = Z_concat.detach().cpu().numpy()
        self.final_weights = F.softmax(self.view_weights, dim=0).detach().cpu().numpy()
        print(self.final_weights)

    def cluster(self, peer_views):
        """
        peer_views: List[List[Tensor]], 每个元素是一个 peer 的多视图特征
        """
        peer_views = [[torch.tensor(v, dtype=torch.float32) if not isinstance(v, torch.Tensor) else v
                       for v in peer] for peer in peer_views]

        # 解包每种视图
        views_by_type = list(zip(*peer_views))
        features = [torch.stack(view_list).to(self.device) for view_list in views_by_type]

        # ✅ flatten 所有高维视图为 [B, D]
        flattened_features = []
        for i, f in enumerate(features):
            if f.ndim > 2:
                f = f.view(f.shape[0], -1)
                # print(f"[Flatten] View {i} flattened to shape {f.shape}")
            # else:
            #     print(f"[Flatten] View {i} kept as shape {f.shape}")
            flattened_features.append(f)
        features = flattened_features

        # ✅ 执行 PCA 降维
        reduced_features = []
        for i, feat in enumerate(features):
            feat_np = feat.cpu().numpy()
            target_dim = self.pca_dims[i]

            if target_dim is not None:
                max_allowed_dim = min(feat_np.shape[0], feat_np.shape[1])  # min(n_samples, n_features)
                actual_dim = min(target_dim, max_allowed_dim)
                if actual_dim < 1:
                    raise ValueError(f"[PCA] View {i} too small for PCA: shape {feat_np.shape}")

                # 在 PCA 降维前，加上标准化：
                feat_np = StandardScaler().fit_transform(feat_np)
                pca = PCA(n_components=actual_dim)
                reduced = pca.fit_transform(feat_np)

                # print(f"[PCA] View {i}: {feat_np.shape} -> ({feat_np.shape[0]}, {actual_dim})")
                reduced_features.append(torch.tensor(reduced, dtype=torch.float32).to(self.device))
            else:
                # print(f"[PCA] View {i}: {feat_np.shape} -> [skip]")
                reduced_features.append(feat)

        features = reduced_features
        view_dims = [f.shape[1] for f in features]

        if self.model is None:
            self.build_model(view_dims)

        self.model.eval()
        self.train(features)  # ⬅️ 必须保留计算图，允许 loss.backward()
        Z_scaled = StandardScaler().fit_transform(self.Z_final)
        pred = KMeans(n_clusters=self.n_clusters).fit_predict(Z_scaled)

        return pred

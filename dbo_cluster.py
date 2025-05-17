# dbo_clusterer.py
import torch
from dbo_net.model import DBONet
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

class DBOClusterer:
    def __init__(self, n_clusters=2, device='cpu'):
        self.n_clusters = n_clusters
        self.device = device
        self.model = None

    def build_model(self, view_dims, n_blocks=2):
        self.model = DBONet(nfeats=view_dims, n_view=len(view_dims), n_clusters=self.n_clusters, blocks=n_blocks, device=self.device).to(self.device)

    def cluster(self, peer_views):
        # peer_views: [ [v1, v2, v3], ... ] 每个元素是peer的3个视图
        # 构建每个视图的特征矩阵
        n_peers = len(peer_views)
        n_views = len(peer_views[0])
        features = []
        for v in range(n_views):
            view_data = [peer_views[i][v].flatten() for i in range(n_peers)]
            view_data = StandardScaler().fit_transform(view_data)
            view_data = torch.tensor(view_data, dtype=torch.float32, device=self.device)
            features.append(view_data)

        if self.model is None:
            self.build_model([f.shape[1] for f in features])

        with torch.no_grad():
            Z = self.model(features)
            pred = self.model.get_cluster_labels(Z)
        return pred.cpu().numpy().tolist()

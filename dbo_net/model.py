# ✅ 优化后的 DBONet 模型实现（支持 explicit + implicit 双损失）
import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Block, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


class DBONet(nn.Module):
    def __init__(self, nfeats, n_view, n_clusters, blocks=3, para=0.1, Z_init=None, device='cpu'):
        super(DBONet, self).__init__()
        self.n_view = n_view
        self.n_clusters = n_clusters
        self.device = device

        # 每个视图对应一个 block 序列（输入 -> 聚类空间）
        self.blocks = nn.ModuleList()
        self.U = nn.ParameterList()
        for v in range(n_view):
            block_layers = []
            input_dim = nfeats[v]
            for _ in range(blocks):
                block_layers.append(Block(input_dim, 128))
                input_dim = 128
            block_layers.append(nn.Linear(128, n_clusters))
            self.blocks.append(nn.Sequential(*block_layers))
            self.U.append(nn.Parameter(torch.randn(n_clusters, nfeats[v], device=device)))

        # 初始化共享聚类表示 Z
        if Z_init is not None:
            self.Z = nn.Parameter(torch.tensor(Z_init, dtype=torch.float32, device=device), requires_grad=True)
        else:
            self.Z = nn.Parameter(torch.randn(100, n_clusters, device=device), requires_grad=True)

    def forward(self, features, adjs=None, return_H=False):
        # 输入：features 是每视图输入特征列表
        H_list = []
        for v in range(self.n_view):
            H_v = self.blocks[v](features[v])  # shape: (num_clients, n_clusters)
            H_list.append(H_v)

        # Z 是共享聚类表示
        if return_H:
            return self.Z, H_list
        else:
            return self.Z

    def compute_loss(self, Z, H_list, laplacians, alpha=0.1, beta=0.1):
        # (1) 显式特征一致性损失
        consistency_loss = sum(torch.norm(Z - H, p=2) for H in H_list)

        # (2) 隐式图结构保持损失
        graph_loss = sum(torch.trace(Z.T @ L @ Z) for L in laplacians)

        return alpha * consistency_loss + beta * graph_loss

    def get_cluster_labels(self, Z):
        return torch.argmax(Z, dim=1)

    def compute_implicit_loss(self, Z, X_list, alpha=1.0, beta=0.1):
        """
        L_imp = ∑_v ||ZZᵗ - X^vX^vᵗ||²
        可选扩展为 ∑_v ||ZZᵗ - W^v||²
        """
        ZZ = Z @ Z.T
        loss = 0
        for X in X_list:
            XX = X @ X.T
            loss += torch.norm(ZZ - XX, p='fro') ** 2
        return alpha * loss

    def encode_each_view(self, features):
        Z_list = []
        for i in range(self.n_view):
            Zi = self.blocks[i](features[i])  # ✅ 正确调用每个视图的编码器（即 block）
            Z_list.append(Zi)
        return Z_list


# dbo_loss.py
import torch
import torch.nn.functional as F

def reconstruction_loss(X, U, Z):
    # ‖X - UZ‖²
    recon = torch.matmul(Z, U)
    return F.mse_loss(recon, X)

def sparsity_loss(Z, l1_weight=1e-3):
    # λ * ‖Z‖₁
    return l1_weight * torch.norm(Z, p=1)

def laplacian_loss(Z, L):
    # Tr(Z L Z^T)
    return torch.trace(Z.T @ L @ Z)

def view_alignment_loss(Z_list):
    # ∑‖Zᵢ - Z̄‖²
    Z_bar = sum(Z_list) / len(Z_list)
    loss = 0
    for Z in Z_list:
        loss += F.mse_loss(Z, Z_bar)
    return loss

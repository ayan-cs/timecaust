import torch
import numpy as np

def compute_css(pred_clean, pred_adv, target_dim, gamma=1e-8):
    """
    Causal Selectivity Score (post-hoc metric).
    CSS = ||f(XX_hat)^(j) - f(X)^(j)||_2 / (||f(X_hat)^(-j) - f(X)^(-j)||_2 + gamma)
    """
    M = pred_clean.shape[-1]
    non_target_mask = torch.ones(M, dtype=torch.bool, device=pred_clean.device)
    non_target_mask[target_dim] = False
 
    tgt_shift = torch.norm(pred_adv[:, :, target_dim] - pred_clean[:, :, target_dim], p=2).item()
    nt_shift = torch.norm(pred_adv[:, :, non_target_mask] - pred_clean[:, :, non_target_mask], p=2).item()
 
    css = tgt_shift / (nt_shift + gamma)
    return css

def compute_mse_tgt(pred_adv, ground_truth, target_dim):
    """
    MSEtgt = (1/H) \SUM (f(X_hat)^(j)_h - Y^(j)_h)^2
    Measures attack strength on target channel. Higher = stronger attack.
    """
    H = pred_adv.shape[1]
    mse = ((pred_adv[:, :, target_dim] - ground_truth[:, :, target_dim]) ** 2).sum() / (pred_adv.shape[0] * H)
    return mse.item()

def compute_mae_tgt(pred_adv, ground_truth, target_dim):
    """
    MAEtgt = (1/H) \SUM |f(X_hat)^(j)_h - Y^(j)_h|
    L1 counterpart of MSEtgt. Higher = stronger attack.
    """
    H = pred_adv.shape[1]
    mae = (torch.abs(pred_adv[:, :, target_dim] - ground_truth[:, :, target_dim])).sum() / (pred_adv.shape[0] * H)
    return mae.item()

def compute_prs(pred_clean, pred_adv, ground_truth, target_dim, gamma=1e-8):
    """
    PRS = min(exp(1 - RMSE_adv / (RMSE_clean + gamma)), 1)
    Dataset-agnostic degradation score. Lower = stronger attack.
    """
    clean_residuals = pred_clean[:, :, target_dim] - ground_truth[:, :, target_dim]
    adv_residuals = pred_adv[:, :, target_dim] - ground_truth[:, :, target_dim]

    rmse_clean = torch.sqrt((clean_residuals ** 2).mean()).item()
    rmse_adv = torch.sqrt((adv_residuals ** 2).mean()).item()

    prs = min(np.exp(1 - rmse_adv / (rmse_clean + gamma)), 1.0)
    return prs

def compute_mse_nt(pred_clean, pred_adv, target_dim):
    """
    MSEnt = (1/(H*(M-1))) \SUM_{k \neq j} \SUM_h (f(X_hat)^(k)_h - f(X)^(k)_h)^2
    Spillover: adv vs CLEAN predictions (not ground truth). Lower = better selectivity.
    """
    M = pred_clean.shape[-1]
    H = pred_clean.shape[1]
    nt_mask = torch.ones(M, dtype=torch.bool, device=pred_clean.device)
    nt_mask[target_dim] = False

    diff = pred_adv[:, :, nt_mask] - pred_clean[:, :, nt_mask]
    mse_nt = (diff ** 2).sum() / (pred_clean.shape[0] * H * (M - 1))
    return mse_nt.item()

def compute_mae_nt(pred_clean, pred_adv, target_dim):
    """
    MAEnt = (1/(H*(M-1))) \SUM_{k \neq j} \SUM_h |f(X_hat)^(k)_h - f(X)^(k)_h|
    L1 counterpart of MSEnt. Adv vs CLEAN. Lower = better selectivity.
    """
    M = pred_clean.shape[-1]
    H = pred_clean.shape[1]
    nt_mask = torch.ones(M, dtype=torch.bool, device=pred_clean.device)
    nt_mask[target_dim] = False

    diff = pred_adv[:, :, nt_mask] - pred_clean[:, :, nt_mask]
    mae_nt = torch.abs(diff).sum() / (pred_clean.shape[0] * H * (M - 1))
    return mae_nt.item()

def compute_cga(delta, perturb_mask, gamma=1e-8):
    """
    CGA = ||\delta^(Pj)||_F / (||\delta*||_F + gamma)
    Structural compliance metric. = 1.0 for TimeCAT by construction.
    """
    perturb_mask_t = torch.tensor(perturb_mask, dtype=torch.bool, device=delta.device)
    parent_norm = torch.norm(delta[:, :, perturb_mask_t], p='fro').item()
    total_norm = torch.norm(delta, p='fro').item()
    cga = parent_norm / (total_norm + gamma)
    return cga

def compute_pe(pred_clean, pred_adv, delta, target_dim, gamma=1e-8):
    """
    PE = ||f(X_hat)^(j) - f(X)^(j)||_2 / (||\delta*||_F + gamma)
    Perturbation efficiency: damage per unit budget. Higher = more efficient.
    """
    tgt_shift = torch.norm(pred_adv[:, :, target_dim] - pred_clean[:, :, target_dim], p=2).item()
    delta_norm = torch.norm(delta, p='fro').item()
    pe = tgt_shift / (delta_norm + gamma)
    return pe
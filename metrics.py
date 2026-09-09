import torch
import numpy as np

def compute_css(pred_clean, pred_adv, target_dim, gamma=1e-8):
    M = pred_clean.shape[-1]
    non_target_mask = torch.ones(M, dtype=torch.bool, device=pred_clean.device)
    non_target_mask[target_dim] = False

    tgt_shift = torch.norm(pred_adv[:, :, target_dim] - pred_clean[:, :, target_dim], p=2).item()
    nt_shift = torch.norm(pred_adv[:, :, non_target_mask] - pred_clean[:, :, non_target_mask], p=2).item()

    css = tgt_shift / (nt_shift + gamma)
    return css

def compute_mse_tgt(pred_adv, ground_truth, target_dim):
    H = pred_adv.shape[1]
    mse = ((pred_adv[:, :, target_dim] - ground_truth[:, :, target_dim]) ** 2).sum() / (pred_adv.shape[0] * H)
    return mse.item()

def compute_mae_tgt(pred_adv, ground_truth, target_dim):
    H = pred_adv.shape[1]
    mae = (torch.abs(pred_adv[:, :, target_dim] - ground_truth[:, :, target_dim])).sum() / (pred_adv.shape[0] * H)
    return mae.item()

def compute_prs(pred_clean, pred_adv, ground_truth, target_dim, gamma=1e-8):
    clean_residuals = pred_clean[:, :, target_dim] - ground_truth[:, :, target_dim]
    adv_residuals = pred_adv[:, :, target_dim] - ground_truth[:, :, target_dim]

    rmse_clean = torch.sqrt((clean_residuals ** 2).mean()).item()
    rmse_adv = torch.sqrt((adv_residuals ** 2).mean()).item()

    prs = min(np.exp(1 - rmse_adv / (rmse_clean + gamma)), 1.0)
    return prs

def compute_mse_nt(pred_clean, pred_adv, target_dim):
    M = pred_clean.shape[-1]
    H = pred_clean.shape[1]
    nt_mask = torch.ones(M, dtype=torch.bool, device=pred_clean.device)
    nt_mask[target_dim] = False

    diff = pred_adv[:, :, nt_mask] - pred_clean[:, :, nt_mask]
    mse_nt = (diff ** 2).sum() / (pred_clean.shape[0] * H * (M - 1))
    return mse_nt.item()

def compute_mae_nt(pred_clean, pred_adv, target_dim):
    M = pred_clean.shape[-1]
    H = pred_clean.shape[1]
    nt_mask = torch.ones(M, dtype=torch.bool, device=pred_clean.device)
    nt_mask[target_dim] = False

    diff = pred_adv[:, :, nt_mask] - pred_clean[:, :, nt_mask]
    mae_nt = torch.abs(diff).sum() / (pred_clean.shape[0] * H * (M - 1))
    return mae_nt.item()

def compute_cga(delta, perturb_mask, gamma=1e-8):
    perturb_mask_t = torch.tensor(perturb_mask, dtype=torch.bool, device=delta.device)
    parent_norm = torch.norm(delta[:, :, perturb_mask_t], p='fro').item()
    total_norm = torch.norm(delta, p='fro').item()
    cga = parent_norm / (total_norm + gamma)
    return cga

def compute_pe(pred_clean, pred_adv, delta, target_dim, gamma=1e-8):
    tgt_shift = torch.norm(pred_adv[:, :, target_dim] - pred_clean[:, :, target_dim], p=2).item()
    delta_norm = torch.norm(delta, p='fro').item()
    pe = tgt_shift / (delta_norm + gamma)
    return pe

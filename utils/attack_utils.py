from __future__ import annotations

import os
import time

import numpy as np
import torch

from metrics import (
    compute_css, compute_mse_tgt, compute_mae_tgt,
    compute_prs, compute_mse_nt, compute_mae_nt,
    compute_cga, compute_pe,
)


def pgd_attack(model, X_left, X_right, target_dim, perturb_mask_t, epsilon, alpha, pgd_steps, lam_ntgt, lam_smooth):
    model.eval()
    B, T, M = X_left.shape
    H = X_right.shape[1]

    # Clean prediction (compute once, frozen)
    with torch.no_grad():
        pred_clean, _ = model(X_left, future=H)

    # Non-target channel mask
    non_target_mask = torch.ones(M, dtype=torch.bool, device=X_left.device)
    non_target_mask[target_dim] = False

    # Initialise perturbation
    delta = torch.rand_like(X_left, requires_grad=True)

    loss_history = {'total': [], 'tgt': [], 'nt': [], 'smooth': []}

    torch.backends.cudnn.enabled = False  # Disable CuDNN for deterministic behavior (optional, may slow down)
    for step in range(pgd_steps):
        if delta.grad is not None:
            delta.grad.zero_()

        X_adv = X_left + delta
        pred_adv, _ = model(X_adv, future=H)

        # L_tgt: maximise forecast error on target dim
        Ltgt = ((pred_adv[:, :, target_dim] - X_right[:, :, target_dim]) ** 2).mean()

        # L_nt: penalise spillover to non-target forecasts (vs clean, not ground truth)
        Lnt = ((pred_adv[:, :, non_target_mask] - pred_clean[:, :, non_target_mask]) ** 2).mean()

        # L_smooth: temporal smoothness of perturbation
        Lsmooth = ((delta[:, 1:, :] - delta[:, :-1, :]) ** 2).mean()

        model.zero_grad()

        # Minimise: L(delta) = -L_tgt + lam1*L_nt + lam2*L_smooth
        loss = -Ltgt + lam_ntgt * Lnt + lam_smooth * Lsmooth
        loss.backward()

        with torch.no_grad():
            grad_sign = delta.grad.sign()
            delta_new = delta - alpha * grad_sign

            # === Two-stage projection onto S(G,j), epsilon ===
            # Stage 1: L-inf clipping on allowed channels
            delta_new[:, :, perturb_mask_t] = torch.clamp(
                delta_new[:, :, perturb_mask_t], -epsilon, epsilon
            )
            # Stage 2: Structural zeroing of disallowed channels
            delta_new[:, :, ~perturb_mask_t] = 0.0

            delta = delta_new.detach().requires_grad_(True)

        loss_history['total'].append(loss.item())
        loss_history['tgt'].append(Ltgt.item())
        loss_history['nt'].append(Lnt.item())
        loss_history['smooth'].append(Lsmooth.item())

    torch.backends.cudnn.enabled = True  # Re-enable CuDNN after attack loop (optional)

    # Final adversarial prediction
    X_adv_final = X_left + delta.detach()
    with torch.no_grad():
        pred_adv_final, _ = model(X_adv_final, future=H)

    return delta.detach(), pred_clean, pred_adv_final, loss_history


def run_condition(model, val_dl, X_val_right, target_dim, perturb_mask, condition_name, device, epsilon, alpha, pgd_steps, lam_ntgt, lam_smooth):
    perturb_mask_t = torch.tensor(perturb_mask, dtype=torch.bool, device=device)

    agg_pred_clean = []
    agg_pred_adv = []
    agg_deltas = []
    agg_inputs = []
    agg_gt = []
    batch_loss_histories = []
    start = time.time()
    for batch_idx, (X_l, X_r) in enumerate(val_dl):
        X_l, X_r = X_l.to(device), X_r.to(device)

        delta, pred_clean, pred_adv, loss_hist = pgd_attack(
            model, X_l, X_r, target_dim, perturb_mask_t,
            epsilon, alpha, pgd_steps, lam_ntgt, lam_smooth
        )

        agg_pred_clean.append(pred_clean.cpu())
        agg_pred_adv.append(pred_adv.cpu())
        agg_deltas.append(delta.cpu())
        agg_inputs.append(X_l.cpu())
        agg_gt.append(X_r.cpu())
        batch_loss_histories.append(loss_hist)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    end = time.time()

    # Aggregate across batches
    agg_pred_clean = torch.cat(agg_pred_clean, dim=0)
    agg_pred_adv = torch.cat(agg_pred_adv, dim=0)
    agg_deltas = torch.cat(agg_deltas, dim=0)
    agg_inputs = torch.cat(agg_inputs, dim=0)
    agg_gt = torch.cat(agg_gt, dim=0)
    X_right_t = torch.FloatTensor(np.array(X_val_right))

    # --- Existing metrics ---
    mse_tgt = compute_mse_tgt(agg_pred_adv, X_right_t, target_dim)
    mae_tgt = compute_mae_tgt(agg_pred_adv, X_right_t, target_dim)
    clean_mse_tgt = compute_mse_tgt(agg_pred_clean, X_right_t, target_dim)  # for PRS baseline
    clean_mae_tgt = compute_mae_tgt(agg_pred_clean, X_right_t, target_dim)
    prs = compute_prs(agg_pred_clean, agg_pred_adv, X_right_t, target_dim)
    mse_nt = compute_mse_nt(agg_pred_clean, agg_pred_adv, target_dim)  # adv vs clean
    clean_mse_nt = compute_mse_nt(agg_pred_clean, X_right_t, target_dim)
    mae_nt = compute_mae_nt(agg_pred_clean, agg_pred_adv, target_dim)  # adv vs clean
    clean_mae_nt = compute_mae_nt(agg_pred_clean, X_right_t, target_dim)

    # --- Novel metrics ---
    css = compute_css(agg_pred_clean, agg_pred_adv, target_dim)
    cga = compute_cga(agg_deltas, perturb_mask)
    pe = compute_pe(agg_pred_clean, agg_pred_adv, agg_deltas, target_dim)

    # Perturbation stats
    delta_l2_avg = (torch.norm(agg_deltas, p=2) / agg_deltas.shape[0]).item()
    delta_linf = torch.max(torch.abs(agg_deltas)).item()
    delta_fro = torch.norm(agg_deltas, p='fro').item()

    h, m, s = _hms(end - start)

    print(f"  Clean MSEtgt: {clean_mse_tgt:.6f} | Clean MAEtgt: {clean_mae_tgt:.6f}", flush=True)
    print(f"  MSEtgt: {mse_tgt:.6f} | MAEtgt: {mae_tgt:.6f} | PRS: {prs:.6f}", flush=True)
    print(f"  Clean MSEnt: {clean_mse_nt:.6f} | Clean MAEnt: {clean_mae_nt:.6f}", flush=True)
    print(f"  MSEnt:  {mse_nt:.6f} | MAEnt:  {mae_nt:.6f}", flush=True)
    print(f"  CSS: {css:.4f} | CGA: {cga:.4f} | PE: {pe:.4f}", flush=True)
    print(f"  Delta L2-Avg: {delta_l2_avg:.4f} | Delta Fro: {delta_fro:.4f} | Delta Linf: {delta_linf:.4f}", flush=True)

    results = {
        'condition': condition_name,
        'mse_tgt': mse_tgt, 'mae_tgt': mae_tgt,
        'clean_mse_tgt': clean_mse_tgt, 'clean_mae_tgt': clean_mae_tgt,
        'prs': prs,
        'mse_nt': mse_nt, 'mae_nt': mae_nt,
        'clean_mse_nt': clean_mse_nt, 'clean_mae_nt': clean_mae_nt,
        'css': css, 'cga': cga, 'pe': pe,
        'delta_l2_avg': delta_l2_avg, 'delta_fro': delta_fro, 'delta_linf': delta_linf,
        'perturbed_channels': np.where(perturb_mask)[0].tolist(),
        'num_perturbed_channels': int(perturb_mask.sum()),
        'inputs': agg_inputs.numpy(),
        'attack_vectors': agg_deltas.numpy(),
        'ground_truths': agg_gt.numpy(),
        'pred_clean': agg_pred_clean.numpy(),
        'pred_adv': agg_pred_adv.numpy(),
        'loss_convergence': {
            'total': np.mean([h_['total'] for h_ in batch_loss_histories], axis=0).tolist(),
            'tgt':   np.mean([h_['tgt']   for h_ in batch_loss_histories], axis=0).tolist(),
            'nt':    np.mean([h_['nt']    for h_ in batch_loss_histories], axis=0).tolist(),
            'smooth': np.mean([h_['smooth'] for h_ in batch_loss_histories], axis=0).tolist(),
        },
        'time': {'hr': h, 'mins': m, 'sec': s},
    }

    return results


def _hms(elapsed_seconds):
    h = int(elapsed_seconds / 3600)
    m = int((elapsed_seconds - h * 3600) / 60)
    s = elapsed_seconds - (m * 60 + h * 3600)
    return h, m, s


def save_attack_artifacts(results, metadata_dir, prefix):
    _ARRAY_KEYS = ['inputs', 'attack_vectors', 'ground_truths', 'pred_clean', 'pred_adv']
    arrays = {k: results[k] for k in _ARRAY_KEYS if k in results}
    if not arrays:
        return None

    npz_path = os.path.join(metadata_dir, f'{prefix}_vectors.npz')
    np.savez(npz_path, **arrays)

    for k in _ARRAY_KEYS:
        if k in results:
            results[k] = npz_path

    return npz_path

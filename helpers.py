import numpy as np
import torch, time, os

from metrics import (compute_css, compute_mse_tgt, compute_mae_tgt,
                     compute_prs, compute_mse_nt, compute_mae_nt,
                     compute_cga, compute_pe)
from utils import epoch_time

def pgd_attack(model, X_left, X_right, target_dim, perturb_mask_t, epsilon, alpha, pgd_steps, lam_ntgt, lam_smooth):
    """
    Per-batch PGD attack with two-stage causal projection.
    Args:
        model       : frozen forecasting model
        X_left      : (B, T, M) input context window
        X_right     : (B, H, M) ground truth future
        target_dim  : int, target channel j
        perturb_mask_t : (M,) bool tensor on device, channels allowed to perturb
        epsilon     : perturbation budget (L_infinity norm)
        alpha       : PGD step size
        pgd_steps   : number of PGD iterations
        lam_ntgt    : weight for non-target penalty
        lam_smooth  : weight for temporal smoothness
    Returns:
        delta, pred_clean, pred_adv_final, loss_history
    """
    model.eval()
    _, _, M = X_left.shape
    H = X_right.shape[1]

    with torch.no_grad():
        pred_clean, _ = model(X_left, future=H)

    non_target_mask = torch.ones(M, dtype=torch.bool, device=X_left.device)
    non_target_mask[target_dim] = False

    delta = torch.rand_like(X_left, requires_grad=True)
 
    loss_history = {'total': [], 'tgt': [], 'nt': [], 'smooth': []}

    # Disable CuDNN for deterministic behavior (optional, may slow down)
    torch.backends.cudnn.enabled = False
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

        loss = -Ltgt + lam_ntgt * Lnt + lam_smooth * Lsmooth
        loss.backward()
 
        with torch.no_grad():
            grad_sign = delta.grad.sign()
            delta_new = delta - alpha * grad_sign
 
            # === Two-stage projection ===
            delta_new[:, :, perturb_mask_t] = torch.clamp(
                delta_new[:, :, perturb_mask_t], -epsilon, epsilon
            )
            delta_new[:, :, ~perturb_mask_t] = 0.0
 
            delta = delta_new.detach().requires_grad_(True)
 
        loss_history['total'].append(loss.item())
        loss_history['tgt'].append(Ltgt.item())
        loss_history['nt'].append(Lnt.item())
        loss_history['smooth'].append(Lsmooth.item())
    
    # Re-enable CuDNN after attack loop
    torch.backends.cudnn.enabled = True
 
    # Final adversarial prediction
    X_adv_final = X_left + delta.detach()
    with torch.no_grad():
        pred_adv_final, _ = model(X_adv_final, future=H)
 
    return delta.detach(), pred_clean, pred_adv_final, loss_history

def run_condition(model, val_dl, X_val_right, target_dim, perturb_mask, condition_name, epsilon, alpha, pgd_steps, lam_ntgt, lam_smooth):
    """
    Run PGD attack under a specific perturbation condition across all val batches.
    Returns aggregated results dict.
    """
    perturb_mask_t = torch.tensor(perturb_mask, dtype=torch.bool, device='cuda')
 
    agg_pred_clean = []
    agg_pred_adv = []
    agg_deltas = []
    agg_inputs = []
    agg_gt = []
    batch_loss_histories = []
    start = time.time()
    for batch_idx, (X_l, X_r) in enumerate(val_dl):
        X_l, X_r = X_l.cuda(), X_r.cuda()
 
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
 
        # if (batch_idx + 1) % 10 == 0:
        #     print(f"  [{condition_name}] Batch {batch_idx+1}/{len(val_dl)} | CSS: {batch_css:.4f}", flush=True)
 
        torch.cuda.empty_cache()
 
    end = time.time()

    # Aggregate across batches
    agg_pred_clean = torch.cat(agg_pred_clean, dim=0)
    agg_pred_adv = torch.cat(agg_pred_adv, dim=0)
    agg_deltas = torch.cat(agg_deltas, dim=0)
    agg_inputs = torch.cat(agg_inputs, dim=0)
    agg_gt = torch.cat(agg_gt, dim=0)
    X_right_t = torch.FloatTensor(np.array(X_val_right))

    # --- Existing metrics (5) ---
    mse_tgt = compute_mse_tgt(agg_pred_adv, X_right_t, target_dim)
    mae_tgt = compute_mae_tgt(agg_pred_adv, X_right_t, target_dim)
    clean_mse_tgt = compute_mse_tgt(agg_pred_clean, X_right_t, target_dim) # for PRS baseline
    clean_mae_tgt = compute_mae_tgt(agg_pred_clean, X_right_t, target_dim)
    prs = compute_prs(agg_pred_clean, agg_pred_adv, X_right_t, target_dim)
    mse_nt = compute_mse_nt(agg_pred_clean, agg_pred_adv, target_dim) # adv vs clean
    clean_mse_nt = compute_mse_nt(agg_pred_clean, X_right_t, target_dim)
    mae_nt = compute_mae_nt(agg_pred_clean, agg_pred_adv, target_dim) # adv vs clean
    clean_mae_nt = compute_mae_nt(agg_pred_clean, X_right_t, target_dim)

    # --- Novel metrics (3) ---
    css = compute_css(agg_pred_clean, agg_pred_adv, target_dim)
    cga = compute_cga(agg_deltas, perturb_mask)
    pe = compute_pe(agg_pred_clean, agg_pred_adv, agg_deltas, target_dim)

    # Perturbation stats (unchanged)
    delta_l2_avg = (torch.norm(agg_deltas, p=2) / agg_deltas.shape[0]).item()
    delta_linf = torch.max(torch.abs(agg_deltas)).item()
    delta_fro = torch.norm(agg_deltas, p='fro').item()

    h, m, s = epoch_time(start, end)

    print(f"  Clean MSEtgt: {clean_mse_tgt:.6f} | Clean MAEtgt: {clean_mae_tgt:.6f}", flush=True)
    print(f"  MSEtgt: {mse_tgt:.6f} | MAEtgt: {mae_tgt:.6f} | PRS: {prs:.6f}", flush=True)
    print(f"  Clean MSEnt: {clean_mse_nt:.6f} | Clean MAEnt: {clean_mae_nt:.6f}", flush=True)
    print(f"  MSEnt:  {mse_nt:.6f} | MAEnt:  {mae_nt:.6f}", flush=True)
    print(f"  CSS: {css:.4f} | CGA: {cga:.4f} | PE: {pe:.4f}", flush=True)
    print(f"  Delta L2-Avg: {delta_l2_avg:.4f} | Delta Fro: {delta_fro:.4f} | Delta Linf: {delta_linf:.4f}", flush=True)

    results = {
        'condition': condition_name,
        # Existing metrics
        'mse_tgt': mse_tgt,
        'mae_tgt': mae_tgt,
        'clean_mse_tgt': clean_mse_tgt,
        'clean_mae_tgt': clean_mae_tgt,
        'prs': prs,
        'mse_nt': mse_nt,
        'mae_nt': mae_nt,
        'clean_mse_nt': clean_mse_nt,
        'clean_mae_nt': clean_mae_nt,
        # Novel metrics
        'css': css,
        'cga': cga,
        'pe': pe,
        # Perturbation stats
        'delta_l2_avg': delta_l2_avg,
        'delta_fro': delta_fro,
        'delta_linf': delta_linf,
        'perturbed_channels': np.where(perturb_mask)[0].tolist(),
        'num_perturbed_channels': int(perturb_mask.sum()),
        'inputs': agg_inputs.numpy(),
        'attack_vectors': agg_deltas.numpy(),
        'ground_truths': agg_gt.numpy(),
        'pred_clean': agg_pred_clean.numpy(),
        'pred_adv': agg_pred_adv.numpy(),
        # Averaged convergence curves across all batches (per PGD step)
        'loss_convergence': {
            'total': np.mean([h['total'] for h in batch_loss_histories], axis=0).tolist(),
            'tgt':   np.mean([h['tgt']   for h in batch_loss_histories], axis=0).tolist(),
            'nt':    np.mean([h['nt']    for h in batch_loss_histories], axis=0).tolist(),
            'smooth':np.mean([h['smooth']for h in batch_loss_histories], axis=0).tolist()
        },
        'time': {'hr': h, 'mins': m, 'sec': s}
    }
 
    return results

def get_parent_mask(causal_graph, target_dim):
    """
    Returns boolean mask of parent channels for target_dim j.
    Parent set P_j = {k : A_{j,k} = 1} (column j of the adjacency matrix).
    """
    mask = causal_graph[target_dim] != 0
    return mask.astype(bool)
 
 
def get_nonparent_mask(causal_graph, target_dim):
    """
    Returns boolean mask of non-parent, non-target channels.
    These are channels k ∉ P_j and k ≠ j.
    """
    n_dim = causal_graph.shape[0]
    parent_mask = get_parent_mask(causal_graph, target_dim)
    nonparent_mask = ~parent_mask
    nonparent_mask[target_dim] = False  # exclude target itself
    return nonparent_mask
 
 
def get_random_mask(n_dim, n_select, exclude_dim=None, seed=None):
    """
    Returns boolean mask selecting n_select random channels.
    Optionally excludes a specific dimension (the target) from selection pool.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
 
    candidates = list(range(n_dim))
    if exclude_dim is not None:
        candidates.remove(exclude_dim)
 
    n_select = min(n_select, len(candidates))
    chosen = rng.choice(candidates, size=n_select, replace=False)
 
    mask = np.zeros(n_dim, dtype=bool)
    mask[chosen] = True
    return mask

def save_attack_artifacts(results, metadata_dir, prefix):
    """
    Extract numpy arrays from a results dict, save them as a single .npz,
    and replace the dict entries with the file path so the dict stays
    JSON-serializable.

    Saved .npz contents (all aligned on axis 0 by sample index):
        inputs[i]          — (T, M) input context window for sample i
        attack_vectors[i]  — (T, M) adversarial perturbation δ for sample i
        ground_truths[i]   — (H, M) ground-truth future for sample i
        pred_clean[i]      — (H, M) clean model forecast for sample i
        pred_adv[i]        — (H, M) adversarial model forecast for sample i

    Args:
        results   : dict returned by run_condition() — modified in place
        save_dir  : directory to save the .npz file
        prefix    : filename prefix

    Returns:
        path to the saved .npz file
    """
    # Keys in the results dict that hold numpy arrays (not JSON-serializable).
    _ARRAY_KEYS = ['inputs', 'attack_vectors', 'ground_truths', 'pred_clean', 'pred_adv']
    arrays = {k: results[k] for k in _ARRAY_KEYS if k in results}
    if not arrays:
        return None

    # os.makedirs(save_dir, exist_ok=True)
    npz_path = os.path.join(metadata_dir, f'{prefix}_vectors.npz')
    np.savez(npz_path, **arrays)

    # Replace arrays with path string so the dict can be JSON-serialized
    for k in _ARRAY_KEYS:
        if k in results:
            results[k] = npz_path # all keys point to the same file

    return npz_path
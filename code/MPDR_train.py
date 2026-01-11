#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import copy
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# =======================
# Hyperparameters
# =======================
SEED = 42
EPOCHS = 1000
LR = 1e-2
MB = 20
DEVICE = "cpu"
THRESHOLD = 1e-6

# =======================
# Reproducibility
# =======================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

set_seed(SEED)

# =======================
# CLI
# =======================
def parse_args():
    p = argparse.ArgumentParser(description="MPDR Dirichlet-MLP (diet+microbiome concat) training and recommendation")
    p.add_argument("--p_train", required=True, help="Training target microbiome composition (healthy), CSV")
    p.add_argument("--z_train", required=True, help="Training baseline microbiome state (healthy), CSV")
    p.add_argument("--q_train", required=True, help="Training diet matrix (healthy), CSV")

    p.add_argument("--z_start", required=True, help="Baseline microbiome state to start from (disease), CSV")
    p.add_argument("--q_start", required=True, help="Initial diet for optimization (disease random), CSV")
    p.add_argument("--p_target", required=True, help="Desired target microbiome composition (disease desired), CSV")

    p.add_argument("--out_dir", required=True, help="Output directory root")
    p.add_argument("--tag", default="MPDR", help="Subfolder tag under out_dir")
    p.add_argument("--max_reco", type=int, default=400, help="Max number of samples to recommend")

    return p.parse_args()

# =======================
# Safe helper
# =======================
def _safe(t: torch.Tensor, *, min_val: float = -1e6, max_val: float = 1e6, fill: float = 0.0) -> torch.Tensor:
    t = torch.nan_to_num(t, nan=fill, posinf=max_val, neginf=min_val)
    return torch.clamp(t, min=min_val, max=max_val)

# =======================
# Utilities
# =======================
def process_data(P: np.ndarray) -> torch.Tensor:
    """
    Input: P of shape (N_s, n_samples).
    Normalize each column to sum 1 and return tensor of shape (n_samples, N_s).
    """
    col_sums = P.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    P = P / col_sums
    P = P.astype(np.float32)
    return torch.from_numpy(P.T)

def compute_microbiome_mask(prev_microbiome: torch.Tensor, threshold_val: float = 0.0) -> torch.Tensor:
    return (prev_microbiome > threshold_val).float()

def get_batch(ztrn: torch.Tensor, ptrn: torch.Tensor, Q1: torch.Tensor, mb_size: int):
    n = ptrn.size(0)
    mb = min(mb_size, n)
    idx = np.random.choice(np.arange(n, dtype=np.int64), mb, replace=False)
    s = torch.from_numpy(idx)
    batch_p = ztrn[s, :].to(DEVICE)
    batch_q = ptrn[s, :].to(DEVICE)
    batch_d = Q1[s, :].to(DEVICE)
    return batch_p, batch_q, batch_d

def bray_curtis(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = _safe(p, min_val=0.0, max_val=1.0)
    q = _safe(q, min_val=0.0, max_val=1.0)
    num = (p - q).abs().sum(dim=1)
    den = (p + q).sum(dim=1).clamp(min=eps)
    return num / den

def tensor_to_np_row(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().reshape(-1)

# =======================
# Loss: masked Dirichlet
# =======================
def masked_dirichlet_loss(pred_alpha: torch.Tensor, true_abund: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_alpha = _safe(pred_alpha, min_val=eps, max_val=1e6, fill=eps)
    true_abund = _safe(true_abund, min_val=0.0, max_val=1e6, fill=0.0)
    mask = (mask > 0).float()

    msum = mask.sum(dim=1, keepdim=True)
    all_zeros = (msum == 0)
    if all_zeros.any():
        mask = mask.clone()
        mask[all_zeros.squeeze(-1)] = 1.0
        msum = mask.sum(dim=1, keepdim=True)

    true = true_abund * mask
    denom_true = true.sum(dim=1, keepdim=True).clamp(min=eps)
    true = true / denom_true
    true = _safe(true, min_val=eps, max_val=1.0, fill=eps)
    true = true / true.sum(dim=1, keepdim=True).clamp(min=eps)

    alpha = pred_alpha * mask + (1.0 - mask) * eps
    alpha = _safe(alpha, min_val=eps, max_val=1e6, fill=eps)

    dist = torch.distributions.Dirichlet(alpha)
    nll = -dist.log_prob(true)
    return nll.mean()

# =======================
# Model
# =======================
class DietMicrobiomeMLPConcat(nn.Module):
    """
    Concatenate diet and previous microbiome abundances and predict Dirichlet alpha.
    """
    def __init__(self, diet_input_dim: int, microbe_output_dim: int, hidden: int = 256):
        super().__init__()
        in_dim = diet_input_dim + microbe_output_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Linear(hidden, microbe_output_dim)
        )

    def forward(self, diet_input: torch.Tensor, prev_microbiome: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        diet_input = _safe(diet_input.float())
        prev_microbiome = _safe(prev_microbiome)
        mask = (mask > 0).float()
        x = torch.cat([diet_input, prev_microbiome], dim=1)
        alpha_raw = _safe(self.mlp(x))
        alpha = F.softplus(torch.clamp(alpha_raw, min=-50.0, max=50.0), beta=1.0, threshold=20.0) + 1e-6
        alpha = _safe(alpha, min_val=1e-6, max_val=1e6)
        alpha = alpha * mask + (1.0 - mask) * 1e-6
        return alpha

# =======================
# Train and select best model
# =======================
def train_and_select_best(max_epochs, mb, lr,
                          ztrn, ptrn, zval, pval, Q1, Q3):
    diet_dim = Q1.shape[1]
    microbe_dim = ptrn.shape[1]
    model = DietMicrobiomeMLPConcat(diet_input_dim=diet_dim, microbe_output_dim=microbe_dim, hidden=256).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float("inf")
    best_model = None
    train_curve, val_curve = [], []

    for _ in range(max_epochs):
        model.train()
        optimizer.zero_grad()

        batch_p, batch_q, batch_d = get_batch(ztrn, ptrn, Q1, mb)
        mask = (compute_microbiome_mask(batch_p) > 0).float()
        alpha_pred = model(batch_d, batch_p, mask)
        loss = masked_dirichlet_loss(alpha_pred, batch_q, mask)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_curve.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            mask_val = (compute_microbiome_mask(zval) > 0).float()
            alpha_val = model(_safe(Q3.to(DEVICE)), _safe(zval.to(DEVICE)), mask_val.to(DEVICE))
            l_val = masked_dirichlet_loss(alpha_val, _safe(pval.to(DEVICE), min_val=0.0), mask_val.to(DEVICE))
            v = float(l_val.item())
            val_curve.append(v)
            if v < best_loss:
                best_loss = v
                best_model = copy.deepcopy(model)

    if best_model is None:
        best_model = model
    best_model.eval()
    return best_model, np.array(train_curve, dtype=np.float32), np.array(val_curve, dtype=np.float32)

# =======================
# Inference helper
# =======================
def predict_probs(model, diet_row: torch.Tensor, prev_micro: torch.Tensor) -> torch.Tensor:
    mask = (compute_microbiome_mask(prev_micro) > 0).float()
    msum = mask.sum(dim=1, keepdim=True)
    all_zero = (msum == 0)
    if all_zero.any():
        idx = prev_micro.argmax(dim=1)
        mask = mask.clone()
        mask[all_zero.squeeze(-1)] = 0.0
        mask[torch.arange(mask.size(0), device=mask.device), idx] = 1.0

    alpha = model(_safe(diet_row), _safe(prev_micro), mask)
    probs = _safe(alpha * mask, min_val=0.0, max_val=1e6)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return probs

# =======================
# Diet optimization (gradient)
# =======================
def optimize_diet_for_target(
    model,
    prev_micro_start: torch.Tensor,
    target_micro_end: torch.Tensor,
    init_diet: torch.Tensor,
    steps: int = 100,
    step_size: float = 0.05,
    early_stop_patience: int = 30,
    normalize: bool = True,
):
    device = init_diet.device

    diet = init_diet.clone().to(device)
    diet = torch.clamp(diet, min=0.0)
    if normalize:
        diet = diet / (diet.sum(dim=1, keepdim=True) + 1e-12)

    diet_var = diet.detach().clone().requires_grad_(True)

    best_loss = float("inf")
    best_diet = None
    patience_left = early_stop_patience
    hist = []

    for _ in range(steps):
        if diet_var.grad is not None:
            diet_var.grad.zero_()

        pred_end = predict_probs(model, diet_var, prev_micro_start)
        loss = bray_curtis(pred_end, target_micro_end).mean()
        loss.backward()

        cur = float(loss.item())
        hist.append(cur)

        if cur + 1e-9 < best_loss:
            best_loss = cur
            best_diet = diet_var.detach().clone()
            patience_left = early_stop_patience
        else:
            if early_stop_patience is not None:
                patience_left -= 1
                if patience_left <= 0:
                    break

        with torch.no_grad():
            g = diet_var.grad
            if g is None:
                break
            g = torch.where(torch.isfinite(g), g, torch.zeros_like(g))

            max_abs = g.abs().amax(dim=1, keepdim=True)
            max_abs = torch.clamp(max_abs, min=1e-8)

            step = -step_size * g / max_abs
            new_diet = diet_var + step
            new_diet = torch.clamp(new_diet, min=0.0)
            if normalize:
                new_diet = new_diet / (new_diet.sum(dim=1, keepdim=True) + 1e-12)

            diet_var.data.copy_(new_diet)

    if best_diet is None:
        best_diet = diet_var.detach()

    return best_diet, {
        "bc_history": np.array(hist, np.float32),
        "best_bc": float(best_loss),
        "steps_ran": len(hist),
    }

# =======================
# Main
# =======================
def main():
    args = parse_args()

    out_dir = os.path.join(args.out_dir, args.tag)
    os.makedirs(out_dir, exist_ok=True)

    # Load train P/Z/target
    P1 = np.loadtxt(args.p_train, delimiter=",")
    Z1 = np.loadtxt(args.z_train, delimiter=",")
    if P1.ndim == 1:
        P1 = P1.reshape(-1, 1)
    if Z1.ndim == 1:
        Z1 = Z1.reshape(-1, 1)

    if P1.shape[0] >= P1.shape[1]:
        P1 = P1.T
        if Z1.shape[0] >= Z1.shape[1]:
            Z1 = Z1.T
        if Z1.shape[0] != P1.shape[0] and Z1.shape[1] == P1.shape[0]:
            Z1 = Z1.T

    N_s = P1.shape[0]
    n_samples = P1.shape[1]
    assert Z1.shape[0] == N_s

    # Load target microbiome endpoints
    PTARGET = np.loadtxt(args.p_target, delimiter=",")
    if PTARGET.ndim == 1:
        PTARGET = PTARGET.reshape(-1, 1)
    if PTARGET.shape[0] != N_s and PTARGET.shape[1] == N_s:
        PTARGET = PTARGET.T
    if PTARGET.shape[0] >= PTARGET.shape[1] and PTARGET.shape[0] == n_samples:
        # if accidentally sample x species, fix
        PTARGET = PTARGET.T
    assert PTARGET.shape[0] == N_s

    # Load train Q (healthy diet) raw
    Qtrain_raw = np.loadtxt(args.q_train, delimiter=",")
    if Qtrain_raw.ndim == 1:
        Qtrain_raw = Qtrain_raw.reshape(1, -1)

    # Ensure (N_f, n_samples): rows = foods, cols = samples
    if Qtrain_raw.shape[1] != n_samples and Qtrain_raw.shape[0] == n_samples:
        Qtrain_raw = Qtrain_raw.T
    assert Qtrain_raw.shape[1] == n_samples

    N_f = Qtrain_raw.shape[0]

    # Log-transform for training (log10(count + 1))
    Q1_raw_all = np.log10(Qtrain_raw + 1.0)

    P1[P1 < THRESHOLD] = 0.0

    # Train/val split
    val_size = max(1, int(0.2 * n_samples))
    random_indices = np.random.choice(n_samples, size=val_size, replace=False)
    train_indices = np.setdiff1d(np.arange(n_samples), random_indices)

    P_val = P1[:, random_indices]
    Z_val = Z1[:, random_indices]
    Q3_raw = Q1_raw_all[:, random_indices]

    P_train = P1[:, train_indices]
    Z_train = Z1[:, train_indices]
    Q1_raw_train = Q1_raw_all[:, train_indices]

    ptrn = process_data(P_train)
    ztrn = process_data(Z_train)
    pval = process_data(P_val)
    zval = process_data(Z_val)

    Q1 = torch.from_numpy(Q1_raw_train.T.astype(np.float32))
    Q3 = torch.from_numpy(Q3_raw.T.astype(np.float32))

    print(f"[INFO] N_s={N_s} species, N_f={N_f} diet features, n_train={ptrn.size(0)}, n_val={pval.size(0)}")

    # Load start Z (baseline microbiome for recommendation)
    Z_start = np.loadtxt(args.z_start, delimiter=",")
    if Z_start.ndim == 1:
        Z_start = Z_start.reshape(-1, 1)
    if Z_start.shape[0] != N_s and Z_start.shape[1] == N_s:
        Z_start = Z_start.T
    Z_start[Z_start < THRESHOLD] = 0.0
    ztst = process_data(Z_start)

    # Load start Q (initial diet for recommendation)
    Q_start = np.loadtxt(args.q_start, delimiter=",")
    Q_start = np.log10(Q_start + 1.0)
    if Q_start.ndim == 1:
        Q_start = Q_start.reshape(1, -1)
    if Q_start.shape[0] != N_f and Q_start.shape[1] == N_f:
        Q_start = Q_start.T
    assert Q_start.shape[0] == N_f
    assert Q_start.shape[1] == ztst.size(0)
    Q2 = torch.from_numpy(Q_start.T.astype(np.float32))

    # Train model
    best_model, train_curve, val_curve = train_and_select_best(
        max_epochs=EPOCHS, mb=MB, lr=LR,
        ztrn=ztrn, ptrn=ptrn, zval=zval, pval=pval, Q1=Q1, Q3=Q3
    )

    np.savetxt(os.path.join(out_dir, "self_train_loss.csv"), train_curve, delimiter=",")
    np.savetxt(os.path.join(out_dir, "self_val_loss.csv"), val_curve, delimiter=",")

    ckpt_path = os.path.join(out_dir, "Train_best_model.pt")
    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "model_type": "MLP_concat",
            "N_s": N_s,
            "N_f": N_f,
            "seed": SEED,
            "epochs": EPOCHS,
            "lr": LR,
            "mb": MB,
        },
        ckpt_path,
    )

    # =======================
    # Multi-sample recommendation
    # =======================
    def multi_sample_recommendations(max_reco: int):
        n_start = ztst.size(0)
        n_target = PTARGET.shape[1]
        n_reco = min(max_reco, n_start, n_target)

        diets_mat = np.zeros((n_reco, N_f), dtype=np.float32)
        preds_mat = np.zeros((n_reco, N_s), dtype=np.float32)

        for i in range(n_reco):
            prev_micro_start = ztst[i : i + 1, :].to(DEVICE)
            target_end_tensor = process_data(PTARGET[:, i : i + 1]).to(DEVICE)

            if Q2.size(0) > i:
                init_diet = Q2[i : i + 1, :].clone().to(DEVICE)
            else:
                init_diet = torch.zeros((1, N_f), dtype=torch.float32, device=DEVICE)

            diet_opt, opt_hist = optimize_diet_for_target(
                best_model,
                prev_micro_start=prev_micro_start,
                target_micro_end=target_end_tensor,
                init_diet=init_diet,
                normalize=False,
            )

            with torch.no_grad():
                pred_endpoint = predict_probs(best_model, diet_opt, prev_micro_start)

            diets_mat[i, :] = tensor_to_np_row(diet_opt)
            preds_mat[i, :] = tensor_to_np_row(pred_endpoint)

            if (i % 20 == 0) or (i == n_reco - 1):
                tail = opt_hist["bc_history"][-5:] if len(opt_hist["bc_history"]) >= 5 else opt_hist["bc_history"]
                print(f"[Reco][{i}/{n_reco}] best_BC={opt_hist['best_bc']:.6f}; tail {np.array(tail)}")

        np.savetxt(os.path.join(out_dir, "self_diet_recommendations.csv"), diets_mat, delimiter=",")
        np.savetxt(os.path.join(out_dir, "self_pred_endpoints.csv"), preds_mat, delimiter=",")
        print(f"[OK] Wrote diet_recommendations.csv ({diets_mat.shape}) and pred_endpoints.csv ({preds_mat.shape})")

    multi_sample_recommendations(max_reco=args.max_reco)
    print(f"[OK] Results in: {out_dir}")

if __name__ == "__main__":
    main()

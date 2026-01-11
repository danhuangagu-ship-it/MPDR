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

# -----------------------
# CLI arguments (file-input interface)
# -----------------------
parser = argparse.ArgumentParser(
    description="composition mapping with masked Dirichlet loss (MLP on [diet||prev_micro]) + diet recommendation (DMAS-stable)"
)

parser.add_argument("--p_train", required=True, help="Path to p_healthy.csv")
parser.add_argument("--z_train", required=True, help="Path to z_healthy.csv")
parser.add_argument("--q_train", required=True, help="Path to q_healthy.csv")

parser.add_argument("--p_target", required=True, help="Path to p_target.csv (desired endpoint)")
parser.add_argument("--z_start", required=True, help="Path to z_start.csv (baseline prev microbiome)")
parser.add_argument("--q_start", required=True, help="Path to q_start.csv (initial diet for recommendation)")

parser.add_argument("--out_dir", default="./results", help="Output directory")
parser.add_argument("--tag", default="MPDR_run", help="Tag used in output filenames")

parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--mb", type=int, default=20)

# Diet optimization settings (DMAS-style)
parser.add_argument("--diet_steps", type=int, default=400)
parser.add_argument("--diet_step_size", type=float, default=0.05, help="L-infinity step size per iteration (DMAS-style)")
parser.add_argument("--diet_early_stop", type=int, default=30, help="Early stop patience (DMAS-style)")
parser.add_argument("--diet_normalize", action="store_true", help="If set, normalize diet to sum 1 each step (DMAS default was False)")

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--threshold", type=float, default=1e-6)

args = parser.parse_args()

# -----------------------
# Globals
# -----------------------
DEVICE = "cpu"
THRESHOLD = float(args.threshold)

# -----------------------
# Reproducibility
# -----------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(args.seed)

# -----------------------
# Safe helper
# -----------------------
def _safe(t: torch.Tensor, *, min_val: float = -1e6, max_val: float = 1e6, fill: float = 0.0) -> torch.Tensor:
    t = torch.nan_to_num(t, nan=fill, posinf=max_val, neginf=min_val)
    return torch.clamp(t, min=min_val, max=max_val)

# -----------------------
# IO utilities
# -----------------------
def load_csv(path: str) -> np.ndarray:
    x = np.loadtxt(path, delimiter=",")
    if x.ndim == 0:
        x = x.reshape(1, 1)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# -----------------------
# Data utilities (match DMAS behavior)
# -----------------------
def process_data(P: np.ndarray) -> torch.Tensor:
    """
    Input: P of shape (N_s, n_samples).
    Normalize each column to sum 1 and return tensor of shape (n_samples, N_s).
    """
    col_sums = P.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    P = (P / col_sums).astype(np.float32)
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

# -----------------------
# Loss: masked Dirichlet (match DMAS stable version)
# -----------------------
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

# -----------------------
# Model (match DMAS stable version)
# -----------------------
class DietMicrobiomeMLPConcat(nn.Module):
    """
    Concatenate diet and previous microbiome abundances and predict Dirichlet alpha.
    DMAS-stable: Linear -> Linear (no ReLU), then clamp + softplus.
    """
    def __init__(self, diet_input_dim: int, microbe_output_dim: int, hidden: int = 256):
        super().__init__()
        in_dim = diet_input_dim + microbe_output_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Linear(hidden, microbe_output_dim),
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

# -----------------------
# Train and select best model (match DMAS stable version)
# -----------------------
def train_and_select_best(max_epochs, mb, lr, ztrn, ptrn, zval, pval, Q1, Q3):
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

# -----------------------
# Inference helper (IMPORTANT: must be differentiable for diet optimization)
# -----------------------
def predict_probs(model, diet_row: torch.Tensor, prev_micro: torch.Tensor) -> torch.Tensor:
    """
    Differentiable w.r.t. diet_row. Do NOT wrap with @torch.no_grad().
    Use torch.no_grad() only at call sites where you do not need gradients.
    """
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

# -----------------------
# Diet optimization (match DMAS stable version)
# -----------------------
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

# -----------------------
# Shape alignment helpers (robust)
# -----------------------
def ensure_microbe_shape_S_by_n(X: np.ndarray, N_s: int = None) -> np.ndarray:
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if N_s is None:
        return X
    if X.shape[0] == N_s:
        return X
    if X.shape[1] == N_s:
        return X.T
    raise ValueError(f"Cannot align microbiome matrix to N_s={N_s}: shape={X.shape}")

def ensure_diet_shape_F_by_n(Q: np.ndarray, n: int = None) -> np.ndarray:
    if Q.ndim == 1:
        Q = Q.reshape(1, -1)
    if n is None:
        return Q
    if Q.shape[1] == n:
        return Q
    if Q.shape[0] == n:
        return Q.T
    raise ValueError(f"Cannot align diet matrix to n={n}: shape={Q.shape}")

# -----------------------
# Main
# -----------------------
def main():
    ensure_dir(args.out_dir)

    # Load microbiome matrices
    P1 = load_csv(args.p_train)
    Z1 = load_csv(args.z_train)
    PTARGET = load_csv(args.p_target)
    Z_start_np = load_csv(args.z_start)

    # DMAS-style transpose heuristic for microbiomes
    if P1.shape[0] >= P1.shape[1]:
        P1 = P1.T
        Z1 = Z1.T if Z1.shape[0] >= Z1.shape[1] else Z1
        if Z1.shape[0] != P1.shape[0] and Z1.shape[1] == P1.shape[0]:
            Z1 = Z1.T
        if PTARGET.shape[0] >= PTARGET.shape[1]:
            PTARGET = PTARGET.T

    N_s, n_samples = P1.shape
    if Z1.shape[0] != N_s:
        if Z1.shape[1] == N_s:
            Z1 = Z1.T
        else:
            raise ValueError(f"z_train incompatible with p_train: p_train={P1.shape}, z_train={Z1.shape}")

    # Load diet train, align, and apply log10(+1)
    Qtrain_raw = load_csv(args.q_train)
    Qtrain_raw = ensure_diet_shape_F_by_n(Qtrain_raw, n=n_samples)
    N_f = Qtrain_raw.shape[0]
    Q1_raw_all = np.log10(Qtrain_raw + 1.0)

    # Threshold P
    P1 = P1.copy()
    P1[P1 < THRESHOLD] = 0.0

    # Train/val split
    val_size = max(1, int(0.2 * n_samples))
    vidx = np.random.choice(n_samples, size=val_size, replace=False)
    tidx = np.setdiff1d(np.arange(n_samples), vidx)

    P_train = P1[:, tidx]
    Z_train = Z1[:, tidx]
    Q_train = Q1_raw_all[:, tidx]

    P_val = P1[:, vidx]
    Z_val = Z1[:, vidx]
    Q_val = Q1_raw_all[:, vidx]

    ptrn = process_data(P_train)
    ztrn = process_data(Z_train)
    pval = process_data(P_val)
    zval = process_data(Z_val)

    Q1 = torch.from_numpy(Q_train.T.astype(np.float32))
    Q3 = torch.from_numpy(Q_val.T.astype(np.float32))

    print(f"[INFO] N_s={N_s} N_f={N_f} n_train={ptrn.size(0)} n_val={pval.size(0)}")

    # Load z_start and align to N_s
    Z_start_np = ensure_microbe_shape_S_by_n(Z_start_np, N_s=N_s)
    ztst = process_data(Z_start_np)

    # Align p_target to (N_s, n_target)
    if PTARGET.shape[0] != N_s and PTARGET.shape[1] == N_s:
        PTARGET = PTARGET.T
    if PTARGET.shape[0] != N_s:
        raise ValueError(f"p_target incompatible with N_s={N_s}: p_target={PTARGET.shape}")

    # Load q_start and align to N_f x n_test, apply log10(+1)
    Q_start_np = load_csv(args.q_start)
    Q_start_np = np.log10(Q_start_np + 1.0)
    Q_start_np = ensure_diet_shape_F_by_n(Q_start_np, n=ztst.size(0))
    if Q_start_np.shape[0] != N_f:
        raise ValueError(f"q_start incompatible with N_f={N_f}: q_start={Q_start_np.shape}")
    Q2 = torch.from_numpy(Q_start_np.T.astype(np.float32))

    # Train model
    best_model, train_curve, val_curve = train_and_select_best(
        max_epochs=args.epochs,
        mb=args.mb,
        lr=args.lr,
        ztrn=ztrn,
        ptrn=ptrn,
        zval=zval,
        pval=pval,
        Q1=Q1,
        Q3=Q3,
    )

    # Save curves + checkpoint
    np.savetxt(os.path.join(args.out_dir, f"train_loss_{args.tag}.csv"), train_curve, delimiter=",")
    np.savetxt(os.path.join(args.out_dir, f"val_loss_{args.tag}.csv"), val_curve, delimiter=",")

    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "model_type": "MLP_concat_DMAS_stable",
            "N_s": N_s,
            "N_f": N_f,
            "args": vars(args),
        },
        os.path.join(args.out_dir, f"best_model_{args.tag}.pt"),
    )

    # Recommendation
    n_reco = min(400, ztst.size(0), PTARGET.shape[1])
    diets_mat = np.zeros((n_reco, N_f), dtype=np.float32)
    preds_mat = np.zeros((n_reco, N_s), dtype=np.float32)

    for i in range(n_reco):
        prev_micro_start = ztst[i : i + 1, :].to(DEVICE)
        target_end_tensor = process_data(PTARGET[:, i : i + 1]).to(DEVICE)

        init_diet = Q2[i : i + 1, :].clone().to(DEVICE) if Q2.size(0) > i else torch.zeros((1, N_f), dtype=torch.float32, device=DEVICE)

        diet_opt, _ = optimize_diet_for_target(
            best_model,
            prev_micro_start=prev_micro_start,
            target_micro_end=target_end_tensor,
            init_diet=init_diet,
            steps=args.diet_steps,
            step_size=args.diet_step_size,
            early_stop_patience=args.diet_early_stop,
            normalize=bool(args.diet_normalize),  # DMAS default was False
        )

        with torch.no_grad():
            pred_endpoint = predict_probs(best_model, diet_opt, prev_micro_start)

        diets_mat[i, :] = tensor_to_np_row(diet_opt)
        preds_mat[i, :] = tensor_to_np_row(pred_endpoint)

    np.savetxt(os.path.join(args.out_dir, f"diet_recommendations_{args.tag}.csv"), diets_mat, delimiter=",")
    np.savetxt(os.path.join(args.out_dir, f"pred_endpoints_{args.tag}.csv"), preds_mat, delimiter=",")

    print("[OK] Finished")

if __name__ == "__main__":
    main()

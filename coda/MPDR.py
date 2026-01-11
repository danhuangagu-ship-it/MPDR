#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =======================
# Config (fixed defaults)
# =======================
DEVICE = "cpu"
THRESHOLD = 1e-6

MAX_RECO = 400
STEPS = 100
STEP_SIZE = 0.05
EARLY_STOP_PATIENCE = 30
NORMALIZE_DIET = False  # keep consistent with your current script


# =======================
# Helpers
# =======================
def _safe(t: torch.Tensor, *, min_val: float = -1e6, max_val: float = 1e6, fill: float = 0.0) -> torch.Tensor:
    t = torch.nan_to_num(t, nan=fill, posinf=max_val, neginf=min_val)
    return torch.clamp(t, min=min_val, max=max_val)

def process_data(P: np.ndarray) -> torch.Tensor:
    # P: (N_s, n_samples) -> (n_samples, N_s), col-normalized
    col_sums = P.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    P = (P / col_sums).astype(np.float32)
    return torch.from_numpy(P.T)

def compute_microbiome_mask(prev_micro: torch.Tensor, threshold_val: float = 0.0) -> torch.Tensor:
    return (prev_micro > threshold_val).float()

def bray_curtis(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = _safe(p, min_val=0.0, max_val=1.0)
    q = _safe(q, min_val=0.0, max_val=1.0)
    num = (p - q).abs().sum(dim=1)
    den = (p + q).sum(dim=1).clamp(min=eps)
    return num / den

def tensor_to_np_row(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().reshape(-1)

def ensure_orientation_species_by_samples(mat: np.ndarray, N_s: int = None) -> np.ndarray:
    # Make it (N_s, n_samples). If N_s known, use it; else use heuristic.
    if mat.ndim == 1:
        mat = mat.reshape(-1, 1)
    if N_s is not None:
        if mat.shape[0] != N_s and mat.shape[1] == N_s:
            mat = mat.T
        return mat
    # heuristic
    if mat.shape[0] >= mat.shape[1]:
        mat = mat.T
    return mat


# =======================
# Model (MUST match training)
# =======================
class DietMicrobiomeMLPConcat(nn.Module):
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
        return alpha * mask + (1.0 - mask) * 1e-6


# IMPORTANT:
# Do NOT decorate with @torch.no_grad(), because we need gradients w.r.t. diet during optimization.
def predict_probs(model: nn.Module, diet_row: torch.Tensor, prev_micro: torch.Tensor) -> torch.Tensor:
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


def optimize_diet_for_target(
    model: nn.Module,
    prev_micro_start: torch.Tensor,
    target_micro_end: torch.Tensor,
    init_diet: torch.Tensor,
    steps: int = 100,
    step_size: float = 0.05,
    early_stop_patience: int = 30,
    normalize: bool = False,
):
    diet = init_diet.clone().to(DEVICE)
    diet = torch.clamp(diet, min=0.0)
    if normalize:
        diet = diet / (diet.sum(dim=1, keepdim=True) + 1e-12)

    diet_var = diet.detach().clone().requires_grad_(True)

    best_loss = float("inf")
    best_diet = None
    patience_left = early_stop_patience

    for _ in range(steps):
        if diet_var.grad is not None:
            diet_var.grad.zero_()

        pred_end = predict_probs(model, diet_var, prev_micro_start)
        loss = bray_curtis(pred_end, target_micro_end).mean()
        loss.backward()

        cur = float(loss.item())
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

            max_abs = g.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
            step = -step_size * g / max_abs

            new_diet = diet_var + step
            new_diet = torch.clamp(new_diet, min=0.0)
            if normalize:
                new_diet = new_diet / (new_diet.sum(dim=1, keepdim=True) + 1e-12)

            diet_var.data.copy_(new_diet)

    if best_diet is None:
        best_diet = diet_var.detach()
    return best_diet


# =======================
# CLI
# =======================
def build_argparser():
    p = argparse.ArgumentParser(
        description="Diet recommendation using a trained MPDR checkpoint (explicit inputs via CLI)"
    )
    p.add_argument("--model_path", required=True, help="Trained checkpoint .pt (e.g., MPDR_model.pt)")
    p.add_argument("--z_start", required=True, help="Baseline microbiome z (CSV)")
    p.add_argument("--q_start", required=True, help="Initial diet q (CSV)")
    p.add_argument("--p_target", required=True, help="Target microbiome p* (CSV)")
    p.add_argument("--out_dir", default="./results", help="Output directory")
    p.add_argument("--max_reco", type=int, default=MAX_RECO)
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--step_size", type=float, default=STEP_SIZE)
    p.add_argument("--early_stop_patience", type=int, default=EARLY_STOP_PATIENCE)
    p.add_argument("--normalize_diet", action="store_true", help="If set, enforce diet sum-to-1 normalization")
    return p


# =======================
# Main
# =======================
def main():
    args = build_argparser().parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.model_path, map_location=DEVICE)
    N_s = int(ckpt["N_s"])
    N_f = int(ckpt["N_f"])

    model = DietMicrobiomeMLPConcat(diet_input_dim=N_f, microbe_output_dim=N_s, hidden=256).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    # ---- load recommendation inputs ----
    Z_start = np.loadtxt(args.z_start, delimiter=",")
    P_target = np.loadtxt(args.p_target, delimiter=",")
    Q_init = np.loadtxt(args.q_start, delimiter=",")

    Z_start = ensure_orientation_species_by_samples(Z_start, N_s)
    P_target = ensure_orientation_species_by_samples(P_target, N_s)

    if Q_init.ndim == 1:
        Q_init = Q_init.reshape(1, -1)

    # make Q_init shape (N_f, n_samples)
    if Q_init.shape[0] != N_f and Q_init.shape[1] == N_f:
        Q_init = Q_init.T

    if Q_init.shape[0] != N_f:
        raise ValueError(f"q_start rows must be N_f={N_f}, got {Q_init.shape}")

    if not (Z_start.shape[0] == N_s and P_target.shape[0] == N_s):
        raise ValueError(f"z_start/p_target must have N_s={N_s} rows, got z_start={Z_start.shape}, p_target={P_target.shape}")

    # threshold + normalization for microbiome matrices
    Z_start[Z_start < THRESHOLD] = 0.0
    P_target[P_target < THRESHOLD] = 0.0

    ztst = process_data(Z_start).to(DEVICE)  # (n_start, N_s)
    ptar = P_target                          # (N_s, n_target) numpy

    # diet preprocessing MUST match training/test usage: log10(q+1)
    Q_init = np.log10(Q_init + 1.0)
    Q2 = torch.from_numpy(Q_init.T.astype(np.float32)).to(DEVICE)  # (n_start, N_f)

    n_reco = min(args.max_reco, ztst.size(0), ptar.shape[1])
    diets_mat = np.zeros((n_reco, N_f), dtype=np.float32)
    preds_mat = np.zeros((n_reco, N_s), dtype=np.float32)

    for i in range(n_reco):
        prev = ztst[i:i+1, :]
        target = process_data(ptar[:, i:i+1]).to(DEVICE)  # (1, N_s)

        init_diet = Q2[i:i+1, :] if i < Q2.size(0) else torch.zeros((1, N_f), dtype=torch.float32, device=DEVICE)

        diet_opt = optimize_diet_for_target(
            model,
            prev_micro_start=prev,
            target_micro_end=target,
            init_diet=init_diet,
            steps=args.steps,
            step_size=args.step_size,
            early_stop_patience=args.early_stop_patience,
            normalize=args.normalize_diet,
        )

        with torch.no_grad():
            pred_end = predict_probs(model, diet_opt, prev)

        diets_mat[i, :] = tensor_to_np_row(diet_opt)
        preds_mat[i, :] = tensor_to_np_row(pred_end)

        if (i % 20 == 0) or (i == n_reco - 1):
            bc = float(bray_curtis(pred_end, target).mean().item())
            print(f"[Reco][{i}/{n_reco}] BC={bc:.6f}")

    np.savetxt(os.path.join(args.out_dir, "DMAS_diet_recommendations.csv"), diets_mat, delimiter=",")
    np.savetxt(os.path.join(args.out_dir, "DMAS_pred_endpoints.csv"), preds_mat, delimiter=",")
    print(f"[OK] Wrote outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()

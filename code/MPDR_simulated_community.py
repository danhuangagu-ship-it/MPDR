import os
import copy
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------
# CLI arguments
# -----------------------
parser = argparse.ArgumentParser(
    description="composition mapping with masked Dirichlet loss (MLP on [diet||prev_micro]) + diet recommendation"
)

parser.add_argument("--p_train", required=True, help="Path to p_healthy.csv")
parser.add_argument("--z_train", required=True, help="Path to z_healthy.csv")
parser.add_argument("--q_train", required=True, help="Path to q_healthy.csv")
parser.add_argument("--p_target", required=True, help="Path to p_target.csv (e.g., p_disease.csv)")
parser.add_argument("--z_start", required=True, help="Path to z_start.csv (e.g., z_disease.csv)")
parser.add_argument("--q_start", required=True, help="Path to q_start.csv (e.g., q_disease_perm.csv)")

parser.add_argument("--out_dir", default="./results", help="Output directory")
parser.add_argument("--tag", default="MPDR_run", help="Tag used in output filenames")

parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--mb", type=int, default=20)

parser.add_argument("--diet_steps", type=int, default=400)
parser.add_argument("--diet_lr", type=float, default=1e-1)
parser.add_argument("--diet_l1", type=float, default=0.0)
parser.add_argument("--diet_l2", type=float, default=1e-3)
parser.add_argument("--diet_nonneg", action="store_true")
parser.add_argument("--diet_sum1", action="store_true")

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--threshold", type=float, default=1e-6)

args = parser.parse_args()


# -----------------------
# Globals
# -----------------------
device = "cpu"


# -----------------------
# Reproducibility
# -----------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(args.seed)


# -----------------------
# Sanitizer
# -----------------------
def _safe(t: torch.Tensor, *, min_val=-1e6, max_val=1e6, fill=0.0):
    t = torch.nan_to_num(t, nan=fill, posinf=max_val, neginf=min_val)
    return torch.clamp(t, min=min_val, max=max_val)


# -----------------------
# Utilities
# -----------------------
def process_data(P: np.ndarray) -> torch.Tensor:
    col_sums = P.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    P = (P / col_sums).astype(np.float32)
    return torch.from_numpy(P.T)

def compute_microbiome_mask(prev_microbiome: torch.Tensor):
    return (prev_microbiome > 0).float()

def get_batch(ztrn, ptrn, Q1, mb):
    n = ptrn.size(0)
    idx = np.random.choice(n, min(mb, n), replace=False)
    s = torch.from_numpy(idx)
    return ztrn[s], ptrn[s], Q1[s]

def project_simplex(x, eps=1e-12):
    v, _ = torch.sort(x, dim=1, descending=True)
    cssv = torch.cumsum(v, dim=1) - 1
    ind = torch.arange(1, x.size(1) + 1, device=x.device).float()
    cond = v - cssv / ind > 0
    rho = cond.sum(dim=1).clamp(min=1).long() - 1
    theta = cssv[torch.arange(x.size(0)), rho] / (rho.float() + 1)
    w = torch.clamp(x - theta.unsqueeze(1), min=0.0)
    return w / w.sum(dim=1, keepdim=True).clamp(min=eps)

def bray_curtis(p, q, eps=1e-12):
    p = _safe(p, min_val=0, max_val=1)
    q = _safe(q, min_val=0, max_val=1)
    return (p - q).abs().sum(1) / (p + q).sum(1).clamp(min=eps)

def tensor_to_np_row(x):
    return x.detach().cpu().numpy().reshape(-1)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_csv(path: str) -> np.ndarray:
    x = np.loadtxt(path, delimiter=",")
    if x.ndim == 0:
        x = x.reshape(1, 1)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x


# -----------------------
# Loss
# -----------------------
def masked_dirichlet_loss(alpha, true, mask, eps=1e-6):
    alpha = _safe(alpha, min_val=eps)
    true = _safe(true, min_val=0)
    mask = (mask > 0).float()

    true = true * mask
    true = true / true.sum(dim=1, keepdim=True).clamp(min=eps)

    alpha = alpha * mask + (1 - mask) * eps
    dist = torch.distributions.Dirichlet(alpha)
    return (-dist.log_prob(true)).mean()


# -----------------------
# Model
# -----------------------
class DietMicrobiomeMLPConcat(nn.Module):
    def __init__(self, diet_dim, microbe_dim, hidden=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(diet_dim + microbe_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, microbe_dim),
        )

    def forward(self, diet, prev, mask):
        x = torch.cat([_safe(diet), _safe(prev)], dim=1)
        alpha = F.softplus(self.mlp(x)) + 1e-6
        return alpha * mask + (1 - mask) * 1e-6


# -----------------------
# Training
# -----------------------
def train_and_select_best(epochs, mb, lr, ztrn, ptrn, zval, pval, Q1, Q3):
    model = DietMicrobiomeMLPConcat(Q1.shape[1], ptrn.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best, best_loss = None, float("inf")
    train_curve, val_curve = [], []

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        bp, bq, bd = get_batch(ztrn, ptrn, Q1, mb)
        mask = compute_microbiome_mask(bp)
        loss = masked_dirichlet_loss(model(bd, bp, mask), bq, mask)
        loss.backward()
        opt.step()
        train_curve.append(loss.item())

        model.eval()
        with torch.no_grad():
            maskv = compute_microbiome_mask(zval)
            lv = masked_dirichlet_loss(model(Q3, zval, maskv), pval, maskv).item()
            val_curve.append(lv)
            if lv < best_loss:
                best_loss = lv
                best = copy.deepcopy(model)

    return best, np.array(train_curve), np.array(val_curve)


# -----------------------
# Inference
# -----------------------
@torch.no_grad()
def predict_probs(model, diet, prev):
    mask = compute_microbiome_mask(prev)
    alpha = model(diet, prev, mask)
    p = alpha * mask
    return p / p.sum(dim=1, keepdim=True).clamp(min=1e-8)


# -----------------------
# Diet optimization
# -----------------------
def optimize_diet_for_target(model, prev, target, init_diet, steps, lr, l1, l2, nonneg, simplex):
    diet = init_diet.clone().detach().requires_grad_(True)
    opt = torch.optim.SGD([diet], lr=lr, momentum=0.9)

    for _ in range(steps):
        opt.zero_grad()
        pred = predict_probs(model, diet, prev)
        loss = bray_curtis(pred, target).mean()
        loss = loss + l1 * diet.abs().sum() + l2 * (diet ** 2).sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            if nonneg:
                diet.clamp_(min=0)
            if simplex:
                diet.copy_(project_simplex(diet))

    return diet.detach()


# -----------------------
# Load data (explicit input files)
# -----------------------
P1 = load_csv(args.p_train)
Z1 = load_csv(args.z_train)
Q1_raw = load_csv(args.q_train)

PTARGET = load_csv(args.p_target)
Z_start = load_csv(args.z_start)
Q_start = load_csv(args.q_start)

# Ensure microbiome matrices are (S x n)
if P1.shape[0] >= P1.shape[1]:
    P1 = P1.T
    Z1 = Z1.T
    PTARGET = PTARGET.T

P1[P1 < args.threshold] = 0
N_s, n = P1.shape

# Align q_train to (F x n)
if Q1_raw.shape[1] == n:
    Q1_aligned = Q1_raw
elif Q1_raw.shape[0] == n:
    Q1_aligned = Q1_raw.T
else:
    raise ValueError(f"q_train shape incompatible with p_train: q_train={Q1_raw.shape}, p_train={P1.shape}")

N_f = Q1_aligned.shape[0]

# Train/val split
val_n = max(1, int(0.2 * n))
vidx = np.random.choice(n, val_n, replace=False)
tidx = np.setdiff1d(np.arange(n), vidx)

ptrn = process_data(P1[:, tidx])
ztrn = process_data(Z1[:, tidx])
pval = process_data(P1[:, vidx])
zval = process_data(Z1[:, vidx])

Q1 = torch.from_numpy(Q1_aligned[:, tidx].T.astype(np.float32))
Q3 = torch.from_numpy(Q1_aligned[:, vidx].T.astype(np.float32))

# Test prev microbiome
if Z_start.shape[0] == N_s:
    ztst = process_data(Z_start)
elif Z_start.shape[1] == N_s:
    ztst = process_data(Z_start.T)
else:
    raise ValueError(f"z_start shape incompatible with N_s={N_s}: z_start={Z_start.shape}")

# Align p_target to (S x n_target)
if PTARGET.shape[0] != N_s and PTARGET.shape[1] == N_s:
    PTARGET_use = PTARGET.T
else:
    PTARGET_use = PTARGET
if PTARGET_use.shape[0] != N_s:
    raise ValueError(f"p_target shape incompatible with N_s={N_s}: p_target={PTARGET.shape}")

# Align q_start to (n_test x F)
if Q_start.shape[0] == N_f:
    Q2 = torch.from_numpy(Q_start.T.astype(np.float32))
elif Q_start.shape[1] == N_f:
    Q2 = torch.from_numpy(Q_start.astype(np.float32))
else:
    raise ValueError(f"q_start shape incompatible with q_train F={N_f}: q_start={Q_start.shape}")


# -----------------------
# Train
# -----------------------
best_model, train_curve, val_curve = train_and_select_best(
    args.epochs, args.mb, args.lr,
    ztrn, ptrn, zval, pval, Q1, Q3
)


# -----------------------
# Outputs
# -----------------------
ensure_dir(args.out_dir)
tag = args.tag

np.savetxt(f"{args.out_dir}/train_loss_{tag}.csv", train_curve, delimiter=",")
np.savetxt(f"{args.out_dir}/val_loss_{tag}.csv", val_curve, delimiter=",")

torch.save(
    {
        "model_state_dict": best_model.state_dict(),
        "N_s": N_s,
        "N_f": N_f,
        "args": vars(args),
    },
    f"{args.out_dir}/best_model_{tag}.pt",
)


# -----------------------
# Recommendation
# -----------------------
n_reco = min(400, ztst.size(0), PTARGET_use.shape[1])
diets = np.zeros((n_reco, N_f), dtype=np.float32)
preds = np.zeros((n_reco, N_s), dtype=np.float32)

for i in range(n_reco):
    prev = ztst[i:i+1]
    target = process_data(PTARGET_use[:, i:i+1])

    init_diet = Q2[i:i+1] if i < Q2.size(0) else torch.zeros((1, N_f), dtype=torch.float32)

    diet = optimize_diet_for_target(
        best_model, prev, target, init_diet,
        args.diet_steps, args.diet_lr,
        args.diet_l1, args.diet_l2,
        args.diet_nonneg or args.diet_sum1,
        args.diet_sum1
    )

    pred = predict_probs(best_model, diet, prev)
    diets[i] = tensor_to_np_row(diet)
    preds[i] = tensor_to_np_row(pred)

np.savetxt(f"{args.out_dir}/diet_recommendations_{tag}.csv", diets, delimiter=",")
np.savetxt(f"{args.out_dir}/pred_endpoints_{tag}.csv", preds, delimiter=",")

print("[OK] Finished")

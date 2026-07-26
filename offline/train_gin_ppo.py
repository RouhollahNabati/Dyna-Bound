#!/usr/bin/env python3
"""Offline GIN-style GNN trainer for CRT trajectories → Java OnlineColonyGnn weights.

Uses reward-advantage BC + structure ranking. Requires IN_DIM=8 (full host emb).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

IN_DIM = 8
HID = 16
CTX_DIM = 8
READOUT = HID + CTX_DIM
EPS_GIN = 0.1

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def load_traj(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pad_x(x: np.ndarray) -> np.ndarray:
    """Accept legacy 6-d host rows by zero-padding to IN_DIM."""
    if x.shape[1] == IN_DIM:
        return x
    if x.shape[1] < IN_DIM:
        out = np.zeros((x.shape[0], IN_DIM), dtype=np.float64)
        out[:, : x.shape[1]] = x
        return out
    return x[:, :IN_DIM]


def valid_sample(s: dict) -> bool:
    try:
        x = pad_x(np.asarray(s["x"], dtype=np.float64))
        ctx = np.asarray(s["ctx"], dtype=np.float64)
        adj = np.asarray(s["adj"], dtype=np.float64)
        a = int(s["action"])
        return (
            x.ndim == 2
            and x.shape[1] == IN_DIM
            and ctx.shape == (x.shape[0], CTX_DIM)
            and adj.shape == (x.shape[0], x.shape[0])
            and 0 <= a < x.shape[0]
            and x.shape[0] >= 2  # singleton decisions teach nothing
        )
    except (KeyError, TypeError, ValueError):
        return False


def structure_score(x_row: np.ndarray, ctx_row: np.ndarray) -> float:
    """Graph-SLA proxy: producer pull + coloc under fan, plus host msgs."""
    neigh = float(x_row[6]) if len(x_row) > 6 else 0.0
    prod_host = float(x_row[7]) if len(x_row) > 7 else 0.0
    pull = float(ctx_row[5]) if len(ctx_row) > 5 else 0.0
    fan = float(ctx_row[6]) if len(ctx_row) > 6 else 0.0
    coloc = float(ctx_row[7]) if len(ctx_row) > 7 else 0.0
    return 0.40 * pull + 0.30 * coloc + 0.15 * (fan * max(pull, coloc)) + 0.10 * prod_host + 0.05 * neigh


if HAS_TORCH:

    class TorchGnnPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1_self = nn.Parameter(torch.randn(HID, IN_DIM) * 0.08)
            self.w1_neigh = nn.Parameter(torch.randn(HID, IN_DIM) * 0.05)
            self.w2_self = nn.Parameter(torch.eye(HID) * 0.15 + torch.randn(HID, HID) * 0.02)
            self.w2_neigh = nn.Parameter(torch.randn(HID, HID) * 0.04)
            ro = torch.zeros(READOUT)
            ro[HID + 5] = 0.55
            ro[HID + 6] = 0.20
            ro[HID + 7] = 0.60
            self.readout = nn.Parameter(ro)

        def scores(self, x, ctx, adj):
            agg1 = adj @ x
            h1 = torch.tanh((1.0 + EPS_GIN) * (x @ self.w1_self.T) + (agg1 @ self.w1_neigh.T))
            agg2 = adj @ h1
            h2 = torch.tanh((1.0 + EPS_GIN) * (h1 @ self.w2_self.T) + (agg2 @ self.w2_neigh.T))
            return torch.cat([h2, ctx], dim=-1) @ self.readout

        def to_java_dict(self) -> dict:
            scale = 1.0 + EPS_GIN
            return {
                "in_dim": IN_DIM,
                "hid": HID,
                "ctx_dim": CTX_DIM,
                "encoder": "gin_struct_adv_torch",
                "w1_self": (self.w1_self.detach().cpu().numpy() * scale).tolist(),
                "w1_neigh": self.w1_neigh.detach().cpu().numpy().tolist(),
                "w2_self": (self.w2_self.detach().cpu().numpy() * scale).tolist(),
                "w2_neigh": self.w2_neigh.detach().cpu().numpy().tolist(),
                "readout": self.readout.detach().cpu().numpy().tolist(),
            }

    def train_torch(data, epochs, batch, lr, seed):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        policy = TorchGnnPolicy()
        opt = torch.optim.Adam(policy.parameters(), lr=lr, weight_decay=1e-5)

        rewards = np.asarray([float(s["reward"]) for s in data], dtype=np.float64)
        r_mean = float(rewards.mean())
        r_std = float(max(1e-4, rewards.std()))

        for ep in range(epochs):
            random.shuffle(data)
            losses = []
            for i in range(0, len(data), batch):
                chunk = data[i : i + batch]
                opt.zero_grad()
                loss = torch.tensor(0.0)
                n_ok = 0
                for s in chunk:
                    x_np = pad_x(np.asarray(s["x"], dtype=np.float64))
                    ctx_np = np.asarray(s["ctx"], dtype=np.float64)
                    x = torch.tensor(x_np, dtype=torch.float32)
                    ctx = torch.tensor(ctx_np, dtype=torch.float32)
                    adj = torch.tensor(s["adj"], dtype=torch.float32)
                    action = int(s["action"])
                    reward = float(s["reward"])
                    adv = (reward - r_mean) / r_std
                    logits = policy.scores(x, ctx, adj)
                    logp = F.log_softmax(logits, dim=0)
                    probs = torch.softmax(logits, dim=0)

                    st = structure_score(x_np[action], ctx_np[action])
                    w = 1.0 + 6.0 * max(0.0, adv) + 2.5 * max(0.0, reward - 0.5) + 1.5 * st
                    loss = loss + (-w * logp[action])

                    st_all = [structure_score(x_np[j], ctx_np[j]) for j in range(ctx_np.shape[0])]
                    j_best = int(np.argmax(st_all))
                    j_worst = int(np.argmin(st_all))
                    if j_best != j_worst and (st_all[j_best] - st_all[j_worst]) > 0.02:
                        loss = loss + 0.4 * F.softplus(-(logits[j_best] - logits[j_worst]))

                    if "base" in s and s["base"] is not None:
                        base = np.asarray(s["base"], dtype=np.float64)
                        if base.shape[0] == logits.shape[0]:
                            jb = int(np.argmax(base))
                            jw = int(np.argmin(base))
                            if jb != jw and (base[jb] - base[jw]) > 1e-4:
                                loss = loss + 0.75 * F.softplus(-(logits[jb] - logits[jw]))
                            if reward > 0.55 and action != jw:
                                loss = loss + 0.8 * F.softplus(-(logits[action] - logits[jw]))

                    loss = loss - 0.01 * (-(probs * logp).sum())
                    n_ok += 1
                if n_ok == 0:
                    continue
                loss = loss / n_ok
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 2.0)
                opt.step()
                losses.append(float(loss.detach()))
            print(f"epoch {ep + 1}/{epochs} loss={np.mean(losses):.4f} backend=torch", flush=True)
        return policy


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.008)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = [s for s in load_traj(args.traj) if valid_sample(s)]
    if not data:
        raise SystemExit(f"no usable trajectories in {args.traj}")
    # Diagnostics for feature plumbing
    ctx = np.vstack([np.asarray(s["ctx"]) for s in data])
    x = np.vstack([pad_x(np.asarray(s["x"], dtype=np.float64)) for s in data])
    print(
        f"Loaded {len(data)} multi-candidate decisions "
        f"(torch={HAS_TORCH}) ctx[3:] nonzero frac="
        f"{float((np.abs(ctx[:, 3:]) > 1e-8).any(axis=1).mean()):.3f} "
        f"x[6:8] nonzero frac={float((np.abs(x[:, 6:8]) > 1e-8).any(axis=1).mean()):.3f}",
        flush=True,
    )

    if not HAS_TORCH:
        raise SystemExit("PyTorch required. pip install torch")

    policy = train_torch(data, args.epochs, args.batch, args.lr, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(policy.to_java_dict(), indent=2), encoding="utf-8")
    print(f"Wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

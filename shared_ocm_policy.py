from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


def _flatten_f32(x: object) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def _pad_1d(x: np.ndarray, target: int) -> np.ndarray:
    out = np.zeros((int(target),), dtype=np.float32)
    n = min(int(target), int(x.shape[0]))
    if n > 0:
        out[:n] = x[:n]
    return out


def _build_op_feat_for_legal_ops(
    obs: Dict[str, object],
    legal_ops: List[int],
    target_dim: int,
) -> np.ndarray:
    op_node = np.asarray(obs.get("op_node_feat", []), dtype=np.float32)
    pair_feat = np.asarray(obs.get("pair_feat", []), dtype=np.float32)
    pairs = list(obs.get("candidate_pairs", []))

    op_feats: List[np.ndarray] = []
    for op in legal_ops:
        if op_node.ndim == 2 and 0 <= int(op) < int(op_node.shape[0]):
            base = np.asarray(op_node[int(op)], dtype=np.float32).reshape(-1)
        else:
            base = np.zeros((0,), dtype=np.float32)

        idx = [i for i, (op_id, _m) in enumerate(pairs) if int(op_id) == int(op)]
        if len(idx) > 0 and pair_feat.ndim == 2:
            pf = pair_feat[np.asarray(idx, dtype=np.int64)]
            ect = pf[:, 4] if pf.shape[1] > 4 else np.zeros((pf.shape[0],), dtype=np.float32)
            pt = pf[:, 0] if pf.shape[1] > 0 else np.zeros((pf.shape[0],), dtype=np.float32)
            aug = np.asarray(
                [
                    float(np.min(ect)),
                    float(np.mean(ect)),
                    float(np.std(ect)),
                    float(np.min(pt)),
                    float(np.mean(pt)),
                    float(np.std(pt)),
                    float(len(idx)),
                ],
                dtype=np.float32,
            )
        else:
            aug = np.zeros((7,), dtype=np.float32)

        feat = np.concatenate([base, aug], axis=0)
        if feat.shape[0] < int(target_dim):
            pad = np.zeros((int(target_dim),), dtype=np.float32)
            pad[: feat.shape[0]] = feat
            feat = pad
        else:
            feat = feat[: int(target_dim)]
        op_feats.append(feat.astype(np.float32))

    if len(op_feats) == 0:
        return np.zeros((0, int(target_dim)), dtype=np.float32)
    return np.stack(op_feats, axis=0).astype(np.float32)


def pack_shared_obs(
    obs: Dict[str, object],
    *,
    global_dim: int,
    pair_feat_dim: int,
    op_feat_dim: int,
    max_pairs: int,
    max_ops: int,
) -> Dict[str, object]:
    global_feat = _flatten_f32(obs.get("global_feat", np.zeros((0,), dtype=np.float32)))
    if "candidate_set_feat" in obs:
        global_feat = np.concatenate([global_feat, _flatten_f32(obs.get("candidate_set_feat", []))], axis=0)
    global_feat = _pad_1d(global_feat, int(global_dim))

    pairs = list(obs.get("candidate_pairs", []))
    pair_feat_raw = np.asarray(obs.get("pair_feat", []), dtype=np.float32)
    if pair_feat_raw.ndim != 2:
        pair_feat_raw = np.zeros((0, int(pair_feat_dim)), dtype=np.float32)

    pair_count = min(int(len(pairs)), int(max_pairs))
    pair_feat = np.zeros((int(max_pairs), int(pair_feat_dim)), dtype=np.float32)
    pair_mask = np.zeros((int(max_pairs),), dtype=bool)
    pair_op_idx = np.zeros((int(max_pairs),), dtype=np.int64)
    pair_mch_idx = np.zeros((int(max_pairs),), dtype=np.int64)

    kept_pairs: List[Tuple[int, int]] = []
    for i in range(pair_count):
        op_id, mch_id = pairs[i]
        kept_pairs.append((int(op_id), int(mch_id)))
        if i < pair_feat_raw.shape[0]:
            cols = min(int(pair_feat_dim), int(pair_feat_raw.shape[1]))
            if cols > 0:
                pair_feat[i, :cols] = pair_feat_raw[i, :cols]
        pair_mask[i] = True
        pair_op_idx[i] = int(op_id)
        pair_mch_idx[i] = int(mch_id)

    legal_ops = sorted({int(op) for op, _m in kept_pairs})
    op_feat_legal = _build_op_feat_for_legal_ops(obs, legal_ops=legal_ops, target_dim=int(op_feat_dim))

    op_count = min(int(len(legal_ops)), int(max_ops))
    op_feat = np.zeros((int(max_ops), int(op_feat_dim)), dtype=np.float32)
    op_mask = np.zeros((int(max_ops),), dtype=bool)

    legal_ops_kept = legal_ops[:op_count]
    if op_count > 0:
        op_feat[:op_count] = op_feat_legal[:op_count]
        op_mask[:op_count] = True

    return {
        "global_feat": global_feat.astype(np.float32),
        "pair_feat": pair_feat.astype(np.float32),
        "pair_mask": pair_mask,
        "pair_op_idx": pair_op_idx,
        "pair_mch_idx": pair_mch_idx,
        "candidate_pairs": kept_pairs,
        "op_feat": op_feat.astype(np.float32),
        "op_mask": op_mask,
        "legal_ops": [int(x) for x in legal_ops_kept],
        "original_pair_count": int(len(pairs)),
        "used_pair_count": int(pair_count),
    }


class SharedTrunkOCMPolicy(nn.Module):
    def __init__(
        self,
        *,
        global_dim: int,
        pair_feat_dim: int,
        op_feat_dim: int,
        hidden_dim: int,
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)

        self.global_proj = nn.Sequential(
            nn.Linear(int(global_dim), int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
        )
        self.pair_proj = nn.Sequential(
            nn.Linear(int(pair_feat_dim), int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
        )
        self.op_proj = nn.Sequential(
            nn.Linear(int(op_feat_dim), int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
        )

        n_heads = int(max(1, heads))
        while n_heads > 1 and (int(hidden_dim) % n_heads) != 0:
            n_heads -= 1

        enc_layer = nn.TransformerEncoderLayer(
            d_model=int(hidden_dim),
            nhead=int(n_heads),
            dim_feedforward=int(hidden_dim) * 4,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(max(1, layers)))

        self.o_head = nn.Linear(int(hidden_dim), 1)
        self.o_uncertainty_head = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )
        self.m_head = nn.Linear(int(hidden_dim), 1)
        self.c_head = nn.Linear(int(hidden_dim), 1)

    def encode(
        self,
        global_feat: torch.Tensor,
        pair_feat: torch.Tensor,
        pair_mask: torch.Tensor,
        op_feat: torch.Tensor,
        op_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g = self.global_proj(global_feat).unsqueeze(1)
        p = self.pair_proj(pair_feat)
        o = self.op_proj(op_feat)

        tokens = torch.cat([g, p, o], dim=1)
        bsz, pair_n = pair_mask.shape
        op_n = op_mask.shape[1]

        g_mask = torch.ones((bsz, 1), dtype=torch.bool, device=pair_mask.device)
        valid_mask = torch.cat([g_mask, pair_mask, op_mask], dim=1)
        key_padding_mask = ~valid_mask

        encoded = self.encoder(tokens, src_key_padding_mask=key_padding_mask)

        g_h = encoded[:, 0, :]
        p_h = encoded[:, 1 : 1 + pair_n, :]
        o_h = encoded[:, 1 + pair_n : 1 + pair_n + op_n, :]

        denom = valid_mask.sum(dim=1, keepdim=True).clamp(min=1).to(dtype=encoded.dtype)
        pooled = (encoded * valid_mask.unsqueeze(-1).to(dtype=encoded.dtype)).sum(dim=1) / denom
        pooled = 0.5 * (pooled + g_h)
        return pooled, p_h, o_h

    def o_logits(self, op_h: torch.Tensor) -> torch.Tensor:
        return self.o_head(op_h).squeeze(-1)

    def o_uncertainty(self, pooled: torch.Tensor) -> torch.Tensor:
        raw = self.o_uncertainty_head(pooled).squeeze(-1)
        return torch.nn.functional.softplus(raw)

    def c_logits(self, pair_h: torch.Tensor) -> torch.Tensor:
        return self.c_head(pair_h).squeeze(-1)

    def m_logits_for_op(
        self,
        pair_h_op: torch.Tensor,
        op_ctx: torch.Tensor,
    ) -> torch.Tensor:
        if pair_h_op.ndim == 1:
            pair_h_op = pair_h_op.unsqueeze(0)
        if op_ctx.ndim == 1:
            op_ctx = op_ctx.unsqueeze(0)
        fused = pair_h_op + op_ctx.expand_as(pair_h_op)
        return self.m_head(fused).squeeze(-1)


class SharedCentralCritic(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        h = int(hidden_dim)
        self.backbone = nn.Sequential(
            nn.Linear(h, h),
            nn.GELU(),
            nn.LayerNorm(h),
            nn.Linear(h, h),
            nn.GELU(),
            nn.LayerNorm(h),
        )
        self.v_c = nn.Linear(h, 1)
        self.v_o = nn.Linear(h, 1)
        self.v_m = nn.Linear(h, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.v_c(h), self.v_o(h), self.v_m(h)


@dataclass
class SharedHeadEvalRecord:
    log_prob: torch.Tensor
    entropy: torch.Tensor

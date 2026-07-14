from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# =========================
# Config / basic utilities
# =========================

@dataclass
class AgentCConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    hidden_dim: int = 128
    attn_heads: int = 4
    attn_layers: int = 1
    dropout: float = 0.0
    use_candidate_set_feat: bool = True

    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 64
    target_kl: Optional[float] = None

    max_candidates: int = 16
    overflow_strategy: str = "truncate"   # "truncate" or "raise"
    overflow_warning_limit: int = 20

    # Top-K optional fields
    topk_prefilter_enabled: bool = False
    topk_k: int = 32
    topk_keep_eet: int = 2
    topk_keep_pt: int = 2
    topk_keep_preferred_rule: int = 2
    topk_preferred_rule: str = "FIFO_SPT"
    topk_machine_diversity: bool = True
    topk_score_weights: Dict[str, float] = field(
        default_factory=lambda: {"eet": 1.0, "pt": 0.3, "queue": 0.1, "slack": 0.2}
    )

    use_graph_encoder: bool = False
    use_edge_rule_msg: bool = True
    use_edge_opmch_msg: bool = True
    gnn_hidden_dim: int = 64
    gnn_layers: int = 2
    op_node_dim: int = 10
    machine_node_dim: int = 6
    pair_graph_gate_init: float = 0.1
    rule_edge_gate_init: float = 0.1
    opmch_edge_gate_init: float = 0.1
    edge_message_dropout: float = 0.0
    use_separate_edge_gates: bool = True
    use_adaptive_edge_gates: bool = False
    adaptive_gate_hidden_dim: int = 64


def _flatten_f32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(-1)


def pad_global_feat(global_feat: np.ndarray, target_dim: int) -> np.ndarray:
    global_feat = _flatten_f32(global_feat)
    if global_feat.shape[0] > target_dim:
        raise ValueError(f"global_feat dim={global_feat.shape[0]} exceeds target_dim={target_dim}")
    out = np.zeros(target_dim, dtype=np.float32)
    out[: global_feat.shape[0]] = global_feat
    return out


def pad_pair_feat(
    pair_feat: np.ndarray,
    target_candidates: int,
    pair_feat_dim: int,
    overflow_strategy: str = "truncate",
) -> Tuple[np.ndarray, np.ndarray, int]:
    pair_feat = np.asarray(pair_feat, dtype=np.float32)
    if pair_feat.ndim != 2:
        raise ValueError(f"pair_feat must be rank-2, got shape={pair_feat.shape}")
    if pair_feat.shape[1] != pair_feat_dim:
        raise ValueError(f"pair_feat dim mismatch: got {pair_feat.shape[1]}, expected {pair_feat_dim}")

    original_count = int(pair_feat.shape[0])
    overflow_count = max(0, original_count - int(target_candidates))
    if overflow_count > 0:
        if overflow_strategy == "raise":
            raise ValueError(f"candidate count={original_count} exceeds max_candidates={target_candidates}")
        if overflow_strategy != "truncate":
            raise ValueError(f"Unsupported overflow_strategy={overflow_strategy}")
        pair_feat = pair_feat[:target_candidates]

    used_count = int(pair_feat.shape[0])
    feat_out = np.zeros((target_candidates, pair_feat_dim), dtype=np.float32)
    mask_out = np.zeros((target_candidates,), dtype=bool)
    if used_count > 0:
        feat_out[:used_count] = pair_feat
        mask_out[:used_count] = True
    return feat_out, mask_out, overflow_count


def pad_index_array(index_arr: np.ndarray, target_candidates: int, fill_value: int = 0) -> np.ndarray:
    idx = np.asarray(index_arr, dtype=np.int64).reshape(-1)
    out = np.full((int(target_candidates),), int(fill_value), dtype=np.int64)
    used = min(int(target_candidates), int(idx.shape[0]))
    if used > 0:
        out[:used] = idx[:used]
    return out


def pad_pair_matrix(mat: np.ndarray, target_candidates: int, feat_dim: int) -> np.ndarray:
    arr = np.asarray(mat, dtype=np.float32)
    if arr.ndim != 2:
        arr = np.zeros((0, int(feat_dim)), dtype=np.float32)
    out = np.zeros((int(target_candidates), int(feat_dim)), dtype=np.float32)
    used = min(int(target_candidates), int(arr.shape[0]))
    used_dim = min(int(feat_dim), int(arr.shape[1]) if arr.ndim == 2 else 0)
    if used > 0 and used_dim > 0:
        out[:used, :used_dim] = arr[:used, :used_dim]
    return out


def _extract_edge_feats_from_pair_feat(pair_feat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pair_feat = np.asarray(pair_feat, dtype=np.float32)
    if pair_feat.ndim != 2:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)

    if pair_feat.shape[1] > 24:
        edge_rule = np.asarray(pair_feat[:, 24:], dtype=np.float32)
    else:
        edge_rule = np.zeros((pair_feat.shape[0], 0), dtype=np.float32)

    edge_opmch = np.zeros((pair_feat.shape[0], 4), dtype=np.float32)
    cols = [0, 4, 13, 15]
    for i, col in enumerate(cols):
        if pair_feat.shape[1] > col:
            edge_opmch[:, i] = pair_feat[:, col]
    return edge_rule, edge_opmch


# =========================
# Base observation packing
# =========================

def preprocess_obs(
    obs: Dict[str, object],
    global_dim: int,
    pair_feat_dim: int,
    max_candidates: int,
    use_candidate_set_feat: bool = True,
    overflow_strategy: str = "truncate",
) -> Dict[str, object]:
    candidate_pairs = list(obs["candidate_pairs"])
    if len(candidate_pairs) == 0:
        raise RuntimeError("No candidate_pairs found in current observation.")

    global_feat = _flatten_f32(obs["global_feat"])
    if use_candidate_set_feat and "candidate_set_feat" in obs:
        global_feat = np.concatenate([global_feat, _flatten_f32(obs["candidate_set_feat"])], axis=0)
    global_feat = pad_global_feat(global_feat, global_dim)

    original_candidate_count = len(candidate_pairs)
    pair_feat, pair_mask, overflow_count = pad_pair_feat(
        np.asarray(obs["pair_feat"], dtype=np.float32),
        max_candidates,
        pair_feat_dim,
        overflow_strategy=overflow_strategy,
    )

    used_candidate_count = int(pair_mask.sum())
    candidate_pairs = candidate_pairs[:used_candidate_count]

    pair_sources = list(obs.get("pair_sources", []))
    if len(pair_sources) < original_candidate_count:
        pair_sources = pair_sources + [[] for _ in range(original_candidate_count - len(pair_sources))]
    pair_sources = pair_sources[:used_candidate_count]

    pair_op_idx_raw = np.asarray(obs.get("pair_op_idx", [p[0] for p in candidate_pairs]), dtype=np.int64)
    pair_mch_idx_raw = np.asarray(obs.get("pair_mch_idx", [p[1] for p in candidate_pairs]), dtype=np.int64)
    pair_op_idx = pad_index_array(pair_op_idx_raw[:used_candidate_count], max_candidates, fill_value=0)
    pair_mch_idx = pad_index_array(pair_mch_idx_raw[:used_candidate_count], max_candidates, fill_value=0)

    edge_rule_raw = obs.get("edge_rule_to_pair_feat", None)
    edge_opmch_raw = obs.get("edge_opmch_to_pair_feat", None)
    if edge_rule_raw is None or edge_opmch_raw is None:
        derived_rule, derived_opmch = _extract_edge_feats_from_pair_feat(np.asarray(obs["pair_feat"], dtype=np.float32))
        if edge_rule_raw is None:
            edge_rule_raw = derived_rule
        if edge_opmch_raw is None:
            edge_opmch_raw = derived_opmch

    edge_rule_raw = np.asarray(edge_rule_raw, dtype=np.float32)
    edge_opmch_raw = np.asarray(edge_opmch_raw, dtype=np.float32)
    edge_rule_dim = int(edge_rule_raw.shape[1]) if edge_rule_raw.ndim == 2 else 0
    edge_opmch_dim = int(edge_opmch_raw.shape[1]) if edge_opmch_raw.ndim == 2 else 4
    edge_rule_feat = pad_pair_matrix(edge_rule_raw[:used_candidate_count], max_candidates, edge_rule_dim)
    edge_opmch_feat = pad_pair_matrix(edge_opmch_raw[:used_candidate_count], max_candidates, edge_opmch_dim)

    op_node_feat = None
    op_adj = None
    machine_node_feat = None
    if "op_node_feat" in obs:
        op_node_feat = np.asarray(obs["op_node_feat"], dtype=np.float32)
    if "op_adj" in obs:
        op_adj = np.asarray(obs["op_adj"], dtype=np.float32)
    if "machine_node_feat" in obs:
        machine_node_feat = np.asarray(obs["machine_node_feat"], dtype=np.float32)

    return {
        "global_feat": global_feat,
        "pair_feat": pair_feat,
        "pair_mask": pair_mask,
        "candidate_pairs": candidate_pairs,
        "pair_sources": pair_sources,
        "pair_op_idx": pair_op_idx,
        "pair_mch_idx": pair_mch_idx,
        "edge_rule_to_pair_feat": edge_rule_feat,
        "edge_opmch_to_pair_feat": edge_opmch_feat,
        "op_node_feat": op_node_feat,
        "op_adj": op_adj,
        "machine_node_feat": machine_node_feat,
        "original_candidate_count": int(original_candidate_count),
        "used_candidate_count": int(used_candidate_count),
        "overflow_count": int(overflow_count),
        "was_truncated": bool(overflow_count > 0),
    }


def infer_obs_dims_from_env(env, batch_idx: int = 0, use_candidate_set_feat: bool = True) -> Tuple[int, int, int]:
    obs = env.get_agent_c_obs(batch_idx=batch_idx)
    global_feat = _flatten_f32(obs["global_feat"])
    if use_candidate_set_feat and "candidate_set_feat" in obs:
        global_feat = np.concatenate([global_feat, _flatten_f32(obs["candidate_set_feat"])], axis=0)
    pair_feat = np.asarray(obs["pair_feat"], dtype=np.float32)
    if pair_feat.ndim != 2 or pair_feat.shape[1] <= 0:
        raise RuntimeError(f"Invalid pair_feat shape: {pair_feat.shape}")
    max_candidates = max(1, len(obs["candidate_pairs"]))
    return int(global_feat.shape[0]), int(pair_feat.shape[1]), int(max_candidates)


# =========================
# Model
# =========================

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HeteroLiteLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.op_msg = nn.Linear(hidden_dim, hidden_dim)
        self.op_norm = nn.LayerNorm(hidden_dim)

        self.mch_from_pair = nn.Linear(hidden_dim, hidden_dim)
        self.mch_norm = nn.LayerNorm(hidden_dim)

        self.pair_from_op = nn.Linear(hidden_dim, hidden_dim)
        self.pair_from_mch = nn.Linear(hidden_dim, hidden_dim)
        self.pair_self = nn.Linear(hidden_dim, hidden_dim)
        self.pair_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        op_repr: torch.Tensor,
        mch_repr: torch.Tensor,
        pair_repr: torch.Tensor,
        op_adj: torch.Tensor,
        pair_op_idx: torch.Tensor,
        pair_mch_idx: torch.Tensor,
        pair_mask: torch.Tensor,
    ):
        deg = op_adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        op_agg = torch.bmm(op_adj / deg, op_repr)
        op_repr = self.op_norm(op_repr + self.dropout(F.gelu(self.op_msg(op_agg))))

        bsz, mch_n, hid = mch_repr.shape
        _, pair_n, _ = pair_repr.shape
        pair_mask_f = pair_mask.float().unsqueeze(-1)

        mch_acc = torch.zeros((bsz, mch_n, hid), dtype=pair_repr.dtype, device=pair_repr.device)
        mch_cnt = torch.zeros((bsz, mch_n, 1), dtype=pair_repr.dtype, device=pair_repr.device)
        mch_idx_expand = pair_mch_idx.unsqueeze(-1).expand(-1, -1, hid)
        mch_acc.scatter_add_(dim=1, index=mch_idx_expand, src=pair_repr * pair_mask_f)
        mch_cnt.scatter_add_(dim=1, index=pair_mch_idx.unsqueeze(-1), src=pair_mask_f)
        mch_agg = mch_acc / mch_cnt.clamp(min=1.0)
        mch_repr = self.mch_norm(mch_repr + self.dropout(F.gelu(self.mch_from_pair(mch_agg))))

        op_idx_expand = pair_op_idx.unsqueeze(-1).expand(-1, -1, hid)
        mch_idx_expand = pair_mch_idx.unsqueeze(-1).expand(-1, -1, hid)
        pair_op = torch.gather(op_repr, dim=1, index=op_idx_expand)
        pair_mch = torch.gather(mch_repr, dim=1, index=mch_idx_expand)

        pair_delta = self.pair_self(pair_repr) + self.pair_from_op(pair_op) + self.pair_from_mch(pair_mch)
        pair_repr = self.pair_norm(pair_repr + self.dropout(F.gelu(pair_delta)))
        pair_repr = pair_repr * pair_mask_f
        return op_repr, mch_repr, pair_repr


class HeteroLiteEncoder(nn.Module):
    def __init__(
        self,
        op_node_dim: int,
        machine_node_dim: int,
        pair_node_dim: int,
        hidden_dim: int,
        gnn_layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.op_proj = nn.Sequential(nn.Linear(op_node_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.mch_proj = nn.Sequential(nn.Linear(machine_node_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.pair_proj = nn.Sequential(nn.Linear(pair_node_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList([HeteroLiteLayer(hidden_dim=hidden_dim, dropout=dropout) for _ in range(int(max(1, gnn_layers)))])

    def forward(
        self,
        op_node_feat: torch.Tensor,
        machine_node_feat: torch.Tensor,
        pair_node_feat: torch.Tensor,
        op_adj: torch.Tensor,
        pair_op_idx: torch.Tensor,
        pair_mch_idx: torch.Tensor,
        pair_mask: torch.Tensor,
    ):
        op_repr = self.op_proj(op_node_feat)
        mch_repr = self.mch_proj(machine_node_feat)
        pair_repr = self.pair_proj(pair_node_feat)
        for layer in self.layers:
            op_repr, mch_repr, pair_repr = layer(
                op_repr=op_repr,
                mch_repr=mch_repr,
                pair_repr=pair_repr,
                op_adj=op_adj,
                pair_op_idx=pair_op_idx,
                pair_mch_idx=pair_mch_idx,
                pair_mask=pair_mask,
            )
        return pair_repr


class ResidualSelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class FiLMModulator(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.to_gamma = nn.Linear(hidden_dim, hidden_dim)
        self.to_beta = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, token_repr: torch.Tensor, global_repr: torch.Tensor) -> torch.Tensor:
        gamma = self.to_gamma(global_repr).unsqueeze(1)
        beta = self.to_beta(global_repr).unsqueeze(1)
        return token_repr * (1.0 + gamma) + beta


class MaskedAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, token_repr: torch.Tensor, valid_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(token_repr).squeeze(-1)
        logits = logits.masked_fill(~valid_mask, -1e9)
        weight = torch.softmax(logits, dim=1)
        weight = weight * valid_mask.float()
        weight = weight / weight.sum(dim=1, keepdim=True).clamp(min=1e-8)
        pooled = torch.sum(token_repr * weight.unsqueeze(-1), dim=1)
        return pooled, weight


class CandidateActorCritic(nn.Module):
    def __init__(
        self,
        global_dim: int,
        pair_feat_dim: int,
        hidden_dim: int = 128,
        attn_heads: int = 4,
        attn_layers: int = 1,
        dropout: float = 0.0,
        use_graph_encoder: bool = False,
        op_node_dim: int = 10,
        machine_node_dim: int = 6,
        gnn_hidden_dim: int = 64,
        gnn_layers: int = 2,
        pair_graph_gate_init: float = 0.1,
        use_edge_rule_msg: bool = True,
        use_edge_opmch_msg: bool = True,
        rule_edge_gate_init: float = 0.1,
        opmch_edge_gate_init: float = 0.1,
        edge_message_dropout: float = 0.0,
        use_separate_edge_gates: bool = True,
        use_adaptive_edge_gates: bool = False,
        adaptive_gate_hidden_dim: int = 64,
    ):
        super().__init__()
        self.use_graph_encoder = bool(use_graph_encoder)
        self.use_edge_rule_msg = bool(use_edge_rule_msg)
        self.use_edge_opmch_msg = bool(use_edge_opmch_msg)
        self.gnn_hidden_dim = int(gnn_hidden_dim)

        pair_encoder_input_dim = int(pair_feat_dim)
        if self.use_graph_encoder:
            self.graph_encoder = HeteroLiteEncoder(
                op_node_dim=int(op_node_dim),
                machine_node_dim=int(machine_node_dim),
                pair_node_dim=int(pair_feat_dim),
                hidden_dim=self.gnn_hidden_dim,
                gnn_layers=int(gnn_layers),
                dropout=dropout,
            )
            pair_encoder_input_dim += self.gnn_hidden_dim
            self.pair_graph_gate = nn.Parameter(torch.tensor(float(pair_graph_gate_init), dtype=torch.float32))
            self.use_separate_edge_gates = bool(use_separate_edge_gates)
            self.use_adaptive_edge_gates = bool(use_adaptive_edge_gates)

            self.rule_edge_start = 24
            self.rule_edge_dim = int(max(0, int(pair_feat_dim) - int(self.rule_edge_start)))
            self.opmch_edge_dim = 4

            if self.rule_edge_dim > 0:
                self.rule_edge_proj = nn.Linear(self.rule_edge_dim, self.gnn_hidden_dim)
            else:
                self.rule_edge_proj = None
            self.opmch_edge_proj = nn.Linear(self.opmch_edge_dim, self.gnn_hidden_dim)

            if self.use_separate_edge_gates:
                self.rule_edge_gate = nn.Parameter(torch.tensor(float(rule_edge_gate_init), dtype=torch.float32))
                self.opmch_edge_gate = nn.Parameter(torch.tensor(float(opmch_edge_gate_init), dtype=torch.float32))
            else:
                self.rule_edge_gate = None
                self.opmch_edge_gate = None

            if self.use_adaptive_edge_gates:
                hid = int(max(8, adaptive_gate_hidden_dim))
                self.adaptive_edge_gate = nn.Sequential(
                    nn.Linear(hidden_dim, hid),
                    nn.GELU(),
                    nn.Linear(hid, 2),
                )
            else:
                self.adaptive_edge_gate = None

            self.edge_message_dropout = nn.Dropout(float(edge_message_dropout))

        self.global_encoder = MLP(global_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.pair_encoder = MLP(pair_encoder_input_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.global_norm = nn.LayerNorm(hidden_dim)
        self.pair_norm = nn.LayerNorm(hidden_dim)
        self.self_attn_blocks = nn.ModuleList(
            [ResidualSelfAttentionBlock(hidden_dim, attn_heads, dropout=dropout) for _ in range(attn_layers)]
        )
        self.film = FiLMModulator(hidden_dim)
        self.pool = MaskedAttentionPool(hidden_dim)

        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(
        self,
        global_feat: torch.Tensor,
        pair_feat: torch.Tensor,
        pair_mask: torch.Tensor,
        op_node_feat: Optional[torch.Tensor] = None,
        op_adj: Optional[torch.Tensor] = None,
        machine_node_feat: Optional[torch.Tensor] = None,
        pair_op_idx: Optional[torch.Tensor] = None,
        pair_mch_idx: Optional[torch.Tensor] = None,
        edge_rule_to_pair_feat: Optional[torch.Tensor] = None,
        edge_opmch_to_pair_feat: Optional[torch.Tensor] = None,
    ):
        global_repr = self.global_norm(self.global_encoder(global_feat))

        pair_input = pair_feat
        if self.use_graph_encoder:
            if any(v is None for v in [op_node_feat, op_adj, machine_node_feat, pair_op_idx, pair_mch_idx]):
                raise RuntimeError("Graph encoder enabled but graph tensors are missing in observation/batch.")
            pair_graph_repr = self.graph_encoder(
                op_node_feat=op_node_feat,
                machine_node_feat=machine_node_feat,
                pair_node_feat=pair_feat,
                op_adj=op_adj,
                pair_op_idx=pair_op_idx,
                pair_mch_idx=pair_mch_idx,
                pair_mask=pair_mask,
            )

            bsz, k, _ = pair_feat.shape
            if edge_rule_to_pair_feat is None:
                if pair_feat.shape[-1] > self.rule_edge_start:
                    edge_rule_to_pair_feat = pair_feat[:, :, self.rule_edge_start:]
                else:
                    edge_rule_to_pair_feat = pair_feat.new_zeros((bsz, k, 0))

            if edge_opmch_to_pair_feat is None:
                edge_opmch_to_pair_feat = pair_feat.new_zeros((bsz, k, 4))
                col_ids = [0, 4, 13, 15]
                for idx_col, col in enumerate(col_ids):
                    if pair_feat.shape[-1] > col:
                        edge_opmch_to_pair_feat[:, :, idx_col] = pair_feat[:, :, col]

            if self.rule_edge_dim > 0 and edge_rule_to_pair_feat.shape[-1] > 0 and self.rule_edge_proj is not None:
                if edge_rule_to_pair_feat.shape[-1] != self.rule_edge_dim:
                    aligned = pair_feat.new_zeros((bsz, k, self.rule_edge_dim))
                    take = min(int(edge_rule_to_pair_feat.shape[-1]), int(self.rule_edge_dim))
                    if take > 0:
                        aligned[:, :, :take] = edge_rule_to_pair_feat[:, :, :take]
                    edge_rule_to_pair_feat = aligned
                rule_msg = self.rule_edge_proj(edge_rule_to_pair_feat)
            else:
                rule_msg = pair_graph_repr.new_zeros(pair_graph_repr.shape)
            if not self.use_edge_rule_msg:
                rule_msg = pair_graph_repr.new_zeros(pair_graph_repr.shape)

            if edge_opmch_to_pair_feat.shape[-1] != self.opmch_edge_dim:
                aligned = pair_feat.new_zeros((bsz, k, self.opmch_edge_dim))
                take = min(int(edge_opmch_to_pair_feat.shape[-1]), int(self.opmch_edge_dim))
                if take > 0:
                    aligned[:, :, :take] = edge_opmch_to_pair_feat[:, :, :take]
                edge_opmch_to_pair_feat = aligned
            opmch_msg = self.opmch_edge_proj(edge_opmch_to_pair_feat)
            if not self.use_edge_opmch_msg:
                opmch_msg = pair_graph_repr.new_zeros(pair_graph_repr.shape)

            rule_msg = self.edge_message_dropout(rule_msg)
            opmch_msg = self.edge_message_dropout(opmch_msg)

            base_graph_gate = torch.sigmoid(self.pair_graph_gate)
            if self.use_separate_edge_gates and self.rule_edge_gate is not None and self.opmch_edge_gate is not None:
                rule_gate = torch.sigmoid(self.rule_edge_gate)
                opmch_gate = torch.sigmoid(self.opmch_edge_gate)
            else:
                rule_gate = base_graph_gate
                opmch_gate = base_graph_gate

            if self.use_adaptive_edge_gates and self.adaptive_edge_gate is not None:
                adapt = torch.sigmoid(self.adaptive_edge_gate(global_repr))
                rule_gate = rule_gate * adapt[:, 0].unsqueeze(-1)
                opmch_gate = opmch_gate * adapt[:, 1].unsqueeze(-1)

            def _gate_to_3d(g: torch.Tensor) -> torch.Tensor:
                # Support both scalar gates and batch-conditioned gates.
                if g.ndim == 0:
                    return g.view(1, 1, 1)
                if g.ndim == 1:
                    return g.view(-1, 1, 1)
                if g.ndim == 2:
                    return g.unsqueeze(1)
                return g

            rule_gate_3d = _gate_to_3d(rule_gate)
            opmch_gate_3d = _gate_to_3d(opmch_gate)

            graph_msg = (
                base_graph_gate * pair_graph_repr
                + rule_gate_3d * rule_msg
                + opmch_gate_3d * opmch_msg
            )
            graph_msg = self.edge_message_dropout(graph_msg)
            pair_input = torch.cat([pair_feat, graph_msg], dim=-1)

        token_repr = self.pair_norm(self.pair_encoder(pair_input))

        key_padding_mask = ~pair_mask
        for block in self.self_attn_blocks:
            token_repr = block(token_repr, key_padding_mask=key_padding_mask)

        token_repr = self.film(token_repr, global_repr)
        set_repr, attn_weight = self.pool(token_repr, pair_mask)
        return global_repr, token_repr, set_repr, attn_weight

    def forward(
        self,
        global_feat: torch.Tensor,
        pair_feat: torch.Tensor,
        pair_mask: torch.Tensor,
        op_node_feat: Optional[torch.Tensor] = None,
        op_adj: Optional[torch.Tensor] = None,
        machine_node_feat: Optional[torch.Tensor] = None,
        pair_op_idx: Optional[torch.Tensor] = None,
        pair_mch_idx: Optional[torch.Tensor] = None,
        edge_rule_to_pair_feat: Optional[torch.Tensor] = None,
        edge_opmch_to_pair_feat: Optional[torch.Tensor] = None,
    ):
        global_repr, token_repr, set_repr, attn_weight = self.encode(
            global_feat,
            pair_feat,
            pair_mask,
            op_node_feat=op_node_feat,
            op_adj=op_adj,
            machine_node_feat=machine_node_feat,
            pair_op_idx=pair_op_idx,
            pair_mch_idx=pair_mch_idx,
            edge_rule_to_pair_feat=edge_rule_to_pair_feat,
            edge_opmch_to_pair_feat=edge_opmch_to_pair_feat,
        )
        global_expand = global_repr.unsqueeze(1).expand_as(token_repr)
        set_expand = set_repr.unsqueeze(1).expand_as(token_repr)
        interaction = token_repr * set_expand

        actor_in = torch.cat([token_repr, global_expand, set_expand, interaction], dim=-1)
        logits = self.actor_head(actor_in).squeeze(-1)
        logits = logits.masked_fill(~pair_mask, -1e9)

        masked_token = token_repr * pair_mask.unsqueeze(-1).float()
        mean_repr = masked_token.sum(dim=1) / pair_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        critic_in = torch.cat([global_repr, set_repr, mean_repr], dim=-1)
        values = self.critic_head(critic_in).squeeze(-1)
        return logits, values, attn_weight

    def act(
        self,
        global_feat: torch.Tensor,
        pair_feat: torch.Tensor,
        pair_mask: torch.Tensor,
        deterministic: bool = False,
        op_node_feat: Optional[torch.Tensor] = None,
        op_adj: Optional[torch.Tensor] = None,
        machine_node_feat: Optional[torch.Tensor] = None,
        pair_op_idx: Optional[torch.Tensor] = None,
        pair_mch_idx: Optional[torch.Tensor] = None,
        edge_rule_to_pair_feat: Optional[torch.Tensor] = None,
        edge_opmch_to_pair_feat: Optional[torch.Tensor] = None,
    ):
        logits, values, attn_weight = self.forward(
            global_feat,
            pair_feat,
            pair_mask,
            op_node_feat=op_node_feat,
            op_adj=op_adj,
            machine_node_feat=machine_node_feat,
            pair_op_idx=pair_op_idx,
            pair_mch_idx=pair_mch_idx,
            edge_rule_to_pair_feat=edge_rule_to_pair_feat,
            edge_opmch_to_pair_feat=edge_opmch_to_pair_feat,
        )
        dist = torch.distributions.Categorical(logits=logits)
        actions = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return actions, log_prob, entropy, values, attn_weight

    def evaluate_actions(
        self,
        global_feat: torch.Tensor,
        pair_feat: torch.Tensor,
        pair_mask: torch.Tensor,
        actions: torch.Tensor,
        op_node_feat: Optional[torch.Tensor] = None,
        op_adj: Optional[torch.Tensor] = None,
        machine_node_feat: Optional[torch.Tensor] = None,
        pair_op_idx: Optional[torch.Tensor] = None,
        pair_mch_idx: Optional[torch.Tensor] = None,
        edge_rule_to_pair_feat: Optional[torch.Tensor] = None,
        edge_opmch_to_pair_feat: Optional[torch.Tensor] = None,
    ):
        logits, values, attn_weight = self.forward(
            global_feat,
            pair_feat,
            pair_mask,
            op_node_feat=op_node_feat,
            op_adj=op_adj,
            machine_node_feat=machine_node_feat,
            pair_op_idx=pair_op_idx,
            pair_mch_idx=pair_mch_idx,
            edge_rule_to_pair_feat=edge_rule_to_pair_feat,
            edge_opmch_to_pair_feat=edge_opmch_to_pair_feat,
        )
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, values, attn_weight


# =========================
# Rollout data
# =========================

@dataclass
class StepRecord:
    global_feat: np.ndarray
    pair_feat: np.ndarray
    pair_mask: np.ndarray
    action_idx: int
    log_prob: float
    value: float
    reward: float
    done: bool
    op_node_feat: Optional[np.ndarray] = None
    op_adj: Optional[np.ndarray] = None
    machine_node_feat: Optional[np.ndarray] = None
    pair_op_idx: Optional[np.ndarray] = None
    pair_mch_idx: Optional[np.ndarray] = None
    edge_rule_to_pair_feat: Optional[np.ndarray] = None
    edge_opmch_to_pair_feat: Optional[np.ndarray] = None


class RolloutBuffer:
    def __init__(self) -> None:
        self.records: List[Dict[str, np.ndarray]] = []

    def clear(self) -> None:
        self.records.clear()

    def __len__(self) -> int:
        return len(self.records)

    def add_episode(self, episode_steps: List[StepRecord], gamma: float, gae_lambda: float) -> None:
        if len(episode_steps) == 0:
            return

        rewards = np.asarray([s.reward for s in episode_steps], dtype=np.float32)
        values = np.asarray([s.value for s in episode_steps], dtype=np.float32)
        dones = np.asarray([s.done for s in episode_steps], dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_adv = 0.0
        bootstrap_value = 0.0 if bool(dones[-1]) else float(values[-1])
        next_value = float(bootstrap_value)

        for t in reversed(range(len(episode_steps))):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * nonterminal - values[t]
            last_adv = delta + gamma * gae_lambda * nonterminal * last_adv
            advantages[t] = last_adv
            next_value = values[t]

        returns = advantages + values

        for step, adv, ret in zip(episode_steps, advantages, returns):
            item = {
                "global_feat": step.global_feat.astype(np.float32),
                "pair_feat": step.pair_feat.astype(np.float32),
                "pair_mask": step.pair_mask.astype(bool),
                "action_idx": np.int64(step.action_idx),
                "old_log_prob": np.float32(step.log_prob),
                "old_value": np.float32(step.value),
                "advantage": np.float32(adv),
                "return": np.float32(ret),
            }
            if step.op_node_feat is not None:
                item["op_node_feat"] = np.asarray(step.op_node_feat, dtype=np.float32)
            if step.op_adj is not None:
                item["op_adj"] = np.asarray(step.op_adj, dtype=np.float32)
            if step.machine_node_feat is not None:
                item["machine_node_feat"] = np.asarray(step.machine_node_feat, dtype=np.float32)
            if step.pair_op_idx is not None:
                item["pair_op_idx"] = np.asarray(step.pair_op_idx, dtype=np.int64)
            if step.pair_mch_idx is not None:
                item["pair_mch_idx"] = np.asarray(step.pair_mch_idx, dtype=np.int64)
            if step.edge_rule_to_pair_feat is not None:
                item["edge_rule_to_pair_feat"] = np.asarray(step.edge_rule_to_pair_feat, dtype=np.float32)
            if step.edge_opmch_to_pair_feat is not None:
                item["edge_opmch_to_pair_feat"] = np.asarray(step.edge_opmch_to_pair_feat, dtype=np.float32)
            self.records.append(item)

    def as_tensors(self, device: torch.device) -> Dict[str, torch.Tensor]:
        if len(self.records) == 0:
            raise RuntimeError("RolloutBuffer is empty.")

        pair_arrs = [np.asarray(r["pair_feat"], dtype=np.float32) for r in self.records]
        pair_masks = [np.asarray(r["pair_mask"], dtype=bool).reshape(-1) for r in self.records]
        pair_max = max(1, max(int(a.shape[0]) if a.ndim == 2 else 0 for a in pair_arrs))
        pair_dim = max(1, max(int(a.shape[1]) if a.ndim == 2 and a.shape[1] > 0 else 0 for a in pair_arrs))

        pair_np = np.zeros((len(self.records), pair_max, pair_dim), dtype=np.float32)
        pair_mask_np = np.zeros((len(self.records), pair_max), dtype=bool)
        for i, arr in enumerate(pair_arrs):
            if arr.ndim != 2 or arr.shape[0] == 0:
                continue
            rows = min(pair_max, int(arr.shape[0]))
            cols = min(pair_dim, int(arr.shape[1]))
            pair_np[i, :rows, :cols] = arr[:rows, :cols]
            if pair_masks[i].shape[0] > 0:
                valid_rows = min(rows, int(pair_masks[i].shape[0]))
                pair_mask_np[i, :valid_rows] = pair_masks[i][:valid_rows]

        out = {
            "global_feat": torch.tensor(np.stack([r["global_feat"] for r in self.records]), dtype=torch.float32, device=device),
            "pair_feat": torch.tensor(pair_np, dtype=torch.float32, device=device),
            "pair_mask": torch.tensor(pair_mask_np, dtype=torch.bool, device=device),
            "action_idx": torch.tensor(np.asarray([r["action_idx"] for r in self.records]), dtype=torch.long, device=device),
            "old_log_prob": torch.tensor(np.asarray([r["old_log_prob"] for r in self.records]), dtype=torch.float32, device=device),
            "old_value": torch.tensor(np.asarray([r["old_value"] for r in self.records]), dtype=torch.float32, device=device),
            "advantage": torch.tensor(np.asarray([r["advantage"] for r in self.records]), dtype=torch.float32, device=device),
            "return": torch.tensor(np.asarray([r["return"] for r in self.records]), dtype=torch.float32, device=device),
        }
        has_graph = all(
            key in self.records[0] and all(key in r for r in self.records)
            for key in [
                "op_node_feat",
                "op_adj",
                "machine_node_feat",
                "pair_op_idx",
                "pair_mch_idx",
                "edge_rule_to_pair_feat",
                "edge_opmch_to_pair_feat",
            ]
        )
        if has_graph:
            n = len(self.records)
            k = int(out["pair_feat"].shape[1])

            op_nodes = [np.asarray(r["op_node_feat"], dtype=np.float32) for r in self.records]
            op_adjs = [np.asarray(r["op_adj"], dtype=np.float32) for r in self.records]
            mch_nodes = [np.asarray(r["machine_node_feat"], dtype=np.float32) for r in self.records]
            edge_rule = [np.asarray(r["edge_rule_to_pair_feat"], dtype=np.float32) for r in self.records]
            edge_opmch = [np.asarray(r["edge_opmch_to_pair_feat"], dtype=np.float32) for r in self.records]

            max_ops = max(arr.shape[0] for arr in op_nodes)
            max_mch = max(arr.shape[0] for arr in mch_nodes)
            op_dim = op_nodes[0].shape[1]
            mch_dim = mch_nodes[0].shape[1]
            edge_rule_dim = edge_rule[0].shape[1] if edge_rule[0].ndim == 2 else 0
            edge_opmch_dim = edge_opmch[0].shape[1] if edge_opmch[0].ndim == 2 else 4

            op_node_np = np.zeros((n, max_ops, op_dim), dtype=np.float32)
            op_adj_np = np.zeros((n, max_ops, max_ops), dtype=np.float32)
            mch_node_np = np.zeros((n, max_mch, mch_dim), dtype=np.float32)
            pair_op_idx_np = np.zeros((n, k), dtype=np.int64)
            pair_mch_idx_np = np.zeros((n, k), dtype=np.int64)
            edge_rule_np = np.zeros((n, k, edge_rule_dim), dtype=np.float32)
            edge_opmch_np = np.zeros((n, k, edge_opmch_dim), dtype=np.float32)

            for i, rec in enumerate(self.records):
                t_i = op_nodes[i].shape[0]
                m_i = mch_nodes[i].shape[0]
                op_node_np[i, :t_i, :] = op_nodes[i]
                op_adj_np[i, :t_i, :t_i] = op_adjs[i]
                mch_node_np[i, :m_i, :] = mch_nodes[i]

                op_idx_raw = np.asarray(rec["pair_op_idx"], dtype=np.int64).reshape(-1)
                mch_idx_raw = np.asarray(rec["pair_mch_idx"], dtype=np.int64).reshape(-1)
                op_idx = np.zeros((k,), dtype=np.int64)
                mch_idx = np.zeros((k,), dtype=np.int64)
                used_k = min(k, op_idx_raw.shape[0], mch_idx_raw.shape[0])
                if used_k > 0:
                    op_idx[:used_k] = op_idx_raw[:used_k]
                    mch_idx[:used_k] = mch_idx_raw[:used_k]

                if edge_rule_dim > 0 and edge_rule[i].ndim == 2:
                    used_rule_rows = min(k, edge_rule[i].shape[0])
                    used_rule_dim = min(edge_rule_dim, edge_rule[i].shape[1])
                    if used_rule_rows > 0 and used_rule_dim > 0:
                        edge_rule_np[i, :used_rule_rows, :used_rule_dim] = edge_rule[i][:used_rule_rows, :used_rule_dim]
                if edge_opmch_dim > 0 and edge_opmch[i].ndim == 2:
                    used_opmch_rows = min(k, edge_opmch[i].shape[0])
                    used_opmch_dim = min(edge_opmch_dim, edge_opmch[i].shape[1])
                    if used_opmch_rows > 0 and used_opmch_dim > 0:
                        edge_opmch_np[i, :used_opmch_rows, :used_opmch_dim] = edge_opmch[i][:used_opmch_rows, :used_opmch_dim]

                valid_mask_i = pair_masks[i]
                if valid_mask_i.shape[0] != k:
                    vm = np.zeros((k,), dtype=bool)
                    vm[: min(k, valid_mask_i.shape[0])] = valid_mask_i[: min(k, valid_mask_i.shape[0])]
                    valid_mask_i = vm
                op_idx[~valid_mask_i] = 0
                mch_idx[~valid_mask_i] = 0

                if t_i > 0:
                    op_idx[valid_mask_i] = np.clip(op_idx[valid_mask_i], 0, t_i - 1)
                else:
                    op_idx[valid_mask_i] = 0
                if m_i > 0:
                    mch_idx[valid_mask_i] = np.clip(mch_idx[valid_mask_i], 0, m_i - 1)
                else:
                    mch_idx[valid_mask_i] = 0

                pair_op_idx_np[i] = op_idx
                pair_mch_idx_np[i] = mch_idx
                if edge_rule_dim > 0:
                    edge_rule_np[i, ~valid_mask_i, :] = 0.0
                if edge_opmch_dim > 0:
                    edge_opmch_np[i, ~valid_mask_i, :] = 0.0

            out["op_node_feat"] = torch.tensor(op_node_np, dtype=torch.float32, device=device)
            out["op_adj"] = torch.tensor(op_adj_np, dtype=torch.float32, device=device)
            out["machine_node_feat"] = torch.tensor(mch_node_np, dtype=torch.float32, device=device)
            out["pair_op_idx"] = torch.tensor(pair_op_idx_np, dtype=torch.long, device=device)
            out["pair_mch_idx"] = torch.tensor(pair_mch_idx_np, dtype=torch.long, device=device)
            out["edge_rule_to_pair_feat"] = torch.tensor(edge_rule_np, dtype=torch.float32, device=device)
            out["edge_opmch_to_pair_feat"] = torch.tensor(edge_opmch_np, dtype=torch.float32, device=device)
        return out


# =========================
# Agent
# =========================

class Agent_C:
    def __init__(self, config: AgentCConfig, global_dim: int, pair_feat_dim: int):
        self.config = config
        self.global_dim = int(global_dim)
        self.pair_feat_dim = int(pair_feat_dim)
        self.device = torch.device(config.device)

        self.policy = CandidateActorCritic(
            global_dim=self.global_dim,
            pair_feat_dim=self.pair_feat_dim,
            hidden_dim=config.hidden_dim,
            attn_heads=config.attn_heads,
            attn_layers=config.attn_layers,
            dropout=config.dropout,
            use_graph_encoder=bool(getattr(config, "use_graph_encoder", False)),
            use_edge_rule_msg=bool(getattr(config, "use_edge_rule_msg", True)),
            use_edge_opmch_msg=bool(getattr(config, "use_edge_opmch_msg", True)),
            op_node_dim=int(getattr(config, "op_node_dim", 10)),
            machine_node_dim=int(getattr(config, "machine_node_dim", 6)),
            gnn_hidden_dim=int(getattr(config, "gnn_hidden_dim", 64)),
            gnn_layers=int(getattr(config, "gnn_layers", 2)),
            pair_graph_gate_init=float(getattr(config, "pair_graph_gate_init", 0.1)),
            rule_edge_gate_init=float(getattr(config, "rule_edge_gate_init", 0.1)),
            opmch_edge_gate_init=float(getattr(config, "opmch_edge_gate_init", 0.1)),
            edge_message_dropout=float(getattr(config, "edge_message_dropout", 0.0)),
            use_separate_edge_gates=bool(getattr(config, "use_separate_edge_gates", True)),
            use_adaptive_edge_gates=bool(getattr(config, "use_adaptive_edge_gates", False)),
            adaptive_gate_hidden_dim=int(getattr(config, "adaptive_gate_hidden_dim", 64)),
        ).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.lr)

        self.overflow_events = 0
        self.overflow_candidates = 0
        self._overflow_warning_count = 0

    def _obs_to_tensors(self, obs: Dict[str, object]):
        packed = preprocess_obs(
            obs,
            global_dim=self.global_dim,
            pair_feat_dim=self.pair_feat_dim,
            max_candidates=self.config.max_candidates,
            use_candidate_set_feat=self.config.use_candidate_set_feat,
            overflow_strategy=self.config.overflow_strategy,
        )
        if packed.get("overflow_count", 0) > 0:
            self.overflow_events += 1
            self.overflow_candidates += int(packed["overflow_count"])
            if self._overflow_warning_count < int(self.config.overflow_warning_limit):
                print(
                    f"[Agent_C warning] observation candidate count={packed['original_candidate_count']} "
                    f"exceeds max_candidates={self.config.max_candidates}; "
                    f"truncated by {packed['overflow_count']}."
                )
                self._overflow_warning_count += 1

        global_tensor = torch.tensor(packed["global_feat"], dtype=torch.float32, device=self.device).unsqueeze(0)
        pair_tensor = torch.tensor(packed["pair_feat"], dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_tensor = torch.tensor(packed["pair_mask"], dtype=torch.bool, device=self.device).unsqueeze(0)
        return packed, global_tensor, pair_tensor, mask_tensor

    def _agent_obs_list_to_tensors(self, obs_list: List[Dict[str, object]]):
        packed_list = []
        for obs in obs_list:
            packed = preprocess_obs(
                obs,
                global_dim=self.global_dim,
                pair_feat_dim=self.pair_feat_dim,
                max_candidates=self.config.max_candidates,
                use_candidate_set_feat=self.config.use_candidate_set_feat,
                overflow_strategy=self.config.overflow_strategy,
            )
            if packed.get("overflow_count", 0) > 0:
                self.overflow_events += 1
                self.overflow_candidates += int(packed["overflow_count"])
                if self._overflow_warning_count < int(self.config.overflow_warning_limit):
                    print(
                        f"[Agent_C warning] observation candidate count={packed['original_candidate_count']} "
                        f"exceeds max_candidates={self.config.max_candidates}; "
                        f"truncated by {packed['overflow_count']}."
                    )
                    self._overflow_warning_count += 1
            packed_list.append(packed)

        global_np = np.stack([p["global_feat"] for p in packed_list], axis=0).astype(np.float32)
        pair_np = np.stack([p["pair_feat"] for p in packed_list], axis=0).astype(np.float32)
        mask_np = np.stack([p["pair_mask"] for p in packed_list], axis=0).astype(bool)

        global_tensor = torch.tensor(global_np, dtype=torch.float32, device=self.device)
        pair_tensor = torch.tensor(pair_np, dtype=torch.float32, device=self.device)
        mask_tensor = torch.tensor(mask_np, dtype=torch.bool, device=self.device)
        return packed_list, global_tensor, pair_tensor, mask_tensor

    @torch.no_grad()
    def select_action(self, obs: Dict[str, object], deterministic: bool = False) -> Dict[str, object]:
        packed, global_tensor, pair_tensor, mask_tensor, graph_tensors = self._obs_to_tensors(obs)
        action, log_prob, entropy, value, attn_weight = self.policy.act(
            global_tensor,
            pair_tensor,
            mask_tensor,
            deterministic=deterministic,
            **graph_tensors,
        )
        action_idx = int(action.item())
        op_id, mch_id = packed["candidate_pairs"][action_idx]
        return {
            "action_idx": action_idx,
            "op_id": int(op_id),
            "mch_id": int(mch_id),
            "log_prob": float(log_prob.item()),
            "entropy": float(entropy.item()),
            "value": float(value.item()),
            "candidate_attention": attn_weight.squeeze(0).detach().cpu().numpy(),
            "packed_obs": packed,
            "was_truncated": bool(packed.get("was_truncated", False)),
            "overflow_count": int(packed.get("overflow_count", 0)),
            "original_candidate_count": int(packed.get("original_candidate_count", len(obs.get("candidate_pairs", [])))),
            "used_candidate_count": int(packed.get("used_candidate_count", len(packed["candidate_pairs"]))),
        }

    def act(self, obs: Dict[str, object], deterministic: bool = False) -> Tuple[int, int, Dict[str, object]]:
        info = self.select_action(obs, deterministic=deterministic)
        return info["op_id"], info["mch_id"], info

    @torch.no_grad()
    def select_action_batch(self, obs_list: List[Dict[str, object]], deterministic: bool = False):
        if len(obs_list) == 0:
            return []

        packed_list, global_tensor, pair_tensor, mask_tensor, graph_tensors = self._agent_obs_list_to_tensors(obs_list)
        actions, log_probs, entropies, values, attn_weights = self.policy.act(
            global_tensor,
            pair_tensor,
            mask_tensor,
            deterministic=deterministic,
            **graph_tensors,
        )

        infos = []
        for idx, packed in enumerate(packed_list):
            action_idx = int(actions[idx].item())
            op_id, mch_id = packed["candidate_pairs"][action_idx]
            infos.append(
                {
                    "action_idx": action_idx,
                    "op_id": int(op_id),
                    "mch_id": int(mch_id),
                    "log_prob": float(log_probs[idx].item()),
                    "entropy": float(entropies[idx].item()),
                    "value": float(values[idx].item()),
                    "candidate_attention": attn_weights[idx].detach().cpu().numpy(),
                    "packed_obs": packed,
                    "was_truncated": bool(packed.get("was_truncated", False)),
                    "overflow_count": int(packed.get("overflow_count", 0)),
                    "original_candidate_count": int(packed.get("original_candidate_count", len(obs_list[idx].get("candidate_pairs", [])))),
                    "used_candidate_count": int(packed.get("used_candidate_count", len(packed["candidate_pairs"]))),
                }
            )
        return infos

    def act_batch(self, obs_list: List[Dict[str, object]], deterministic: bool = False):
        infos = self.select_action_batch(obs_list, deterministic=deterministic)
        return [(info["op_id"], info["mch_id"], info) for info in infos]

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        batch = buffer.as_tensors(self.device)
        raw_advantages = batch["advantage"]
        adv_std = float(raw_advantages.std(unbiased=False).item())
        batch_size = int(batch["global_feat"].shape[0])
        minibatch_size = int(min(self.config.minibatch_size, batch_size))

        if batch_size == 0:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
                "clip_frac": 0.0,
                "total_loss": 0.0,
                "effective_ppo_epochs": 0.0,
                "early_stop_triggered": 0.0,
                "skip_update_reason": "empty_batch",
            }

        if batch_size < max(8, minibatch_size // 2):
            effective_ppo_epochs = 1
            skip_reason = "small_buffer"
        elif adv_std < 1e-6:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
                "clip_frac": 0.0,
                "total_loss": 0.0,
                "effective_ppo_epochs": 0.0,
                "early_stop_triggered": 0.0,
                "skip_update_reason": "flat_advantage",
            }
        else:
            effective_ppo_epochs = int(self.config.ppo_epochs)
            skip_reason = ""
            if batch_size < minibatch_size * 2:
                effective_ppo_epochs = min(effective_ppo_epochs, 2)

        advantages = (raw_advantages - raw_advantages.mean()) / raw_advantages.std(unbiased=False).clamp(min=1e-6)
        returns = batch["return"]
        logs = {k: [] for k in ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "total_loss"]}
        early_stop_triggered = False

        for _ in range(effective_ppo_epochs):
            perm = torch.randperm(batch_size, device=self.device)
            epoch_kls = []

            for start in range(0, batch_size, minibatch_size):
                idx = perm[start : start + minibatch_size]

                mb_global = batch["global_feat"][idx]
                mb_pair = batch["pair_feat"][idx]
                mb_mask = batch["pair_mask"][idx]
                mb_action = batch["action_idx"][idx]
                mb_old_log_prob = batch["old_log_prob"][idx]
                mb_adv = advantages[idx]
                mb_ret = returns[idx]

                mb_graph = {}
                for key in [
                    "op_node_feat",
                    "op_adj",
                    "machine_node_feat",
                    "pair_op_idx",
                    "pair_mch_idx",
                    "edge_rule_to_pair_feat",
                    "edge_opmch_to_pair_feat",
                ]:
                    if key in batch:
                        mb_graph[key] = batch[key][idx]

                new_log_prob, entropy, values, _ = self.policy.evaluate_actions(
                    mb_global,
                    mb_pair,
                    mb_mask,
                    mb_action,
                    **mb_graph,
                )
                ratio = torch.exp(new_log_prob - mb_old_log_prob)
                ratio_clipped = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio)

                policy_loss = -torch.min(ratio * mb_adv, ratio_clipped * mb_adv).mean()
                value_loss = F.mse_loss(values, mb_ret)
                entropy_mean = entropy.mean()
                total_loss = policy_loss + self.config.value_coef * value_loss - self.config.ent_coef * entropy_mean

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                approx_kl = float((mb_old_log_prob - new_log_prob).mean().abs().item())
                clip_frac = float(((ratio - ratio_clipped).abs() > 1e-8).float().mean().item())
                epoch_kls.append(approx_kl)

                logs["policy_loss"].append(float(policy_loss.item()))
                logs["value_loss"].append(float(value_loss.item()))
                logs["entropy"].append(float(entropy_mean.item()))
                logs["approx_kl"].append(approx_kl)
                logs["clip_frac"].append(clip_frac)
                logs["total_loss"].append(float(total_loss.item()))

            target_kl = getattr(self.config, "target_kl", None)
            if target_kl is not None and len(epoch_kls) > 0 and float(np.mean(epoch_kls)) > float(target_kl):
                early_stop_triggered = True
                break

        result = {k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in logs.items()}
        result["effective_ppo_epochs"] = float(effective_ppo_epochs)
        result["early_stop_triggered"] = float(early_stop_triggered)
        result["skip_update_reason"] = skip_reason
        return result

    def save(self, path: str) -> None:
        torch.save(
            {
                "config": self.config.__dict__,
                "global_dim": self.global_dim,
                "pair_feat_dim": self.pair_feat_dim,
                "state_dict": self.policy.state_dict(),
            },
            path,
        )

    def load(self, path: str, strict: bool = True) -> None:
        try:
            payload = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(payload["state_dict"], strict=strict)


# =========================
# Top-K prefilter support
# =========================

TOPK_PREFILTER_DEFAULTS = {
    "topk_prefilter_enabled": False,
    "topk_k": 32,
    "topk_keep_eet": 2,
    "topk_keep_pt": 2,
    "topk_keep_preferred_rule": 2,
    "topk_preferred_rule": "FIFO_SPT",
    "topk_machine_diversity": True,
    "topk_score_weights": {"eet": 1.0, "pt": 0.3, "queue": 0.1, "slack": 0.2},
}


def _cfg_get(cfg, key):
    if hasattr(cfg, key):
        return getattr(cfg, key)
    return TOPK_PREFILTER_DEFAULTS[key]


def _safe_col(arr: np.ndarray, col: int, default: float = 0.0) -> np.ndarray:
    if arr.ndim != 2 or arr.shape[1] <= col:
        return np.full((arr.shape[0],), float(default), dtype=np.float32)
    return arr[:, col].astype(np.float32)


def _topk_rank_indices(
    candidate_pairs,
    pair_feat: np.ndarray,
    pair_sources,
    topk_k: int,
    keep_eet: int,
    keep_pt: int,
    keep_preferred_rule: int,
    preferred_rule: str,
    machine_diversity: bool,
    score_weights: Dict[str, float],
):
    n = int(len(candidate_pairs))
    if n == 0:
        return []
    if topk_k >= n:
        return list(range(n))

    eet = _safe_col(pair_feat, 4, default=0.0)
    pt = _safe_col(pair_feat, 0, default=0.0)
    queue = _safe_col(pair_feat, 6, default=0.0)
    slack = _safe_col(pair_feat, 8, default=0.0)

    w_eet = float(score_weights.get("eet", 1.0))
    w_pt = float(score_weights.get("pt", 0.3))
    w_queue = float(score_weights.get("queue", 0.1))
    w_slack = float(score_weights.get("slack", 0.2))

    score = w_eet * eet + w_pt * pt + w_queue * queue + w_slack * slack
    order = list(np.argsort(score, kind="mergesort").tolist())

    keep = set()
    if keep_eet > 0:
        for i in np.argsort(eet, kind="mergesort")[: int(keep_eet)].tolist():
            keep.add(int(i))
    if keep_pt > 0:
        for i in np.argsort(pt, kind="mergesort")[: int(keep_pt)].tolist():
            keep.add(int(i))

    if keep_preferred_rule > 0 and pair_sources is not None:
        preferred = []
        for i, src in enumerate(pair_sources):
            src_set = set(src) if isinstance(src, (list, tuple, set)) else set()
            if preferred_rule in src_set:
                preferred.append(i)
        preferred_sorted = sorted(preferred, key=lambda i: (score[i], i))
        for i in preferred_sorted[: int(keep_preferred_rule)]:
            keep.add(int(i))

    selected = []
    for i in order:
        if i in keep:
            selected.append(i)
    for i in order:
        if len(selected) >= int(topk_k):
            break
        if i not in keep:
            selected.append(i)

    if machine_diversity and len(selected) > 0:
        selected_set = set(selected)
        selected_machines = set(int(candidate_pairs[i][1]) for i in selected)
        all_machines = set(int(m) for _, m in candidate_pairs)
        missing = sorted(list(all_machines - selected_machines))
        if len(missing) > 0:
            selected_keep = set(i for i in selected if i in keep)
            replaceable = [i for i in reversed(selected) if i not in selected_keep]
            for mch in missing:
                if len(replaceable) == 0:
                    break
                cand_i = None
                for i in order:
                    if i in selected_set:
                        continue
                    if int(candidate_pairs[i][1]) == int(mch):
                        cand_i = i
                        break
                if cand_i is None:
                    continue
                drop_i = replaceable.pop(0)
                if drop_i in selected_set:
                    selected_set.remove(drop_i)
                selected_set.add(cand_i)
            selected = sorted(list(selected_set), key=lambda i: (score[i], i))[: int(topk_k)]

    return selected


def preprocess_obs_topk(
    obs: Dict[str, object],
    global_dim: int,
    pair_feat_dim: int,
    max_candidates: int,
    use_candidate_set_feat: bool = True,
    overflow_strategy: str = "truncate",
    topk_prefilter_enabled: bool = False,
    topk_k: int = 32,
    topk_keep_eet: int = 2,
    topk_keep_pt: int = 2,
    topk_keep_preferred_rule: int = 2,
    topk_preferred_rule: str = "FIFO_SPT",
    topk_machine_diversity: bool = True,
    topk_score_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    candidate_pairs = list(obs["candidate_pairs"])
    if len(candidate_pairs) == 0:
        raise RuntimeError("No candidate_pairs found in current observation.")

    global_feat = _flatten_f32(obs["global_feat"])
    if use_candidate_set_feat and "candidate_set_feat" in obs:
        global_feat = np.concatenate([global_feat, _flatten_f32(obs["candidate_set_feat"])], axis=0)
    global_feat = pad_global_feat(global_feat, global_dim)

    pair_feat_raw = np.asarray(obs["pair_feat"], dtype=np.float32)
    pair_op_idx_raw = np.asarray(obs.get("pair_op_idx", [p[0] for p in candidate_pairs]), dtype=np.int64)
    pair_mch_idx_raw = np.asarray(obs.get("pair_mch_idx", [p[1] for p in candidate_pairs]), dtype=np.int64)
    edge_rule_raw = obs.get("edge_rule_to_pair_feat", None)
    edge_opmch_raw = obs.get("edge_opmch_to_pair_feat", None)
    if edge_rule_raw is None or edge_opmch_raw is None:
        derived_rule, derived_opmch = _extract_edge_feats_from_pair_feat(pair_feat_raw)
        if edge_rule_raw is None:
            edge_rule_raw = derived_rule
        if edge_opmch_raw is None:
            edge_opmch_raw = derived_opmch
    edge_rule_raw = np.asarray(edge_rule_raw, dtype=np.float32)
    edge_opmch_raw = np.asarray(edge_opmch_raw, dtype=np.float32)

    pair_sources = list(obs.get("pair_sources", []))
    if len(pair_sources) < len(candidate_pairs):
        pair_sources = pair_sources + [[] for _ in range(len(candidate_pairs) - len(pair_sources))]

    topk_selected = None
    original_candidate_count = int(len(candidate_pairs))

    if bool(topk_prefilter_enabled):
        limit_k = int(max(1, min(int(topk_k), int(max_candidates))))
        topk_selected = _topk_rank_indices(
            candidate_pairs=candidate_pairs,
            pair_feat=pair_feat_raw,
            pair_sources=pair_sources,
            topk_k=limit_k,
            keep_eet=int(max(0, topk_keep_eet)),
            keep_pt=int(max(0, topk_keep_pt)),
            keep_preferred_rule=int(max(0, topk_keep_preferred_rule)),
            preferred_rule=str(topk_preferred_rule),
            machine_diversity=bool(topk_machine_diversity),
            score_weights=topk_score_weights or TOPK_PREFILTER_DEFAULTS["topk_score_weights"],
        )
        candidate_pairs = [candidate_pairs[i] for i in topk_selected]
        pair_feat_raw = pair_feat_raw[topk_selected]
        pair_op_idx_raw = pair_op_idx_raw[topk_selected]
        pair_mch_idx_raw = pair_mch_idx_raw[topk_selected]
        if edge_rule_raw.ndim == 2 and edge_rule_raw.shape[0] >= len(topk_selected):
            edge_rule_raw = edge_rule_raw[topk_selected]
        if edge_opmch_raw.ndim == 2 and edge_opmch_raw.shape[0] >= len(topk_selected):
            edge_opmch_raw = edge_opmch_raw[topk_selected]
        pair_sources = [pair_sources[i] for i in topk_selected]

    pair_feat, pair_mask, overflow_count = pad_pair_feat(
        pair_feat_raw,
        max_candidates,
        pair_feat_dim,
        overflow_strategy=overflow_strategy,
    )

    used_candidate_count = int(pair_mask.sum())
    candidate_pairs = candidate_pairs[:used_candidate_count]
    pair_op_idx = pad_index_array(pair_op_idx_raw[:used_candidate_count], max_candidates, fill_value=0)
    pair_mch_idx = pad_index_array(pair_mch_idx_raw[:used_candidate_count], max_candidates, fill_value=0)
    pair_sources = pair_sources[:used_candidate_count]

    edge_rule_dim = int(edge_rule_raw.shape[1]) if edge_rule_raw.ndim == 2 else 0
    edge_opmch_dim = int(edge_opmch_raw.shape[1]) if edge_opmch_raw.ndim == 2 else 4
    edge_rule_feat = pad_pair_matrix(edge_rule_raw[:used_candidate_count], max_candidates, edge_rule_dim)
    edge_opmch_feat = pad_pair_matrix(edge_opmch_raw[:used_candidate_count], max_candidates, edge_opmch_dim)

    op_node_feat = None
    op_adj = None
    machine_node_feat = None
    if "op_node_feat" in obs:
        op_node_feat = np.asarray(obs["op_node_feat"], dtype=np.float32)
    if "op_adj" in obs:
        op_adj = np.asarray(obs["op_adj"], dtype=np.float32)
    if "machine_node_feat" in obs:
        machine_node_feat = np.asarray(obs["machine_node_feat"], dtype=np.float32)

    return {
        "global_feat": global_feat,
        "pair_feat": pair_feat,
        "pair_mask": pair_mask,
        "candidate_pairs": candidate_pairs,
        "pair_sources": pair_sources,
        "pair_op_idx": pair_op_idx,
        "pair_mch_idx": pair_mch_idx,
        "edge_rule_to_pair_feat": edge_rule_feat,
        "edge_opmch_to_pair_feat": edge_opmch_feat,
        "op_node_feat": op_node_feat,
        "op_adj": op_adj,
        "machine_node_feat": machine_node_feat,
        "original_candidate_count": int(original_candidate_count),
        "used_candidate_count": int(used_candidate_count),
        "overflow_count": int(overflow_count),
        "was_truncated": bool(overflow_count > 0),
        "topk_prefilter_enabled": bool(topk_prefilter_enabled),
        "topk_selected_count": int(len(topk_selected) if topk_selected is not None else original_candidate_count),
        "topk_reduced_count": int(original_candidate_count - (len(topk_selected) if topk_selected is not None else original_candidate_count)),
    }


def _required_candidates_for_obs_with_topk(obs: Dict[str, object], topk_prefilter_enabled: bool, topk_k: int) -> int:
    candidate_pairs = list(obs.get("candidate_pairs", []))
    pair_feat = np.asarray(obs.get("pair_feat", []), dtype=np.float32)
    pair_rows = int(pair_feat.shape[0]) if pair_feat.ndim == 2 else 0
    required = max(int(len(candidate_pairs)), int(pair_rows), 1)
    if bool(topk_prefilter_enabled):
        required = min(required, int(max(1, topk_k)))
    return int(max(1, required))


def _agent_obs_to_tensors_with_topk(self, obs: Dict[str, object]):
    topk_prefilter_enabled = bool(getattr(self.config, "topk_prefilter_enabled", False))
    topk_k = int(getattr(self.config, "topk_k", TOPK_PREFILTER_DEFAULTS["topk_k"]))
    required_candidates = _required_candidates_for_obs_with_topk(
        obs=obs,
        topk_prefilter_enabled=topk_prefilter_enabled,
        topk_k=topk_k,
    )

    configured_cap = int(getattr(self.config, "max_candidates", 1))
    if required_candidates > configured_cap:
        notice_count = int(getattr(self, "_dynamic_candidate_notice_count", 0))
        if notice_count < int(self.config.overflow_warning_limit):
            print(
                f"[Agent_C info] observation candidate count={required_candidates} "
                f"exceeds configured max_candidates={configured_cap}; "
                "using dynamic padding without truncation."
            )
        setattr(self, "_dynamic_candidate_notice_count", notice_count + 1)

    packed = preprocess_obs_topk(
        obs,
        global_dim=self.global_dim,
        pair_feat_dim=self.pair_feat_dim,
        max_candidates=required_candidates,
        use_candidate_set_feat=self.config.use_candidate_set_feat,
        overflow_strategy=self.config.overflow_strategy,
        topk_prefilter_enabled=topk_prefilter_enabled,
        topk_k=topk_k,
        topk_keep_eet=int(getattr(self.config, "topk_keep_eet", TOPK_PREFILTER_DEFAULTS["topk_keep_eet"])),
        topk_keep_pt=int(getattr(self.config, "topk_keep_pt", TOPK_PREFILTER_DEFAULTS["topk_keep_pt"])),
        topk_keep_preferred_rule=int(getattr(self.config, "topk_keep_preferred_rule", TOPK_PREFILTER_DEFAULTS["topk_keep_preferred_rule"])),
        topk_preferred_rule=str(getattr(self.config, "topk_preferred_rule", TOPK_PREFILTER_DEFAULTS["topk_preferred_rule"])),
        topk_machine_diversity=bool(getattr(self.config, "topk_machine_diversity", TOPK_PREFILTER_DEFAULTS["topk_machine_diversity"])),
        topk_score_weights=dict(getattr(self.config, "topk_score_weights", TOPK_PREFILTER_DEFAULTS["topk_score_weights"])),
    )
    if packed.get("overflow_count", 0) > 0:
        self.overflow_events += 1
        self.overflow_candidates += int(packed["overflow_count"])
        if self._overflow_warning_count < int(self.config.overflow_warning_limit):
            print(
                f"[Agent_C warning] observation candidate count={packed['original_candidate_count']} "
                f"exceeds max_candidates={self.config.max_candidates}; "
                f"truncated by {packed['overflow_count']}."
            )
            self._overflow_warning_count += 1

    global_tensor = torch.tensor(packed["global_feat"], dtype=torch.float32, device=self.device).unsqueeze(0)
    pair_tensor = torch.tensor(packed["pair_feat"], dtype=torch.float32, device=self.device).unsqueeze(0)
    mask_tensor = torch.tensor(packed["pair_mask"], dtype=torch.bool, device=self.device).unsqueeze(0)
    graph_tensors = {}
    if bool(getattr(self.config, "use_graph_encoder", False)):
        graph_tensors = {
            "op_node_feat": torch.tensor(np.asarray(packed["op_node_feat"], dtype=np.float32), dtype=torch.float32, device=self.device).unsqueeze(0),
            "op_adj": torch.tensor(np.asarray(packed["op_adj"], dtype=np.float32), dtype=torch.float32, device=self.device).unsqueeze(0),
            "machine_node_feat": torch.tensor(np.asarray(packed["machine_node_feat"], dtype=np.float32), dtype=torch.float32, device=self.device).unsqueeze(0),
            "pair_op_idx": torch.tensor(np.asarray(packed["pair_op_idx"], dtype=np.int64), dtype=torch.long, device=self.device).unsqueeze(0),
            "pair_mch_idx": torch.tensor(np.asarray(packed["pair_mch_idx"], dtype=np.int64), dtype=torch.long, device=self.device).unsqueeze(0),
            "edge_rule_to_pair_feat": torch.tensor(np.asarray(packed["edge_rule_to_pair_feat"], dtype=np.float32), dtype=torch.float32, device=self.device).unsqueeze(0),
            "edge_opmch_to_pair_feat": torch.tensor(np.asarray(packed["edge_opmch_to_pair_feat"], dtype=np.float32), dtype=torch.float32, device=self.device).unsqueeze(0),
        }
    return packed, global_tensor, pair_tensor, mask_tensor, graph_tensors


def _agent_obs_list_to_tensors_with_topk(self, obs_list: List[Dict[str, object]]):
    topk_prefilter_enabled = bool(getattr(self.config, "topk_prefilter_enabled", False))
    topk_k = int(getattr(self.config, "topk_k", TOPK_PREFILTER_DEFAULTS["topk_k"]))
    required_candidates_list = [
        _required_candidates_for_obs_with_topk(
            obs=obs,
            topk_prefilter_enabled=topk_prefilter_enabled,
            topk_k=topk_k,
        )
        for obs in obs_list
    ]
    batch_max_candidates = int(max(required_candidates_list)) if len(required_candidates_list) > 0 else 1

    configured_cap = int(getattr(self.config, "max_candidates", 1))
    if batch_max_candidates > configured_cap:
        notice_count = int(getattr(self, "_dynamic_candidate_notice_count", 0))
        if notice_count < int(self.config.overflow_warning_limit):
            print(
                f"[Agent_C info] batch candidate max={batch_max_candidates} "
                f"exceeds configured max_candidates={configured_cap}; "
                "using dynamic batch padding without truncation."
            )
        setattr(self, "_dynamic_candidate_notice_count", notice_count + 1)

    packed_list = []
    for obs in obs_list:
        packed = preprocess_obs_topk(
            obs,
            global_dim=self.global_dim,
            pair_feat_dim=self.pair_feat_dim,
            max_candidates=batch_max_candidates,
            use_candidate_set_feat=self.config.use_candidate_set_feat,
            overflow_strategy=self.config.overflow_strategy,
            topk_prefilter_enabled=topk_prefilter_enabled,
            topk_k=topk_k,
            topk_keep_eet=int(getattr(self.config, "topk_keep_eet", TOPK_PREFILTER_DEFAULTS["topk_keep_eet"])),
            topk_keep_pt=int(getattr(self.config, "topk_keep_pt", TOPK_PREFILTER_DEFAULTS["topk_keep_pt"])),
            topk_keep_preferred_rule=int(getattr(self.config, "topk_keep_preferred_rule", TOPK_PREFILTER_DEFAULTS["topk_keep_preferred_rule"])),
            topk_preferred_rule=str(getattr(self.config, "topk_preferred_rule", TOPK_PREFILTER_DEFAULTS["topk_preferred_rule"])),
            topk_machine_diversity=bool(getattr(self.config, "topk_machine_diversity", TOPK_PREFILTER_DEFAULTS["topk_machine_diversity"])),
            topk_score_weights=dict(getattr(self.config, "topk_score_weights", TOPK_PREFILTER_DEFAULTS["topk_score_weights"])),
        )
        packed_list.append(packed)

    global_np = np.stack([p["global_feat"] for p in packed_list], axis=0).astype(np.float32)
    pair_np = np.stack([p["pair_feat"] for p in packed_list], axis=0).astype(np.float32)
    mask_np = np.stack([p["pair_mask"] for p in packed_list], axis=0).astype(bool)

    global_tensor = torch.tensor(global_np, dtype=torch.float32, device=self.device)
    pair_tensor = torch.tensor(pair_np, dtype=torch.float32, device=self.device)
    mask_tensor = torch.tensor(mask_np, dtype=torch.bool, device=self.device)
    graph_tensors = {}
    if bool(getattr(self.config, "use_graph_encoder", False)):
        op_nodes = [np.asarray(p["op_node_feat"], dtype=np.float32) for p in packed_list]
        op_adjs = [np.asarray(p["op_adj"], dtype=np.float32) for p in packed_list]
        mch_nodes = [np.asarray(p["machine_node_feat"], dtype=np.float32) for p in packed_list]
        edge_rule = [np.asarray(p["edge_rule_to_pair_feat"], dtype=np.float32) for p in packed_list]
        edge_opmch = [np.asarray(p["edge_opmch_to_pair_feat"], dtype=np.float32) for p in packed_list]

        bsz = len(packed_list)
        max_ops = max(arr.shape[0] for arr in op_nodes)
        max_mch = max(arr.shape[0] for arr in mch_nodes)
        op_dim = op_nodes[0].shape[1]
        mch_dim = mch_nodes[0].shape[1]
        edge_rule_dim = edge_rule[0].shape[1] if edge_rule[0].ndim == 2 else 0
        edge_opmch_dim = edge_opmch[0].shape[1] if edge_opmch[0].ndim == 2 else 4

        op_node_np = np.zeros((bsz, max_ops, op_dim), dtype=np.float32)
        op_adj_np = np.zeros((bsz, max_ops, max_ops), dtype=np.float32)
        mch_node_np = np.zeros((bsz, max_mch, mch_dim), dtype=np.float32)
        edge_rule_np = np.zeros((bsz, int(mask_np.shape[1]), edge_rule_dim), dtype=np.float32)
        edge_opmch_np = np.zeros((bsz, int(mask_np.shape[1]), edge_opmch_dim), dtype=np.float32)

        pair_op_idx_np = np.stack([np.asarray(p["pair_op_idx"], dtype=np.int64) for p in packed_list], axis=0)
        pair_mch_idx_np = np.stack([np.asarray(p["pair_mch_idx"], dtype=np.int64) for p in packed_list], axis=0)

        for i in range(bsz):
            t_i = op_nodes[i].shape[0]
            m_i = mch_nodes[i].shape[0]
            op_node_np[i, :t_i, :] = op_nodes[i]
            op_adj_np[i, :t_i, :t_i] = op_adjs[i]
            mch_node_np[i, :m_i, :] = mch_nodes[i]

            valid_mask_i = np.asarray(packed_list[i]["pair_mask"], dtype=bool)
            pair_op_idx_np[i, ~valid_mask_i] = 0
            pair_mch_idx_np[i, ~valid_mask_i] = 0
            if edge_rule_dim > 0:
                edge_rule_i = edge_rule[i]
                rows = min(edge_rule_i.shape[0], edge_rule_np.shape[1])
                cols = min(edge_rule_i.shape[1], edge_rule_dim) if edge_rule_i.ndim == 2 else 0
                if rows > 0 and cols > 0:
                    edge_rule_np[i, :rows, :cols] = edge_rule_i[:rows, :cols]
                edge_rule_np[i, ~valid_mask_i, :] = 0.0
            if edge_opmch_dim > 0:
                edge_opmch_i = edge_opmch[i]
                rows = min(edge_opmch_i.shape[0], edge_opmch_np.shape[1])
                cols = min(edge_opmch_i.shape[1], edge_opmch_dim) if edge_opmch_i.ndim == 2 else 0
                if rows > 0 and cols > 0:
                    edge_opmch_np[i, :rows, :cols] = edge_opmch_i[:rows, :cols]
                edge_opmch_np[i, ~valid_mask_i, :] = 0.0
            if t_i > 0:
                pair_op_idx_np[i, valid_mask_i] = np.clip(pair_op_idx_np[i, valid_mask_i], 0, t_i - 1)
            else:
                pair_op_idx_np[i, valid_mask_i] = 0
            if m_i > 0:
                pair_mch_idx_np[i, valid_mask_i] = np.clip(pair_mch_idx_np[i, valid_mask_i], 0, m_i - 1)
            else:
                pair_mch_idx_np[i, valid_mask_i] = 0

        graph_tensors = {
            "op_node_feat": torch.tensor(op_node_np, dtype=torch.float32, device=self.device),
            "op_adj": torch.tensor(op_adj_np, dtype=torch.float32, device=self.device),
            "machine_node_feat": torch.tensor(mch_node_np, dtype=torch.float32, device=self.device),
            "pair_op_idx": torch.tensor(pair_op_idx_np, dtype=torch.long, device=self.device),
            "pair_mch_idx": torch.tensor(pair_mch_idx_np, dtype=torch.long, device=self.device),
            "edge_rule_to_pair_feat": torch.tensor(edge_rule_np, dtype=torch.float32, device=self.device),
            "edge_opmch_to_pair_feat": torch.tensor(edge_opmch_np, dtype=torch.float32, device=self.device),
        }
    return packed_list, global_tensor, pair_tensor, mask_tensor, graph_tensors


def _apply_topk_settings_to_agent_config_from_train_cfg(cfg):
    for k, default_v in TOPK_PREFILTER_DEFAULTS.items():
        setattr(AgentCConfig, k, getattr(cfg, k, default_v))


# enable optimized / top-k capable methods
Agent_C._obs_to_tensors = _agent_obs_to_tensors_with_topk
Agent_C._agent_obs_list_to_tensors = _agent_obs_list_to_tensors_with_topk


__all__ = [
    "AgentCConfig",
    "Agent_C",
    "RolloutBuffer",
    "StepRecord",
    "TOPK_PREFILTER_DEFAULTS",
    "_apply_topk_settings_to_agent_config_from_train_cfg",
    "_flatten_f32",
    "preprocess_obs",
    "preprocess_obs_topk",
    "pad_global_feat",
    "pad_pair_feat",
    "infer_obs_dims_from_env",
]

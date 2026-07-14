from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def _flatten_f32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(-1)


@dataclass
class AgentOConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    hidden_dim: int = 128
    dropout: float = 0.05

    lr: float = 2e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    ent_coef: float = 0.005
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 256
    target_kl: Optional[float] = 0.02

    max_ops: int = 32
    use_candidate_set_feat: bool = True
    overflow_strategy: str = "truncate"  # truncate | raise

    # model_type: mlp | hybrid_transformer | moe | hybrid_transformer_moe
    model_type: str = "mlp"
    transformer_layers: int = 1
    transformer_heads: int = 4
    transformer_dropout: float = 0.1
    moe_num_experts: int = 4
    moe_topk: int = 1
    moe_temperature: float = 1.0


@dataclass
class StepRecordO:
    global_feat: np.ndarray
    op_feat: np.ndarray
    op_mask: np.ndarray
    action_idx: int
    log_prob: float
    value: float
    reward: float
    done: bool


class RolloutBufferO:
    def __init__(self) -> None:
        self.records: List[Dict[str, np.ndarray]] = []

    def clear(self) -> None:
        self.records.clear()

    def __len__(self) -> int:
        return len(self.records)

    def add_episode(self, episode_steps: List[StepRecordO], gamma: float, gae_lambda: float) -> None:
        if len(episode_steps) == 0:
            return

        rewards = np.asarray([s.reward for s in episode_steps], dtype=np.float32)
        values = np.asarray([s.value for s in episode_steps], dtype=np.float32)
        dones = np.asarray([s.done for s in episode_steps], dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_adv = 0.0
        next_value = 0.0 if bool(dones[-1]) else float(values[-1])

        for t in reversed(range(len(episode_steps))):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * nonterminal - values[t]
            last_adv = delta + gamma * gae_lambda * nonterminal * last_adv
            advantages[t] = last_adv
            next_value = values[t]

        returns = advantages + values

        for step, adv, ret in zip(episode_steps, advantages, returns):
            self.records.append(
                {
                    "global_feat": np.asarray(step.global_feat, dtype=np.float32),
                    "op_feat": np.asarray(step.op_feat, dtype=np.float32),
                    "op_mask": np.asarray(step.op_mask, dtype=bool),
                    "action_idx": np.int64(step.action_idx),
                    "old_log_prob": np.float32(step.log_prob),
                    "old_value": np.float32(step.value),
                    "advantage": np.float32(adv),
                    "return": np.float32(ret),
                }
            )

    def as_tensors(self, device: torch.device) -> Dict[str, torch.Tensor]:
        if len(self.records) == 0:
            raise RuntimeError("RolloutBufferO is empty.")

        out = {
            "global_feat": torch.tensor(np.stack([r["global_feat"] for r in self.records]), dtype=torch.float32, device=device),
            "op_feat": torch.tensor(np.stack([r["op_feat"] for r in self.records]), dtype=torch.float32, device=device),
            "op_mask": torch.tensor(np.stack([r["op_mask"] for r in self.records]), dtype=torch.bool, device=device),
            "action_idx": torch.tensor(np.asarray([r["action_idx"] for r in self.records]), dtype=torch.long, device=device),
            "old_log_prob": torch.tensor(np.asarray([r["old_log_prob"] for r in self.records]), dtype=torch.float32, device=device),
            "old_value": torch.tensor(np.asarray([r["old_value"] for r in self.records]), dtype=torch.float32, device=device),
            "advantage": torch.tensor(np.asarray([r["advantage"] for r in self.records]), dtype=torch.float32, device=device),
            "return": torch.tensor(np.asarray([r["return"] for r in self.records]), dtype=torch.float32, device=device),
        }
        return out


def _pad_global_feat(global_feat: np.ndarray, target_dim: int) -> np.ndarray:
    global_feat = _flatten_f32(global_feat)
    if global_feat.shape[0] > target_dim:
        raise ValueError(f"global_feat dim={global_feat.shape[0]} exceeds target_dim={target_dim}")
    out = np.zeros((target_dim,), dtype=np.float32)
    out[: global_feat.shape[0]] = global_feat
    return out


def _pad_op_feat(op_feat: np.ndarray, target_ops: int, op_feat_dim: int, overflow_strategy: str) -> Tuple[np.ndarray, np.ndarray, int]:
    op_feat = np.asarray(op_feat, dtype=np.float32)
    if op_feat.ndim != 2:
        raise ValueError(f"op_feat must be rank-2, got shape={op_feat.shape}")
    if op_feat.shape[1] != op_feat_dim:
        raise ValueError(f"op_feat dim mismatch: got {op_feat.shape[1]}, expected {op_feat_dim}")

    original_count = int(op_feat.shape[0])
    overflow_count = max(0, original_count - int(target_ops))
    if overflow_count > 0:
        if overflow_strategy == "raise":
            raise ValueError(f"op count={original_count} exceeds max_ops={target_ops}")
        if overflow_strategy != "truncate":
            raise ValueError(f"Unsupported overflow_strategy={overflow_strategy}")
        op_feat = op_feat[:target_ops]

    used_count = int(op_feat.shape[0])
    feat_out = np.zeros((target_ops, op_feat_dim), dtype=np.float32)
    mask_out = np.zeros((target_ops,), dtype=bool)
    if used_count > 0:
        feat_out[:used_count] = op_feat
        mask_out[:used_count] = True
    return feat_out, mask_out, overflow_count


def _build_legal_ops(candidate_pairs: List[Tuple[int, int]]) -> List[int]:
    legal_ops: List[int] = []
    seen = set()
    for op_id, _ in candidate_pairs:
        op_id = int(op_id)
        if op_id not in seen:
            seen.add(op_id)
            legal_ops.append(op_id)
    return legal_ops


def _build_pair_index_by_op(candidate_pairs: List[Tuple[int, int]], legal_ops: List[int]) -> Dict[int, List[int]]:
    op_to_idx: Dict[int, List[int]] = {int(op): [] for op in legal_ops}
    for i, (op_id, _) in enumerate(candidate_pairs):
        op_to_idx[int(op_id)].append(int(i))
    return op_to_idx


def _safe_stat_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size > 0 else 0.0


def _safe_stat_min(x: np.ndarray) -> float:
    return float(np.min(x)) if x.size > 0 else 0.0


def _build_op_aug_features(obs: Dict[str, object], legal_ops: List[int]) -> np.ndarray:
    candidate_pairs = list(obs.get("candidate_pairs", []))
    pair_feat = np.asarray(obs.get("pair_feat", []), dtype=np.float32)

    if len(candidate_pairs) == 0:
        return np.zeros((0, 10), dtype=np.float32)

    if pair_feat.ndim != 2 or pair_feat.shape[0] != len(candidate_pairs):
        return np.zeros((len(legal_ops), 10), dtype=np.float32)

    op_to_pair_idx = _build_pair_index_by_op(candidate_pairs, legal_ops)

    ect_col = 4 if pair_feat.shape[1] > 4 else None
    pt_col = 0 if pair_feat.shape[1] > 0 else None
    queue_col = 19 if pair_feat.shape[1] > 19 else None

    all_best_ect = []
    for op in legal_ops:
        idx = np.asarray(op_to_pair_idx.get(int(op), []), dtype=np.int64)
        if idx.size == 0 or ect_col is None:
            all_best_ect.append(0.0)
        else:
            all_best_ect.append(_safe_stat_min(pair_feat[idx, ect_col]))
    min_best_ect = min(all_best_ect) if len(all_best_ect) > 0 else 1.0

    feats = []
    for op in legal_ops:
        idx = np.asarray(op_to_pair_idx.get(int(op), []), dtype=np.int64)
        if idx.size == 0:
            feats.append([0.0] * 10)
            continue

        row_pf = pair_feat[idx]
        best_ect = _safe_stat_min(row_pf[:, ect_col]) if ect_col is not None else 0.0
        mean_ect = _safe_stat_mean(row_pf[:, ect_col]) if ect_col is not None else 0.0
        best_pt = _safe_stat_min(row_pf[:, pt_col]) if pt_col is not None else 0.0
        mean_pt = _safe_stat_mean(row_pf[:, pt_col]) if pt_col is not None else 0.0
        mean_queue = _safe_stat_mean(row_pf[:, queue_col]) if queue_col is not None else 0.0

        machines = {int(candidate_pairs[j][1]) for j in idx.tolist()}
        num_pairs = float(idx.size)
        num_machines = float(len(machines))

        ect_rel = best_ect / max(min_best_ect, 1e-6)
        compact_hint = 1.0 / max(num_pairs, 1.0)
        mch_div = num_machines / max(num_pairs, 1.0)

        feats.append(
            [
                best_ect,
                mean_ect,
                best_pt,
                mean_pt,
                mean_queue,
                num_pairs,
                num_machines,
                ect_rel,
                compact_hint,
                mch_div,
            ]
        )

    return np.asarray(feats, dtype=np.float32)


def preprocess_obs_o(
    obs: Dict[str, object],
    global_dim: int,
    op_feat_dim: int,
    max_ops: int,
    use_candidate_set_feat: bool = True,
    overflow_strategy: str = "truncate",
) -> Dict[str, object]:
    candidate_pairs = list(obs.get("candidate_pairs", []))
    if len(candidate_pairs) == 0:
        raise RuntimeError("No candidate_pairs found in current observation.")

    op_node_feat = np.asarray(obs.get("op_node_feat", None), dtype=np.float32)
    if op_node_feat.ndim != 2 or op_node_feat.shape[1] <= 0:
        raise RuntimeError("obs['op_node_feat'] is missing or invalid for AgentO.")

    global_feat = _flatten_f32(obs["global_feat"])
    if use_candidate_set_feat and "candidate_set_feat" in obs:
        global_feat = np.concatenate([global_feat, _flatten_f32(obs["candidate_set_feat"])], axis=0)
    global_feat = _pad_global_feat(global_feat, global_dim)

    legal_ops = _build_legal_ops(candidate_pairs)
    if len(legal_ops) == 0:
        raise RuntimeError("No legal operations derived from candidate_pairs.")

    op_base = op_node_feat[np.asarray(legal_ops, dtype=np.int64)]
    op_aug = _build_op_aug_features(obs=obs, legal_ops=legal_ops)
    op_feat_raw = np.concatenate([op_base, op_aug], axis=1)

    op_feat, op_mask, overflow_count = _pad_op_feat(
        op_feat=op_feat_raw,
        target_ops=max_ops,
        op_feat_dim=op_feat_dim,
        overflow_strategy=overflow_strategy,
    )

    used_count = int(op_mask.sum())
    legal_ops = legal_ops[:used_count]

    return {
        "global_feat": global_feat,
        "op_feat": op_feat,
        "op_mask": op_mask,
        "legal_ops": legal_ops,
        "original_op_count": int(len(_build_legal_ops(candidate_pairs))),
        "used_op_count": int(used_count),
        "overflow_count": int(overflow_count),
        "was_truncated": bool(overflow_count > 0),
    }


class OperationSelectorActorCritic(nn.Module):
    def __init__(self, global_dim: int, op_feat_dim: int, hidden_dim: int, dropout: float = 0.05):
        super().__init__()
        self.global_enc = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.op_enc = nn.Sequential(
            nn.Linear(op_feat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_feat: torch.Tensor, op_feat: torch.Tensor, op_mask: torch.Tensor):
        global_repr = self.global_enc(global_feat)
        op_repr = self.op_enc(op_feat)

        global_expand = global_repr.unsqueeze(1).expand_as(op_repr)
        interact = op_repr * global_expand
        actor_in = torch.cat([op_repr, global_expand, interact], dim=-1)

        logits = self.actor_head(actor_in).squeeze(-1)
        logits = logits.masked_fill(~op_mask, -1e9)

        valid = op_mask.float().unsqueeze(-1)
        pooled = (op_repr * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        critic_in = torch.cat([global_repr, pooled], dim=-1)
        values = self.critic_head(critic_in).squeeze(-1)
        return logits, values

    def act(self, global_feat: torch.Tensor, op_feat: torch.Tensor, op_mask: torch.Tensor, deterministic: bool = False):
        logits, values = self.forward(global_feat, op_feat, op_mask)
        dist = torch.distributions.Categorical(logits=logits)
        actions = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return actions, log_prob, entropy, values, logits

    def evaluate_actions(self, global_feat: torch.Tensor, op_feat: torch.Tensor, op_mask: torch.Tensor, actions: torch.Tensor):
        logits, values = self.forward(global_feat, op_feat, op_mask)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, values


class FlexibleOperationSelectorActorCritic(nn.Module):
    def __init__(
        self,
        global_dim: int,
        op_feat_dim: int,
        hidden_dim: int,
        dropout: float = 0.05,
        use_transformer: bool = False,
        transformer_layers: int = 1,
        transformer_heads: int = 4,
        transformer_dropout: float = 0.1,
        use_moe: bool = False,
        moe_num_experts: int = 4,
        moe_topk: int = 1,
        moe_temperature: float = 1.0,
    ):
        super().__init__()
        self.use_transformer = bool(use_transformer)
        self.use_moe = bool(use_moe)
        self.moe_topk = int(max(1, moe_topk))
        self.moe_temperature = float(max(1e-4, moe_temperature))

        self.global_enc = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.op_enc = nn.Sequential(
            nn.Linear(op_feat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        if self.use_transformer:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=int(max(1, transformer_heads)),
                dim_feedforward=hidden_dim * 4,
                dropout=float(transformer_dropout),
                batch_first=True,
                activation="gelu",
            )
            self.op_context = nn.TransformerEncoder(enc_layer, num_layers=int(max(1, transformer_layers)))
        else:
            self.op_context = None

        if self.use_moe:
            expert_count = int(max(2, moe_num_experts))
            self.expert_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_dim * 3, hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, 1),
                    )
                    for _ in range(expert_count)
                ]
            )
            self.gate_head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, expert_count),
            )
            self.actor_head = None
        else:
            self.expert_heads = None
            self.gate_head = None
            self.actor_head = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _masked_pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask.float().unsqueeze(-1)
        return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    def _moe_gate_probs(self, gate_logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(gate_logits / self.moe_temperature, dim=-1)
        if self.moe_topk >= probs.shape[-1]:
            return probs
        topv, topi = torch.topk(probs, k=int(self.moe_topk), dim=-1)
        sparse = torch.zeros_like(probs)
        sparse.scatter_(dim=-1, index=topi, src=topv)
        denom = sparse.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return sparse / denom

    def forward(self, global_feat: torch.Tensor, op_feat: torch.Tensor, op_mask: torch.Tensor):
        global_repr = self.global_enc(global_feat)
        op_repr = self.op_enc(op_feat)

        if self.op_context is not None:
            key_padding_mask = ~op_mask
            op_repr = self.op_context(op_repr, src_key_padding_mask=key_padding_mask)

        global_expand = global_repr.unsqueeze(1).expand_as(op_repr)
        interact = op_repr * global_expand
        actor_in = torch.cat([op_repr, global_expand, interact], dim=-1)

        if self.use_moe and self.expert_heads is not None and self.gate_head is not None:
            pooled = self._masked_pool(op_repr, op_mask)
            gate_in = torch.cat([global_repr, pooled], dim=-1)
            gate_probs = self._moe_gate_probs(self.gate_head(gate_in))
            expert_logits = torch.stack([head(actor_in).squeeze(-1) for head in self.expert_heads], dim=-1)
            logits = (expert_logits * gate_probs.unsqueeze(1)).sum(dim=-1)
        else:
            logits = self.actor_head(actor_in).squeeze(-1)

        logits = logits.masked_fill(~op_mask, -1e9)

        pooled = self._masked_pool(op_repr, op_mask)
        critic_in = torch.cat([global_repr, pooled], dim=-1)
        values = self.critic_head(critic_in).squeeze(-1)
        return logits, values

    def act(self, global_feat: torch.Tensor, op_feat: torch.Tensor, op_mask: torch.Tensor, deterministic: bool = False):
        logits, values = self.forward(global_feat, op_feat, op_mask)
        dist = torch.distributions.Categorical(logits=logits)
        actions = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return actions, log_prob, entropy, values, logits

    def evaluate_actions(self, global_feat: torch.Tensor, op_feat: torch.Tensor, op_mask: torch.Tensor, actions: torch.Tensor):
        logits, values = self.forward(global_feat, op_feat, op_mask)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, values


class Agent_O:
    def __init__(self, config: AgentOConfig, global_dim: int, op_feat_dim: int):
        self.config = config
        self.global_dim = int(global_dim)
        self.op_feat_dim = int(op_feat_dim)
        self.device = torch.device(config.device)

        model_type = str(getattr(config, "model_type", "mlp")).strip().lower()
        if model_type == "mlp":
            self.policy = OperationSelectorActorCritic(
                global_dim=self.global_dim,
                op_feat_dim=self.op_feat_dim,
                hidden_dim=int(config.hidden_dim),
                dropout=float(config.dropout),
            ).to(self.device)
        elif model_type in {"hybrid_transformer", "moe", "hybrid_transformer_moe"}:
            self.policy = FlexibleOperationSelectorActorCritic(
                global_dim=self.global_dim,
                op_feat_dim=self.op_feat_dim,
                hidden_dim=int(config.hidden_dim),
                dropout=float(config.dropout),
                use_transformer=(model_type in {"hybrid_transformer", "hybrid_transformer_moe"}),
                transformer_layers=int(getattr(config, "transformer_layers", 1)),
                transformer_heads=int(getattr(config, "transformer_heads", 4)),
                transformer_dropout=float(getattr(config, "transformer_dropout", 0.1)),
                use_moe=(model_type in {"moe", "hybrid_transformer_moe"}),
                moe_num_experts=int(getattr(config, "moe_num_experts", 4)),
                moe_topk=int(getattr(config, "moe_topk", 1)),
                moe_temperature=float(getattr(config, "moe_temperature", 1.0)),
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported AgentO model_type={model_type}")

        self.optimizer = optim.AdamW(self.policy.parameters(), lr=float(config.lr), weight_decay=1e-4)

    def _obs_to_tensors(self, obs: Dict[str, object]):
        packed = preprocess_obs_o(
            obs=obs,
            global_dim=self.global_dim,
            op_feat_dim=self.op_feat_dim,
            max_ops=int(self.config.max_ops),
            use_candidate_set_feat=bool(self.config.use_candidate_set_feat),
            overflow_strategy=str(self.config.overflow_strategy),
        )
        global_tensor = torch.tensor(packed["global_feat"], dtype=torch.float32, device=self.device).unsqueeze(0)
        op_tensor = torch.tensor(packed["op_feat"], dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_tensor = torch.tensor(packed["op_mask"], dtype=torch.bool, device=self.device).unsqueeze(0)
        return packed, global_tensor, op_tensor, mask_tensor

    @torch.no_grad()
    def select_action(self, obs: Dict[str, object], deterministic: bool = False, topk_k: int = 2) -> Dict[str, object]:
        was_training = self.policy.training
        if deterministic:
            self.policy.eval()

        packed, global_tensor, op_tensor, mask_tensor = self._obs_to_tensors(obs)
        action, log_prob, entropy, value, logits = self.policy.act(
            global_feat=global_tensor,
            op_feat=op_tensor,
            op_mask=mask_tensor,
            deterministic=deterministic,
        )

        legal_ops = list(packed["legal_ops"])
        valid_n = len(legal_ops)
        if valid_n <= 0:
            raise RuntimeError("No valid legal_ops after preprocessing.")

        action_idx = int(action.item())
        action_idx = int(np.clip(action_idx, 0, valid_n - 1))

        logits_np = logits.squeeze(0).detach().cpu().numpy()[:valid_n]
        topk_k = int(max(1, min(int(topk_k), valid_n)))

        sorted_idx = list(np.argsort(-logits_np, kind="mergesort").tolist())
        selected_idx = [action_idx]
        for idx in sorted_idx:
            if len(selected_idx) >= topk_k:
                break
            if idx != action_idx:
                selected_idx.append(int(idx))

        selected_ops = [int(legal_ops[i]) for i in selected_idx]

        if deterministic and was_training:
            self.policy.train()

        return {
            "action_idx": int(action_idx),
            "action_op_id": int(legal_ops[action_idx]),
            "selected_ops": selected_ops,
            "log_prob": float(log_prob.item()),
            "entropy": float(entropy.item()),
            "value": float(value.item()),
            "valid_logits": np.asarray(logits_np, dtype=np.float32),
            "packed_obs": packed,
        }

    def update(self, buffer: RolloutBufferO) -> Dict[str, float]:
        batch = buffer.as_tensors(self.device)
        batch_size = int(batch["global_feat"].shape[0])
        if batch_size == 0:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
                "clip_frac": 0.0,
                "total_loss": 0.0,
            }

        advantages = batch["advantage"]
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp(min=1e-6)
        returns = batch["return"]

        minibatch_size = int(min(max(8, self.config.minibatch_size), batch_size))
        logs = {k: [] for k in ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "total_loss"]}

        for _ in range(int(self.config.ppo_epochs)):
            perm = torch.randperm(batch_size, device=self.device)
            epoch_kls = []

            for start in range(0, batch_size, minibatch_size):
                idx = perm[start : start + minibatch_size]

                mb_global = batch["global_feat"][idx]
                mb_op = batch["op_feat"][idx]
                mb_mask = batch["op_mask"][idx]
                mb_action = batch["action_idx"][idx]
                mb_old_log_prob = batch["old_log_prob"][idx]
                mb_adv = advantages[idx]
                mb_ret = returns[idx]

                new_log_prob, entropy, values = self.policy.evaluate_actions(
                    global_feat=mb_global,
                    op_feat=mb_op,
                    op_mask=mb_mask,
                    actions=mb_action,
                )

                ratio = torch.exp(new_log_prob - mb_old_log_prob)
                ratio_clipped = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio)
                policy_loss = -torch.min(ratio * mb_adv, ratio_clipped * mb_adv).mean()
                value_loss = F.mse_loss(values, mb_ret)
                entropy_mean = entropy.mean()
                total_loss = policy_loss + self.config.value_coef * value_loss - self.config.ent_coef * entropy_mean

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), float(self.config.max_grad_norm))
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
                break

        return {k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in logs.items()}

    def save(self, path: str) -> None:
        torch.save(
            {
                "config": self.config.__dict__,
                "global_dim": self.global_dim,
                "op_feat_dim": self.op_feat_dim,
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


__all__ = [
    "AgentOConfig",
    "Agent_O",
    "RolloutBufferO",
    "StepRecordO",
    "preprocess_obs_o",
    "_flatten_f32",
]

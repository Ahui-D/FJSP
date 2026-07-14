from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


@dataclass
class AgentMLearnConfig:
    device: str = "cpu"
    hidden_dim: int = 128
    dropout: float = 0.05
    lr: float = 2e-4
    weight_decay: float = 1e-4

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 256
    target_kl: Optional[float] = 0.02

    feature_mode: str = "full"  # base | full | machine_heavy
    strategy_mode: str = "ppo"  # ppo | ac_basic | ppo_entropy_anneal

    # Stage-2 style adaptive backup
    enable_entropy_backup: bool = False
    backup_entropy_threshold: float = 0.75
    backup_max_extra_pairs: int = 1


@dataclass
class StepRecordM:
    global_feat: np.ndarray
    pair_feat: np.ndarray
    pair_mask: np.ndarray
    pair_op_idx: np.ndarray
    op_order: List[int]
    action_pair_indices: np.ndarray
    old_log_prob: float
    old_value: float
    reward: float
    done: bool


class RolloutBufferM:
    def __init__(self) -> None:
        self.records: List[Dict[str, object]] = []

    def clear(self) -> None:
        self.records.clear()

    def __len__(self) -> int:
        return len(self.records)

    def add_episode(self, episode_steps: List[StepRecordM], gamma: float, gae_lambda: float) -> None:
        if len(episode_steps) == 0:
            return

        rewards = np.asarray([s.reward for s in episode_steps], dtype=np.float32)
        values = np.asarray([s.old_value for s in episode_steps], dtype=np.float32)
        dones = np.asarray([float(s.done) for s in episode_steps], dtype=np.float32)

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
                    "pair_feat": np.asarray(step.pair_feat, dtype=np.float32),
                    "pair_mask": np.asarray(step.pair_mask, dtype=bool),
                    "pair_op_idx": np.asarray(step.pair_op_idx, dtype=np.int64),
                    "op_order": [int(x) for x in step.op_order],
                    "action_pair_indices": np.asarray(step.action_pair_indices, dtype=np.int64),
                    "old_log_prob": np.float32(step.old_log_prob),
                    "old_value": np.float32(step.old_value),
                    "advantage": np.float32(adv),
                    "return": np.float32(ret),
                }
            )


class MachineSelectorActorCritic(nn.Module):
    def __init__(self, global_dim: int, pair_feat_dim: int, hidden_dim: int, dropout: float) -> None:
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
        self.pair_enc = nn.Sequential(
            nn.Linear(pair_feat_dim, hidden_dim),
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

    def forward(self, global_feat: torch.Tensor, pair_feat: torch.Tensor, pair_mask: torch.Tensor):
        # global_feat: [G], pair_feat: [P, D], pair_mask: [P]
        g = self.global_enc(global_feat.unsqueeze(0)).squeeze(0)
        p = self.pair_enc(pair_feat)

        g_expand = g.unsqueeze(0).expand_as(p)
        interact = p * g_expand
        actor_in = torch.cat([p, g_expand, interact], dim=-1)
        pair_logits = self.actor_head(actor_in).squeeze(-1)
        pair_logits = pair_logits.masked_fill(~pair_mask, -1e9)

        valid = pair_mask.float().unsqueeze(-1)
        pooled = (p * valid).sum(dim=0) / valid.sum(dim=0).clamp(min=1.0)
        critic_in = torch.cat([g, pooled], dim=-1)
        value = self.critic_head(critic_in).squeeze(-1)
        return pair_logits, value


class Agent_M_Learn:
    def __init__(self, config: AgentMLearnConfig, global_dim: int, pair_feat_dim: int):
        self.config = config
        self.device = torch.device(config.device)
        self.global_dim = int(global_dim)
        self.raw_pair_feat_dim = int(pair_feat_dim)
        self.policy_pair_feat_dim = int(self._feature_dim_from_mode(config.feature_mode))

        self.policy = MachineSelectorActorCritic(
            global_dim=self.global_dim,
            pair_feat_dim=self.policy_pair_feat_dim,
            hidden_dim=int(config.hidden_dim),
            dropout=float(config.dropout),
        ).to(self.device)
        self.optimizer = optim.AdamW(
            self.policy.parameters(),
            lr=float(config.lr),
            weight_decay=float(config.weight_decay),
        )

    @staticmethod
    def _feature_cols(mode: str) -> Optional[List[int]]:
        mode = str(mode).strip().lower()
        if mode == "base":
            return [0, 4, 8, 13, 14, 19, 20, 21, 22, 23]
        if mode in {
            "machine_heavy",
            "machine_heavy_rel_ect",
            "machine_heavy_balance",
            "machine_heavy_uncertainty",
            "machine_heavy_struct",
            "machine_heavy_rel_ect_uncertainty",
        }:
            return [0, 1, 3, 4, 8, 9, 10, 14, 15, 16, 18, 19, 20, 21, 22, 23]
        return None

    @staticmethod
    def _extra_dim_from_mode(mode: str) -> int:
        mode = str(mode).strip().lower()
        if mode in {"machine_heavy_rel_ect", "machine_heavy_balance", "machine_heavy_uncertainty"}:
            return 1
        if mode in {"machine_heavy_struct", "machine_heavy_rel_ect_uncertainty"}:
            return 2
        return 0

    def _feature_dim_from_mode(self, mode: str) -> int:
        cols = self._feature_cols(mode)
        if cols is None:
            return int(self.raw_pair_feat_dim + self._extra_dim_from_mode(mode))
        return int(len(cols) + self._extra_dim_from_mode(mode))

    def _project_pair_feat(self, pair_feat: np.ndarray, pair_op_idx: Optional[np.ndarray] = None) -> np.ndarray:
        cols = self._feature_cols(self.config.feature_mode)
        if cols is None:
            out = np.asarray(pair_feat, dtype=np.float32)
        else:
            valid_cols = [c for c in cols if 0 <= int(c) < pair_feat.shape[1]]
            if len(valid_cols) == 0:
                out = np.asarray(pair_feat, dtype=np.float32)
            else:
                out = np.asarray(pair_feat[:, valid_cols], dtype=np.float32)

        mode = str(self.config.feature_mode).strip().lower()
        extras: List[np.ndarray] = []

        # Relative ECT inside each operation group helps compare machine choices for the same op.
        if mode in {"machine_heavy_rel_ect", "machine_heavy_rel_ect_uncertainty"} and pair_feat.shape[1] > 4 and pair_op_idx is not None:
            ect = np.asarray(pair_feat[:, 4], dtype=np.float32)
            rel = np.zeros_like(ect, dtype=np.float32)
            for op in np.unique(pair_op_idx):
                m = pair_op_idx == op
                if np.any(m):
                    rel[m] = ect[m] - np.min(ect[m])
            extras.append(rel.reshape(-1, 1))

        # A lightweight machine-load z-score feature stabilizes load balancing preferences.
        if mode == "machine_heavy_balance" and pair_feat.shape[1] > 10:
            load = np.asarray(pair_feat[:, 10], dtype=np.float32)
            z = (load - float(np.mean(load))) / max(float(np.std(load)), 1e-6)
            extras.append(z.reshape(-1, 1))

        # Ambiguity gap (2nd-best vs best ECT) per op captures uncertainty in machine selection.
        if mode in {"machine_heavy_uncertainty", "machine_heavy_rel_ect_uncertainty"} and pair_feat.shape[1] > 4 and pair_op_idx is not None:
            ect = np.asarray(pair_feat[:, 4], dtype=np.float32)
            amb = np.zeros_like(ect, dtype=np.float32)
            for op in np.unique(pair_op_idx):
                m = pair_op_idx == op
                vals = np.sort(ect[m])
                gap = float(vals[1] - vals[0]) if vals.size >= 2 else 0.0
                amb[m] = gap
            extras.append(amb.reshape(-1, 1))

        # Structural density features: candidates per operation and per machine.
        if mode == "machine_heavy_struct" and pair_op_idx is not None:
            op_cnt = np.zeros((pair_feat.shape[0],), dtype=np.float32)
            for op in np.unique(pair_op_idx):
                m = pair_op_idx == op
                op_cnt[m] = float(np.sum(m))
            op_cnt = op_cnt / max(float(np.max(op_cnt)), 1.0)

            if pair_feat.shape[1] > 1:
                mch_idx = np.asarray(pair_feat[:, 1], dtype=np.float32)
                mch_cnt = np.zeros((pair_feat.shape[0],), dtype=np.float32)
                for mch in np.unique(mch_idx):
                    m = mch_idx == mch
                    mch_cnt[m] = float(np.sum(m))
                mch_cnt = mch_cnt / max(float(np.max(mch_cnt)), 1.0)
            else:
                mch_cnt = np.zeros((pair_feat.shape[0],), dtype=np.float32)
            extras.extend([op_cnt.reshape(-1, 1), mch_cnt.reshape(-1, 1)])

        if len(extras) > 0:
            out = np.concatenate([out] + extras, axis=1)

        # Per-step robust scaling for stability.
        if out.size > 0:
            mu = out.mean(axis=0, keepdims=True)
            sd = out.std(axis=0, keepdims=True)
            out = (out - mu) / np.clip(sd, 1e-6, None)
        return out.astype(np.float32)

    def _resolve_selected_pair_indices(
        self,
        pairs: List[tuple],
        pair_feat: np.ndarray,
        selected_ops: List[int],
        pair_indices: Optional[List[int]] = None,
    ) -> List[int]:
        n_pairs = int(len(pairs))
        if n_pairs <= 0:
            return []

        if pair_indices is not None:
            idx_keep: List[int] = []
            for x in pair_indices:
                i = int(x)
                if 0 <= i < n_pairs:
                    idx_keep.append(i)
            if len(idx_keep) > 0:
                return sorted(set(idx_keep))

        selected_set = {int(x) for x in selected_ops}
        idx_keep = [i for i, (op, _m) in enumerate(pairs) if int(op) in selected_set]
        if len(idx_keep) == 0:
            idx_keep = list(range(n_pairs))

        if len(idx_keep) == 0:
            if pair_feat.ndim == 2 and pair_feat.shape[0] > 0 and pair_feat.shape[1] > 4:
                return [int(np.argmin(pair_feat[:, 4]))]
            return [0]

        return idx_keep

    def build_packed_obs(
        self,
        obs: Dict[str, object],
        selected_ops: List[int],
        pair_indices: Optional[List[int]] = None,
    ) -> Dict[str, object]:
        pairs = list(obs.get("candidate_pairs", []))
        pair_feat = np.asarray(obs.get("pair_feat", []), dtype=np.float32)
        global_feat = np.asarray(obs.get("global_feat", []), dtype=np.float32).reshape(-1)

        if len(global_feat) == 0:
            global_feat = np.zeros((self.global_dim,), dtype=np.float32)
        elif global_feat.shape[0] < self.global_dim:
            pad = np.zeros((self.global_dim,), dtype=np.float32)
            pad[: global_feat.shape[0]] = global_feat
            global_feat = pad
        else:
            global_feat = global_feat[: self.global_dim]

        idx_keep = self._resolve_selected_pair_indices(
            pairs=pairs,
            pair_feat=pair_feat,
            selected_ops=selected_ops,
            pair_indices=pair_indices,
        )

        if len(idx_keep) == 0:
            # No valid pair, caller should handle fallback action.
            return {
                "global_feat": global_feat,
                "pair_feat": np.zeros((0, self.policy_pair_feat_dim), dtype=np.float32),
                "pair_mask": np.zeros((0,), dtype=bool),
                "pair_op_idx": np.zeros((0,), dtype=np.int64),
                "candidate_pairs": [],
                "full_pair_indices": np.zeros((0,), dtype=np.int64),
                "op_order": [],
            }

        m_pairs = [pairs[i] for i in idx_keep]
        m_feat_raw = pair_feat[idx_keep]
        m_feat = self._project_pair_feat(m_feat_raw, pair_op_idx=np.asarray([int(op) for op, _ in m_pairs], dtype=np.int64))
        m_op = np.asarray([int(op) for op, _ in m_pairs], dtype=np.int64)

        unique_ops = sorted({int(x) for x in m_op.tolist()})
        # 只对“至少有两个机器可选”的操作进行策略决策，避免单可选项把熵虚假拉低。
        op_choice_counts = {op_i: int(np.sum(m_op == op_i)) for op_i in unique_ops}
        eligible_ops = {op_i for op_i, cnt in op_choice_counts.items() if int(cnt) >= 2}

        op_order: List[int] = []
        seen = set()
        for op in selected_ops:
            op_i = int(op)
            if op_i in seen:
                continue
            if op_i in eligible_ops:
                op_order.append(op_i)
                seen.add(op_i)

        if len(op_order) == 0:
            # 若当前没有多机可选操作，回退到原行为，保证动作总是可定义。
            op_order = sorted(unique_ops)

        return {
            "global_feat": global_feat.astype(np.float32),
            "pair_feat": m_feat.astype(np.float32),
            "pair_mask": np.ones((m_feat.shape[0],), dtype=bool),
            "pair_op_idx": m_op,
            "candidate_pairs": m_pairs,
            "full_pair_indices": np.asarray(idx_keep, dtype=np.int64),
            "op_order": op_order,
        }

    def _act_from_logits(
        self,
        pair_logits: torch.Tensor,
        pair_op_idx: torch.Tensor,
        pair_mask: torch.Tensor,
        op_order: List[int],
        deterministic: bool,
    ) -> Dict[str, object]:
        chosen_abs: List[int] = []
        backup_abs: List[int] = []
        log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []

        for op in op_order:
            op_val = int(op)
            idx = torch.where((pair_op_idx == op_val) & pair_mask)[0]
            if idx.numel() == 0:
                continue

            logits = pair_logits[idx]
            dist = torch.distributions.Categorical(logits=logits)
            if deterministic:
                local_choice = torch.argmax(logits)
            else:
                local_choice = dist.sample()

            abs_idx = int(idx[int(local_choice.item())].item())
            chosen_abs.append(abs_idx)
            log_probs.append(dist.log_prob(local_choice))
            ent = dist.entropy()
            entropies.append(ent)

            if self.config.enable_entropy_backup and idx.numel() >= 2:
                ent_v = float(ent.item())
                if ent_v >= float(self.config.backup_entropy_threshold):
                    topk = torch.topk(logits, k=min(2, logits.numel()))
                    for local_j in topk.indices.tolist():
                        abs_j = int(idx[int(local_j)].item())
                        if abs_j != abs_idx:
                            backup_abs.append(abs_j)
                            if len(backup_abs) >= int(self.config.backup_max_extra_pairs):
                                break

        if len(log_probs) == 0:
            return {
                "chosen_abs": [],
                "backup_abs": [],
                "log_prob": torch.tensor(0.0, device=self.device),
                "entropy": torch.tensor(0.0, device=self.device),
            }

        # 采用按决策数归一化的 log-prob，减少不同步长样本对 PPO 比率的尺度偏置。
        log_prob = torch.stack(log_probs).mean()
        entropy = torch.stack(entropies).mean()
        return {
            "chosen_abs": chosen_abs,
            "backup_abs": backup_abs,
            "log_prob": log_prob,
            "entropy": entropy,
        }

    def act(self, packed_obs: Dict[str, object], deterministic: bool = False) -> Dict[str, object]:
        pair_feat = np.asarray(packed_obs["pair_feat"], dtype=np.float32)
        if pair_feat.shape[0] == 0:
            return {
                "keep_indices": [],
                "action_pair_indices": np.zeros((0,), dtype=np.int64),
                "log_prob": 0.0,
                "entropy": 0.0,
                "value": 0.0,
            }

        g = torch.tensor(packed_obs["global_feat"], dtype=torch.float32, device=self.device)
        p = torch.tensor(pair_feat, dtype=torch.float32, device=self.device)
        m = torch.tensor(packed_obs["pair_mask"], dtype=torch.bool, device=self.device)
        op_idx = torch.tensor(packed_obs["pair_op_idx"], dtype=torch.long, device=self.device)

        with torch.no_grad():
            pair_logits, value = self.policy(g, p, m)
            action = self._act_from_logits(
                pair_logits=pair_logits,
                pair_op_idx=op_idx,
                pair_mask=m,
                op_order=list(packed_obs["op_order"]),
                deterministic=bool(deterministic),
            )

        keep = sorted(set(action["chosen_abs"] + action["backup_abs"]))
        return {
            "keep_indices": keep,
            "action_pair_indices": np.asarray(action["chosen_abs"], dtype=np.int64),
            "log_prob": float(action["log_prob"].item()),
            "entropy": float(action["entropy"].item()),
            "value": float(value.item()),
        }

    def evaluate_action(self, record: Dict[str, object]) -> Dict[str, torch.Tensor]:
        g = torch.tensor(record["global_feat"], dtype=torch.float32, device=self.device)
        p = torch.tensor(record["pair_feat"], dtype=torch.float32, device=self.device)
        m = torch.tensor(record["pair_mask"], dtype=torch.bool, device=self.device)
        op_idx = torch.tensor(record["pair_op_idx"], dtype=torch.long, device=self.device)

        pair_logits, value = self.policy(g, p, m)

        log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        chosen_abs = np.asarray(record["action_pair_indices"], dtype=np.int64)
        op_order = [int(x) for x in record["op_order"]]

        for i, op in enumerate(op_order):
            if i >= len(chosen_abs):
                break
            idx = torch.where((op_idx == int(op)) & m)[0]
            if idx.numel() == 0:
                continue

            logits = pair_logits[idx]
            dist = torch.distributions.Categorical(logits=logits)
            abs_choice = int(chosen_abs[i])

            try:
                local_choice = int((idx == abs_choice).nonzero(as_tuple=False)[0].item())
            except Exception:
                local_choice = int(torch.argmax(logits).item())

            local_t = torch.tensor(local_choice, dtype=torch.long, device=self.device)
            log_probs.append(dist.log_prob(local_t))
            entropies.append(dist.entropy())

        if len(log_probs) == 0:
            log_prob = torch.tensor(0.0, device=self.device)
            entropy = torch.tensor(0.0, device=self.device)
        else:
            log_prob = torch.stack(log_probs).mean()
            entropy = torch.stack(entropies).mean()

        return {
            "log_prob": log_prob,
            "entropy": entropy,
            "value": value,
        }

    def update(self, buffer: RolloutBufferM, epoch_idx: int = 1, total_epochs: int = 1) -> Dict[str, float]:
        if len(buffer) == 0:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
                "clip_frac": 0.0,
                "total_loss": 0.0,
                "effective_ent_coef": float(self.config.ent_coef),
            }

        adv = np.asarray([float(r["advantage"]) for r in buffer.records], dtype=np.float32)
        ret = np.asarray([float(r["return"]) for r in buffer.records], dtype=np.float32)
        old_lp = np.asarray([float(r["old_log_prob"]) for r in buffer.records], dtype=np.float32)

        adv_mu = float(np.mean(adv))
        adv_sd = float(np.std(adv))
        adv = (adv - adv_mu) / max(adv_sd, 1e-6)

        idx_all = np.arange(len(buffer.records), dtype=np.int64)
        mb = int(min(max(8, int(self.config.minibatch_size)), len(idx_all)))

        ent_coef = float(self.config.ent_coef)
        if str(self.config.strategy_mode).lower() == "ac_basic":
            ent_coef = max(0.001, ent_coef * 0.5)
        elif str(self.config.strategy_mode).lower() == "ppo_entropy_anneal":
            prog = float(max(0.0, min(1.0, (epoch_idx - 1) / max(total_epochs - 1, 1))))
            ent_coef = ent_coef * (1.0 - 0.7 * prog)

        logs = {k: [] for k in ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "total_loss"]}

        ppo_epochs = int(max(1, self.config.ppo_epochs))
        if str(self.config.strategy_mode).lower() == "ac_basic":
            ppo_epochs = 1

        for _ in range(ppo_epochs):
            np.random.shuffle(idx_all)
            epoch_kls: List[float] = []

            for s in range(0, len(idx_all), mb):
                batch_idx = idx_all[s : s + mb]
                if len(batch_idx) == 0:
                    continue

                pol_losses = []
                val_losses = []
                ents = []
                kls = []
                cfracs = []

                for bi in batch_idx.tolist():
                    rec = buffer.records[int(bi)]
                    out = self.evaluate_action(rec)

                    new_lp = out["log_prob"]
                    ent = out["entropy"]
                    val = out["value"]

                    old_lp_i = torch.tensor(old_lp[int(bi)], dtype=torch.float32, device=self.device)
                    adv_i = torch.tensor(adv[int(bi)], dtype=torch.float32, device=self.device)
                    ret_i = torch.tensor(ret[int(bi)], dtype=torch.float32, device=self.device)

                    ratio = torch.exp(new_lp - old_lp_i)
                    ratio_clip = torch.clamp(ratio, 1.0 - float(self.config.clip_ratio), 1.0 + float(self.config.clip_ratio))
                    pol = -torch.min(ratio * adv_i, ratio_clip * adv_i)
                    vloss = F.mse_loss(val, ret_i)

                    approx_kl = torch.abs(old_lp_i - new_lp)
                    clip_frac = (torch.abs(ratio - ratio_clip) > 1e-8).float()

                    pol_losses.append(pol)
                    val_losses.append(vloss)
                    ents.append(ent)
                    kls.append(approx_kl)
                    cfracs.append(clip_frac)

                policy_loss = torch.stack(pol_losses).mean()
                value_loss = torch.stack(val_losses).mean()
                entropy_mean = torch.stack(ents).mean()
                approx_kl = torch.stack(kls).mean()
                clip_frac = torch.stack(cfracs).mean()

                total_loss = policy_loss + float(self.config.value_coef) * value_loss - ent_coef * entropy_mean

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), float(self.config.max_grad_norm))
                self.optimizer.step()

                epoch_kls.append(float(approx_kl.item()))
                logs["policy_loss"].append(float(policy_loss.item()))
                logs["value_loss"].append(float(value_loss.item()))
                logs["entropy"].append(float(entropy_mean.item()))
                logs["approx_kl"].append(float(approx_kl.item()))
                logs["clip_frac"].append(float(clip_frac.item()))
                logs["total_loss"].append(float(total_loss.item()))

            if self.config.target_kl is not None and len(epoch_kls) > 0:
                if float(np.mean(epoch_kls)) > float(self.config.target_kl):
                    break

        out = {k: (float(np.mean(v)) if len(v) else 0.0) for k, v in logs.items()}
        out["effective_ent_coef"] = float(ent_coef)
        return out

    def save(self, path: str) -> None:
        torch.save(
            {
                "config": self.config.__dict__,
                "global_dim": int(self.global_dim),
                "raw_pair_feat_dim": int(self.raw_pair_feat_dim),
                "policy_pair_feat_dim": int(self.policy_pair_feat_dim),
                "state_dict": self.policy.state_dict(),
            },
            path,
        )


__all__ = [
    "AgentMLearnConfig",
    "StepRecordM",
    "RolloutBufferM",
    "Agent_M_Learn",
]

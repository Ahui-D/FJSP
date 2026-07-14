from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from Agent_C import AgentCConfig, Agent_C
from Train_C import build_env_for_case, make_case_split, seed_everything


def _load_agent_o_symbols():
    agent_o_path = Path(__file__).resolve().with_name("Agent_O_frozenC.py")
    module_name = "agent_o_frozen_c_runtime_module"
    spec = importlib.util.spec_from_file_location(module_name, str(agent_o_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load Agent_O_frozenC module from {agent_o_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return (
        module.AgentOConfig,
        module.Agent_O,
        module.RolloutBufferO,
        module.StepRecordO,
        module._flatten_f32,
        module.preprocess_obs_o,
    )


AgentOConfig, Agent_O, RolloutBufferO, StepRecordO, _flatten_f32, preprocess_obs_o = _load_agent_o_symbols()


@dataclass
class TrainOConfig:
    train_cases: List[str]
    val_cases: List[str]
    test_cases: List[str]

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    reset_rule: str = "FIFO_SPT"

    epochs: int = 80
    episodes_per_update: int = 8
    eval_interval: int = 1

    save_dir: str = "checkpoints_agent_o_frozen_c"
    checkpoint_name: str = "agent_o_best.pt"
    quiet_env: bool = True
    extra_step_budget: int = 40

    c_checkpoint_path: str = ""
    c_deterministic: bool = True
    disable_c_topk_prefilter: bool = False
    c_max_candidates_floor: int = 0

    op_topk: int = 5
    op_topk_min: int = 3
    op_topk_ratio: float = 0.35
    op_topk_max: int = 10
    min_selected_ops_floor: int = 4
    keep_c_full_top1: bool = True
    keep_anchor_best_ect: bool = True
    keep_anchor_best_pt: bool = True
    keep_anchor_preferred_rule: bool = True
    anchor_preferred_rule: str = "FIFO_SPT"
    op_hidden_dim: int = 128
    op_dropout: float = 0.05
    op_lr: float = 2e-4
    op_gamma: float = 0.99
    op_gae_lambda: float = 0.95
    op_clip_ratio: float = 0.2
    op_value_coef: float = 0.5
    op_ent_coef: float = 0.005
    op_max_grad_norm: float = 0.5
    op_ppo_epochs: int = 4
    op_minibatch_size: int = 256
    op_target_kl: float = 0.02

    op_auto_max_ops: bool = True
    op_max_ops: int = 32
    op_safety_margin: int = 2

    # AgentO structure: mlp | hybrid_transformer | moe | hybrid_transformer_moe
    op_model_type: str = "mlp"
    op_transformer_layers: int = 1
    op_transformer_heads: int = 4
    op_transformer_dropout: float = 0.1
    op_moe_num_experts: int = 4
    op_moe_topk: int = 1
    op_moe_temperature: float = 1.0

    # Reward terms for O-only selection quality
    reward_retain_pos: float = 0.20
    reward_retain_neg: float = -0.45
    reward_quality_coef: float = 0.30
    reward_quality_clip: float = 0.30
    reward_quality_hard_threshold: float = 0.08
    reward_quality_hard_penalty: float = 0.18
    reward_compact_coef: float = 0.0
    reward_consistency_coef: float = 0.04
    reward_mismatch_penalty: float = 0.08
    reward_makespan_increase_coef: float = 0.25
    reward_makespan_decrease_coef: float = 0.05
    reward_makespan_terminal_coef: float = 0.01
    reward_c_value_delta_coef: float = 0.05
    reward_c_entropy_increase_penalty: float = 0.03

    c_entropy_fallback_threshold: float = 1.0
    c_entropy_fallback_extra_ops: int = 2

    # Robust operation scoring and blending
    rule_score_enabled: bool = False
    rule_score_gate_min_ops: int = 4
    rule_score_gate_max_ops: int = 5
    rule_score_gate_min_entropy: float = 1.0
    rule_score_gate_max_entropy: float = 10.0
    rule_blend_addon_only: bool = True
    rule_blend_addon_count: int = 1
    rule_score_tau: float = 2.0
    rule_score_trim_lowest: int = 1
    rule_score_var_penalty: float = 0.15
    rule_policy_blend_alpha: float = 0.6
    topk_c_entropy_gain: float = 0.0
    topk_rule_disagreement_gain: float = 0.0
    small_instance_ops_threshold: int = 5
    small_instance_extra_keep: int = 0
    mk6_protect_enabled: bool = True
    mk6_quality_gap_threshold: float = 0.20
    mk6_extra_keep: int = 1


class FrozenAgentC:
    def __init__(
        self,
        ckpt_path: str,
        device: str = "cpu",
        disable_topk_prefilter: bool = True,
        max_candidates_floor: int = 64,
    ):
        self.ckpt_path = str(ckpt_path)
        self.device = str(device)
        self.disable_topk_prefilter = bool(disable_topk_prefilter)
        self.max_candidates_floor = int(max_candidates_floor)

        payload = torch.load(self.ckpt_path, map_location=self.device)
        cfg_dict = dict(payload.get("config", {}))
        cfg_dict["device"] = self.device

        if self.disable_topk_prefilter:
            cfg_dict["topk_prefilter_enabled"] = False
        if self.max_candidates_floor > 0:
            cfg_dict["max_candidates"] = max(int(cfg_dict.get("max_candidates", 0)), self.max_candidates_floor)

        valid_fields = {f.name for f in dataclasses.fields(AgentCConfig)}
        cfg_filtered = {k: v for k, v in cfg_dict.items() if k in valid_fields}
        cfg = AgentCConfig(**cfg_filtered)

        self.agent_c = Agent_C(
            config=cfg,
            global_dim=int(payload["global_dim"]),
            pair_feat_dim=int(payload["pair_feat_dim"]),
        )
        self.agent_c.load(self.ckpt_path, strict=True)
        self.agent_c.policy.eval()
        for p in self.agent_c.policy.parameters():
            p.requires_grad = False

    def act(self, obs: Dict[str, object], deterministic: bool = True):
        with torch.no_grad():
            return self.agent_c.act(obs, deterministic=deterministic)


def _resolve_pair_sources(obs: Dict[str, object]) -> List[List[str]]:
    pairs = list(obs.get("candidate_pairs", []))
    src = list(obs.get("pair_sources", []))
    if len(src) < len(pairs):
        src = src + [[] for _ in range(len(pairs) - len(src))]
    return src[: len(pairs)]


def _safe_slice_candidate_obs(obs: Dict[str, object], idx_keep: List[int]) -> Dict[str, object]:
    idx_keep = [int(i) for i in idx_keep]
    out = dict(obs)

    pairs = list(obs["candidate_pairs"])
    src = _resolve_pair_sources(obs)
    out["candidate_pairs"] = [pairs[i] for i in idx_keep]
    out["pair_sources"] = [src[i] for i in idx_keep]

    pair_feat = np.asarray(obs["pair_feat"], dtype=np.float32)
    out["pair_feat"] = pair_feat[idx_keep]
    out["pair_node_feat"] = np.asarray(out["pair_feat"], dtype=np.float32)

    if "pair_op_idx" in obs:
        out["pair_op_idx"] = np.asarray(obs["pair_op_idx"], dtype=np.int64)[idx_keep]
    else:
        out["pair_op_idx"] = np.asarray([int(a) for a, _ in out["candidate_pairs"]], dtype=np.int64)

    if "pair_mch_idx" in obs:
        out["pair_mch_idx"] = np.asarray(obs["pair_mch_idx"], dtype=np.int64)[idx_keep]
    else:
        out["pair_mch_idx"] = np.asarray([int(b) for _, b in out["candidate_pairs"]], dtype=np.int64)

    out["edge_op_to_pair"] = np.asarray(out["pair_op_idx"], dtype=np.int64)
    out["edge_mch_to_pair"] = np.asarray(out["pair_mch_idx"], dtype=np.int64)

    if "edge_rule_to_pair_feat" in obs:
        out["edge_rule_to_pair_feat"] = np.asarray(obs["edge_rule_to_pair_feat"], dtype=np.float32)[idx_keep]
    if "edge_opmch_to_pair_feat" in obs:
        out["edge_opmch_to_pair_feat"] = np.asarray(obs["edge_opmch_to_pair_feat"], dtype=np.float32)[idx_keep]

    out["pair_mask"] = np.ones((len(out["candidate_pairs"]),), dtype=bool)
    out["candidate_set_feat"] = np.asarray(
        [
            float(len(out["candidate_pairs"])),
            float(len({int(a) for a, _ in out["candidate_pairs"]})),
            float(len({int(b) for _, b in out["candidate_pairs"]})),
        ],
        dtype=np.float32,
    )
    out["legal_pairs_set"] = {(int(a), int(b)) for a, b in out["candidate_pairs"]}
    return out


def filter_obs_by_selected_ops(obs: Dict[str, object], selected_ops: List[int]) -> Dict[str, object]:
    selected_set = {int(x) for x in selected_ops}
    pairs = list(obs.get("candidate_pairs", []))
    if len(pairs) == 0:
        return obs

    idx_keep = [i for i, (op_id, _) in enumerate(pairs) if int(op_id) in selected_set]

    if len(idx_keep) == 0:
        pair_feat = np.asarray(obs.get("pair_feat", []), dtype=np.float32)
        if pair_feat.ndim == 2 and pair_feat.shape[0] > 0 and pair_feat.shape[1] > 4:
            idx_keep = [int(np.argmin(pair_feat[:, 4]))]
        else:
            idx_keep = [0]

    return _safe_slice_candidate_obs(obs, idx_keep)


def _best_ect(obs: Dict[str, object]) -> float:
    pair_feat = np.asarray(obs.get("pair_feat", []), dtype=np.float32)
    if pair_feat.ndim == 2 and pair_feat.shape[0] > 0 and pair_feat.shape[1] > 4:
        return float(np.min(pair_feat[:, 4]))
    return 0.0


def _reward_terms(
    full_obs: Dict[str, object],
    filtered_obs: Dict[str, object],
    selected_ops: List[int],
    c_full_op: int,
    c_filtered_op: int,
    prev_makespan: float,
    new_makespan: float,
    is_terminal: bool,
    initial_lb_makespan: float,
    c_value_full: float,
    c_value_filtered: float,
    c_entropy_full: float,
    c_entropy_filtered: float,
    cfg: TrainOConfig,
) -> Dict[str, float]:
    selected_set = {int(x) for x in selected_ops}

    retain_bonus = float(cfg.reward_retain_pos) if int(c_full_op) in selected_set else float(cfg.reward_retain_neg)

    ect_full = _best_ect(full_obs)
    ect_filtered = _best_ect(filtered_obs)
    quality_gap = 0.0
    if abs(ect_full) > 1e-8:
        quality_gap = max(0.0, (ect_filtered - ect_full) / abs(ect_full))
    quality_penalty = float(cfg.reward_quality_coef) * float(
        np.clip(quality_gap, 0.0, abs(float(cfg.reward_quality_clip)))
    )
    hard_quality_penalty = float(cfg.reward_quality_hard_penalty) if quality_gap > float(cfg.reward_quality_hard_threshold) else 0.0

    full_pairs = list(full_obs.get("candidate_pairs", []))
    full_unique_ops = len({int(op) for op, _ in full_pairs})
    compact_ratio = 1.0 - float(len(selected_set)) / float(max(full_unique_ops, 1))
    compact_bonus = float(cfg.reward_compact_coef) * float(np.clip(compact_ratio, 0.0, 1.0))
    if quality_gap > float(cfg.reward_quality_hard_threshold):
        compact_bonus *= 0.25

    consistency_bonus = float(cfg.reward_consistency_coef) if int(c_full_op) == int(c_filtered_op) else 0.0
    mismatch_penalty = float(cfg.reward_mismatch_penalty) if int(c_full_op) != int(c_filtered_op) else 0.0

    denom_prev = max(abs(float(prev_makespan)), 1e-6)
    ms_delta_ratio = (float(new_makespan) - float(prev_makespan)) / denom_prev
    makespan_increase_penalty = float(cfg.reward_makespan_increase_coef) * max(0.0, ms_delta_ratio)
    makespan_decrease_bonus = float(cfg.reward_makespan_decrease_coef) * max(0.0, -ms_delta_ratio)

    denom_init = max(abs(float(initial_lb_makespan)), 1e-6)
    terminal_gap_ratio = 0.0
    terminal_penalty = 0.0
    if bool(is_terminal):
        terminal_gap_ratio = max(0.0, (float(new_makespan) - float(initial_lb_makespan)) / denom_init)
        terminal_penalty = float(cfg.reward_makespan_terminal_coef) * terminal_gap_ratio

    c_value_delta = float(c_value_filtered) - float(c_value_full)
    c_value_delta_bonus = float(cfg.reward_c_value_delta_coef) * c_value_delta
    c_entropy_increase = max(0.0, float(c_entropy_filtered) - float(c_entropy_full))
    c_entropy_increase_penalty = float(cfg.reward_c_entropy_increase_penalty) * c_entropy_increase

    total = (
        retain_bonus
        - quality_penalty
        - hard_quality_penalty
        + compact_bonus
        + consistency_bonus
        - mismatch_penalty
        - makespan_increase_penalty
        + makespan_decrease_bonus
        - terminal_penalty
        + c_value_delta_bonus
        - c_entropy_increase_penalty
    )
    return {
        "total": float(total),
        "retain_bonus": float(retain_bonus),
        "quality_penalty": float(quality_penalty),
        "hard_quality_penalty": float(hard_quality_penalty),
        "quality_gap": float(quality_gap),
        "compact_bonus": float(compact_bonus),
        "compact_ratio": float(compact_ratio),
        "consistency_bonus": float(consistency_bonus),
        "mismatch_penalty": float(mismatch_penalty),
        "makespan_delta_ratio": float(ms_delta_ratio),
        "makespan_increase_penalty": float(makespan_increase_penalty),
        "makespan_decrease_bonus": float(makespan_decrease_bonus),
        "terminal_gap_ratio": float(terminal_gap_ratio),
        "terminal_penalty": float(terminal_penalty),
        "c_value_delta": float(c_value_delta),
        "c_value_delta_bonus": float(c_value_delta_bonus),
        "c_entropy_increase": float(c_entropy_increase),
        "c_entropy_increase_penalty": float(c_entropy_increase_penalty),
    }


def _safe_zscore_map(values: Dict[int, float]) -> Dict[int, float]:
    if len(values) == 0:
        return {}
    arr = np.asarray(list(values.values()), dtype=np.float32)
    mu = float(arr.mean())
    sd = float(arr.std())
    if sd < 1e-6:
        return {k: 0.0 for k in values}
    return {k: float((v - mu) / sd) for k, v in values.items()}


def _compute_op_rule_score_map(full_obs: Dict[str, object], cfg: TrainOConfig) -> Tuple[Dict[int, float], float]:
    pairs = list(full_obs.get("candidate_pairs", []))
    if len(pairs) == 0:
        return {}, 0.0

    pair_sources = _resolve_pair_sources(full_obs)
    pair_feat = np.asarray(full_obs.get("pair_feat", []), dtype=np.float32)
    ect_col = 4 if pair_feat.ndim == 2 and pair_feat.shape[1] > 4 else None

    # per-op, per-rule best score (lower is better for ECT-like quantity)
    op_rule_best: Dict[int, Dict[str, float]] = {}
    for i, (op_id, _) in enumerate(pairs):
        op_id = int(op_id)
        rule_list = pair_sources[i] if i < len(pair_sources) else []
        if len(rule_list) == 0:
            rule_list = ["_NO_RULE_"]

        if ect_col is not None and pair_feat.shape[0] == len(pairs):
            score_raw = float(pair_feat[i, ect_col])
        else:
            score_raw = float(i)

        op_rule_best.setdefault(op_id, {})
        for r in rule_list:
            r = str(r)
            prev = op_rule_best[op_id].get(r, None)
            if prev is None or score_raw < prev:
                op_rule_best[op_id][r] = score_raw

    # global rule normalization
    rule_mu_sd: Dict[str, Tuple[float, float]] = {}
    all_rules = sorted({r for m in op_rule_best.values() for r in m.keys()})
    for r in all_rules:
        vals = [m[r] for m in op_rule_best.values() if r in m]
        arr = np.asarray(vals, dtype=np.float32)
        mu = float(arr.mean()) if arr.size > 0 else 0.0
        sd = float(arr.std()) if arr.size > 0 else 1.0
        if sd < 1e-6:
            sd = 1.0
        rule_mu_sd[r] = (mu, sd)

    op_scores: Dict[int, float] = {}
    per_op_var: Dict[int, float] = {}
    for op_id, rmap in op_rule_best.items():
        vals = []
        for r, v in rmap.items():
            mu, sd = rule_mu_sd[r]
            vals.append(float((v - mu) / sd))

        if len(vals) == 0:
            op_scores[op_id] = 0.0
            per_op_var[op_id] = 0.0
            continue

        vals = sorted(vals)
        trim = int(max(0, cfg.rule_score_trim_lowest))
        if trim > 0 and len(vals) > (trim + 1):
            vals = vals[:-trim]

        arr = np.asarray(vals, dtype=np.float32)
        # rank-aware exponential weighting on ascending normalized values
        order = np.argsort(arr)
        tau = max(float(cfg.rule_score_tau), 1e-6)
        weights = np.exp(-np.arange(order.size, dtype=np.float32) / tau)
        ordered_vals = arr[order]
        agg = float(np.sum(weights * ordered_vals) / max(np.sum(weights), 1e-6))

        # lower normalized score is better -> negate to maximize
        op_scores[op_id] = float(-agg)
        per_op_var[op_id] = float(arr.var()) if arr.size > 1 else 0.0

    # variance-penalized score
    lambda_var = float(max(0.0, cfg.rule_score_var_penalty))
    for op_id in list(op_scores.keys()):
        op_scores[op_id] = float(op_scores[op_id] - lambda_var * per_op_var.get(op_id, 0.0))

    # disagreement scalar for dynamic K
    disagreement = float(np.mean(list(per_op_var.values()))) if len(per_op_var) > 0 else 0.0
    return op_scores, disagreement


def _compute_dynamic_topk(
    full_obs: Dict[str, object],
    cfg: TrainOConfig,
    c_entropy_full: float = 0.0,
    rule_disagreement: float = 0.0,
) -> int:
    pairs = list(full_obs.get("candidate_pairs", []))
    n_ops = len({int(op) for op, _ in pairs})
    if n_ops <= 0:
        return 1

    k_base = int(max(1, cfg.op_topk))
    k_ratio = int(np.ceil(float(cfg.op_topk_ratio) * float(n_ops)))
    k = max(int(cfg.op_topk_min), k_base, k_ratio)
    k += int(np.ceil(max(0.0, float(cfg.topk_c_entropy_gain)) * max(0.0, float(c_entropy_full))))
    k += int(np.ceil(max(0.0, float(cfg.topk_rule_disagreement_gain)) * max(0.0, float(rule_disagreement))))

    # Keep more operations for very small instances to reduce over-pruning risk.
    if n_ops <= int(max(1, cfg.small_instance_ops_threshold)):
        k += int(max(0, cfg.small_instance_extra_keep))

    k = min(int(max(1, cfg.op_topk_max)), k)
    return int(max(1, min(n_ops, k)))


def _should_enable_rule_blend(
    cfg: TrainOConfig,
    n_ops: int,
    c_entropy_full: float,
) -> bool:
    if not bool(cfg.rule_score_enabled):
        return False
    if int(n_ops) < int(max(1, cfg.rule_score_gate_min_ops)):
        return False
    if int(n_ops) > int(max(1, cfg.rule_score_gate_max_ops)):
        return False
    if float(c_entropy_full) < float(cfg.rule_score_gate_min_entropy):
        return False
    if float(c_entropy_full) > float(max(cfg.rule_score_gate_min_entropy, cfg.rule_score_gate_max_entropy)):
        return False
    return True


def _blend_policy_rule_selected_ops(
    full_obs: Dict[str, object],
    o_info: Dict[str, object],
    topk_k: int,
    cfg: TrainOConfig,
    rule_score_map: Optional[Dict[int, float]] = None,
    base_selected_ops: Optional[List[int]] = None,
) -> List[int]:
    packed = dict(o_info.get("packed_obs", {}))
    legal_ops = [int(x) for x in packed.get("legal_ops", [])]
    logits = np.asarray(o_info.get("valid_logits", []), dtype=np.float32)
    if len(legal_ops) == 0:
        return list(dict.fromkeys([int(x) for x in o_info.get("selected_ops", [])]))

    topk_k = int(max(1, min(int(topk_k), len(legal_ops))))

    policy_raw = {int(op): float(logits[i]) if i < logits.shape[0] else 0.0 for i, op in enumerate(legal_ops)}
    policy_z = _safe_zscore_map(policy_raw)

    if rule_score_map is not None:
        rule_raw = {int(k): float(v) for k, v in rule_score_map.items()}
    elif bool(cfg.rule_score_enabled):
        rule_raw, _ = _compute_op_rule_score_map(full_obs, cfg)
    else:
        rule_raw = {int(op): 0.0 for op in legal_ops}
    rule_z = _safe_zscore_map({int(op): float(rule_raw.get(int(op), 0.0)) for op in legal_ops})

    alpha = float(np.clip(cfg.rule_policy_blend_alpha, 0.0, 1.0))
    blended: List[Tuple[float, int]] = []
    for op in legal_ops:
        score = alpha * float(policy_z.get(int(op), 0.0)) + (1.0 - alpha) * float(rule_z.get(int(op), 0.0))
        blended.append((float(score), int(op)))

    blended = sorted(blended, key=lambda x: (-x[0], x[1]))

    action_op = int(o_info.get("action_op_id", legal_ops[0]))
    if bool(cfg.rule_blend_addon_only):
        selected = list(dict.fromkeys([int(x) for x in (base_selected_ops or o_info.get("selected_ops", []))]))
        if len(selected) == 0:
            selected = [action_op]
        target = int(max(len(selected), min(topk_k, len(selected) + int(max(0, cfg.rule_blend_addon_count)))))
    else:
        selected = [action_op]
        target = int(topk_k)

    selected_set = set(selected)
    for _, op in blended:
        if len(selected) >= target:
            break
        if int(op) not in selected_set:
            selected.append(int(op))
            selected_set.add(int(op))
    return list(dict.fromkeys(selected))


def _mk6_min_keep_boost(case_path: str, full_obs: Dict[str, object], cfg: TrainOConfig) -> int:
    if not bool(cfg.mk6_protect_enabled):
        return 0
    case_name = Path(case_path).stem
    if "Mk6" not in str(case_name):
        return 0

    pair_feat = np.asarray(full_obs.get("pair_feat", []), dtype=np.float32)
    if pair_feat.ndim != 2 or pair_feat.shape[0] <= 1 or pair_feat.shape[1] <= 4:
        return 0

    ect = pair_feat[:, 4]
    best = float(np.min(ect))
    p25 = float(np.percentile(ect, 25.0))
    denom = max(abs(best), 1e-6)
    gap = max(0.0, (p25 - best) / denom)
    boost = 0
    if gap > float(cfg.mk6_quality_gap_threshold):
        boost += int(max(0, cfg.mk6_extra_keep))

    # Additional safeguard for Mk6 when candidates are tight and pruning risk is high.
    n_ops = len({int(op) for op, _ in full_obs.get("candidate_pairs", [])})
    if n_ops <= 6:
        boost += 1
    return int(max(0, boost))


def _best_pair_index_by_col(obs: Dict[str, object], col: int) -> Optional[int]:
    pair_feat = np.asarray(obs.get("pair_feat", []), dtype=np.float32)
    if pair_feat.ndim != 2 or pair_feat.shape[0] <= 0 or pair_feat.shape[1] <= int(col):
        return None
    return int(np.argmin(pair_feat[:, int(col)]))


def _collect_anchor_ops(full_obs: Dict[str, object], c_full_op: int, cfg: TrainOConfig) -> List[int]:
    pairs = list(full_obs.get("candidate_pairs", []))
    pair_sources = _resolve_pair_sources(full_obs)
    pair_feat = np.asarray(full_obs.get("pair_feat", []), dtype=np.float32)
    if len(pairs) == 0:
        return [int(c_full_op)]

    anchors: List[int] = [int(c_full_op)] if bool(cfg.keep_c_full_top1) else []

    if bool(cfg.keep_anchor_best_ect):
        idx = _best_pair_index_by_col(full_obs, col=4)
        if idx is not None:
            anchors.append(int(pairs[idx][0]))

    if bool(cfg.keep_anchor_best_pt):
        idx = _best_pair_index_by_col(full_obs, col=0)
        if idx is not None:
            anchors.append(int(pairs[idx][0]))

    if bool(cfg.keep_anchor_preferred_rule):
        pref = str(cfg.anchor_preferred_rule)
        pref_idx = [i for i, src in enumerate(pair_sources) if pref in set(src)]
        if len(pref_idx) > 0 and pair_feat.ndim == 2 and pair_feat.shape[1] > 4:
            local = int(pref_idx[int(np.argmin(pair_feat[np.asarray(pref_idx, dtype=np.int64), 4]))])
            anchors.append(int(pairs[local][0]))

    # Deduplicate while preserving order.
    return list(dict.fromkeys([int(x) for x in anchors]))


def _expand_ops_for_safety(full_obs: Dict[str, object], selected_ops: List[int], min_ops: int) -> List[int]:
    selected = list(dict.fromkeys([int(x) for x in selected_ops]))
    if len(selected) >= int(min_ops):
        return selected

    pairs = list(full_obs.get("candidate_pairs", []))
    pair_feat = np.asarray(full_obs.get("pair_feat", []), dtype=np.float32)
    if len(pairs) == 0:
        return selected

    op_best_ect: Dict[int, float] = {}
    if pair_feat.ndim == 2 and pair_feat.shape[0] == len(pairs) and pair_feat.shape[1] > 4:
        for i, (op_id, _) in enumerate(pairs):
            op_id = int(op_id)
            ect = float(pair_feat[i, 4])
            if op_id not in op_best_ect or ect < op_best_ect[op_id]:
                op_best_ect[op_id] = ect
    else:
        for op_id, _ in pairs:
            op_best_ect[int(op_id)] = 0.0

    ranked_ops = [op for op, _ in sorted(op_best_ect.items(), key=lambda kv: (kv[1], kv[0]))]
    for op in ranked_ops:
        if len(selected) >= int(min_ops):
            break
        if int(op) not in set(selected):
            selected.append(int(op))
    return selected


def _maybe_uncertainty_fallback(
    full_obs: Dict[str, object],
    selected_ops: List[int],
    c_entropy_filtered: float,
    cfg: TrainOConfig,
) -> List[int]:
    selected = list(dict.fromkeys([int(x) for x in selected_ops]))
    if float(c_entropy_filtered) <= float(cfg.c_entropy_fallback_threshold):
        return selected

    target_min = len(selected) + int(max(0, cfg.c_entropy_fallback_extra_ops))
    return _expand_ops_for_safety(full_obs=full_obs, selected_ops=selected, min_ops=target_min)


def infer_o_dims(
    case_paths: List[str],
    reset_rule: str,
    quiet: bool = True,
    use_candidate_set_feat: bool = True,
) -> Tuple[int, int, int]:
    max_global_dim = 1
    op_feat_dim = 1
    max_ops = 1

    for case_path in case_paths:
        env, _ = build_env_for_case(case_path, reset_rule=reset_rule, quiet=quiet)
        obs = env.get_agent_c_obs(batch_idx=0)

        global_feat = _flatten_f32(obs["global_feat"])
        if use_candidate_set_feat and "candidate_set_feat" in obs:
            global_feat = np.concatenate([global_feat, _flatten_f32(obs["candidate_set_feat"])], axis=0)

        # infer op_feat_dim by reusing preprocess pipeline with temporary max_ops
        legal_ops = len({int(op) for op, _ in obs.get("candidate_pairs", [])})
        temp_max_ops = max(1, legal_ops)

        packed = preprocess_obs_o(
            obs=obs,
            global_dim=max(1, global_feat.shape[0]),
            op_feat_dim=np.asarray(obs["op_node_feat"]).shape[1] + 10,
            max_ops=temp_max_ops,
            use_candidate_set_feat=use_candidate_set_feat,
            overflow_strategy="truncate",
        )

        max_global_dim = max(max_global_dim, int(global_feat.shape[0]))
        op_feat_dim = max(op_feat_dim, int(packed["op_feat"].shape[1]))
        max_ops = max(max_ops, int(legal_ops))

    return int(max_global_dim), int(op_feat_dim), int(max_ops)


def run_episode_o_with_frozen_c(
    agent_o: Any,
    frozen_c: FrozenAgentC,
    case_path: str,
    reset_rule: str,
    op_topk: int,
    deterministic_o: bool,
    deterministic_c: bool,
    quiet: bool,
    extra_step_budget: int,
    cfg: TrainOConfig,
    collect_records: bool,
) -> Dict[str, object]:
    env, total_tasks = build_env_for_case(case_path, reset_rule=reset_rule, quiet=quiet)
    initial_lb_makespan = float(env.LBm[0].max())

    max_steps = int(total_tasks) + int(extra_step_budget)
    done = False
    episode_reward = 0.0
    episode_train_reward = 0.0
    executed_steps = 0

    records: List[Any] = []

    stat_retain = 0.0
    stat_quality_gap = 0.0
    stat_compact = 0.0
    stat_consistency = 0.0
    stat_hard_quality = 0.0
    stat_mismatch_penalty = 0.0
    stat_makespan_delta_ratio = 0.0
    stat_makespan_increase_penalty = 0.0
    stat_makespan_decrease_bonus = 0.0
    stat_terminal_penalty = 0.0
    stat_c_value_delta = 0.0
    stat_c_value_delta_bonus = 0.0
    stat_c_entropy_increase = 0.0
    stat_c_entropy_increase_penalty = 0.0
    stat_fallback_trigger_rate = 0.0

    for _ in range(max_steps):
        full_obs = env.get_agent_c_obs(batch_idx=0)
        prev_makespan = float(env.LBm[0].max())
        c_full_op, _, c_full_info = frozen_c.act(full_obs, deterministic=True)

        c_entropy_full = float(c_full_info.get("entropy", 0.0))
        n_ops = len({int(op) for op, _ in full_obs.get("candidate_pairs", [])})
        rule_blend_enabled = _should_enable_rule_blend(
            cfg=cfg,
            n_ops=n_ops,
            c_entropy_full=c_entropy_full,
        )
        if rule_blend_enabled:
            rule_score_map, rule_disagreement = _compute_op_rule_score_map(full_obs, cfg)
        else:
            rule_score_map, rule_disagreement = {}, 0.0
        dynamic_topk = _compute_dynamic_topk(
            full_obs=full_obs,
            cfg=cfg,
            c_entropy_full=c_entropy_full,
            rule_disagreement=rule_disagreement,
        )
        o_info = agent_o.select_action(full_obs, deterministic=deterministic_o, topk_k=max(int(op_topk), dynamic_topk))

        selected_ops = list(dict.fromkeys([int(x) for x in o_info.get("selected_ops", [])]))
        if rule_blend_enabled:
            selected_ops = _blend_policy_rule_selected_ops(
                full_obs=full_obs,
                o_info=o_info,
                topk_k=max(int(op_topk), dynamic_topk),
                cfg=cfg,
                rule_score_map=rule_score_map,
                base_selected_ops=selected_ops,
            )
        anchors = _collect_anchor_ops(full_obs=full_obs, c_full_op=int(c_full_op), cfg=cfg)
        selected_ops.extend(anchors)
        selected_ops = list(dict.fromkeys([int(x) for x in selected_ops]))

        min_keep = max(int(cfg.min_selected_ops_floor), int(dynamic_topk))
        min_keep += int(_mk6_min_keep_boost(case_path=case_path, full_obs=full_obs, cfg=cfg))
        selected_ops = _expand_ops_for_safety(full_obs=full_obs, selected_ops=selected_ops, min_ops=min_keep)

        filtered_obs = filter_obs_by_selected_ops(full_obs, selected_ops)

        c_op, c_mch, c_filtered_info = frozen_c.act(filtered_obs, deterministic=deterministic_c)

        c_entropy_filtered = float(c_filtered_info.get("entropy", 0.0))
        selected_after_fallback = _maybe_uncertainty_fallback(
            full_obs=full_obs,
            selected_ops=selected_ops,
            c_entropy_filtered=c_entropy_filtered,
            cfg=cfg,
        )
        fallback_triggered = 1.0 if len(selected_after_fallback) > len(selected_ops) else 0.0
        if fallback_triggered > 0.0:
            selected_ops = selected_after_fallback
            filtered_obs = filter_obs_by_selected_ops(full_obs, selected_ops)
            c_op, c_mch, c_filtered_info = frozen_c.act(filtered_obs, deterministic=deterministic_c)

        step_out = env.step_with_pair(c_op, c_mch, batch_idx=0)
        base_reward = float(step_out[2][0])
        done = bool(step_out[3][0])
        new_makespan = float(env.LBm[0].max())

        terms = _reward_terms(
            full_obs=full_obs,
            filtered_obs=filtered_obs,
            selected_ops=selected_ops,
            c_full_op=int(c_full_op),
            c_filtered_op=int(c_op),
            prev_makespan=prev_makespan,
            new_makespan=new_makespan,
            is_terminal=done,
            initial_lb_makespan=initial_lb_makespan,
            c_value_full=float(c_full_info.get("value", 0.0)),
            c_value_filtered=float(c_filtered_info.get("value", 0.0)),
            c_entropy_full=float(c_full_info.get("entropy", 0.0)),
            c_entropy_filtered=float(c_filtered_info.get("entropy", 0.0)),
            cfg=cfg,
        )

        train_reward = float(base_reward + terms["total"])

        episode_reward += base_reward
        episode_train_reward += train_reward
        executed_steps += 1

        stat_retain += float(terms["retain_bonus"])
        stat_quality_gap += float(terms["quality_gap"])
        stat_hard_quality += float(terms["hard_quality_penalty"])
        stat_compact += float(terms["compact_ratio"])
        stat_consistency += float(terms["consistency_bonus"])
        stat_mismatch_penalty += float(terms["mismatch_penalty"])
        stat_makespan_delta_ratio += float(terms["makespan_delta_ratio"])
        stat_makespan_increase_penalty += float(terms["makespan_increase_penalty"])
        stat_makespan_decrease_bonus += float(terms["makespan_decrease_bonus"])
        stat_terminal_penalty += float(terms["terminal_penalty"])
        stat_c_value_delta += float(terms["c_value_delta"])
        stat_c_value_delta_bonus += float(terms["c_value_delta_bonus"])
        stat_c_entropy_increase += float(terms["c_entropy_increase"])
        stat_c_entropy_increase_penalty += float(terms["c_entropy_increase_penalty"])
        stat_fallback_trigger_rate += float(fallback_triggered)

        if collect_records:
            packed = o_info["packed_obs"]
            records.append(
                StepRecordO(
                    global_feat=np.asarray(packed["global_feat"], dtype=np.float32),
                    op_feat=np.asarray(packed["op_feat"], dtype=np.float32),
                    op_mask=np.asarray(packed["op_mask"], dtype=bool),
                    action_idx=int(o_info["action_idx"]),
                    log_prob=float(o_info["log_prob"]),
                    value=float(o_info["value"]),
                    reward=float(train_reward),
                    done=bool(done),
                )
            )

        if done:
            break

    makespan = float(env.LBm[0].max())
    denom = float(max(executed_steps, 1))

    return {
        "case": Path(case_path).name,
        "episode_reward": float(episode_reward),
        "episode_train_reward": float(episode_train_reward),
        "makespan": float(makespan),
        "steps": int(executed_steps),
        "done": bool(done),
        "records": records,
        "mean_retain_bonus": float(stat_retain / denom),
        "mean_quality_gap": float(stat_quality_gap / denom),
        "mean_hard_quality_penalty": float(stat_hard_quality / denom),
        "mean_compact_ratio": float(stat_compact / denom),
        "mean_consistency_bonus": float(stat_consistency / denom),
        "mean_mismatch_penalty": float(stat_mismatch_penalty / denom),
        "mean_makespan_delta_ratio": float(stat_makespan_delta_ratio / denom),
        "mean_makespan_increase_penalty": float(stat_makespan_increase_penalty / denom),
        "mean_makespan_decrease_bonus": float(stat_makespan_decrease_bonus / denom),
        "mean_terminal_penalty": float(stat_terminal_penalty / denom),
        "mean_c_value_delta": float(stat_c_value_delta / denom),
        "mean_c_value_delta_bonus": float(stat_c_value_delta_bonus / denom),
        "mean_c_entropy_increase": float(stat_c_entropy_increase / denom),
        "mean_c_entropy_increase_penalty": float(stat_c_entropy_increase_penalty / denom),
        "mean_fallback_trigger_rate": float(stat_fallback_trigger_rate / denom),
    }


def evaluate_o_with_frozen_c(
    agent_o: Any,
    frozen_c: FrozenAgentC,
    case_paths: List[str],
    cfg: TrainOConfig,
) -> Dict[str, object]:
    rows = []
    errors = []

    for case_path in case_paths:
        try:
            ep = run_episode_o_with_frozen_c(
                agent_o=agent_o,
                frozen_c=frozen_c,
                case_path=case_path,
                reset_rule=cfg.reset_rule,
                op_topk=cfg.op_topk,
                deterministic_o=True,
                deterministic_c=cfg.c_deterministic,
                quiet=cfg.quiet_env,
                extra_step_budget=cfg.extra_step_budget,
                cfg=cfg,
                collect_records=False,
            )
            rows.append(ep)
        except Exception as exc:
            errors.append({"case": str(case_path), "reason": repr(exc)})

    if len(rows) == 0:
        return {
            "num_cases": int(len(case_paths)),
            "num_success": 0,
            "num_errors": int(len(errors)),
            "mean_reward": 0.0,
            "mean_train_reward": 0.0,
            "mean_makespan": float("inf"),
            "mean_steps": 0.0,
            "rows": rows,
            "errors": errors,
        }

    return {
        "num_cases": int(len(case_paths)),
        "num_success": int(len(rows)),
        "num_errors": int(len(errors)),
        "mean_reward": float(np.mean([x["episode_reward"] for x in rows])),
        "mean_train_reward": float(np.mean([x["episode_train_reward"] for x in rows])),
        "mean_makespan": float(np.mean([x["makespan"] for x in rows])),
        "mean_steps": float(np.mean([x["steps"] for x in rows])),
        "mean_quality_gap": float(np.mean([x.get("mean_quality_gap", 0.0) for x in rows])),
        "mean_compact_ratio": float(np.mean([x.get("mean_compact_ratio", 0.0) for x in rows])),
        "rows": rows,
        "errors": errors,
    }


def train_agent_o_with_frozen_c(cfg: TrainOConfig) -> Dict[str, object]:
    if len(cfg.train_cases) == 0:
        raise ValueError("train_cases is empty")
    if len(cfg.val_cases) == 0:
        raise ValueError("val_cases is empty")
    if len(cfg.test_cases) == 0:
        raise ValueError("test_cases is empty")
    if not cfg.c_checkpoint_path:
        raise ValueError("c_checkpoint_path is required")

    seed_everything(int(cfg.seed))

    all_scan_cases = list(cfg.train_cases) + list(cfg.val_cases)
    global_dim, op_feat_dim, observed_max_ops = infer_o_dims(
        case_paths=all_scan_cases,
        reset_rule=cfg.reset_rule,
        quiet=cfg.quiet_env,
        use_candidate_set_feat=True,
    )

    max_ops = int(observed_max_ops + max(0, int(cfg.op_safety_margin)))
    if not bool(cfg.op_auto_max_ops):
        max_ops = max(int(cfg.op_max_ops), int(observed_max_ops))

    print(
        f"[init-o] global_dim={global_dim}, op_feat_dim={op_feat_dim}, "
        f"observed_max_ops={observed_max_ops}, max_ops={max_ops}"
    )

    agent_o_cfg = AgentOConfig(
        device=cfg.device,
        hidden_dim=cfg.op_hidden_dim,
        dropout=cfg.op_dropout,
        lr=cfg.op_lr,
        gamma=cfg.op_gamma,
        gae_lambda=cfg.op_gae_lambda,
        clip_ratio=cfg.op_clip_ratio,
        value_coef=cfg.op_value_coef,
        ent_coef=cfg.op_ent_coef,
        max_grad_norm=cfg.op_max_grad_norm,
        ppo_epochs=cfg.op_ppo_epochs,
        minibatch_size=cfg.op_minibatch_size,
        target_kl=cfg.op_target_kl,
        max_ops=max_ops,
        use_candidate_set_feat=True,
        overflow_strategy="truncate",
        model_type=cfg.op_model_type,
        transformer_layers=cfg.op_transformer_layers,
        transformer_heads=cfg.op_transformer_heads,
        transformer_dropout=cfg.op_transformer_dropout,
        moe_num_experts=cfg.op_moe_num_experts,
        moe_topk=cfg.op_moe_topk,
        moe_temperature=cfg.op_moe_temperature,
    )

    agent_o = Agent_O(config=agent_o_cfg, global_dim=global_dim, op_feat_dim=op_feat_dim)
    frozen_c = FrozenAgentC(
        ckpt_path=cfg.c_checkpoint_path,
        device=cfg.device,
        disable_topk_prefilter=cfg.disable_c_topk_prefilter,
        max_candidates_floor=cfg.c_max_candidates_floor,
    )

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / cfg.checkpoint_name

    best_val_makespan = float("inf")
    best_epoch = 0
    history = []

    buffer = RolloutBufferO()

    for epoch in range(1, int(cfg.epochs) + 1):
        train_cases = list(cfg.train_cases)
        random.Random(int(cfg.seed) + epoch).shuffle(train_cases)

        train_rows = []
        train_errors = []
        update_logs = []

        for start in range(0, len(train_cases), int(max(1, cfg.episodes_per_update))):
            chunk = train_cases[start : start + int(max(1, cfg.episodes_per_update))]
            buffer.clear()

            for case_path in chunk:
                try:
                    ep = run_episode_o_with_frozen_c(
                        agent_o=agent_o,
                        frozen_c=frozen_c,
                        case_path=case_path,
                        reset_rule=cfg.reset_rule,
                        op_topk=cfg.op_topk,
                        deterministic_o=False,
                        deterministic_c=cfg.c_deterministic,
                        quiet=cfg.quiet_env,
                        extra_step_budget=cfg.extra_step_budget,
                        cfg=cfg,
                        collect_records=True,
                    )
                    train_rows.append(ep)
                    buffer.add_episode(ep["records"], gamma=cfg.op_gamma, gae_lambda=cfg.op_gae_lambda)
                except Exception as exc:
                    train_errors.append({"case": str(case_path), "reason": repr(exc)})

            if len(buffer) > 0:
                logs = agent_o.update(buffer)
                update_logs.append(logs)

        row = {
            "epoch": int(epoch),
            "train_num_success": int(len(train_rows)),
            "train_num_errors": int(len(train_errors)),
            "train_mean_reward": float(np.mean([x["episode_reward"] for x in train_rows])) if train_rows else 0.0,
            "train_mean_train_reward": float(np.mean([x["episode_train_reward"] for x in train_rows])) if train_rows else 0.0,
            "train_mean_makespan": float(np.mean([x["makespan"] for x in train_rows])) if train_rows else float("inf"),
            "train_mean_quality_gap": float(np.mean([x.get("mean_quality_gap", 0.0) for x in train_rows])) if train_rows else 0.0,
            "train_mean_hard_quality_penalty": float(np.mean([x.get("mean_hard_quality_penalty", 0.0) for x in train_rows])) if train_rows else 0.0,
            "train_mean_compact_ratio": float(np.mean([x.get("mean_compact_ratio", 0.0) for x in train_rows])) if train_rows else 0.0,
            "train_mean_mismatch_penalty": float(np.mean([x.get("mean_mismatch_penalty", 0.0) for x in train_rows])) if train_rows else 0.0,
            "train_mean_makespan_delta_ratio": float(np.mean([x.get("mean_makespan_delta_ratio", 0.0) for x in train_rows])) if train_rows else 0.0,
            "train_mean_makespan_increase_penalty": float(np.mean([x.get("mean_makespan_increase_penalty", 0.0) for x in train_rows])) if train_rows else 0.0,
            "train_mean_makespan_decrease_bonus": float(np.mean([x.get("mean_makespan_decrease_bonus", 0.0) for x in train_rows])) if train_rows else 0.0,
            "train_mean_terminal_penalty": float(np.mean([x.get("mean_terminal_penalty", 0.0) for x in train_rows])) if train_rows else 0.0,
            "train_mean_c_value_delta": float(np.mean([x.get("mean_c_value_delta", 0.0) for x in train_rows])) if train_rows else 0.0,
            "train_mean_c_value_delta_bonus": float(np.mean([x.get("mean_c_value_delta_bonus", 0.0) for x in train_rows])) if train_rows else 0.0,
            "train_mean_c_entropy_increase": float(np.mean([x.get("mean_c_entropy_increase", 0.0) for x in train_rows])) if train_rows else 0.0,
            "train_mean_c_entropy_increase_penalty": float(np.mean([x.get("mean_c_entropy_increase_penalty", 0.0) for x in train_rows])) if train_rows else 0.0,
            "train_mean_fallback_trigger_rate": float(np.mean([x.get("mean_fallback_trigger_rate", 0.0) for x in train_rows])) if train_rows else 0.0,
            "update_policy_loss": float(np.mean([x.get("policy_loss", 0.0) for x in update_logs])) if update_logs else 0.0,
            "update_value_loss": float(np.mean([x.get("value_loss", 0.0) for x in update_logs])) if update_logs else 0.0,
            "update_entropy": float(np.mean([x.get("entropy", 0.0) for x in update_logs])) if update_logs else 0.0,
            "update_approx_kl": float(np.mean([x.get("approx_kl", 0.0) for x in update_logs])) if update_logs else 0.0,
        }

        if epoch % int(max(1, cfg.eval_interval)) == 0:
            val_eval = evaluate_o_with_frozen_c(
                agent_o=agent_o,
                frozen_c=frozen_c,
                case_paths=list(cfg.val_cases),
                cfg=cfg,
            )
            row["val_mean_reward"] = float(val_eval["mean_reward"])
            row["val_mean_train_reward"] = float(val_eval["mean_train_reward"])
            row["val_mean_makespan"] = float(val_eval["mean_makespan"])
            row["val_num_errors"] = int(val_eval["num_errors"])

            if float(val_eval["mean_makespan"]) < best_val_makespan:
                best_val_makespan = float(val_eval["mean_makespan"])
                best_epoch = int(epoch)
                agent_o.save(str(ckpt_path))
                row["best_updated"] = 1
            else:
                row["best_updated"] = 0

            epoch_view = {
                "epoch": int(epoch),
                "best_updated": int(row["best_updated"]),
                "best_epoch": int(best_epoch),
                "train": {
                    "num_success": int(row["train_num_success"]),
                    "num_errors": int(row["train_num_errors"]),
                    "mean_ms": float(row["train_mean_makespan"]),
                    "mean_reward": float(row["train_mean_reward"]),
                    "mean_quality_gap": float(row["train_mean_quality_gap"]),
                    "mean_mismatch_penalty": float(row["train_mean_mismatch_penalty"]),
                    "mean_fallback_trigger_rate": float(row["train_mean_fallback_trigger_rate"]),
                    "mean_c_entropy_increase": float(row["train_mean_c_entropy_increase"]),
                },
                "update": {
                    "policy_loss": float(row["update_policy_loss"]),
                    "value_loss": float(row["update_value_loss"]),
                    "entropy": float(row["update_entropy"]),
                    "approx_kl": float(row["update_approx_kl"]),
                },
                "val": {
                    "mean_ms": float(row["val_mean_makespan"]),
                    "mean_reward": float(row["val_mean_reward"]),
                    "num_errors": int(row["val_num_errors"]),
                },
            }

            print(
                f"[epoch {epoch:03d}] train_ms={epoch_view['train']['mean_ms']:.4f}, "
                f"val_ms={epoch_view['val']['mean_ms']:.4f}, best_val={best_val_makespan:.4f} @ {best_epoch}, "
                f"fallback={epoch_view['train']['mean_fallback_trigger_rate']:.4f}, "
                f"qgap={epoch_view['train']['mean_quality_gap']:.4f}, "
                f"mismatch={epoch_view['train']['mean_mismatch_penalty']:.4f}, "
                f"entropy={epoch_view['update']['entropy']:.4f}, kl={epoch_view['update']['approx_kl']:.6f}, "
                f"updated={epoch_view['best_updated']}"
            )
            print("[epoch_json] " + json.dumps(epoch_view, ensure_ascii=False))
        else:
            epoch_view = {
                "epoch": int(epoch),
                "train": {
                    "num_success": int(row["train_num_success"]),
                    "num_errors": int(row["train_num_errors"]),
                    "mean_ms": float(row["train_mean_makespan"]),
                    "mean_reward": float(row["train_mean_reward"]),
                    "mean_quality_gap": float(row["train_mean_quality_gap"]),
                    "mean_mismatch_penalty": float(row["train_mean_mismatch_penalty"]),
                    "mean_fallback_trigger_rate": float(row["train_mean_fallback_trigger_rate"]),
                    "mean_c_entropy_increase": float(row["train_mean_c_entropy_increase"]),
                },
                "update": {
                    "policy_loss": float(row["update_policy_loss"]),
                    "value_loss": float(row["update_value_loss"]),
                    "entropy": float(row["update_entropy"]),
                    "approx_kl": float(row["update_approx_kl"]),
                },
            }
            print(
                f"[epoch {epoch:03d}] train_ms={epoch_view['train']['mean_ms']:.4f}, "
                f"fallback={epoch_view['train']['mean_fallback_trigger_rate']:.4f}, "
                f"qgap={epoch_view['train']['mean_quality_gap']:.4f}, "
                f"mismatch={epoch_view['train']['mean_mismatch_penalty']:.4f}, "
                f"entropy={epoch_view['update']['entropy']:.4f}, kl={epoch_view['update']['approx_kl']:.6f}"
            )
            print("[epoch_json] " + json.dumps(epoch_view, ensure_ascii=False))

        history.append(row)

    if ckpt_path.exists():
        agent_o.load(str(ckpt_path), strict=True)

    val_eval_final = evaluate_o_with_frozen_c(
        agent_o=agent_o,
        frozen_c=frozen_c,
        case_paths=list(cfg.val_cases),
        cfg=cfg,
    )
    test_eval_final = evaluate_o_with_frozen_c(
        agent_o=agent_o,
        frozen_c=frozen_c,
        case_paths=list(cfg.test_cases),
        cfg=cfg,
    )

    summary = {
        "best_val_makespan": float(best_val_makespan),
        "best_epoch": int(best_epoch),
        "checkpoint": str(ckpt_path),
        "val_eval": val_eval_final,
        "test_eval": test_eval_final,
        "history": history,
        "config": dataclasses.asdict(cfg),
    }

    (save_dir / "train_o_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _build_cases(case_dir: str, pattern: str, train_count: int, val_count: int, seed: int):
    all_cases = sorted(Path(case_dir).glob(pattern))
    if len(all_cases) < train_count + val_count + 1:
        raise ValueError("Not enough cases for requested split.")
    random.Random(int(seed)).shuffle(all_cases)
    return make_case_split(all_cases=all_cases, train_count=train_count, val_count=val_count)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Train AgentO with frozen AgentC (new v4 design).")
    parser.add_argument("--case-dir", type=str, default="1_Brandimarte")
    parser.add_argument("--pattern", type=str, default="BrandimarteMk*.fjs")
    parser.add_argument("--train-count", type=int, default=10)
    parser.add_argument("--val-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--episodes-per-update", type=int, default=8)
    parser.add_argument("--eval-interval", type=int, default=1)

    parser.add_argument("--op-topk", type=int, default=5)
    parser.add_argument("--op-model-type", type=str, default="mlp", choices=["mlp", "hybrid_transformer", "moe", "hybrid_transformer_moe"])
    parser.add_argument("--op-transformer-layers", type=int, default=1)
    parser.add_argument("--op-transformer-heads", type=int, default=4)
    parser.add_argument("--op-transformer-dropout", type=float, default=0.1)
    parser.add_argument("--op-moe-num-experts", type=int, default=4)
    parser.add_argument("--op-moe-topk", type=int, default=1)
    parser.add_argument("--op-moe-temperature", type=float, default=1.0)
    parser.add_argument("--save-dir", type=str, default="checkpoints_agent_o_frozen_c")
    parser.add_argument("--checkpoint-name", type=str, default="agent_o_best.pt")
    parser.add_argument("--c-checkpoint-path", type=str, required=True)

    parser.add_argument("--reset-rule", type=str, default="FIFO_SPT")
    parser.add_argument("--extra-step-budget", type=int, default=40)
    parser.add_argument("--quiet-env", action="store_true", default=False)
    return parser.parse_args(args=args)


def main(args=None):
    cli = parse_args(args=args)
    train_cases, val_cases, test_cases = _build_cases(
        case_dir=cli.case_dir,
        pattern=cli.pattern,
        train_count=cli.train_count,
        val_count=cli.val_count,
        seed=cli.seed,
    )

    cfg = TrainOConfig(
        train_cases=train_cases,
        val_cases=val_cases,
        test_cases=test_cases,
        seed=cli.seed,
        device=cli.device,
        reset_rule=cli.reset_rule,
        epochs=cli.epochs,
        episodes_per_update=cli.episodes_per_update,
        eval_interval=cli.eval_interval,
        save_dir=cli.save_dir,
        checkpoint_name=cli.checkpoint_name,
        quiet_env=bool(cli.quiet_env),
        extra_step_budget=cli.extra_step_budget,
        c_checkpoint_path=cli.c_checkpoint_path,
        op_topk=cli.op_topk,
        op_model_type=cli.op_model_type,
        op_transformer_layers=cli.op_transformer_layers,
        op_transformer_heads=cli.op_transformer_heads,
        op_transformer_dropout=cli.op_transformer_dropout,
        op_moe_num_experts=cli.op_moe_num_experts,
        op_moe_topk=cli.op_moe_topk,
        op_moe_temperature=cli.op_moe_temperature,
    )

    summary = train_agent_o_with_frozen_c(cfg)
    print(
        f"[done] best_val={summary['best_val_makespan']:.4f}, "
        f"test={summary['test_eval']['mean_makespan']:.4f}, checkpoint={summary['checkpoint']}"
    )


if __name__ == "__main__":
    main()

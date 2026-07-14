from __future__ import annotations

import argparse
import csv
import json
import sys
import math
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

_ORIG_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]
from Params import configs as env_configs
from Train_OC_MAPPO import (
    OCMAPPOConfig,
    AgentCConfig,
    AgentMLearnConfig,
    AgentOConfig,
    Agent_C,
    Agent_M_Learn,
    Agent_O,
    _all_op_ids_from_obs,
    _best_pair_by_ect,
    _compute_dynamic_topk,
    _expand_pairs_for_safety,
    _expand_ops_for_safety,
    _find_pair_index,
    _load_agent_m_checkpoint,
    _maybe_uncertainty_fallback,
    _parse_case_splits,
    _refine_c_action_deterministic_topk,
    _selected_pair_indices_for_ops,
    _slice_candidate_obs,
    build_env_for_case,
    infer_m_dims,
    infer_o_dims,
    infer_training_dims,
    seed_everything,
)
sys.argv = _ORIG_ARGV


def _cfg_from_summary(summary_path: Path, split_json: str, device: str, max_cases: int | None) -> OCMAPPOConfig:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    raw_cfg = dict(data.get("config", {}))
    if split_json:
        payload = json.loads(Path(split_json).read_text(encoding="utf-8"))
        raw_cfg["train_cases"] = [str(x) for x in payload.get("train_cases", [])]
        raw_cfg["val_cases"] = [str(x) for x in payload.get("val_cases", [])]
        raw_cfg["test_cases"] = [str(x) for x in payload.get("test_cases", [])]
        raw_cfg["split_source"] = split_json
        raw_cfg["scan_case_splits"] = "val,test,train"
        raw_cfg["eval_case_splits"] = "val,test"
    if max_cases is not None and max_cases > 0:
        # Keep config complete; actual trimming is done in main.
        pass
    raw_cfg["device"] = device
    raw_cfg["quiet_env"] = True
    keys = {f.name for f in fields(OCMAPPOConfig)}
    filtered = {k: v for k, v in raw_cfg.items() if k in keys}
    return OCMAPPOConfig(**filtered)


def _bridge_env_config(cfg: OCMAPPOConfig) -> None:
    env_configs.reward_balance_coef = float(getattr(cfg, "reward_balance_coef", 0.0))
    env_configs.reward_wait_coef = float(getattr(cfg, "reward_wait_coef", 0.0))
    env_configs.agent_c_active_job_rules = str(getattr(cfg, "active_job_rules", "")).strip()
    env_configs.agent_c_active_machine_rules = str(getattr(cfg, "active_machine_rules", "")).strip()
    env_configs.agent_c_topk_jobs = int(getattr(cfg, "agent_c_topk_jobs", 3))
    env_configs.agent_c_topk_machines = int(getattr(cfg, "agent_c_topk_machines", 3))
    env_configs.agent_c_pairs_per_rule = int(getattr(cfg, "agent_c_pairs_per_rule", 5))
    env_configs.agent_c_extra_explore_pairs = int(getattr(cfg, "agent_c_extra_explore_pairs", 4))
    env_configs.agent_c_two_stage_refine_enabled = bool(getattr(cfg, "agent_c_two_stage_refine_enabled", True))
    refine_pairs = int(getattr(cfg, "agent_c_refine_max_pairs_per_op", 2))
    if bool(getattr(cfg, "full_soft_widen_enabled", True)) and bool(getattr(cfg, "agent_c_two_stage_refine_enabled", True)):
        refine_pairs += max(0, int(getattr(cfg, "full_soft_widen_refine_pairs_per_op_extra", 1)))
    env_configs.agent_c_refine_max_pairs_per_op = int(refine_pairs)
    env_configs.agent_c_refine_keep_global_best_ect = int(getattr(cfg, "agent_c_refine_keep_global_best_ect", 1))
    env_configs.agent_c_refine_keep_global_best_pt = int(getattr(cfg, "agent_c_refine_keep_global_best_pt", 1))
    env_configs.agent_c_refine_score_w_eet = float(getattr(cfg, "agent_c_refine_score_w_eet", 1.0))
    env_configs.agent_c_refine_score_w_pt = float(getattr(cfg, "agent_c_refine_score_w_pt", 0.3))
    env_configs.agent_c_refine_score_w_queue = float(getattr(cfg, "agent_c_refine_score_w_queue", 0.2))
    env_configs.agent_c_refine_score_w_support = float(getattr(cfg, "agent_c_refine_score_w_support", 0.5))
    env_configs.agent_c_refine_explore_bonus = float(getattr(cfg, "agent_c_refine_explore_bonus", 0.1))
    env_configs.agent_c_candidate_v2_enabled = bool(getattr(cfg, "agent_c_candidate_v2_enabled", True))
    env_configs.agent_c_critical_pairs_enabled = bool(getattr(cfg, "agent_c_critical_pairs_enabled", True))
    env_configs.agent_c_critical_pairs_per_op = int(getattr(cfg, "agent_c_critical_pairs_per_op", 2))
    env_configs.agent_c_refine_max_pairs_per_machine = int(getattr(cfg, "agent_c_refine_max_pairs_per_machine", 0))
    env_configs.agent_c_refine_global_reserve_size = int(getattr(cfg, "agent_c_refine_global_reserve_size", 2))
    env_configs.agent_c_refine_diversity_min_machines = int(getattr(cfg, "agent_c_refine_diversity_min_machines", 2))
    env_configs.agent_c_refine_keep_explore_min = int(getattr(cfg, "agent_c_refine_keep_explore_min", 1))
    env_configs.agent_c_candidate_debug_print = False


def _collect_cases(cfg: OCMAPPOConfig, max_cases: int | None) -> List[str]:
    names = _parse_case_splits(str(getattr(cfg, "eval_case_splits", "val,test")), arg_name="eval", default=("val", "test"))
    case_map = {"train": list(cfg.train_cases), "val": list(cfg.val_cases), "test": list(cfg.test_cases)}
    cases: List[str] = []
    for name in names:
        cases.extend(case_map.get(name, []))
    # de-duplicate while preserving order
    out = []
    seen = set()
    for p in cases:
        if p not in seen:
            out.append(p); seen.add(p)
    if max_cases is not None and max_cases > 0:
        out = out[:max_cases]
    return out


def _make_agents(cfg: OCMAPPOConfig, cases: List[str], summary: Dict[str, object]):
    scan_cases = cases if cases else (cfg.val_cases or cfg.test_cases or cfg.train_cases)
    c_global_dim, pair_feat_dim, observed_max_candidates, _ = infer_training_dims(
        case_paths=scan_cases,
        reset_rule=cfg.reset_rule,
        use_candidate_set_feat=True,
        quiet=True,
        extra_step_budget=cfg.extra_step_budget,
        scan_rule=cfg.reset_rule,
        candidate_safety_margin=cfg.c_candidate_safety_margin,
        round_max_candidates_to_power_of_two=bool(cfg.c_round_max_candidates_to_power_of_two),
    )
    if bool(cfg.c_auto_max_candidates):
        margin = max(int(cfg.c_candidate_safety_margin), int(math.ceil(float(observed_max_candidates) * float(cfg.c_candidate_safety_margin_ratio))))
        c_max_candidates = max(int(observed_max_candidates) + margin, int(cfg.c_min_max_candidates))
    else:
        c_max_candidates = max(int(cfg.c_max_candidates), int(observed_max_candidates), int(cfg.c_min_max_candidates))
    if bool(cfg.full_soft_widen_enabled) and bool(cfg.agent_c_two_stage_refine_enabled):
        c_max_candidates += max(0, int(cfg.full_soft_widen_c_extra))

    o_global_dim, op_feat_dim, observed_max_ops = infer_o_dims(scan_cases, reset_rule=cfg.reset_rule, quiet=True, use_candidate_set_feat=True)
    o_max_ops = int(observed_max_ops + max(0, int(cfg.o_op_safety_margin))) if bool(cfg.o_auto_max_ops) else max(int(cfg.o_max_ops), int(observed_max_ops))
    if bool(cfg.full_soft_widen_enabled) and bool(cfg.agent_c_two_stage_refine_enabled):
        o_max_ops += max(0, int(cfg.full_soft_widen_o_extra))
    m_global_dim, m_pair_feat_dim_raw = infer_m_dims(scan_cases, reset_rule=cfg.reset_rule, quiet=True)

    c_cfg = AgentCConfig(
        device=cfg.device, hidden_dim=cfg.c_hidden_dim, attn_heads=cfg.c_attn_heads, attn_layers=cfg.c_attn_layers,
        dropout=cfg.c_dropout, lr=cfg.c_lr, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
        clip_ratio=cfg.clip_ratio_c, value_coef=cfg.value_coef_c, ent_coef=cfg.c_ent_coef,
        max_grad_norm=cfg.max_grad_norm, ppo_epochs=cfg.ppo_epochs, minibatch_size=cfg.minibatch_size,
        target_kl=cfg.target_kl_c, max_candidates=c_max_candidates, use_candidate_set_feat=True,
        use_graph_encoder=bool(cfg.c_use_graph_encoder), use_edge_rule_msg=bool(cfg.c_use_edge_rule_msg),
        use_edge_opmch_msg=bool(cfg.c_use_edge_opmch_msg), use_adaptive_edge_gates=bool(cfg.c_use_adaptive_edge_gates),
    )
    o_cfg = AgentOConfig(
        device=cfg.device, hidden_dim=cfg.o_hidden_dim, dropout=cfg.o_dropout, lr=cfg.o_lr,
        gamma=cfg.gamma, gae_lambda=cfg.gae_lambda, clip_ratio=cfg.clip_ratio_o, value_coef=cfg.value_coef_o,
        ent_coef=cfg.o_ent_coef, max_grad_norm=cfg.max_grad_norm, ppo_epochs=cfg.ppo_epochs,
        minibatch_size=cfg.minibatch_size, target_kl=cfg.target_kl_o, max_ops=o_max_ops,
        use_candidate_set_feat=True, overflow_strategy="truncate", model_type=cfg.o_model_type,
    )
    m_cfg = AgentMLearnConfig(
        device=cfg.device, hidden_dim=cfg.m_hidden_dim, dropout=cfg.m_dropout, lr=cfg.m_lr,
        weight_decay=cfg.m_weight_decay, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
        clip_ratio=cfg.clip_ratio_m, value_coef=cfg.value_coef_m, ent_coef=cfg.m_ent_coef,
        max_grad_norm=cfg.max_grad_norm, ppo_epochs=cfg.ppo_epochs, minibatch_size=cfg.minibatch_size,
        target_kl=cfg.target_kl_m, feature_mode=cfg.m_feature_mode, strategy_mode=cfg.m_strategy_mode,
        enable_entropy_backup=bool(cfg.m_enable_entropy_backup), backup_entropy_threshold=cfg.m_backup_entropy_threshold,
        backup_max_extra_pairs=cfg.m_backup_max_extra_pairs,
    )
    agent_c = Agent_C(c_cfg, c_global_dim, pair_feat_dim)
    agent_o = Agent_O(o_cfg, o_global_dim, op_feat_dim)
    agent_m = Agent_M_Learn(m_cfg, m_global_dim, m_pair_feat_dim_raw)
    agent_c.load(str(summary["checkpoint_c"]), strict=False)
    agent_o.load(str(summary["checkpoint_o"]), strict=False)
    _load_agent_m_checkpoint(agent_m, str(summary["checkpoint_m"]), strict=False)
    agent_c.policy.eval(); agent_o.policy.eval(); agent_m.policy.eval()
    return agent_c, agent_o, agent_m


def _raw_feasible_pairs(env) -> List[Tuple[int, int]]:
    state = env._get_runtime_state(batch_idx=0, copy_arrays=False)
    pairs = []
    for job in np.asarray(state["available_jobs"], dtype=np.int64).tolist():
        row = int(job)
        col = int(state["job_col"][row])
        if col >= int(state["num_operation"][row]):
            continue
        op_id = int(state["first_col"][row] + col)
        feasible = env._get_feasible_machine_indices_from_state(state, row=row, col=col)
        for m in feasible.tolist():
            pairs.append((op_id, int(m)))
    return pairs


def _best_ect_pair_from_pairs(obs: Dict[str, object], pairs: List[Tuple[int, int]]) -> Tuple[Tuple[int, int], float]:
    pair_feat = np.asarray(obs.get("pair_feat", []), dtype=np.float32)
    obs_pairs = [tuple(map(int, x)) for x in obs.get("candidate_pairs", [])]
    idx = {p: i for i, p in enumerate(obs_pairs)}
    best_pair = (-1, -1); best_ect = float("inf")
    for p in pairs:
        i = idx.get((int(p[0]), int(p[1])))
        if i is None or pair_feat.ndim != 2 or pair_feat.shape[1] <= 4:
            continue
        ect = float(pair_feat[i, 4])
        if (ect, p[0], p[1]) < (best_ect, best_pair[0], best_pair[1]):
            best_ect = ect; best_pair = (int(p[0]), int(p[1]))
    return best_pair, best_ect


def _best_ect(obs: Dict[str, object]) -> float:
    pf = np.asarray(obs.get("pair_feat", []), dtype=np.float32)
    if pf.ndim == 2 and pf.shape[0] > 0 and pf.shape[1] > 4:
        return float(np.min(pf[:, 4]))
    return float("inf")


def _rank_selected_by_ect(obs: Dict[str, object], pair: Tuple[int, int]) -> int:
    pairs = [tuple(map(int, x)) for x in obs.get("candidate_pairs", [])]
    pf = np.asarray(obs.get("pair_feat", []), dtype=np.float32)
    if pf.ndim != 2 or pf.shape[1] <= 4 or not pairs:
        return -1
    rows = sorted([(float(pf[i, 4]), int(pairs[i][0]), int(pairs[i][1])) for i in range(len(pairs))])
    target = (int(pair[0]), int(pair[1]))
    for r, (_, op, m) in enumerate(rows, 1):
        if (op, m) == target:
            return r
    return -1


def _source_flags(obs: Dict[str, object], pair: Tuple[int, int]) -> Dict[str, int]:
    pairs = [tuple(map(int, x)) for x in obs.get("candidate_pairs", [])]
    sources = obs.get("pair_sources", []) or []
    try:
        i = pairs.index((int(pair[0]), int(pair[1])))
        src = [str(x) for x in sources[i]] if i < len(sources) else []
    except ValueError:
        src = []
    return {
        "selected_source_rule": int(any("_" in s for s in src)),
        "selected_source_critical": int(any("CRITICAL" in s for s in src)),
        "selected_source_explore": int(any("EXPLORE" in s for s in src)),
        "selected_source_fallback": int(any("FALLBACK" in s for s in src)),
    }


def analyze_case(case_path: str, cfg: OCMAPPOConfig, agent_c, agent_o, agent_m) -> Tuple[List[Dict[str, object]], float]:
    env, total_tasks = build_env_for_case(case_path, reset_rule=cfg.reset_rule, quiet=True)
    rows = []
    done = False
    for step in range(int(total_tasks) + int(cfg.extra_step_budget)):
        raw_pairs = _raw_feasible_pairs(env)
        full_obs = env.get_agent_c_obs(batch_idx=0)
        full_pairs = [tuple(map(int, x)) for x in full_obs.get("candidate_pairs", [])]
        raw_best_pair, raw_best_ect = _best_ect_pair_from_pairs(full_obs, [p for p in raw_pairs if p in set(full_pairs)])
        c_full_info = agent_c.select_action(full_obs, deterministic=True)
        dynamic_topk = _compute_dynamic_topk(full_obs=full_obs, c_entropy=float(c_full_info.get("entropy", 0.0)), cfg=cfg)
        o_info = agent_o.select_action(full_obs, deterministic=True, topk_k=dynamic_topk)
        selected_ops = list(dict.fromkeys([int(x) for x in o_info.get("selected_ops", [])])) or [int(o_info["action_op_id"])]
        selected_ops = _expand_ops_for_safety(full_obs=full_obs, selected_ops=selected_ops, min_keep=max(int(cfg.o_topk_min), int(dynamic_topk)))
        selected_pair_indices = _selected_pair_indices_for_ops(full_obs, selected_ops)
        o_filtered_obs = _slice_candidate_obs(full_obs, selected_pair_indices)
        selected_after_fallback = _maybe_uncertainty_fallback(full_obs=full_obs, selected_ops=selected_ops, c_entropy_filtered=0.0, cfg=cfg)
        if len(selected_after_fallback) > len(selected_ops):
            selected_ops = selected_after_fallback
            selected_pair_indices = _selected_pair_indices_for_ops(full_obs, selected_ops)
            o_filtered_obs = _slice_candidate_obs(full_obs, selected_pair_indices)

        pm = agent_m.build_packed_obs(full_obs, selected_ops=selected_ops, pair_indices=selected_pair_indices)
        if len(pm["candidate_pairs"]) == 0:
            keep_full_indices = []
        else:
            m_info = agent_m.act(pm, deterministic=True)
            full_idx_map = np.asarray(pm["full_pair_indices"], dtype=np.int64)
            keep_full_indices = [int(full_idx_map[i]) for i in m_info.get("keep_indices", []) if 0 <= int(i) < full_idx_map.shape[0]]
        for pair in [(int(c_full_info["op_id"]), int(c_full_info["mch_id"]))]:
            idx = _find_pair_index(full_obs, pair[0], pair[1])
            if idx >= 0:
                keep_full_indices.append(int(idx))
        keep_full_indices = _expand_pairs_for_safety(
            full_obs=full_obs,
            selected_ops=selected_ops,
            keep_pair_indices=keep_full_indices,
            min_pairs_total=int(getattr(cfg, "m_safety_min_total_pairs", 3)),
            per_op_min_machines=int(getattr(cfg, "m_safety_min_machines_per_op", 3)),
        )
        if not keep_full_indices:
            keep_full_indices = [int(np.argmin(np.asarray(full_obs["pair_feat"])[:, 4]))]
        keep_full_indices = sorted(set(keep_full_indices))
        final_obs = _slice_candidate_obs(full_obs, keep_full_indices)
        c_info = agent_c.select_action(final_obs, deterministic=True)
        c_info = _refine_c_action_deterministic_topk(agent_c=agent_c, obs=final_obs, c_info=c_info, cfg=cfg)
        selected_pair = (int(c_info["op_id"]), int(c_info["mch_id"]))

        o_pairs = [tuple(map(int, x)) for x in o_filtered_obs.get("candidate_pairs", [])]
        final_pairs = [tuple(map(int, x)) for x in final_obs.get("candidate_pairs", [])]
        full_best = tuple(map(int, _best_pair_by_ect(full_obs)))
        best_full_ect = _best_ect(full_obs)
        best_o_ect = _best_ect(o_filtered_obs)
        best_final_ect = _best_ect(final_obs)
        stats = getattr(env, "_last_candidate_refine_stats", {}) or {}
        row = {
            "case_name": Path(case_path).name,
            "step": step,
            "raw_feasible_pairs": len(raw_pairs),
            "full_candidate_pairs": len(full_pairs),
            "o_selected_ops": len(set(selected_ops)),
            "o_filtered_pairs": len(o_pairs),
            "m_kept_pairs": len(keep_full_indices),
            "final_candidate_pairs": len(final_pairs),
            "compression_full_vs_raw": len(full_pairs) / max(1, len(raw_pairs)),
            "compression_o_vs_raw": len(o_pairs) / max(1, len(raw_pairs)),
            "compression_final_vs_raw": len(final_pairs) / max(1, len(raw_pairs)),
            "best_ect_full": best_full_ect,
            "best_ect_o": best_o_ect,
            "best_ect_final": best_final_ect,
            "best_ect_gap_o": max(0.0, (best_o_ect - best_full_ect) / max(abs(best_full_ect), 1e-6)) if np.isfinite(best_o_ect) and np.isfinite(best_full_ect) else 0.0,
            "best_ect_gap_final": max(0.0, (best_final_ect - best_full_ect) / max(abs(best_full_ect), 1e-6)) if np.isfinite(best_final_ect) and np.isfinite(best_full_ect) else 0.0,
            "best_pair_hit_o": int(full_best in set(o_pairs)),
            "best_pair_hit_final": int(full_best in set(final_pairs)),
            "selected_pair_op": selected_pair[0],
            "selected_pair_mch": selected_pair[1],
            "selected_pair_ect_rank_final": _rank_selected_by_ect(final_obs, selected_pair),
            "selected_is_best_ect_final": int(_rank_selected_by_ect(final_obs, selected_pair) == 1),
            "refine_original_candidate_count": float(stats.get("original_candidate_count", 0.0)),
            "refine_refined_candidate_count": float(stats.get("refined_candidate_count", 0.0)),
            "refine_source_rule_pairs": float(stats.get("source_rule_pairs", 0.0)),
            "refine_source_critical_pairs": float(stats.get("source_critical_pairs", 0.0)),
            "refine_source_explore_pairs": float(stats.get("source_explore_pairs", 0.0)),
        }
        row.update(_source_flags(final_obs, selected_pair))
        rows.append(row)
        out = env.step_with_pair(selected_pair[0], selected_pair[1], batch_idx=0)
        done = bool(out[3][0])
        if done:
            break
    return rows, float(env.LBm[0].max())


def summarize(rows: List[Dict[str, object]], case_makespans: Dict[str, float]) -> Dict[str, object]:
    def mean(key: str) -> float:
        vals = [float(r[key]) for r in rows if key in r]
        return float(np.mean(vals)) if vals else 0.0
    def rate(key: str) -> float:
        vals = [float(r[key]) for r in rows if key in r]
        return float(np.mean(vals)) if vals else 0.0
    return {
        "num_cases": len(case_makespans),
        "num_steps": len(rows),
        "mean_makespan": float(np.mean(list(case_makespans.values()))) if case_makespans else 0.0,
        "mean_raw_feasible_pairs": mean("raw_feasible_pairs"),
        "mean_full_candidate_pairs": mean("full_candidate_pairs"),
        "mean_o_filtered_pairs": mean("o_filtered_pairs"),
        "mean_final_candidate_pairs": mean("final_candidate_pairs"),
        "mean_compression_full_vs_raw": mean("compression_full_vs_raw"),
        "mean_compression_o_vs_raw": mean("compression_o_vs_raw"),
        "mean_compression_final_vs_raw": mean("compression_final_vs_raw"),
        "best_pair_hit_o_rate": rate("best_pair_hit_o"),
        "best_pair_hit_final_rate": rate("best_pair_hit_final"),
        "mean_best_ect_gap_o": mean("best_ect_gap_o"),
        "mean_best_ect_gap_final": mean("best_ect_gap_final"),
        "selected_is_best_ect_final_rate": rate("selected_is_best_ect_final"),
        "mean_selected_pair_ect_rank_final": mean("selected_pair_ect_rank_final"),
        "selected_source_rule_rate": rate("selected_source_rule"),
        "selected_source_critical_rate": rate("selected_source_critical"),
        "selected_source_explore_rate": rate("selected_source_explore"),
        "case_makespans": case_makespans,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="Path to train_oc_mappo_summary.json for checkpoint/config")
    ap.add_argument("--split-json", default="", help="Optional eval split override")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-cases", type=int, default=0)
    args = ap.parse_args()

    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cfg = _cfg_from_summary(summary_path, args.split_json, args.device, args.max_cases or None)
    _bridge_env_config(cfg)
    seed_everything(int(cfg.seed))
    cases = _collect_cases(cfg, args.max_cases or None)
    if not cases:
        raise RuntimeError("No eval cases found")
    print(f"[candidate-analysis] cases={len(cases)} device={cfg.device} split={cfg.split_source}", flush=True)
    agent_c, agent_o, agent_m = _make_agents(cfg, cases, summary)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, object]] = []
    case_makespans: Dict[str, float] = {}
    for i, case in enumerate(cases, 1):
        rows, ms = analyze_case(case, cfg, agent_c, agent_o, agent_m)
        all_rows.extend(rows)
        case_makespans[Path(case).name] = ms
        print(f"[{i}/{len(cases)}] {Path(case).name} steps={len(rows)} makespan={ms:.4f}", flush=True)

    csv_path = out_dir / "candidate_quality_steps.csv"
    if all_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader(); writer.writerows(all_rows)
    summary_out = summarize(all_rows, case_makespans)
    summary_out["source_summary"] = str(summary_path)
    summary_out["split_source"] = str(cfg.split_source)
    (out_dir / "candidate_quality_summary.json").write_text(json.dumps(summary_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary_out, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

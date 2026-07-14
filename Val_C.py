from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm.auto import tqdm

from Agent_C import Agent_C, StepRecord


def _aggregate_results(results: List[Dict[str, object]], errors: List[Dict[str, str]], split_name: str) -> Dict[str, object]:
    total_cases = len(results) + len(errors)
    if len(results) == 0:
        return {
            "split": split_name,
            "results": [],
            "errors": errors,
            "num_cases": 0,
            "num_errors": len(errors),
            "total_requested_cases": int(total_cases),
            "success_rate": 0.0,
            "mean_reward": 0.0,
            "mean_raw_reward": 0.0,
            "mean_makespan": float("inf"),
            "mean_steps": 0.0,
            "mean_candidate_count_last": 0.0,
            "mean_max_original_candidate_count": 0.0,
            "mean_num_truncated_steps": 0.0,
            "done_ratio": 0.0,
        }

    return {
        "split": split_name,
        "results": results,
        "errors": errors,
        "num_cases": len(results),
        "num_errors": len(errors),
        "total_requested_cases": int(total_cases),
        "success_rate": float(len(results) / max(total_cases, 1)),
        "mean_reward": float(np.mean([item["episode_reward"] for item in results])),
        "mean_raw_reward": float(np.mean([item["episode_raw_reward"] for item in results])),
        "mean_makespan": float(np.mean([item["makespan"] for item in results])),
        "mean_steps": float(np.mean([item["steps"] for item in results])),
        "mean_candidate_count_last": float(np.mean([item["candidate_count_last"] for item in results])),
        "mean_max_original_candidate_count": float(np.mean([item.get("max_original_candidate_count", item["candidate_count_last"]) for item in results])),
        "mean_num_truncated_steps": float(np.mean([item.get("num_truncated_steps", 0) for item in results])),
        "done_ratio": float(np.mean([float(item["done"]) for item in results])),
    }


def evaluate_actor(
    actor,
    case_paths: List[str],
    reset_rule: str,
    split_name: str,
    reward_normalize: bool = True,
    deterministic: bool = True,
    quiet: bool = True,
    extra_step_budget: int = 40,
    collect_records: bool = False,
    show_progress: bool = False,
) -> Dict[str, object]:
    from Train_C import run_episode

    results = []
    errors = []

    case_iter = tqdm(case_paths, desc=f"eval:{split_name}", leave=False) if show_progress else case_paths
    for case_path in case_iter:
        try:
            result = run_episode(
                actor,
                case_path=case_path,
                reset_rule=reset_rule,
                reward_normalize=reward_normalize,
                deterministic=deterministic,
                quiet=quiet,
                extra_step_budget=extra_step_budget,
                collect_records=collect_records,
            )
            results.append(result)
        except Exception as e:
            errors.append({"case": str(case_path), "reason": repr(e)})

    return _aggregate_results(results, errors, split_name=split_name)


def evaluate_rule_baselines(
    case_paths: List[str],
    reset_rule: str,
    split_name: str,
    reward_normalize: bool = True,
    quiet: bool = True,
    extra_step_budget: int = 40,
    baseline_rules: Optional[List[str]] = None,
    show_progress: bool = False,
):
    from Train_C import RuleBasedCandidateAgent

    baseline_rules = baseline_rules or [
        "FIFO_SPT",
        "FIFO_EET",
        "LWKR_SPT",
        "LWKR_EET",
        "MWKR_SPT",
        "MWKR_EET",
    ]
    baseline_scores = {}

    rule_iter = tqdm(baseline_rules, desc=f"baseline:{split_name}", leave=False) if show_progress else baseline_rules
    for rule_name in rule_iter:
        baseline_agent = RuleBasedCandidateAgent(preferred_rule=rule_name)
        baseline_scores[rule_name] = evaluate_actor(
            baseline_agent,
            case_paths=case_paths,
            split_name=split_name,
            reset_rule=reset_rule,
            reward_normalize=reward_normalize,
            deterministic=True,
            quiet=quiet,
            extra_step_budget=extra_step_budget,
            collect_records=False,
            show_progress=show_progress,
        )

    return baseline_scores


def _extract_case_makespan_map(eval_log: Dict[str, object]) -> Dict[str, float]:
    mapping = {}
    for item in eval_log.get("results", []):
        mapping[item["case"]] = float(item["makespan"])
    return mapping


def build_casewise_comparison(agent_eval: Dict[str, object], baseline_eval: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    agent_map = _extract_case_makespan_map(agent_eval)
    baseline_maps = {rule: _extract_case_makespan_map(log) for rule, log in baseline_eval.items()}

    all_cases = sorted(agent_map.keys())
    for case_name in all_cases:
        row = {
            "case": case_name,
            "agent_c_makespan": agent_map.get(case_name, np.nan),
        }
        baseline_values = []
        for rule, cmap in baseline_maps.items():
            v = cmap.get(case_name, np.nan)
            row[f"{rule}_makespan"] = v
            if np.isfinite(v):
                baseline_values.append((rule, v))
        if baseline_values:
            best_rule, best_val = min(baseline_values, key=lambda x: x[1])
            row["best_rule"] = best_rule
            row["best_rule_makespan"] = best_val
            row["agent_minus_best_rule"] = row["agent_c_makespan"] - best_val
        rows.append(row)
    return rows


def summarize_baseline_gap(agent_eval_log: Dict[str, object], baseline_eval: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    out = {}
    agent_ms = float(agent_eval_log.get("mean_makespan", np.inf))
    for rule_name, log in baseline_eval.items():
        base_ms = float(log.get("mean_makespan", np.inf))
        gap_pct = None
        if np.isfinite(agent_ms) and np.isfinite(base_ms) and abs(base_ms) > 1e-8:
            gap_pct = (agent_ms - base_ms) / base_ms * 100.0
        out[rule_name] = {
            "agent_mean_makespan": agent_ms,
            "baseline_mean_makespan": base_ms,
            "gap_pct": gap_pct,
        }
    return out

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import io
import json
import math
import random
import shutil
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []

from Val_C import (
    evaluate_actor,
    evaluate_rule_baselines,
    build_casewise_comparison,
    summarize_baseline_gap,
)
from FJSP_Env_Agent import FJSP
from DataRead import getdata
from Agent_C import (
    AgentCConfig,
    Agent_C,
    RolloutBuffer,
    StepRecord,
    _flatten_f32,
)

# 如果 Agent_C.py 里提供了这个函数，就用；没有也不报错
try:
    from Agent_C import _apply_topk_settings_to_agent_config_from_train_cfg
except Exception:
    _apply_topk_settings_to_agent_config_from_train_cfg = None


if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float


_CASE_INPUT_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray, int, int, int]] = {}
_CASE_ENV_CACHE: Dict[str, FJSP] = {}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_jsonable(obj: Any):
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if dataclasses.is_dataclass(obj):
        return _to_jsonable(dataclasses.asdict(obj))
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return obj.item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return _to_jsonable(vars(obj))
    return str(obj)


def call_silently(func, *args, quiet: bool = True, **kwargs):
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            return func(*args, **kwargs)
    return func(*args, **kwargs)


def build_env_inputs_from_fjs(fjs_path: Path, batch_size: int = 1):
    data_dict = getdata(str(fjs_path))

    n_j = int(data_dict["n"])
    n_m = int(data_dict["m"])

    num_operation = [data_dict["OJ"][job][-1] for job in data_dict["J"]]
    max_operation = int(np.max(num_operation))

    time_window = np.zeros((n_j, max_operation, n_m), dtype=np.float32)

    for job in range(1, n_j + 1):
        for op in data_dict["OJ"][job]:
            for mch in data_dict["operations_machines"][(job, op)]:
                time_window[job - 1, op - 1, mch - 1] = data_dict["operations_times"][(job, op, mch)]

    each_job_num_operation = np.array([num_operation], dtype=np.int32)
    data = np.repeat(time_window[None, ...], repeats=batch_size, axis=0)
    return data, each_job_num_operation, n_j, n_m


@dataclass
class TrainConfig:
    train_cases: List[str]
    val_cases: List[str]
    test_cases: List[str]

    dataset_name: str = "1_Brandimarte"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    reset_rule: str = "FIFO_SPT"

    epochs: int = 120
    episodes_per_update: int = 8
    eval_interval: int = 1
    run_test_on_every_eval: bool = False

    save_dir: str = "checkpoints_agent_c_aligned_v2"
    checkpoint_name: str = "agent_c_aligned_v2.pt"
    quiet_env: bool = True
    deterministic_eval: bool = True
    extra_step_budget: int = 40
    reward_normalize: bool = True
    use_candidate_set_feat: bool = True

    baseline_rules: List[str] = field(
        default_factory=lambda: [
            "FIFO_SPT",
            "FIFO_EET",
            "LWKR_SPT",
            "LWKR_EET",
            "MWKR_SPT",
            "MWKR_EET",
        ]
    )

    hidden_dim: int = 128
    attn_heads: int = 4
    attn_layers: int = 1
    dropout: float = 0.0
    lr: float = 2e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    ent_coef: float = 0.005
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 64
    target_kl: Optional[float] = None

    max_candidates: int = 16
    auto_max_candidates: bool = True
    candidate_scan_rule: str = "FIFO_SPT"
    candidate_safety_margin: int = 4
    round_max_candidates_to_power_of_two: bool = True

    train_full_eval_interval: int = 3
    baseline_cache_enabled: bool = True
    evaluate_rule_baselines_on_cache_miss: bool = True
    rollout_parallel_envs: int = 1
    eval_parallel_envs: int = 1
    val_patience: int = 0
    min_epochs_before_early_stop: int = 1
    val_improve_epsilon: float = 1e-9
    best_last_n_epochs: int = 20
    plot_checkpoint_variant: str = "best_overall"

    # 可选 Top-K 字段，Train_C 只透传，不在这里实现 patch
    topk_prefilter_enabled: bool = False
    topk_k: int = 16
    topk_keep_eet: int = 2
    topk_keep_pt: int = 2
    topk_keep_preferred_rule: int = 2
    topk_preferred_rule: str = "FIFO_SPT"
    topk_machine_diversity: bool = True
    topk_score_weights: Dict[str, float] = field(
        default_factory=lambda: {"eet": 1.0, "pt": 0.3, "queue": 0.1, "slack": 0.2}
    )
    use_graph_encoder: bool = False
    use_hetero_lite: bool = False
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
    init_checkpoint_path: Optional[str] = None
    init_checkpoint_strict: bool = True


class RuleBasedCandidateAgent:
    def __init__(self, preferred_rule: str = "FIFO_SPT"):
        self.preferred_rule = preferred_rule

    def act(self, obs: Dict[str, object], deterministic: bool = True):
        del deterministic
        candidate_pairs = list(obs["candidate_pairs"])
        pair_sources = list(obs.get("pair_sources", []))
        pair_feat = np.asarray(obs["pair_feat"], dtype=np.float32)

        if len(candidate_pairs) == 0:
            raise RuntimeError("No candidate pairs available for rule baseline.")

        preferred_indices = []
        for idx, sources in enumerate(pair_sources):
            if self.preferred_rule in set(sources):
                preferred_indices.append(idx)

        choose_from = preferred_indices if preferred_indices else list(range(len(candidate_pairs)))
        best_idx = min(
            choose_from,
            key=lambda i: (
                pair_feat[i, 4],
                pair_feat[i, 0],
                candidate_pairs[i][0],
                candidate_pairs[i][1],
            ),
        )
        op_id, mch_id = candidate_pairs[best_idx]
        return int(op_id), int(mch_id), {
            "action_idx": int(best_idx),
            "op_id": int(op_id),
            "mch_id": int(mch_id),
            "log_prob": 0.0,
            "value": 0.0,
            "packed_obs": None,
        }


def _get_reward_scale(env: FJSP) -> float:
    return 1.0


def build_env_for_case(
    case_path: str,
    reset_rule: str,
    quiet: bool = True,
    reuse_env: bool = True,
) -> Tuple[FJSP, int]:
    case_key = str(Path(case_path).resolve())
    cached = _CASE_INPUT_CACHE.get(case_key)

    if cached is None:
        data, each_job_num_operation, n_j, n_m = build_env_inputs_from_fjs(Path(case_path), batch_size=1)
        total_tasks = int(each_job_num_operation.sum())
        cached = (data, each_job_num_operation, int(n_j), int(n_m), int(total_tasks))
        _CASE_INPUT_CACHE[case_key] = cached

    data, each_job_num_operation, n_j, n_m, total_tasks = cached

    if reuse_env:
        env = _CASE_ENV_CACHE.get(case_key)
        if env is None:
            env = FJSP(n_j, n_m, each_job_num_operation)
            _CASE_ENV_CACHE[case_key] = env
    else:
        env = FJSP(n_j, n_m, each_job_num_operation)

    call_silently(env.reset, data, rule=reset_rule, quiet=quiet)
    return env, int(total_tasks)


def _round_up_power_of_two(x: int) -> int:
    x = int(max(1, x))
    return 1 << (x - 1).bit_length()


def scan_case_runtime_stats(
    case_path: str,
    reset_rule: str,
    use_candidate_set_feat: bool,
    quiet: bool = True,
    extra_step_budget: int = 40,
    scan_rule: str = "FIFO_SPT",
) -> Dict[str, object]:
    env, total_tasks = build_env_for_case(case_path, reset_rule=reset_rule, quiet=quiet)
    scan_actor = RuleBasedCandidateAgent(preferred_rule=scan_rule)

    max_candidates = 0
    max_global_dim = 0
    pair_feat_dim = 0
    steps = 0
    done = False
    errors: List[str] = []

    for _ in range(int(total_tasks) + int(extra_step_budget)):
        try:
            obs = env.get_agent_c_obs(batch_idx=0)
        except Exception as e:
            errors.append(f"get_obs_failed: {repr(e)}")
            break

        candidate_pairs = list(obs.get("candidate_pairs", []))
        if len(candidate_pairs) == 0:
            errors.append("empty_candidate_pairs")
            break

        pair_feat = np.asarray(obs["pair_feat"], dtype=np.float32)
        gfeat = _flatten_f32(obs["global_feat"])
        if use_candidate_set_feat and "candidate_set_feat" in obs:
            gfeat = np.concatenate([gfeat, _flatten_f32(obs["candidate_set_feat"])], axis=0)

        max_candidates = max(max_candidates, int(len(candidate_pairs)))
        max_global_dim = max(max_global_dim, int(gfeat.shape[0]))
        if pair_feat.ndim == 2:
            pair_feat_dim = max(pair_feat_dim, int(pair_feat.shape[1]))

        try:
            op_id, mch_id, _ = scan_actor.act(obs, deterministic=True)
            step_out = call_silently(env.step_with_pair, op_id, mch_id, batch_idx=0, quiet=quiet)
            done = bool(step_out[3][0])
            steps += 1
            if done:
                break
        except Exception as e:
            errors.append(f"step_failed: {repr(e)}")
            break

    return {
        "case": Path(case_path).name,
        "max_candidates": int(max_candidates),
        "max_global_dim": int(max_global_dim),
        "pair_feat_dim": int(pair_feat_dim),
        "steps": int(steps),
        "done": bool(done),
        "errors": errors,
    }


def infer_training_dims(
    case_paths: List[str],
    reset_rule: str,
    use_candidate_set_feat: bool,
    quiet: bool = True,
    extra_step_budget: int = 40,
    scan_rule: str = "FIFO_SPT",
    candidate_safety_margin: int = 4,
    round_max_candidates_to_power_of_two: bool = True,
) -> Tuple[int, int, int, Dict[str, object]]:
    max_global_dim = 0
    pair_feat_dim = 0
    observed_max_candidates = 1
    per_case_stats = []

    for case_path in case_paths:
        stats = scan_case_runtime_stats(
            case_path=case_path,
            reset_rule=reset_rule,
            use_candidate_set_feat=use_candidate_set_feat,
            quiet=quiet,
            extra_step_budget=extra_step_budget,
            scan_rule=scan_rule,
        )
        per_case_stats.append(stats)
        max_global_dim = max(max_global_dim, int(stats["max_global_dim"]))
        pair_feat_dim = max(pair_feat_dim, int(stats["pair_feat_dim"]))
        observed_max_candidates = max(observed_max_candidates, int(stats["max_candidates"]))

    recommended = int(observed_max_candidates) + int(max(0, candidate_safety_margin))
    if round_max_candidates_to_power_of_two:
        recommended = _round_up_power_of_two(recommended)

    report = {
        "observed_max_candidates": int(observed_max_candidates),
        "recommended_max_candidates": int(recommended),
        "candidate_safety_margin": int(max(0, candidate_safety_margin)),
        "round_max_candidates_to_power_of_two": bool(round_max_candidates_to_power_of_two),
        "num_scan_errors": int(sum(1 for x in per_case_stats if len(x["errors"]) > 0)),
        "per_case_stats": per_case_stats,
    }
    return max_global_dim, pair_feat_dim, observed_max_candidates, report


def run_episode(
    actor,
    case_path: str,
    reset_rule: str,
    reward_normalize: bool = True,
    deterministic: bool = False,
    quiet: bool = True,
    extra_step_budget: int = 40,
    collect_records: bool = True,
) -> Dict[str, object]:
    env, total_tasks = build_env_for_case(case_path, reset_rule=reset_rule, quiet=quiet)
    reward_scale = _get_reward_scale(env) if reward_normalize else 1.0

    episode_steps: List[StepRecord] = []
    episode_reward = 0.0
    episode_raw_reward = 0.0
    max_steps = int(total_tasks) + int(extra_step_budget)
    done = False
    last_obs = None
    executed_steps = 0
    max_original_candidate_count = 0
    max_used_candidate_count = 0
    num_truncated_steps = 0
    total_overflow_candidates = 0

    for _ in range(max_steps):
        obs = env.get_agent_c_obs(batch_idx=0)
        last_obs = obs
        max_original_candidate_count = max(max_original_candidate_count, int(len(obs.get("candidate_pairs", []))))

        op_id, mch_id, info = actor.act(obs, deterministic=deterministic)
        out = call_silently(env.step_with_pair, op_id, mch_id, batch_idx=0, quiet=quiet)

        raw_reward = float(out[2][0])
        reward = raw_reward / max(reward_scale, 1e-8)
        done = bool(out[3][0])

        episode_reward += reward
        episode_raw_reward += raw_reward
        executed_steps += 1

        if isinstance(actor, Agent_C) and collect_records:
            packed_obs = info["packed_obs"]
            max_used_candidate_count = max(max_used_candidate_count, int(packed_obs.get("used_candidate_count", 0)))
            if info.get("was_truncated", False):
                num_truncated_steps += 1
                total_overflow_candidates += int(info.get("overflow_count", 0))
            episode_steps.append(
                StepRecord(
                    global_feat=packed_obs["global_feat"],
                    pair_feat=packed_obs["pair_feat"],
                    pair_mask=packed_obs["pair_mask"],
                    action_idx=int(info["action_idx"]),
                    log_prob=float(info["log_prob"]),
                    value=float(info["value"]),
                    reward=reward,
                    done=done,
                    op_node_feat=packed_obs.get("op_node_feat", None),
                    op_adj=packed_obs.get("op_adj", None),
                    machine_node_feat=packed_obs.get("machine_node_feat", None),
                    pair_op_idx=packed_obs.get("pair_op_idx", None),
                    pair_mch_idx=packed_obs.get("pair_mch_idx", None),
                    edge_rule_to_pair_feat=packed_obs.get("edge_rule_to_pair_feat", None),
                    edge_opmch_to_pair_feat=packed_obs.get("edge_opmch_to_pair_feat", None),
                )
            )

        if done:
            break

    makespan = float(env.LBm[0].max())
    candidate_count_last = int(len(last_obs["candidate_pairs"])) if last_obs is not None else 0

    return {
        "case": Path(case_path).name,
        "episode_reward": episode_reward,
        "episode_raw_reward": episode_raw_reward,
        "reward_scale": reward_scale,
        "makespan": makespan,
        "steps": executed_steps,
        "done": done,
        "records": episode_steps if collect_records else [],
        "candidate_count_last": candidate_count_last,
        "max_original_candidate_count": int(max_original_candidate_count),
        "max_used_candidate_count": int(max_used_candidate_count),
        "num_truncated_steps": int(num_truncated_steps),
        "total_overflow_candidates": int(total_overflow_candidates),
    }


def _split_summary(cfg: TrainConfig) -> Dict[str, object]:
    total = len(cfg.train_cases) + len(cfg.val_cases) + len(cfg.test_cases)
    return {
        "train_count": len(cfg.train_cases),
        "val_count": len(cfg.val_cases),
        "test_count": len(cfg.test_cases),
        "total_count": total,
        "train_ratio": len(cfg.train_cases) / total if total else 0.0,
        "val_ratio": len(cfg.val_cases) / total if total else 0.0,
        "test_ratio": len(cfg.test_cases) / total if total else 0.0,
    }


def print_split_summary(cfg: TrainConfig) -> None:
    s = _split_summary(cfg)
    print(
        f"数据集划分: train={s['train_count']} ({s['train_ratio']:.1%}), "
        f"val={s['val_count']} ({s['val_ratio']:.1%}), "
        f"test={s['test_count']} ({s['test_ratio']:.1%}), total={s['total_count']}"
    )
    print(
        f"训练覆盖策略: 每个 epoch 无放回遍历全部 train_cases，"
        f"按 episodes_per_update={cfg.episodes_per_update} 分批做 PPO 更新。"
    )


def iter_train_batches(train_cases: List[str], batch_size: int, seed: int, epoch_idx: int):
    rng = random.Random(seed + epoch_idx)
    shuffled = list(train_cases)
    rng.shuffle(shuffled)
    for start in range(0, len(shuffled), batch_size):
        yield shuffled[start:start + batch_size]


def export_history_csv(history: List[Dict[str, object]], csv_path: Path) -> None:
    if not history:
        return
    all_keys = sorted({k for row in history for k in row.keys() if not isinstance(row.get(k), (list, dict))})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for row in history:
            flat_row = {k: v for k, v in row.items() if k in all_keys}
            writer.writerow(flat_row)


def export_casewise_csv(casewise_rows: List[Dict[str, object]], csv_path: Path) -> None:
    if not casewise_rows:
        return
    keys = sorted({k for row in casewise_rows for k in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(casewise_rows)


def _baseline_cache_path(save_dir: Path, split_name: str) -> Path:
    return save_dir / f"baseline_{split_name}_cache.json"


def _baseline_cache_signature(cfg: TrainConfig, case_paths: List[str], split_name: str) -> Dict[str, object]:
    return {
        "split": split_name,
        "reset_rule": cfg.reset_rule,
        "reward_normalize": bool(cfg.reward_normalize),
        "extra_step_budget": int(cfg.extra_step_budget),
        "baseline_rules": list(cfg.baseline_rules),
        "case_paths": [str(x) for x in case_paths],
    }


def _load_cached_baseline_eval(save_dir: Path, cfg: TrainConfig, case_paths: List[str], split_name: str):
    cache_path = _baseline_cache_path(save_dir, split_name)
    if not cache_path.exists():
        return None

    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if payload.get("signature") != _baseline_cache_signature(cfg, case_paths, split_name):
        return None
    return payload.get("eval")


def _save_cached_baseline_eval(
    save_dir: Path,
    cfg: TrainConfig,
    case_paths: List[str],
    split_name: str,
    eval_payload,
) -> None:
    cache_path = _baseline_cache_path(save_dir, split_name)
    payload = {
        "signature": _baseline_cache_signature(cfg, case_paths, split_name),
        "eval": _to_jsonable(eval_payload),
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_train_full_eval_interval(cfg: TrainConfig) -> int:
    return max(1, int(getattr(cfg, "train_full_eval_interval", 3)))


def _resolve_baseline_cache_enabled(cfg: TrainConfig) -> bool:
    return bool(getattr(cfg, "baseline_cache_enabled", True))


def _resolve_eval_baselines_on_cache_miss(cfg: TrainConfig) -> bool:
    return bool(getattr(cfg, "evaluate_rule_baselines_on_cache_miss", True))


def _infer_structural_global_dim(case_paths: List[str], use_candidate_set_feat: bool) -> int:
    max_dim = 1
    for case_path in case_paths:
        _, _, n_j, n_m = build_env_inputs_from_fjs(Path(case_path), batch_size=1)
        base_dim = int(n_m) + 2 * int(n_j) + 12
        if use_candidate_set_feat:
            base_dim += 3
        max_dim = max(max_dim, base_dim)
    return int(max_dim)


def _chunk_list(items, chunk_size: int):
    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


def _aggregate_results(results: List[Dict[str, object]], errors: List[Dict[str, object]], split_name: str) -> Dict[str, object]:
    num_cases = len(results) + len(errors)
    if len(results) == 0:
        return {
            "split_name": split_name,
            "num_cases": int(num_cases),
            "num_success": 0,
            "num_errors": int(len(errors)),
            "success_rate": 0.0,
            "mean_reward": 0.0,
            "mean_raw_reward": 0.0,
            "mean_makespan": float("inf"),
            "mean_steps": 0.0,
            "done_ratio": 0.0,
            "results": results,
            "errors": errors,
        }

    mean_reward = float(np.mean([x["episode_reward"] for x in results]))
    mean_raw_reward = float(np.mean([x["episode_raw_reward"] for x in results]))
    mean_makespan = float(np.mean([x["makespan"] for x in results]))
    mean_steps = float(np.mean([x["steps"] for x in results]))
    done_ratio = float(np.mean([float(x["done"]) for x in results]))
    success_rate = float(len(results) / max(1, num_cases))

    return {
        "split_name": split_name,
        "num_cases": int(num_cases),
        "num_success": int(len(results)),
        "num_errors": int(len(errors)),
        "success_rate": success_rate,
        "mean_reward": mean_reward,
        "mean_raw_reward": mean_raw_reward,
        "mean_makespan": mean_makespan,
        "mean_steps": mean_steps,
        "done_ratio": done_ratio,
        "results": results,
        "errors": errors,
    }


def run_episodes_batch(
    actor,
    case_paths: List[str],
    reset_rule: str,
    reward_normalize: bool = True,
    deterministic: bool = False,
    quiet: bool = True,
    extra_step_budget: int = 40,
    collect_records: bool = True,
) -> Dict[str, object]:
    env_entries = []
    errors = []

    for case_path in case_paths:
        try:
            env, total_tasks = build_env_for_case(case_path, reset_rule=reset_rule, quiet=quiet)
        except Exception as e:
            errors.append({"case": str(case_path), "reason": repr(e)})
            continue

        reward_scale = _get_reward_scale(env) if reward_normalize else 1.0
        env_entries.append(
            {
                "case_path": str(case_path),
                "env": env,
                "total_tasks": int(total_tasks),
                "reward_scale": float(reward_scale),
                "max_steps": int(total_tasks) + int(extra_step_budget),
                "steps": 0,
                "done": False,
                "failed": False,
                "episode_reward": 0.0,
                "episode_raw_reward": 0.0,
                "records": [],
                "last_obs": None,
                "max_original_candidate_count": 0,
                "max_used_candidate_count": 0,
                "num_truncated_steps": 0,
                "total_overflow_candidates": 0,
            }
        )

    while True:
        active_indices = []
        obs_batch = []

        for idx, entry in enumerate(env_entries):
            if entry["failed"] or entry["done"]:
                continue
            if entry["steps"] >= entry["max_steps"]:
                continue
            try:
                obs = entry["env"].get_agent_c_obs(batch_idx=0)
            except Exception as e:
                entry["failed"] = True
                errors.append({"case": entry["case_path"], "reason": repr(e)})
                continue

            entry["last_obs"] = obs
            entry["max_original_candidate_count"] = max(
                int(entry["max_original_candidate_count"]),
                int(len(obs.get("candidate_pairs", []))),
            )
            active_indices.append(idx)
            obs_batch.append(obs)

        if len(active_indices) == 0:
            break

        if isinstance(actor, Agent_C):
            infos = actor.select_action_batch(obs_batch, deterministic=deterministic)
        else:
            infos = []
            for obs in obs_batch:
                op_id, mch_id, info = actor.act(obs, deterministic=deterministic)
                info = dict(info)
                info["op_id"] = int(op_id)
                info["mch_id"] = int(mch_id)
                infos.append(info)

        for entry_idx, info in zip(active_indices, infos):
            entry = env_entries[entry_idx]
            try:
                out = call_silently(
                    entry["env"].step_with_pair,
                    int(info["op_id"]),
                    int(info["mch_id"]),
                    batch_idx=0,
                    quiet=quiet,
                )
            except Exception as e:
                entry["failed"] = True
                errors.append({"case": entry["case_path"], "reason": repr(e)})
                continue

            raw_reward = float(out[2][0])
            reward = raw_reward / max(float(entry["reward_scale"]), 1e-8)
            done = bool(out[3][0])

            entry["episode_reward"] += reward
            entry["episode_raw_reward"] += raw_reward
            entry["steps"] += 1
            entry["done"] = done

            if isinstance(actor, Agent_C) and collect_records:
                packed_obs = info["packed_obs"]
                entry["max_used_candidate_count"] = max(
                    int(entry["max_used_candidate_count"]),
                    int(packed_obs.get("used_candidate_count", 0)),
                )
                if info.get("was_truncated", False):
                    entry["num_truncated_steps"] += 1
                    entry["total_overflow_candidates"] += int(info.get("overflow_count", 0))
                entry["records"].append(
                    StepRecord(
                        global_feat=packed_obs["global_feat"],
                        pair_feat=packed_obs["pair_feat"],
                        pair_mask=packed_obs["pair_mask"],
                        action_idx=int(info["action_idx"]),
                        log_prob=float(info.get("log_prob", 0.0)),
                        value=float(info.get("value", 0.0)),
                        reward=reward,
                        done=done,
                        op_node_feat=packed_obs.get("op_node_feat", None),
                        op_adj=packed_obs.get("op_adj", None),
                        machine_node_feat=packed_obs.get("machine_node_feat", None),
                        pair_op_idx=packed_obs.get("pair_op_idx", None),
                        pair_mch_idx=packed_obs.get("pair_mch_idx", None),
                        edge_rule_to_pair_feat=packed_obs.get("edge_rule_to_pair_feat", None),
                        edge_opmch_to_pair_feat=packed_obs.get("edge_opmch_to_pair_feat", None),
                    )
                )

    results = []
    for entry in env_entries:
        if entry["failed"]:
            continue
        last_obs = entry["last_obs"]
        candidate_count_last = int(len(last_obs["candidate_pairs"])) if last_obs is not None else 0
        makespan = float(entry["env"].LBm[0].max())
        results.append(
            {
                "case": Path(entry["case_path"]).name,
                "episode_reward": float(entry["episode_reward"]),
                "episode_raw_reward": float(entry["episode_raw_reward"]),
                "reward_scale": float(entry["reward_scale"]),
                "makespan": makespan,
                "steps": int(entry["steps"]),
                "done": bool(entry["done"]),
                "records": entry["records"] if collect_records else [],
                "candidate_count_last": candidate_count_last,
                "max_original_candidate_count": int(entry["max_original_candidate_count"]),
                "max_used_candidate_count": int(entry["max_used_candidate_count"]),
                "num_truncated_steps": int(entry["num_truncated_steps"]),
                "total_overflow_candidates": int(entry["total_overflow_candidates"]),
            }
        )

    return {"results": results, "errors": errors}


def evaluate_actor_parallel(
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
    parallel_envs: int = 1,
) -> Dict[str, object]:
    if parallel_envs <= 1 or len(case_paths) <= 1:
        return evaluate_actor(
            actor,
            case_paths=case_paths,
            split_name=split_name,
            reset_rule=reset_rule,
            reward_normalize=reward_normalize,
            deterministic=deterministic,
            quiet=quiet,
            extra_step_budget=extra_step_budget,
            collect_records=collect_records,
            show_progress=show_progress,
        )

    all_results = []
    all_errors = []
    batches = list(_chunk_list(case_paths, parallel_envs))
    batch_iter = tqdm(batches, desc=f"eval-par:{split_name}", leave=False) if show_progress else batches

    for batch_case_paths in batch_iter:
        out = run_episodes_batch(
            actor,
            case_paths=batch_case_paths,
            reset_rule=reset_rule,
            reward_normalize=reward_normalize,
            deterministic=deterministic,
            quiet=quiet,
            extra_step_budget=extra_step_budget,
            collect_records=collect_records,
        )
        all_results.extend(out["results"])
        all_errors.extend(out["errors"])

    return _aggregate_results(all_results, all_errors, split_name=split_name)


def _build_agent(agent_cfg: AgentCConfig, global_dim: int, pair_feat_dim: int):
    try:
        return Agent_C(global_dim=global_dim, pair_feat_dim=pair_feat_dim, config=agent_cfg)
    except TypeError:
        return Agent_C(agent_cfg, global_dim=global_dim, pair_feat_dim=pair_feat_dim)


def train_agent_c_parallel(cfg: TrainConfig) -> Dict[str, object]:
    if len(cfg.train_cases) == 0:
        raise ValueError("train_cases is empty")
    if len(cfg.val_cases) == 0:
        raise ValueError("val_cases is empty")
    if len(cfg.test_cases) == 0:
        raise ValueError("test_cases is empty")

    seed_everything(cfg.seed)
    print_split_summary(cfg)

    if _apply_topk_settings_to_agent_config_from_train_cfg is not None:
        _apply_topk_settings_to_agent_config_from_train_cfg(cfg)

    structural_global_dim = _infer_structural_global_dim(
        list(cfg.train_cases) + list(cfg.val_cases) + list(cfg.test_cases),
        use_candidate_set_feat=cfg.use_candidate_set_feat,
    )

    scan_case_paths = list(cfg.train_cases) + list(cfg.val_cases)
    global_dim, pair_feat_dim, observed_max_candidates, candidate_scan_report = infer_training_dims(
        scan_case_paths,
        reset_rule=cfg.reset_rule,
        use_candidate_set_feat=cfg.use_candidate_set_feat,
        quiet=cfg.quiet_env,
        extra_step_budget=cfg.extra_step_budget,
        scan_rule=cfg.candidate_scan_rule,
        candidate_safety_margin=cfg.candidate_safety_margin,
        round_max_candidates_to_power_of_two=cfg.round_max_candidates_to_power_of_two,
    )
    global_dim = max(int(global_dim), int(structural_global_dim))
    inferred_recommended = int(candidate_scan_report["recommended_max_candidates"])
    max_candidates = (
        inferred_recommended
        if cfg.auto_max_candidates
        else max(int(cfg.max_candidates), int(observed_max_candidates))
    )

    print(
        f"[init-par] global_dim={global_dim}, pair_feat_dim={pair_feat_dim}, "
        f"max_candidates={max_candidates}, scan_errors={int(candidate_scan_report['num_scan_errors'])}"
    )

    use_hetero_lite = bool(getattr(cfg, "use_hetero_lite", getattr(cfg, "use_graph_encoder", False)))
    agent_cfg = AgentCConfig(
        device=cfg.device,
        hidden_dim=cfg.hidden_dim,
        attn_heads=cfg.attn_heads,
        attn_layers=cfg.attn_layers,
        dropout=cfg.dropout,
        use_candidate_set_feat=cfg.use_candidate_set_feat,
        lr=cfg.lr,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_ratio=cfg.clip_ratio,
        value_coef=cfg.value_coef,
        ent_coef=cfg.ent_coef,
        max_grad_norm=cfg.max_grad_norm,
        ppo_epochs=cfg.ppo_epochs,
        minibatch_size=cfg.minibatch_size,
        target_kl=cfg.target_kl,
        max_candidates=max_candidates,
        topk_prefilter_enabled=bool(getattr(cfg, "topk_prefilter_enabled", False)),
        topk_k=int(getattr(cfg, "topk_k", 16)),
        topk_keep_eet=int(getattr(cfg, "topk_keep_eet", 2)),
        topk_keep_pt=int(getattr(cfg, "topk_keep_pt", 2)),
        topk_keep_preferred_rule=int(getattr(cfg, "topk_keep_preferred_rule", 2)),
        topk_preferred_rule=str(getattr(cfg, "topk_preferred_rule", "FIFO_SPT")),
        topk_machine_diversity=bool(getattr(cfg, "topk_machine_diversity", True)),
        topk_score_weights=dict(getattr(cfg, "topk_score_weights", {"eet": 1.0, "pt": 0.3, "queue": 0.1, "slack": 0.2})),
        use_graph_encoder=use_hetero_lite,
        use_edge_rule_msg=bool(getattr(cfg, "use_edge_rule_msg", True)),
        use_edge_opmch_msg=bool(getattr(cfg, "use_edge_opmch_msg", True)),
        gnn_hidden_dim=int(getattr(cfg, "gnn_hidden_dim", 64)),
        gnn_layers=int(getattr(cfg, "gnn_layers", 2)),
        op_node_dim=int(getattr(cfg, "op_node_dim", 10)),
        machine_node_dim=int(getattr(cfg, "machine_node_dim", 6)),
        pair_graph_gate_init=float(getattr(cfg, "pair_graph_gate_init", 0.1)),
        rule_edge_gate_init=float(getattr(cfg, "rule_edge_gate_init", 0.1)),
        opmch_edge_gate_init=float(getattr(cfg, "opmch_edge_gate_init", 0.1)),
        edge_message_dropout=float(getattr(cfg, "edge_message_dropout", 0.0)),
        use_separate_edge_gates=bool(getattr(cfg, "use_separate_edge_gates", True)),
        use_adaptive_edge_gates=bool(getattr(cfg, "use_adaptive_edge_gates", False)),
        adaptive_gate_hidden_dim=int(getattr(cfg, "adaptive_gate_hidden_dim", 64)),
    )
    agent = _build_agent(agent_cfg, global_dim=global_dim, pair_feat_dim=pair_feat_dim)

    init_checkpoint_path = getattr(cfg, "init_checkpoint_path", None)
    if init_checkpoint_path:
        init_checkpoint_path = str(init_checkpoint_path)
        if not Path(init_checkpoint_path).exists():
            raise FileNotFoundError(f"init checkpoint not found: {init_checkpoint_path}")
        init_checkpoint_strict = bool(getattr(cfg, "init_checkpoint_strict", True))
        agent.load(init_checkpoint_path, strict=init_checkpoint_strict)
        print(f"[warm-start] loaded checkpoint={init_checkpoint_path}, strict={init_checkpoint_strict}")

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_last_n_epochs = int(max(1, getattr(cfg, "best_last_n_epochs", 20)))
    ckpt_path = save_dir / cfg.checkpoint_name
    ckpt_path_last_n_best = save_dir / (
        Path(cfg.checkpoint_name).stem + f"_last{best_last_n_epochs}_best.pt"
    )
    metrics_path = save_dir / (Path(cfg.checkpoint_name).stem + "_metrics.json")
    history_csv_path = save_dir / "train_history.csv"
    test_casewise_csv_path = save_dir / "test_casewise_comparison.csv"

    rollout_parallel_envs = max(1, int(getattr(cfg, "rollout_parallel_envs", cfg.episodes_per_update)))
    eval_parallel_envs = max(1, int(getattr(cfg, "eval_parallel_envs", rollout_parallel_envs)))
    print(f"[parallel] rollout_parallel_envs={rollout_parallel_envs}, eval_parallel_envs={eval_parallel_envs}")

    baseline_eval = {}
    baseline_cache_enabled = _resolve_baseline_cache_enabled(cfg)
    eval_baselines_on_cache_miss = _resolve_eval_baselines_on_cache_miss(cfg)
    for split_name, case_paths in (("val", cfg.val_cases), ("test", cfg.test_cases)):
        cached = _load_cached_baseline_eval(save_dir, cfg, case_paths, split_name) if baseline_cache_enabled else None
        if cached is not None:
            baseline_eval[split_name] = cached
            print(f"[baseline-cache-hit] split={split_name}: reuse cached baseline results")
        else:
            if eval_baselines_on_cache_miss:
                print(f"[baseline-cache-miss] split={split_name}: run baseline evaluation")
                baseline_eval[split_name] = evaluate_rule_baselines(
                    case_paths,
                    split_name=split_name,
                    reset_rule=cfg.reset_rule,
                    reward_normalize=cfg.reward_normalize,
                    quiet=cfg.quiet_env,
                    extra_step_budget=cfg.extra_step_budget,
                    baseline_rules=cfg.baseline_rules,
                    show_progress=True,
                )
                if baseline_cache_enabled:
                    _save_cached_baseline_eval(save_dir, cfg, case_paths, split_name, baseline_eval[split_name])
            else:
                baseline_eval[split_name] = {}
                print(
                    f"[baseline-skip] split={split_name}: cache missing and evaluate_rule_baselines_on_cache_miss=False"
                )

    history: List[Dict[str, object]] = []
    best_val_makespan = float("inf")
    global_update = 0
    final_train_full_eval = None
    final_val_eval = None
    last_train_full_eval = None
    train_full_eval_interval = _resolve_train_full_eval_interval(cfg)
    val_patience = int(max(0, getattr(cfg, "val_patience", 0)))
    min_epochs_before_early_stop = int(max(1, getattr(cfg, "min_epochs_before_early_stop", max(1, cfg.eval_interval))))
    val_improve_epsilon = float(max(0.0, getattr(cfg, "val_improve_epsilon", 1e-9)))
    no_improve_eval_count = 0
    early_stop_triggered = False
    recent_eval_checkpoints: List[Dict[str, object]] = []
    best_last_n_val_makespan = float("inf")
    best_last_n_epoch: Optional[int] = None

    for epoch_idx in tqdm(range(1, cfg.epochs + 1), desc="epochs"):
        train_batches = list(iter_train_batches(cfg.train_cases, cfg.episodes_per_update, cfg.seed, epoch_idx))

        for batch_cases in tqdm(train_batches, desc=f"epoch {epoch_idx} train", leave=False):
            global_update += 1
            buffer = RolloutBuffer()
            train_episodes = []
            skipped_cases = []

            for chunk_cases in _chunk_list(batch_cases, rollout_parallel_envs):
                chunk_out = run_episodes_batch(
                    agent,
                    case_paths=chunk_cases,
                    reset_rule=cfg.reset_rule,
                    reward_normalize=cfg.reward_normalize,
                    deterministic=False,
                    quiet=cfg.quiet_env,
                    extra_step_budget=cfg.extra_step_budget,
                    collect_records=True,
                )
                skipped_cases.extend(chunk_out["errors"])
                train_episodes.extend(chunk_out["results"])

            for result in train_episodes:
                if len(result.get("records", [])) == 0:
                    skipped_cases.append({"case": str(result.get("case", "unknown")), "reason": "empty_records"})
                    continue
                buffer.add_episode(result["records"], gamma=cfg.gamma, gae_lambda=cfg.gae_lambda)

            if len(buffer) == 0:
                history.append(
                    {
                        "epoch": epoch_idx,
                        "global_update": global_update,
                        "batch_cases": [Path(x).name for x in batch_cases],
                        "skipped": True,
                        "skip_reason": "empty_rollout_buffer",
                        "num_success_episodes": 0,
                        "num_skipped_cases": len(skipped_cases),
                        "skipped_cases": skipped_cases,
                    }
                )
                continue

            update_log = agent.update(buffer)
            record = {
                "epoch": epoch_idx,
                "global_update": global_update,
                "batch_cases": [Path(x).name for x in batch_cases],
                "sampled_train_reward": float(np.mean([item["episode_reward"] for item in train_episodes])) if train_episodes else 0.0,
                "sampled_train_makespan": float(np.mean([item["makespan"] for item in train_episodes])) if train_episodes else float("inf"),
                "sampled_train_done_ratio": float(np.mean([float(item["done"]) for item in train_episodes])) if train_episodes else 0.0,
                "sampled_train_steps": float(np.mean([item["steps"] for item in train_episodes])) if train_episodes else 0.0,
                "buffer_size": len(buffer),
                "num_success_episodes": len(train_episodes),
                "num_skipped_cases": len(skipped_cases),
                "sampled_train_mean_max_original_candidate_count": float(
                    np.mean([item.get("max_original_candidate_count", 0) for item in train_episodes])
                ) if train_episodes else 0.0,
                "sampled_train_mean_num_truncated_steps": float(
                    np.mean([item.get("num_truncated_steps", 0) for item in train_episodes])
                ) if train_episodes else 0.0,
                "rollout_parallel_envs": float(rollout_parallel_envs),
                **update_log,
            }
            if skipped_cases:
                record["skipped_cases"] = skipped_cases
            history.append(record)

        if epoch_idx == 1 or epoch_idx % cfg.eval_interval == 0:
            should_run_train_full = (
                epoch_idx == 1
                or last_train_full_eval is None
                or epoch_idx % train_full_eval_interval == 0
            )
            if should_run_train_full:
                train_full_eval = evaluate_actor_parallel(
                    agent,
                    case_paths=cfg.train_cases,
                    split_name="train_full",
                    reset_rule=cfg.reset_rule,
                    reward_normalize=cfg.reward_normalize,
                    deterministic=cfg.deterministic_eval,
                    quiet=cfg.quiet_env,
                    extra_step_budget=cfg.extra_step_budget,
                    collect_records=False,
                    show_progress=True,
                    parallel_envs=eval_parallel_envs,
                )
                last_train_full_eval = dict(train_full_eval)
                last_train_full_eval["throttled"] = False
            else:
                train_full_eval = dict(last_train_full_eval)
                train_full_eval["throttled"] = True

            val_eval = evaluate_actor_parallel(
                agent,
                case_paths=cfg.val_cases,
                split_name="val",
                reset_rule=cfg.reset_rule,
                reward_normalize=cfg.reward_normalize,
                deterministic=cfg.deterministic_eval,
                quiet=cfg.quiet_env,
                extra_step_budget=cfg.extra_step_budget,
                collect_records=False,
                show_progress=True,
                parallel_envs=eval_parallel_envs,
            )

            eval_record = {
                "epoch": epoch_idx,
                "global_update": global_update,
                "train_full_mean_reward": train_full_eval["mean_reward"],
                "train_full_mean_makespan": train_full_eval["mean_makespan"],
                "train_full_done_ratio": train_full_eval["done_ratio"],
                "train_full_throttled": bool(train_full_eval.get("throttled", False)),
                "val_mean_reward": val_eval["mean_reward"],
                "val_mean_makespan": val_eval["mean_makespan"],
                "val_done_ratio": val_eval["done_ratio"],
                "val_success_rate": val_eval.get("success_rate", 0.0),
                "val_num_errors": val_eval["num_errors"],
            }

            if cfg.run_test_on_every_eval:
                test_eval = evaluate_actor_parallel(
                    agent,
                    case_paths=cfg.test_cases,
                    split_name="test",
                    reset_rule=cfg.reset_rule,
                    reward_normalize=cfg.reward_normalize,
                    deterministic=cfg.deterministic_eval,
                    quiet=cfg.quiet_env,
                    extra_step_budget=cfg.extra_step_budget,
                    collect_records=False,
                    show_progress=True,
                    parallel_envs=eval_parallel_envs,
                )
                eval_record["test_mean_reward"] = test_eval["mean_reward"]
                eval_record["test_mean_makespan"] = test_eval["mean_makespan"]
                eval_record["test_done_ratio"] = test_eval["done_ratio"]

            if val_eval["mean_makespan"] < (best_val_makespan - val_improve_epsilon):
                best_val_makespan = val_eval["mean_makespan"]
                agent.save(str(ckpt_path))
                eval_record["saved_checkpoint"] = str(ckpt_path)
                no_improve_eval_count = 0
            else:
                no_improve_eval_count += 1

            eval_snapshot_path = save_dir / f"{Path(cfg.checkpoint_name).stem}_eval_epoch_{epoch_idx:04d}.pt"
            agent.save(str(eval_snapshot_path))
            recent_eval_checkpoints.append(
                {
                    "epoch": int(epoch_idx),
                    "val_mean_makespan": float(val_eval["mean_makespan"]),
                    "path": str(eval_snapshot_path),
                }
            )

            min_epoch_keep = int(epoch_idx - best_last_n_epochs + 1)
            kept_recent: List[Dict[str, object]] = []
            for item in recent_eval_checkpoints:
                if int(item["epoch"]) >= min_epoch_keep:
                    kept_recent.append(item)
                else:
                    with contextlib.suppress(Exception):
                        Path(str(item["path"])).unlink()
            recent_eval_checkpoints = kept_recent

            if recent_eval_checkpoints:
                best_recent = min(recent_eval_checkpoints, key=lambda x: float(x["val_mean_makespan"]))
                best_last_n_val_makespan = float(best_recent["val_mean_makespan"])
                best_last_n_epoch = int(best_recent["epoch"])
                src_path = Path(str(best_recent["path"]))
                if src_path.exists():
                    shutil.copyfile(src_path, ckpt_path_last_n_best)

            eval_record["no_improve_eval_count"] = int(no_improve_eval_count)
            eval_record["best_last_n_val_makespan"] = float(best_last_n_val_makespan)
            eval_record["best_last_n_epoch"] = int(best_last_n_epoch) if best_last_n_epoch is not None else None

            history.append(eval_record)
            print(
                f"[epoch {epoch_idx:02d}] val_ms={eval_record['val_mean_makespan']:.3f}, "
                f"train_ms={eval_record['train_full_mean_makespan']:.3f}, "
                f"best_val_ms={best_val_makespan:.3f}, "
                f"best_last{best_last_n_epochs}_val_ms={best_last_n_val_makespan:.3f}, "
                f"train_full_throttled={eval_record['train_full_throttled']}"
            )

            final_train_full_eval = train_full_eval
            final_val_eval = val_eval

            if (
                val_patience > 0
                and epoch_idx >= min_epochs_before_early_stop
                and no_improve_eval_count >= val_patience
            ):
                early_stop_triggered = True
                print(
                    f"[early-stop] epoch={epoch_idx:02d}, no_improve_eval_count={no_improve_eval_count}, "
                    f"best_val_ms={best_val_makespan:.3f}, patience={val_patience}"
                )
                break

        metrics_payload = {
            "train_config": asdict(cfg),
            "split_summary": _split_summary(cfg),
            "agent_config": asdict(agent_cfg),
            "global_dim": global_dim,
            "pair_feat_dim": pair_feat_dim,
            "candidate_scan_report": candidate_scan_report,
            "baseline_eval": baseline_eval,
            "history": history,
            "best_val_makespan": best_val_makespan,
            "best_last_n_val_makespan": best_last_n_val_makespan,
            "best_last_n_epoch": int(best_last_n_epoch) if best_last_n_epoch is not None else None,
            "best_last_n_epochs_window": int(best_last_n_epochs),
            "early_stop_triggered": bool(early_stop_triggered),
            "no_improve_eval_count": int(no_improve_eval_count),
            "checkpoint": str(ckpt_path),
            "checkpoint_last_n_best": str(ckpt_path_last_n_best),
            "parallel": {
                "rollout_parallel_envs": int(rollout_parallel_envs),
                "eval_parallel_envs": int(eval_parallel_envs),
            },
        }
        metrics_path.write_text(
            json.dumps(_to_jsonable(metrics_payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        export_history_csv(history, history_csv_path)

    agent.load(str(ckpt_path))
    final_test_eval_overall = evaluate_actor_parallel(
        agent,
        case_paths=cfg.test_cases,
        split_name="test",
        reset_rule=cfg.reset_rule,
        reward_normalize=cfg.reward_normalize,
        deterministic=cfg.deterministic_eval,
        quiet=cfg.quiet_env,
        extra_step_budget=cfg.extra_step_budget,
        collect_records=False,
        show_progress=True,
        parallel_envs=eval_parallel_envs,
    )

    if ckpt_path_last_n_best.exists():
        agent.load(str(ckpt_path_last_n_best))
        final_test_eval_last_n = evaluate_actor_parallel(
            agent,
            case_paths=cfg.test_cases,
            split_name="test",
            reset_rule=cfg.reset_rule,
            reward_normalize=cfg.reward_normalize,
            deterministic=cfg.deterministic_eval,
            quiet=cfg.quiet_env,
            extra_step_budget=cfg.extra_step_budget,
            collect_records=False,
            show_progress=True,
            parallel_envs=eval_parallel_envs,
        )
    else:
        final_test_eval_last_n = None

    plot_checkpoint_variant = str(getattr(cfg, "plot_checkpoint_variant", "best_overall")).strip().lower()
    if plot_checkpoint_variant in {"best_last20", "best_last_n", "last20", "last_n"} and final_test_eval_last_n is not None:
        final_test_eval = final_test_eval_last_n
        selected_test_checkpoint = f"best_last{best_last_n_epochs}"
    else:
        final_test_eval = final_test_eval_overall
        selected_test_checkpoint = "best_overall"

    baseline_gap_test = summarize_baseline_gap(final_test_eval, baseline_eval["test"])
    casewise_rows = build_casewise_comparison(final_test_eval, baseline_eval["test"])
    export_casewise_csv(casewise_rows, test_casewise_csv_path)

    for item in recent_eval_checkpoints:
        with contextlib.suppress(Exception):
            Path(str(item["path"])).unlink()

    metrics_payload = {
        "train_config": asdict(cfg),
        "split_summary": _split_summary(cfg),
        "agent_config": asdict(agent_cfg),
        "global_dim": global_dim,
        "pair_feat_dim": pair_feat_dim,
        "candidate_scan_report": candidate_scan_report,
        "baseline_eval": baseline_eval,
        "history": history,
        "best_val_makespan": best_val_makespan,
        "best_last_n_val_makespan": best_last_n_val_makespan,
        "best_last_n_epoch": int(best_last_n_epoch) if best_last_n_epoch is not None else None,
        "best_last_n_epochs_window": int(best_last_n_epochs),
        "early_stop_triggered": bool(early_stop_triggered),
        "no_improve_eval_count": int(no_improve_eval_count),
        "checkpoint": str(ckpt_path),
        "checkpoint_last_n_best": str(ckpt_path_last_n_best),
        "final_train_full_eval": final_train_full_eval,
        "final_val_eval": final_val_eval,
        "final_test_eval": final_test_eval,
        "final_test_eval_overall": final_test_eval_overall,
        "final_test_eval_last20": final_test_eval_last_n,
        "selected_test_checkpoint": selected_test_checkpoint,
        "baseline_gap_test": baseline_gap_test,
        "test_casewise_csv": str(test_casewise_csv_path),
        "history_csv": str(history_csv_path),
        "parallel": {
            "rollout_parallel_envs": int(rollout_parallel_envs),
            "eval_parallel_envs": int(eval_parallel_envs),
        },
    }
    metrics_path.write_text(
        json.dumps(_to_jsonable(metrics_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "checkpoint": str(ckpt_path),
        "metrics": str(metrics_path),
        "history_csv": str(history_csv_path),
        "test_casewise_csv": str(test_casewise_csv_path),
        "best_val_makespan": best_val_makespan,
        "best_last_n_val_makespan": best_last_n_val_makespan,
        "best_last_n_epoch": int(best_last_n_epoch) if best_last_n_epoch is not None else None,
        "best_last_n_epochs_window": int(best_last_n_epochs),
        "baseline_eval": baseline_eval,
        "baseline_gap_test": baseline_gap_test,
        "early_stop_triggered": bool(early_stop_triggered),
        "no_improve_eval_count": int(no_improve_eval_count),
        "history": history,
        "final_train_full_eval": final_train_full_eval,
        "final_val_eval": final_val_eval,
        "final_test_eval": final_test_eval,
        "final_test_eval_overall": final_test_eval_overall,
        "final_test_eval_last20": final_test_eval_last_n,
        "selected_test_checkpoint": selected_test_checkpoint,
        "checkpoint_last_n_best": str(ckpt_path_last_n_best),
    }


def train_agent_c(cfg: TrainConfig) -> Dict[str, object]:
    return train_agent_c_parallel(cfg)


def train_agent_c_with_plots_optimized(cfg, show_plots: bool = True, full_train_eval_interval: int = 3):
    cfg_local = deepcopy(cfg)
    setattr(cfg_local, "train_full_eval_interval", int(max(1, full_train_eval_interval)))

    if not hasattr(cfg_local, "rollout_parallel_envs"):
        setattr(cfg_local, "rollout_parallel_envs", int(max(1, cfg_local.episodes_per_update)))
    if not hasattr(cfg_local, "eval_parallel_envs"):
        setattr(cfg_local, "eval_parallel_envs", int(max(1, min(cfg_local.episodes_per_update, 4))))

    summary = train_agent_c_parallel(cfg_local)

    from Plot import plot_training_and_baselines
    plot_info = plot_training_and_baselines(summary, show=show_plots)
    summary["plot_info"] = plot_info
    summary["optimizations_applied"] = {
        "scan_dims_on": "train+val",
        "train_full_eval_interval": int(max(1, full_train_eval_interval)),
        "baseline_cache_enabled": _resolve_baseline_cache_enabled(cfg_local),
        "parallel_rollout": True,
        "rollout_parallel_envs": int(getattr(cfg_local, "rollout_parallel_envs", 1)),
        "eval_parallel_envs": int(getattr(cfg_local, "eval_parallel_envs", 1)),
        "val_patience": int(getattr(cfg_local, "val_patience", 0)),
        "min_epochs_before_early_stop": int(
            getattr(cfg_local, "min_epochs_before_early_stop", max(1, getattr(cfg_local, "eval_interval", 1)))
        ),
        "val_improve_epsilon": float(getattr(cfg_local, "val_improve_epsilon", 1e-9)),
    }
    return summary


def make_case_split(all_cases: List[Path], train_count: int, val_count: int) -> Tuple[List[str], List[str], List[str]]:
    if train_count <= 0:
        raise ValueError("train_count must be > 0")
    if val_count <= 0:
        raise ValueError("val_count must be > 0")
    if train_count + val_count >= len(all_cases):
        raise ValueError("train_count + val_count must be < len(all_cases)")

    train_cases = [str(path) for path in all_cases[:train_count]]
    val_cases = [str(path) for path in all_cases[train_count:train_count + val_count]]
    test_cases = [str(path) for path in all_cases[train_count + val_count:]]
    return train_cases, val_cases, test_cases


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Train Agent_C PPO with aligned FJSP environment.")
    parser.add_argument("--case_dir", type=str, default="1_Brandimarte", help="Directory containing .fjs instances")
    parser.add_argument("--pattern", type=str, default="BrandimarteMk*.fjs", help="Glob pattern of instances")
    parser.add_argument("--train_count", type=int, default=6, help="Number of train instances")
    parser.add_argument("--val_count", type=int, default=1, help="Number of validation instances")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(args=args)


__all__ = [
    "TrainConfig",
    "RuleBasedCandidateAgent",
    "seed_everything",
    "build_env_inputs_from_fjs",
    "build_env_for_case",
    "scan_case_runtime_stats",
    "infer_training_dims",
    "run_episode",
    "run_episodes_batch",
    "evaluate_actor_parallel",
    "train_agent_c",
    "train_agent_c_parallel",
    "train_agent_c_with_plots_optimized",
    "make_case_split",
    "parse_args",
]
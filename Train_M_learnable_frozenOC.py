from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from Agent_M_learn import AgentMLearnConfig, Agent_M_Learn, RolloutBufferM, StepRecordM

_ORIG_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]
from Train_C import build_env_for_case, make_case_split, seed_everything
from Train_O_frozenC import Agent_O, AgentOConfig, FrozenAgentC, filter_obs_by_selected_ops
sys.argv = _ORIG_ARGV


def _resolve_pair_sources(obs: Dict[str, object]) -> List[List[str]]:
    pairs = list(obs.get("candidate_pairs", []))
    src = list(obs.get("pair_sources", []))
    if len(src) < len(pairs):
        src = src + [[] for _ in range(len(pairs) - len(src))]
    return src[: len(pairs)]


def _slice_candidate_obs(obs: Dict[str, object], idx_keep: List[int]) -> Dict[str, object]:
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


def _pair_best_ect(obs: Dict[str, object]) -> float:
    pf = np.asarray(obs.get("pair_feat", []), dtype=np.float32)
    if pf.ndim == 2 and pf.shape[0] > 0 and pf.shape[1] > 4:
        return float(np.min(pf[:, 4]))
    return 0.0


def _find_pair_index(obs: Dict[str, object], op: int, mch: int) -> int:
    pairs = list(obs.get("candidate_pairs", []))
    for i, (a, b) in enumerate(pairs):
        if int(a) == int(op) and int(b) == int(mch):
            return int(i)
    return -1


def _machine_balance_penalty(obs: Dict[str, object]) -> float:
    pairs = list(obs.get("candidate_pairs", []))
    if len(pairs) == 0:
        return 0.0
    counts: Dict[int, int] = {}
    for _op, mch in pairs:
        m = int(mch)
        counts[m] = counts.get(m, 0) + 1
    arr = np.asarray(list(counts.values()), dtype=np.float32)
    if arr.size <= 1:
        return 0.0
    return float(arr.std() / max(arr.mean(), 1e-6))


@dataclass
class TrainMLearnConfig:
    train_cases: List[str]
    val_cases: List[str]
    test_cases: List[str]

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    reset_rule: str = "FIFO_SPT"

    epochs: int = 50
    eval_interval: int = 1
    extra_step_budget: int = 40
    max_episode_steps: int = 0
    quiet_env: bool = True

    save_dir: str = "checkpoints_agent_m_learn"
    checkpoint_name: str = "agent_m_learn_best.pt"

    o_checkpoint_path: str = ""
    c_checkpoint_path: str = ""
    init_m_checkpoint_path: str = ""

    op_topk: int = 5
    o_train_deterministic: bool = True
    o_eval_deterministic: bool = True

    feature_mode: str = "full"
    strategy_mode: str = "ppo"

    # Reward terms
    env_reward_coef: float = 1.0
    retain_pos: float = 0.20
    retain_neg: float = -0.30
    quality_coef: float = 0.35
    quality_hard_threshold: float = 0.08
    quality_hard_penalty: float = 0.10
    mismatch_penalty: float = 0.08
    overprune_target_keep_ratio: float = 0.30
    overprune_coef: float = 0.10
    machine_balance_coef: float = 0.04
    terminal_gap_coef: float = 0.01

    keep_c_full_top1: bool = True
    keep_c_o_top1: bool = True

    # Optional stage2 improvement
    enable_entropy_backup: bool = False
    backup_entropy_threshold: float = 0.75
    backup_max_extra_pairs: int = 1

    # Learner hyper-parameters
    hidden_dim: int = 128
    dropout: float = 0.05
    lr: float = 2e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 256
    target_kl: float = 0.02


def _load_frozen_o(cfg: TrainMLearnConfig) -> Any:
    payload = torch.load(cfg.o_checkpoint_path, map_location=cfg.device)
    o_cfg = AgentOConfig(**payload["config"])
    o_cfg.device = cfg.device
    agent_o = Agent_O(config=o_cfg, global_dim=int(payload["global_dim"]), op_feat_dim=int(payload["op_feat_dim"]))
    agent_o.load(cfg.o_checkpoint_path, strict=True)
    agent_o.policy.eval()
    return agent_o


def _load_frozen_c(cfg: TrainMLearnConfig) -> FrozenAgentC:
    return FrozenAgentC(
        ckpt_path=cfg.c_checkpoint_path,
        device=cfg.device,
        disable_topk_prefilter=False,
        max_candidates_floor=0,
    )


def _compute_train_reward(
    cfg: TrainMLearnConfig,
    full_obs: Dict[str, object],
    filtered_obs: Dict[str, object],
    selected_ops: List[int],
    c_full_pair: Tuple[int, int],
    c_filtered_pair: Tuple[int, int],
    env_reward: float,
    prev_ms: float,
    new_ms: float,
    initial_lb_ms: float,
    done: bool,
) -> Dict[str, float]:
    c_full_op, c_full_mch = int(c_full_pair[0]), int(c_full_pair[1])
    c_filtered_op, _ = int(c_filtered_pair[0]), int(c_filtered_pair[1])

    retained = (c_full_op, c_full_mch) in filtered_obs.get("legal_pairs_set", set())
    retain_term = float(cfg.retain_pos) if retained else float(cfg.retain_neg)

    ect_full = _pair_best_ect(full_obs)
    ect_filtered = _pair_best_ect(filtered_obs)
    quality_gap = 0.0
    if abs(ect_full) > 1e-8:
        quality_gap = max(0.0, (ect_filtered - ect_full) / abs(ect_full))

    quality_pen = float(cfg.quality_coef) * float(quality_gap)
    hard_quality_pen = float(cfg.quality_hard_penalty) if quality_gap > float(cfg.quality_hard_threshold) else 0.0

    full_n = max(1, len(list(full_obs.get("candidate_pairs", []))))
    keep_n = len(list(filtered_obs.get("candidate_pairs", [])))
    keep_ratio = float(keep_n) / float(full_n)
    overprune_pen = float(cfg.overprune_coef) * max(0.0, float(cfg.overprune_target_keep_ratio) - keep_ratio)

    balance_pen = float(cfg.machine_balance_coef) * _machine_balance_penalty(filtered_obs)

    mismatch_pen = float(cfg.mismatch_penalty) if c_filtered_op != c_full_op else 0.0

    terminal_pen = 0.0
    if bool(done):
        denom = max(abs(float(initial_lb_ms)), 1e-6)
        gap = max(0.0, (float(new_ms) - float(initial_lb_ms)) / denom)
        terminal_pen = float(cfg.terminal_gap_coef) * gap

    total = (
        float(cfg.env_reward_coef) * float(env_reward)
        + retain_term
        - quality_pen
        - hard_quality_pen
        - overprune_pen
        - balance_pen
        - mismatch_pen
        - terminal_pen
    )

    return {
        "total": float(total),
        "retain_term": float(retain_term),
        "quality_gap": float(quality_gap),
        "quality_pen": float(quality_pen),
        "hard_quality_pen": float(hard_quality_pen),
        "keep_ratio": float(keep_ratio),
        "overprune_pen": float(overprune_pen),
        "balance_pen": float(balance_pen),
        "mismatch_pen": float(mismatch_pen),
        "env_reward": float(env_reward),
        "prev_ms": float(prev_ms),
        "new_ms": float(new_ms),
    }


def run_episode_m_trainable(
    agent_o: Any,
    frozen_c: FrozenAgentC,
    agent_m: Agent_M_Learn,
    case_path: str,
    cfg: TrainMLearnConfig,
    deterministic: bool,
    training: bool,
) -> Dict[str, Any]:
    env, total_tasks = build_env_for_case(case_path, reset_rule=cfg.reset_rule, quiet=cfg.quiet_env)
    max_steps = int(total_tasks) + int(cfg.extra_step_budget)
    if int(cfg.max_episode_steps) > 0:
        max_steps = min(max_steps, int(cfg.max_episode_steps))

    ep_env_reward = 0.0
    ep_train_reward = 0.0
    step_count = 0
    episode_steps: List[StepRecordM] = []

    stat_keep_pairs = 0.0
    stat_quality_gap = 0.0
    stat_retain_pair = 0.0
    stat_mismatch_pen = 0.0
    stat_overprune_pen = 0.0
    stat_balance_pen = 0.0

    init_ms = float(env.LBm[0].max())
    done = False

    for _ in range(max_steps):
        full_obs = env.get_agent_c_obs(batch_idx=0)

        o_det = bool(cfg.o_eval_deterministic) if deterministic else bool(cfg.o_train_deterministic)
        o_info = agent_o.select_action(full_obs, deterministic=o_det, topk_k=int(cfg.op_topk))
        selected_ops = [int(x) for x in o_info.get("selected_ops", [])]
        if len(selected_ops) == 0:
            selected_ops = [int(o_info["action_op_id"])]

        c_full_op, c_full_mch, c_full_info = frozen_c.act(full_obs, deterministic=True)
        c_full_pair = (int(c_full_op), int(c_full_mch))
        c_o_pair = c_full_pair
        if bool(cfg.keep_c_o_top1):
            o_filtered_obs = filter_obs_by_selected_ops(full_obs, selected_ops)
            if len(list(o_filtered_obs.get("candidate_pairs", []))) > 0:
                c_o_op, c_o_mch, _ = frozen_c.act(o_filtered_obs, deterministic=True)
                c_o_pair = (int(c_o_op), int(c_o_mch))

        packed = agent_m.build_packed_obs(full_obs, selected_ops=selected_ops)
        if len(packed["candidate_pairs"]) == 0:
            keep_full_indices = [max(0, _find_pair_index(full_obs, c_full_pair[0], c_full_pair[1]))]
            action_info = {
                "action_pair_indices": np.zeros((0,), dtype=np.int64),
                "log_prob": 0.0,
                "value": 0.0,
            }
        else:
            action_info = agent_m.act(packed, deterministic=deterministic)
            local_keep = [int(i) for i in action_info.get("keep_indices", [])]
            full_idx_map = np.asarray(packed["full_pair_indices"], dtype=np.int64)
            keep_full_indices = [int(full_idx_map[i]) for i in local_keep if 0 <= int(i) < full_idx_map.shape[0]]

        if bool(cfg.keep_c_full_top1):
            c_full_idx = _find_pair_index(full_obs, c_full_pair[0], c_full_pair[1])
            if c_full_idx >= 0:
                keep_full_indices.append(int(c_full_idx))

        if bool(cfg.keep_c_o_top1):
            c_o_idx = _find_pair_index(full_obs, c_o_pair[0], c_o_pair[1])
            if c_o_idx >= 0:
                keep_full_indices.append(int(c_o_idx))

        if len(keep_full_indices) == 0:
            pf = np.asarray(full_obs.get("pair_feat", []), dtype=np.float32)
            if pf.ndim == 2 and pf.shape[0] > 0 and pf.shape[1] > 4:
                keep_full_indices = [int(np.argmin(pf[:, 4]))]
            else:
                keep_full_indices = [0]

        keep_full_indices = sorted(set(int(x) for x in keep_full_indices))
        filtered_obs = _slice_candidate_obs(full_obs, keep_full_indices)

        c_op, c_mch, _ = frozen_c.act(filtered_obs, deterministic=True)
        c_filtered_pair = (int(c_op), int(c_mch))

        prev_ms = float(env.LBm[0].max())
        out = env.step_with_pair(c_op, c_mch, batch_idx=0)
        env_reward = float(out[2][0])
        done = bool(out[3][0])
        new_ms = float(env.LBm[0].max())

        terms = _compute_train_reward(
            cfg=cfg,
            full_obs=full_obs,
            filtered_obs=filtered_obs,
            selected_ops=selected_ops,
            c_full_pair=c_full_pair,
            c_filtered_pair=c_filtered_pair,
            env_reward=env_reward,
            prev_ms=prev_ms,
            new_ms=new_ms,
            initial_lb_ms=init_ms,
            done=done,
        )

        ep_env_reward += float(env_reward)
        ep_train_reward += float(terms["total"])
        step_count += 1

        stat_keep_pairs += float(len(filtered_obs.get("candidate_pairs", [])))
        stat_quality_gap += float(terms["quality_gap"])
        stat_retain_pair += 1.0 if (c_full_pair in filtered_obs.get("legal_pairs_set", set())) else 0.0
        stat_mismatch_pen += float(terms["mismatch_pen"])
        stat_overprune_pen += float(terms["overprune_pen"])
        stat_balance_pen += float(terms["balance_pen"])

        if training and len(packed["candidate_pairs"]) > 0:
            episode_steps.append(
                StepRecordM(
                    global_feat=np.asarray(packed["global_feat"], dtype=np.float32),
                    pair_feat=np.asarray(packed["pair_feat"], dtype=np.float32),
                    pair_mask=np.asarray(packed["pair_mask"], dtype=bool),
                    pair_op_idx=np.asarray(packed["pair_op_idx"], dtype=np.int64),
                    op_order=[int(x) for x in packed["op_order"]],
                    action_pair_indices=np.asarray(action_info.get("action_pair_indices", np.zeros((0,), dtype=np.int64)), dtype=np.int64),
                    old_log_prob=float(action_info.get("log_prob", 0.0)),
                    old_value=float(action_info.get("value", 0.0)),
                    reward=float(terms["total"]),
                    done=bool(done),
                )
            )

        if done:
            break

    denom = max(step_count, 1)
    return {
        "episode_env_reward": float(ep_env_reward),
        "episode_train_reward": float(ep_train_reward),
        "makespan": float(env.LBm[0].max()),
        "steps": float(step_count),
        "done_rate": 1.0 if done else 0.0,
        "mean_keep_pairs": float(stat_keep_pairs / denom),
        "mean_quality_gap": float(stat_quality_gap / denom),
        "mean_retain_pair_rate": float(stat_retain_pair / denom),
        "mean_mismatch_penalty": float(stat_mismatch_pen / denom),
        "mean_overprune_penalty": float(stat_overprune_pen / denom),
        "mean_balance_penalty": float(stat_balance_pen / denom),
        "episode_steps_data": episode_steps,
    }


def _aggregate_rows(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if len(rows) == 0:
        return {
            "num_cases": 0,
            "mean_reward": 0.0,
            "mean_train_reward": 0.0,
            "mean_ms": float("inf"),
            "mean_steps": 0.0,
            "mean_keep_pairs": 0.0,
            "mean_quality_gap": 0.0,
            "mean_retain_pair_rate": 0.0,
            "mean_mismatch_penalty": 0.0,
            "mean_overprune_penalty": 0.0,
            "mean_balance_penalty": 0.0,
        }

    def avg(key: str) -> float:
        return float(np.mean([float(x[key]) for x in rows]))

    return {
        "num_cases": int(len(rows)),
        "mean_reward": avg("episode_env_reward"),
        "mean_train_reward": avg("episode_train_reward"),
        "mean_ms": avg("makespan"),
        "mean_steps": avg("steps"),
        "mean_keep_pairs": avg("mean_keep_pairs"),
        "mean_quality_gap": avg("mean_quality_gap"),
        "mean_retain_pair_rate": avg("mean_retain_pair_rate"),
        "mean_mismatch_penalty": avg("mean_mismatch_penalty"),
        "mean_overprune_penalty": avg("mean_overprune_penalty"),
        "mean_balance_penalty": avg("mean_balance_penalty"),
    }


def evaluate_split(
    agent_o: Any,
    frozen_c: FrozenAgentC,
    agent_m: Agent_M_Learn,
    cases: List[str],
    cfg: TrainMLearnConfig,
) -> Dict[str, float]:
    rows = [
        run_episode_m_trainable(
            agent_o=agent_o,
            frozen_c=frozen_c,
            agent_m=agent_m,
            case_path=c,
            cfg=cfg,
            deterministic=True,
            training=False,
        )
        for c in cases
    ]
    return _aggregate_rows(rows)


def train_agent_m_learnable_with_frozen_oc(cfg: TrainMLearnConfig) -> Dict[str, Any]:
    if not cfg.o_checkpoint_path:
        raise ValueError("o_checkpoint_path is required")
    if not cfg.c_checkpoint_path:
        raise ValueError("c_checkpoint_path is required")

    seed_everything(int(cfg.seed))

    agent_o = _load_frozen_o(cfg)
    frozen_c = _load_frozen_c(cfg)

    sample_env, _ = build_env_for_case(cfg.train_cases[0], reset_rule=cfg.reset_rule, quiet=True)
    sample_obs = sample_env.get_agent_c_obs(batch_idx=0)

    global_dim = int(np.asarray(sample_obs.get("global_feat", []), dtype=np.float32).reshape(-1).shape[0])
    pair_feat_dim = int(np.asarray(sample_obs.get("pair_feat", []), dtype=np.float32).shape[1])

    m_cfg = AgentMLearnConfig(
        device=cfg.device,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
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
        feature_mode=cfg.feature_mode,
        strategy_mode=cfg.strategy_mode,
        enable_entropy_backup=cfg.enable_entropy_backup,
        backup_entropy_threshold=cfg.backup_entropy_threshold,
        backup_max_extra_pairs=cfg.backup_max_extra_pairs,
    )

    agent_m = Agent_M_Learn(config=m_cfg, global_dim=global_dim, pair_feat_dim=pair_feat_dim)

    if str(cfg.init_m_checkpoint_path).strip():
        init_path = Path(str(cfg.init_m_checkpoint_path)).expanduser()
        if not init_path.is_absolute():
            init_path = (Path(__file__).resolve().parent / init_path).resolve()
        if not init_path.exists():
            raise FileNotFoundError(f"Missing init M checkpoint: {init_path}")

        payload = torch.load(str(init_path), map_location=cfg.device)
        state_dict = payload.get("state_dict")
        if state_dict is None:
            raise ValueError(f"Invalid init checkpoint without state_dict: {init_path}")
        agent_m.policy.load_state_dict(state_dict, strict=True)
        print(f"[init] loaded_m_checkpoint={init_path}", flush=True)

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_ms = float("inf")
    best_epoch = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, int(cfg.epochs) + 1):
        train_cases = list(cfg.train_cases)
        random.Random(cfg.seed + epoch).shuffle(train_cases)

        buffer = RolloutBufferM()
        train_rows: List[Dict[str, Any]] = []

        for case in train_cases:
            row = run_episode_m_trainable(
                agent_o=agent_o,
                frozen_c=frozen_c,
                agent_m=agent_m,
                case_path=case,
                cfg=cfg,
                deterministic=False,
                training=True,
            )
            train_rows.append(row)
            buffer.add_episode(
                row.get("episode_steps_data", []),
                gamma=float(m_cfg.gamma),
                gae_lambda=float(m_cfg.gae_lambda),
            )

        update_log = agent_m.update(buffer, epoch_idx=epoch, total_epochs=int(cfg.epochs))
        train_metrics = _aggregate_rows(train_rows)

        if epoch % int(max(1, cfg.eval_interval)) == 0:
            val_metrics = evaluate_split(agent_o, frozen_c, agent_m, cfg.val_cases, cfg)
        else:
            val_metrics = {"num_cases": 0, "mean_ms": float("inf")}

        improved = bool(val_metrics.get("mean_ms", float("inf")) < best_val_ms)
        if improved:
            best_val_ms = float(val_metrics["mean_ms"])
            best_epoch = int(epoch)
            agent_m.save(str(save_dir / cfg.checkpoint_name))
            meta = {
                "epoch": int(epoch),
                "best_val_ms": float(best_val_ms),
                "config": asdict(cfg),
                "agent_m_config": asdict(m_cfg),
            }
            (save_dir / "agent_m_learn_best_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        row = {
            "epoch": int(epoch),
            "train": train_metrics,
            "val": val_metrics,
            "update": update_log,
            "best_epoch": int(best_epoch),
            "best_val_ms": float(best_val_ms),
            "best_updated": bool(improved),
        }
        history.append(row)
        print(f"[epoch_json] {json.dumps(row, ensure_ascii=False)}", flush=True)

    test_metrics = evaluate_split(agent_o, frozen_c, agent_m, cfg.test_cases, cfg)

    final_summary = {
        "config": asdict(cfg),
        "agent_m_config": asdict(m_cfg),
        "best_epoch": int(best_epoch),
        "best_val_ms": float(best_val_ms),
        "test": test_metrics,
        "history": history,
    }

    (save_dir / "train_m_learn_summary.json").write_text(json.dumps(final_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return final_summary


def build_default_cfg(
    root: Path,
    epochs: int,
    seed: int,
    save_dir: str,
    train_count: int = 10,
    val_count: int = 2,
    test_count: int = 3,
) -> TrainMLearnConfig:
    all_cases = sorted((root / "1_Brandimarte").glob("BrandimarteMk*.fjs"))
    random.Random(seed).shuffle(all_cases)
    train_cases, val_cases, test_cases = make_case_split(
        all_cases,
        train_count=int(train_count),
        val_count=int(val_count),
    )
    if int(test_count) > 0 and len(test_cases) > int(test_count):
        test_cases = test_cases[: int(test_count)]

    baseline_path = root / "logs" / "next16_20260331" / "oc_report_ranked.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    best_name = baseline["ranked"][0]["name"].lower()
    o_ckpt = root / f"checkpoints_next16_{best_name}" / "agent_o_best.pt"

    c_ckpt = (
        root
        / "checkpoints_edge_msg_ablation_5seed_dualgate"
        / "edge_msg_dualgate_5seed"
        / "A00_1000"
        / "seed_22"
        / "A00_1000_s22_best.pt"
    )

    return TrainMLearnConfig(
        train_cases=[str(x) for x in train_cases],
        val_cases=[str(x) for x in val_cases],
        test_cases=[str(x) for x in test_cases],
        seed=int(seed),
        device="cuda" if torch.cuda.is_available() else "cpu",
        epochs=int(epochs),
        o_checkpoint_path=str(o_ckpt),
        c_checkpoint_path=str(c_ckpt),
        save_dir=str(root / save_dir),
    )


def _apply_profile(cfg: TrainMLearnConfig, profile: str) -> None:
    p = str(profile).strip().lower()
    if p == "base":
        cfg.feature_mode = "base"
        cfg.strategy_mode = "ac_basic"
        cfg.env_reward_coef = 1.0
        cfg.retain_pos = 0.0
        cfg.retain_neg = 0.0
        cfg.quality_coef = 0.0
        cfg.mismatch_penalty = 0.0
        cfg.overprune_coef = 0.0
        cfg.machine_balance_coef = 0.0
        cfg.ent_coef = 0.01
    elif p == "feature_full":
        cfg.feature_mode = "full"
        cfg.strategy_mode = "ppo"
        cfg.env_reward_coef = 1.0
        cfg.retain_pos = 0.0
        cfg.retain_neg = 0.0
        cfg.quality_coef = 0.0
        cfg.mismatch_penalty = 0.0
        cfg.overprune_coef = 0.0
        cfg.machine_balance_coef = 0.0
    elif p == "reward_shaped":
        cfg.feature_mode = "full"
        cfg.strategy_mode = "ppo"
        cfg.env_reward_coef = 1.0
        cfg.retain_pos = 0.25
        cfg.retain_neg = -0.35
        cfg.quality_coef = 0.35
        cfg.mismatch_penalty = 0.08
        cfg.overprune_coef = 0.10
        cfg.machine_balance_coef = 0.04
    elif p == "reward_shaped_heavyfeat":
        cfg.feature_mode = "machine_heavy"
        cfg.strategy_mode = "ppo"
        cfg.env_reward_coef = 1.0
        cfg.retain_pos = 0.28
        cfg.retain_neg = -0.40
        cfg.quality_coef = 0.40
        cfg.mismatch_penalty = 0.10
        cfg.overprune_coef = 0.12
        cfg.machine_balance_coef = 0.06
    elif p == "strategy_entropy_anneal":
        cfg.feature_mode = "machine_heavy"
        cfg.strategy_mode = "ppo_entropy_anneal"
        cfg.env_reward_coef = 1.0
        cfg.retain_pos = 0.28
        cfg.retain_neg = -0.40
        cfg.quality_coef = 0.40
        cfg.mismatch_penalty = 0.10
        cfg.overprune_coef = 0.12
        cfg.machine_balance_coef = 0.06
        cfg.ent_coef = 0.02
    elif p == "stage2_backup":
        cfg.feature_mode = "machine_heavy"
        cfg.strategy_mode = "ppo_entropy_anneal"
        cfg.enable_entropy_backup = True
        cfg.backup_entropy_threshold = 0.70
        cfg.backup_max_extra_pairs = 2
        cfg.retain_pos = 0.32
        cfg.retain_neg = -0.42
        cfg.quality_coef = 0.42
        cfg.quality_hard_penalty = 0.12
        cfg.mismatch_penalty = 0.10
        cfg.overprune_coef = 0.10
        cfg.machine_balance_coef = 0.08
        cfg.ent_coef = 0.02
    elif p == "stage2_stable":
        cfg.feature_mode = "machine_heavy"
        cfg.strategy_mode = "ppo"
        cfg.enable_entropy_backup = True
        cfg.backup_entropy_threshold = 0.78
        cfg.backup_max_extra_pairs = 1
        cfg.retain_pos = 0.30
        cfg.retain_neg = -0.38
        cfg.quality_coef = 0.38
        cfg.quality_hard_penalty = 0.10
        cfg.mismatch_penalty = 0.08
        cfg.overprune_coef = 0.08
        cfg.machine_balance_coef = 0.06
        cfg.ent_coef = 0.015
    elif p == "stage2_aggressive":
        cfg.feature_mode = "machine_heavy"
        cfg.strategy_mode = "ppo_entropy_anneal"
        cfg.enable_entropy_backup = True
        cfg.backup_entropy_threshold = 0.62
        cfg.backup_max_extra_pairs = 2
        cfg.retain_pos = 0.34
        cfg.retain_neg = -0.45
        cfg.quality_coef = 0.45
        cfg.quality_hard_penalty = 0.14
        cfg.mismatch_penalty = 0.12
        cfg.overprune_coef = 0.14
        cfg.machine_balance_coef = 0.10
        cfg.ent_coef = 0.02
    elif p == "stage3_longrun_refine":
        # Long-run refinement from stage2_stable: smoother PPO updates + stricter quality guard.
        cfg.feature_mode = "machine_heavy"
        cfg.strategy_mode = "ppo_entropy_anneal"
        cfg.enable_entropy_backup = True
        cfg.backup_entropy_threshold = 0.72
        cfg.backup_max_extra_pairs = 2
        cfg.retain_pos = 0.33
        cfg.retain_neg = -0.36
        cfg.quality_coef = 0.42
        cfg.quality_hard_threshold = 0.06
        cfg.quality_hard_penalty = 0.12
        cfg.mismatch_penalty = 0.09
        cfg.overprune_target_keep_ratio = 0.34
        cfg.overprune_coef = 0.10
        cfg.machine_balance_coef = 0.08
        cfg.ent_coef = 0.018
        cfg.lr = 1.5e-4
        cfg.target_kl = 0.015
        cfg.ppo_epochs = 5
        cfg.minibatch_size = 320
        cfg.dropout = 0.08
    elif p == "r1_entropy_anneal":
        cfg.feature_mode = "machine_heavy"
        cfg.strategy_mode = "ppo_entropy_anneal"
        cfg.enable_entropy_backup = True
        cfg.backup_entropy_threshold = 0.78
        cfg.backup_max_extra_pairs = 1
        cfg.retain_pos = 0.30
        cfg.retain_neg = -0.38
        cfg.quality_coef = 0.38
        cfg.quality_hard_penalty = 0.10
        cfg.mismatch_penalty = 0.08
        cfg.overprune_coef = 0.08
        cfg.machine_balance_coef = 0.06
        cfg.ent_coef = 0.015
    elif p == "r2_backup_boost":
        cfg.feature_mode = "machine_heavy"
        cfg.strategy_mode = "ppo"
        cfg.enable_entropy_backup = True
        cfg.backup_entropy_threshold = 0.72
        cfg.backup_max_extra_pairs = 2
        cfg.retain_pos = 0.30
        cfg.retain_neg = -0.38
        cfg.quality_coef = 0.38
        cfg.quality_hard_penalty = 0.10
        cfg.mismatch_penalty = 0.08
        cfg.overprune_coef = 0.08
        cfg.machine_balance_coef = 0.06
        cfg.ent_coef = 0.015
    elif p == "r3_quality_guard":
        cfg.feature_mode = "machine_heavy"
        cfg.strategy_mode = "ppo"
        cfg.enable_entropy_backup = True
        cfg.backup_entropy_threshold = 0.78
        cfg.backup_max_extra_pairs = 1
        cfg.retain_pos = 0.30
        cfg.retain_neg = -0.38
        cfg.quality_coef = 0.38
        cfg.quality_hard_threshold = 0.06
        cfg.quality_hard_penalty = 0.12
        cfg.mismatch_penalty = 0.08
        cfg.overprune_coef = 0.08
        cfg.machine_balance_coef = 0.06
        cfg.ent_coef = 0.015
    elif p == "r4_ppo_stable":
        cfg.feature_mode = "machine_heavy"
        cfg.strategy_mode = "ppo"
        cfg.enable_entropy_backup = True
        cfg.backup_entropy_threshold = 0.78
        cfg.backup_max_extra_pairs = 1
        cfg.retain_pos = 0.30
        cfg.retain_neg = -0.38
        cfg.quality_coef = 0.38
        cfg.quality_hard_penalty = 0.10
        cfg.mismatch_penalty = 0.08
        cfg.overprune_coef = 0.08
        cfg.machine_balance_coef = 0.06
        cfg.ent_coef = 0.015
        cfg.lr = 1.5e-4
        cfg.target_kl = 0.015
        cfg.ppo_epochs = 5
        cfg.minibatch_size = 320
        cfg.dropout = 0.08
    elif p == "r5_retain_overprune":
        cfg.feature_mode = "machine_heavy"
        cfg.strategy_mode = "ppo"
        cfg.enable_entropy_backup = True
        cfg.backup_entropy_threshold = 0.78
        cfg.backup_max_extra_pairs = 1
        cfg.retain_pos = 0.33
        cfg.retain_neg = -0.36
        cfg.quality_coef = 0.38
        cfg.quality_hard_penalty = 0.10
        cfg.mismatch_penalty = 0.08
        cfg.overprune_target_keep_ratio = 0.34
        cfg.overprune_coef = 0.10
        cfg.machine_balance_coef = 0.06
        cfg.ent_coef = 0.015
    elif p == "r6_balance_strong":
        cfg.feature_mode = "machine_heavy"
        cfg.strategy_mode = "ppo"
        cfg.enable_entropy_backup = True
        cfg.backup_entropy_threshold = 0.78
        cfg.backup_max_extra_pairs = 1
        cfg.retain_pos = 0.30
        cfg.retain_neg = -0.38
        cfg.quality_coef = 0.38
        cfg.quality_hard_penalty = 0.10
        cfg.mismatch_penalty = 0.08
        cfg.overprune_coef = 0.08
        cfg.machine_balance_coef = 0.08
        cfg.ent_coef = 0.015
    elif p == "c1_r2_r3":
        cfg.feature_mode = "machine_heavy"
        cfg.strategy_mode = "ppo"
        cfg.enable_entropy_backup = True
        cfg.backup_entropy_threshold = 0.72
        cfg.backup_max_extra_pairs = 2
        cfg.retain_pos = 0.30
        cfg.retain_neg = -0.38
        cfg.quality_coef = 0.38
        cfg.quality_hard_threshold = 0.06
        cfg.quality_hard_penalty = 0.12
        cfg.mismatch_penalty = 0.08
        cfg.overprune_coef = 0.08
        cfg.machine_balance_coef = 0.06
        cfg.ent_coef = 0.015
    elif p == "c2_r2_r4":
        cfg.feature_mode = "machine_heavy"
        cfg.strategy_mode = "ppo"
        cfg.enable_entropy_backup = True
        cfg.backup_entropy_threshold = 0.72
        cfg.backup_max_extra_pairs = 2
        cfg.retain_pos = 0.30
        cfg.retain_neg = -0.38
        cfg.quality_coef = 0.38
        cfg.quality_hard_penalty = 0.10
        cfg.mismatch_penalty = 0.08
        cfg.overprune_coef = 0.08
        cfg.machine_balance_coef = 0.06
        cfg.ent_coef = 0.015
        cfg.lr = 1.5e-4
        cfg.target_kl = 0.015
        cfg.ppo_epochs = 5
        cfg.minibatch_size = 320
        cfg.dropout = 0.08
    elif p == "c3_r3_r4":
        cfg.feature_mode = "machine_heavy"
        cfg.strategy_mode = "ppo"
        cfg.enable_entropy_backup = True
        cfg.backup_entropy_threshold = 0.78
        cfg.backup_max_extra_pairs = 1
        cfg.retain_pos = 0.30
        cfg.retain_neg = -0.38
        cfg.quality_coef = 0.38
        cfg.quality_hard_threshold = 0.06
        cfg.quality_hard_penalty = 0.12
        cfg.mismatch_penalty = 0.08
        cfg.overprune_coef = 0.08
        cfg.machine_balance_coef = 0.06
        cfg.ent_coef = 0.015
        cfg.lr = 1.5e-4
        cfg.target_kl = 0.015
        cfg.ppo_epochs = 5
        cfg.minibatch_size = 320
        cfg.dropout = 0.08
    elif p == "c4_r2_r3_r4":
        cfg.feature_mode = "machine_heavy"
        cfg.strategy_mode = "ppo"
        cfg.enable_entropy_backup = True
        cfg.backup_entropy_threshold = 0.72
        cfg.backup_max_extra_pairs = 2
        cfg.retain_pos = 0.30
        cfg.retain_neg = -0.38
        cfg.quality_coef = 0.38
        cfg.quality_hard_threshold = 0.06
        cfg.quality_hard_penalty = 0.12
        cfg.mismatch_penalty = 0.08
        cfg.overprune_coef = 0.08
        cfg.machine_balance_coef = 0.06
        cfg.ent_coef = 0.015
        cfg.lr = 1.5e-4
        cfg.target_kl = 0.015
        cfg.ppo_epochs = 5
        cfg.minibatch_size = 320
        cfg.dropout = 0.08
    elif p == "f1_rel_ect":
        cfg.feature_mode = "machine_heavy_rel_ect"
        cfg.strategy_mode = "ppo"
        cfg.enable_entropy_backup = True
        cfg.backup_entropy_threshold = 0.78
        cfg.backup_max_extra_pairs = 1
        cfg.retain_pos = 0.30
        cfg.retain_neg = -0.38
        cfg.quality_coef = 0.38
        cfg.quality_hard_penalty = 0.10
        cfg.mismatch_penalty = 0.08
        cfg.overprune_coef = 0.08
        cfg.machine_balance_coef = 0.06
    elif p == "f2_balance_feat":
        cfg.feature_mode = "machine_heavy_balance"
        cfg.strategy_mode = "ppo"
        cfg.enable_entropy_backup = True
        cfg.backup_entropy_threshold = 0.78
        cfg.backup_max_extra_pairs = 1
        cfg.retain_pos = 0.30
        cfg.retain_neg = -0.38
        cfg.quality_coef = 0.38
        cfg.quality_hard_penalty = 0.10
        cfg.mismatch_penalty = 0.08
        cfg.overprune_coef = 0.08
        cfg.machine_balance_coef = 0.06
    elif p == "f3_uncertainty_feat":
        cfg.feature_mode = "machine_heavy_uncertainty"
        cfg.strategy_mode = "ppo"
        cfg.enable_entropy_backup = True
        cfg.backup_entropy_threshold = 0.78
        cfg.backup_max_extra_pairs = 1
        cfg.retain_pos = 0.30
        cfg.retain_neg = -0.38
        cfg.quality_coef = 0.38
        cfg.quality_hard_penalty = 0.10
        cfg.mismatch_penalty = 0.08
        cfg.overprune_coef = 0.08
        cfg.machine_balance_coef = 0.06
    elif p == "f4_struct_feat":
        cfg.feature_mode = "machine_heavy_struct"
        cfg.strategy_mode = "ppo"
        cfg.enable_entropy_backup = True
        cfg.backup_entropy_threshold = 0.78
        cfg.backup_max_extra_pairs = 1
        cfg.retain_pos = 0.30
        cfg.retain_neg = -0.38
        cfg.quality_coef = 0.38
        cfg.quality_hard_penalty = 0.10
        cfg.mismatch_penalty = 0.08
        cfg.overprune_coef = 0.08
        cfg.machine_balance_coef = 0.06
    elif p == "f5_rel_uncertainty_feat":
        cfg.feature_mode = "machine_heavy_rel_ect_uncertainty"
        cfg.strategy_mode = "ppo"
        cfg.enable_entropy_backup = True
        cfg.backup_entropy_threshold = 0.78
        cfg.backup_max_extra_pairs = 1
        cfg.retain_pos = 0.30
        cfg.retain_neg = -0.38
        cfg.quality_coef = 0.38
        cfg.quality_hard_penalty = 0.10
        cfg.mismatch_penalty = 0.08
        cfg.overprune_coef = 0.08
        cfg.machine_balance_coef = 0.06
    elif p == "i1_r2_f1":
        cfg.feature_mode = "machine_heavy_rel_ect"
        cfg.strategy_mode = "ppo"
        cfg.enable_entropy_backup = True
        cfg.backup_entropy_threshold = 0.72
        cfg.backup_max_extra_pairs = 2
        cfg.retain_pos = 0.30
        cfg.retain_neg = -0.38
        cfg.quality_coef = 0.38
        cfg.quality_hard_penalty = 0.10
        cfg.mismatch_penalty = 0.08
        cfg.overprune_coef = 0.08
        cfg.machine_balance_coef = 0.06
    elif p == "i2_r2_f3":
        cfg.feature_mode = "machine_heavy_uncertainty"
        cfg.strategy_mode = "ppo"
        cfg.enable_entropy_backup = True
        cfg.backup_entropy_threshold = 0.72
        cfg.backup_max_extra_pairs = 2
        cfg.retain_pos = 0.30
        cfg.retain_neg = -0.38
        cfg.quality_coef = 0.38
        cfg.quality_hard_penalty = 0.10
        cfg.mismatch_penalty = 0.08
        cfg.overprune_coef = 0.08
        cfg.machine_balance_coef = 0.06
    elif p == "i3_r4_f1":
        cfg.feature_mode = "machine_heavy_rel_ect"
        cfg.strategy_mode = "ppo"
        cfg.enable_entropy_backup = True
        cfg.backup_entropy_threshold = 0.78
        cfg.backup_max_extra_pairs = 1
        cfg.retain_pos = 0.30
        cfg.retain_neg = -0.38
        cfg.quality_coef = 0.38
        cfg.quality_hard_penalty = 0.10
        cfg.mismatch_penalty = 0.08
        cfg.overprune_coef = 0.08
        cfg.machine_balance_coef = 0.06
        cfg.lr = 1.5e-4
        cfg.target_kl = 0.015
        cfg.ppo_epochs = 5
        cfg.minibatch_size = 320
        cfg.dropout = 0.08
    elif p == "i4_r3_f5":
        cfg.feature_mode = "machine_heavy_rel_ect_uncertainty"
        cfg.strategy_mode = "ppo"
        cfg.enable_entropy_backup = True
        cfg.backup_entropy_threshold = 0.78
        cfg.backup_max_extra_pairs = 1
        cfg.retain_pos = 0.30
        cfg.retain_neg = -0.38
        cfg.quality_coef = 0.38
        cfg.quality_hard_threshold = 0.06
        cfg.quality_hard_penalty = 0.12
        cfg.mismatch_penalty = 0.08
        cfg.overprune_coef = 0.08
        cfg.machine_balance_coef = 0.06
    else:
        raise ValueError(f"Unsupported profile: {profile}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train learnable AgentM with frozen AgentO and AgentC")
    parser.add_argument("--profile", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--train-count", type=int, default=10)
    parser.add_argument("--val-count", type=int, default=2)
    parser.add_argument("--test-count", type=int, default=3)
    parser.add_argument("--max-episode-steps", type=int, default=0)
    parser.add_argument("--init-m-ckpt", type=str, default="")
    parser.add_argument("--allow-short-epochs", action="store_true")
    args = parser.parse_args()

    if int(args.gpu) >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))

    root = Path(__file__).resolve().parent
    cfg = build_default_cfg(
        root=root,
        epochs=args.epochs,
        seed=args.seed,
        save_dir=args.save_dir,
        train_count=int(args.train_count),
        val_count=int(args.val_count),
        test_count=int(args.test_count),
    )
    _apply_profile(cfg, profile=args.profile)

    cfg.init_m_checkpoint_path = str(args.init_m_ckpt)

    # Keep epoch floor requested by user task.
    if bool(args.allow_short_epochs) or len(str(args.init_m_ckpt).strip()) > 0:
        cfg.epochs = int(max(1, int(args.epochs)))
    else:
        cfg.epochs = int(max(50, int(args.epochs)))
    cfg.max_episode_steps = int(args.max_episode_steps)

    summary = train_agent_m_learnable_with_frozen_oc(cfg)
    print(
        json.dumps(
            {
                "profile": args.profile,
                "best_epoch": int(summary["best_epoch"]),
                "best_val_ms": float(summary["best_val_ms"]),
                "test_mean_ms": float(summary["test"]["mean_ms"]),
                "test_mean_reward": float(summary["test"]["mean_reward"]),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

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

from Agent_M import AgentMConfig, Agent_M

_ORIG_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]
from Train_C import build_env_for_case, make_case_split, seed_everything
from Train_O_frozenC import Agent_O, AgentOConfig, FrozenAgentC
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


def filter_obs_by_ops_and_machines(obs: Dict[str, object], selected_ops: List[int], keep_indices: List[int]) -> Dict[str, object]:
    pairs = list(obs.get("candidate_pairs", []))
    if len(pairs) == 0:
        return obs

    selected_set = {int(x) for x in selected_ops}
    idx_set = {int(i) for i in keep_indices}
    idx_keep = [i for i, (op, _mch) in enumerate(pairs) if i in idx_set and int(op) in selected_set]

    if len(idx_keep) == 0:
        pair_feat = np.asarray(obs.get("pair_feat", []), dtype=np.float32)
        if pair_feat.ndim == 2 and pair_feat.shape[0] > 0 and pair_feat.shape[1] > 4:
            idx_keep = [int(np.argmin(pair_feat[:, 4]))]
        else:
            idx_keep = [0]

    return _slice_candidate_obs(obs, idx_keep)


@dataclass
class TrainMConfig:
    train_cases: List[str]
    val_cases: List[str]
    test_cases: List[str]

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    reset_rule: str = "FIFO_SPT"

    epochs: int = 12
    eval_interval: int = 1
    extra_step_budget: int = 40
    quiet_env: bool = True

    save_dir: str = "checkpoints_agent_m"
    checkpoint_name: str = "agent_m_best.json"

    o_checkpoint_path: str = ""
    c_checkpoint_path: str = ""

    m_mode: str = "hard"
    hard_topk: int = 2
    dynamic_k_base: int = 2
    dynamic_entropy_gain: float = 1.0
    dynamic_k_max: int = 4
    w_eet: float = 1.0
    w_pt: float = 0.35
    w_queue: float = 0.12
    fallback_entropy_threshold: float = 0.95
    fallback_entropy_extra_k: int = 1
    fallback_quality_gap_threshold: float = 0.08
    fallback_quality_extra_k: int = 1


def _load_frozen_o(cfg: TrainMConfig) -> Any:
    payload = torch.load(cfg.o_checkpoint_path, map_location=cfg.device)
    o_cfg = AgentOConfig(**payload["config"])
    o_cfg.device = cfg.device
    agent_o = Agent_O(config=o_cfg, global_dim=int(payload["global_dim"]), op_feat_dim=int(payload["op_feat_dim"]))
    agent_o.load(cfg.o_checkpoint_path, strict=True)
    agent_o.policy.eval()
    return agent_o


def _load_frozen_c(cfg: TrainMConfig) -> FrozenAgentC:
    return FrozenAgentC(
        ckpt_path=cfg.c_checkpoint_path,
        device=cfg.device,
        disable_topk_prefilter=False,
        max_candidates_floor=0,
    )


def run_episode_m_with_frozen_oc(
    agent_o: Any,
    frozen_c: FrozenAgentC,
    agent_m: Agent_M,
    case_path: str,
    cfg: TrainMConfig,
    deterministic: bool = True,
) -> Dict[str, float]:
    env, total_tasks = build_env_for_case(case_path, reset_rule=cfg.reset_rule, quiet=cfg.quiet_env)
    max_steps = int(total_tasks) + int(cfg.extra_step_budget)

    ep_reward = 0.0
    step_count = 0
    stat_keep_pairs = 0.0
    stat_k = 0.0
    stat_quality_gap = 0.0
    stat_fallback = 0.0
    stat_entropy = 0.0
    stat_retain_pair = 0.0

    done = False
    for _ in range(max_steps):
        full_obs = env.get_agent_c_obs(batch_idx=0)

        o_info = agent_o.select_action(full_obs, deterministic=deterministic, topk_k=5)
        selected_ops = [int(x) for x in o_info.get("selected_ops", [])]
        if len(selected_ops) == 0:
            selected_ops = [int(o_info["action_op_id"])]

        c_full_op, c_full_mch, c_full_info = frozen_c.act(full_obs, deterministic=True)
        c_entropy = float(c_full_info.get("entropy", 0.0))

        m_info = agent_m.select_pair_indices(full_obs, selected_ops=selected_ops, c_entropy=c_entropy)
        filtered_obs = filter_obs_by_ops_and_machines(full_obs, selected_ops=selected_ops, keep_indices=m_info["keep_indices"])

        c_op, c_mch, _ = frozen_c.act(filtered_obs, deterministic=True)

        if (int(c_full_op), int(c_full_mch)) in filtered_obs.get("legal_pairs_set", set()):
            stat_retain_pair += 1.0

        out = env.step_with_pair(c_op, c_mch, batch_idx=0)
        reward = float(out[2][0])
        done = bool(out[3][0])

        ep_reward += reward
        step_count += 1
        stat_keep_pairs += float(len(filtered_obs.get("candidate_pairs", [])))
        stat_k += float(m_info.get("k_used", 0.0))
        stat_quality_gap += float(m_info.get("quality_gap", 0.0))
        stat_fallback += float(m_info.get("fallback_triggered", 0.0))
        stat_entropy += c_entropy

        if done:
            break

    denom = max(step_count, 1)
    return {
        "reward": float(ep_reward),
        "makespan": float(env.LBm[0].max()),
        "steps": float(step_count),
        "done_rate": 1.0 if done else 0.0,
        "mean_keep_pairs": float(stat_keep_pairs / denom),
        "mean_k_used": float(stat_k / denom),
        "mean_quality_gap": float(stat_quality_gap / denom),
        "mean_fallback_trigger_rate": float(stat_fallback / denom),
        "mean_c_entropy": float(stat_entropy / denom),
        "mean_retain_pair_rate": float(stat_retain_pair / denom),
    }


def evaluate_split(
    agent_o: Any,
    frozen_c: FrozenAgentC,
    agent_m: Agent_M,
    cases: List[str],
    cfg: TrainMConfig,
) -> Dict[str, float]:
    rows = [run_episode_m_with_frozen_oc(agent_o, frozen_c, agent_m, c, cfg, deterministic=True) for c in cases]
    if len(rows) == 0:
        return {
            "num_cases": 0,
            "mean_reward": 0.0,
            "mean_ms": float("inf"),
            "mean_steps": 0.0,
            "mean_keep_pairs": 0.0,
            "mean_k_used": 0.0,
            "mean_quality_gap": 0.0,
            "mean_fallback_trigger_rate": 0.0,
            "mean_c_entropy": 0.0,
            "mean_retain_pair_rate": 0.0,
        }

    def avg(k: str) -> float:
        return float(np.mean([x[k] for x in rows]))

    return {
        "num_cases": int(len(rows)),
        "mean_reward": avg("reward"),
        "mean_ms": avg("makespan"),
        "mean_steps": avg("steps"),
        "mean_keep_pairs": avg("mean_keep_pairs"),
        "mean_k_used": avg("mean_k_used"),
        "mean_quality_gap": avg("mean_quality_gap"),
        "mean_fallback_trigger_rate": avg("mean_fallback_trigger_rate"),
        "mean_c_entropy": avg("mean_c_entropy"),
        "mean_retain_pair_rate": avg("mean_retain_pair_rate"),
    }


def train_agent_m_with_frozen_oc(cfg: TrainMConfig) -> Dict[str, Any]:
    if not cfg.o_checkpoint_path:
        raise ValueError("o_checkpoint_path is required")
    if not cfg.c_checkpoint_path:
        raise ValueError("c_checkpoint_path is required")

    seed_everything(int(cfg.seed))

    agent_o = _load_frozen_o(cfg)
    frozen_c = _load_frozen_c(cfg)
    m_cfg = AgentMConfig(
        mode=cfg.m_mode,
        hard_topk=cfg.hard_topk,
        dynamic_k_base=cfg.dynamic_k_base,
        dynamic_entropy_gain=cfg.dynamic_entropy_gain,
        dynamic_k_max=cfg.dynamic_k_max,
        w_eet=cfg.w_eet,
        w_pt=cfg.w_pt,
        w_queue=cfg.w_queue,
        fallback_entropy_threshold=cfg.fallback_entropy_threshold,
        fallback_entropy_extra_k=cfg.fallback_entropy_extra_k,
        fallback_quality_gap_threshold=cfg.fallback_quality_gap_threshold,
        fallback_quality_extra_k=cfg.fallback_quality_extra_k,
    )
    agent_m = Agent_M(m_cfg)

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_ms = float("inf")
    best_epoch = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, int(cfg.epochs) + 1):
        train_cases = list(cfg.train_cases)
        random.Random(cfg.seed + epoch).shuffle(train_cases)

        train_rows = [run_episode_m_with_frozen_oc(agent_o, frozen_c, agent_m, c, cfg, deterministic=False) for c in train_cases]

        train_metrics = {
            "num_cases": int(len(train_rows)),
            "mean_reward": float(np.mean([x["reward"] for x in train_rows])),
            "mean_ms": float(np.mean([x["makespan"] for x in train_rows])),
            "mean_steps": float(np.mean([x["steps"] for x in train_rows])),
            "mean_keep_pairs": float(np.mean([x["mean_keep_pairs"] for x in train_rows])),
            "mean_k_used": float(np.mean([x["mean_k_used"] for x in train_rows])),
            "mean_quality_gap": float(np.mean([x["mean_quality_gap"] for x in train_rows])),
            "mean_fallback_trigger_rate": float(np.mean([x["mean_fallback_trigger_rate"] for x in train_rows])),
            "mean_c_entropy": float(np.mean([x["mean_c_entropy"] for x in train_rows])),
            "mean_retain_pair_rate": float(np.mean([x["mean_retain_pair_rate"] for x in train_rows])),
        }

        val_metrics = evaluate_split(agent_o, frozen_c, agent_m, cfg.val_cases, cfg)

        improved = val_metrics["mean_ms"] < best_val_ms
        if improved:
            best_val_ms = float(val_metrics["mean_ms"])
            best_epoch = int(epoch)
            payload = {
                "epoch": int(epoch),
                "config": asdict(cfg),
                "agent_m_config": asdict(m_cfg),
                "best_val_ms": float(best_val_ms),
            }
            (save_dir / cfg.checkpoint_name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        row = {
            "epoch": int(epoch),
            "train": train_metrics,
            "val": val_metrics,
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
    (save_dir / "train_m_summary.json").write_text(json.dumps(final_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return final_summary


def build_default_cfg(root: Path, mode: str, epochs: int, seed: int, save_dir: str) -> TrainMConfig:
    all_cases = sorted((root / "1_Brandimarte").glob("BrandimarteMk*.fjs"))
    random.Random(seed).shuffle(all_cases)
    train_cases, val_cases, test_cases = make_case_split(all_cases, train_count=10, val_count=2)

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

    cfg = TrainMConfig(
        train_cases=[str(x) for x in train_cases],
        val_cases=[str(x) for x in val_cases],
        test_cases=[str(x) for x in test_cases],
        seed=int(seed),
        device="cuda" if torch.cuda.is_available() else "cpu",
        epochs=int(epochs),
        o_checkpoint_path=str(o_ckpt),
        c_checkpoint_path=str(c_ckpt),
        m_mode=str(mode),
        save_dir=str(root / save_dir),
    )

    if mode == "hard":
        cfg.hard_topk = 2
    elif mode == "dynamic":
        cfg.dynamic_k_base = 2
        cfg.dynamic_entropy_gain = 1.0
        cfg.dynamic_k_max = 4
    elif mode == "dynamic_fallback":
        cfg.dynamic_k_base = 2
        cfg.dynamic_entropy_gain = 1.0
        cfg.dynamic_k_max = 4
        cfg.fallback_entropy_threshold = 0.95
        cfg.fallback_entropy_extra_k = 1
        cfg.fallback_quality_gap_threshold = 0.08
        cfg.fallback_quality_extra_k = 1
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AgentM with frozen AgentO and AgentC")
    parser.add_argument("--mode", type=str, required=True, choices=["hard", "dynamic", "dynamic_fallback"])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="checkpoints_agent_m")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cfg = build_default_cfg(root=root, mode=args.mode, epochs=args.epochs, seed=args.seed, save_dir=args.save_dir)
    summary = train_agent_m_with_frozen_oc(cfg)
    print(json.dumps({
        "mode": args.mode,
        "best_epoch": summary["best_epoch"],
        "best_val_ms": summary["best_val_ms"],
        "test_mean_ms": summary["test"]["mean_ms"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

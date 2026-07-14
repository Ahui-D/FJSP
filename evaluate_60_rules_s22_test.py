from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_ORIG_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]
import numpy as np

from Train_C import build_env_for_case, call_silently
from decoupled_dispatch_rules import (
    JOB_RULE_NAMES,
    MACHINE_RULE_NAMES,
    get_job_rule_func,
    get_job_rule_mode,
    get_machine_rule_func,
    prepare_rule_state,
    select_topk_job_indices,
)
sys.argv = _ORIG_ARGV


def _rule_state_from_env(env) -> Dict[str, object]:
    state = env._get_runtime_state(batch_idx=0, copy_arrays=False)
    normalized_mch_time = np.asarray(state["mch_time"], dtype=np.float32)
    normalized_job_time = np.asarray(state["job_time"], dtype=np.float32)
    rule_state = prepare_rule_state(
        job_time=normalized_job_time,
        num_operation=state["num_operation"],
        dispatched_num_opera=state["dispatched_num_opera"],
        input_min=state["input_min"],
        job_col=state["job_col"],
        input_max=state["input_max"],
        dur=state["dur"],
        mask_mch=state["mask_mch"],
    )
    machine_state = {
        "mch_time": normalized_mch_time,
        "job_time": normalized_job_time,
        "machine_queue_len": state["machine_queue_len"],
        "dur": np.asarray(state["dur"], dtype=np.float32),
        "mask_mch": np.asarray(state["mask_mch"]).astype(bool),
        "last_col": np.asarray(state["last_col"], dtype=np.int64),
        "first_col": np.asarray(state["first_col"], dtype=np.int64),
        "task_to_row": np.asarray(state["task_to_row"], dtype=np.int64),
        "task_to_col": np.asarray(state["task_to_col"], dtype=np.int64),
        "number_of_machines": int(state["number_of_machines"]),
    }
    return {"runtime": state, "rule": rule_state, "machine": machine_state}


def select_action_by_compound_rule(env, job_rule: str, machine_rule: str) -> Tuple[int, int]:
    states = _rule_state_from_env(env)
    runtime = states["runtime"]
    rule_state = states["rule"]
    machine_state = states["machine"]
    available_jobs = np.asarray(runtime["available_jobs"], dtype=np.int64)
    if available_jobs.size == 0:
        raise RuntimeError("No available jobs")

    scores = get_job_rule_func(job_rule)(rule_state)
    selected_jobs = select_topk_job_indices(
        scores=scores,
        available_jobs=available_jobs,
        mode=get_job_rule_mode(job_rule),
        topk_jobs=1,
    )
    if selected_jobs.size == 0:
        raise RuntimeError("Job rule selected no job")
    job = int(selected_jobs[0])
    col = int(runtime["job_col"][job])
    op_id = int(runtime["first_col"][job] + col)

    ranked_machines = get_machine_rule_func(machine_rule)(machine_state, op_id)
    if not ranked_machines:
        feasible = env._get_feasible_machine_indices_from_state(runtime, row=job, col=col)
        if feasible.size == 0:
            raise RuntimeError(f"No feasible machine for op={op_id}")
        mch = int(feasible[0])
    else:
        mch = int(ranked_machines[0])
    return int(op_id), int(mch)


def evaluate_rule_on_case(case_path: str, rule_name: str, quiet: bool = True, extra_step_budget: int = 40) -> Dict[str, object]:
    job_rule, machine_rule = rule_name.split("_", 1)
    env, total_tasks = build_env_for_case(case_path, reset_rule=rule_name, quiet=quiet)
    done = False
    total_reward = 0.0
    steps = 0
    error = ""
    try:
        for _ in range(int(total_tasks) + int(extra_step_budget)):
            op_id, mch_id = select_action_by_compound_rule(env, job_rule, machine_rule)
            out = call_silently(env.step, np.asarray([op_id], dtype=np.int64), np.asarray([mch_id], dtype=np.int64), quiet=quiet)
            reward = float(out[2][0])
            done = bool(out[3][0])
            total_reward += reward
            steps += 1
            if done:
                break
    except Exception as exc:
        error = repr(exc)
    return {
        "case_name": Path(case_path).name,
        "case_path": str(case_path),
        "rule": rule_name,
        "makespan": float(env.LBm[0].max()),
        "reward": float(total_reward),
        "steps": int(steps),
        "done": bool(done),
        "error": error,
    }


def load_test_cases(summary_path: Path) -> Tuple[List[str], Dict[str, object]]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    cfg = data.get("config", {})
    cases = [str(x) for x in cfg.get("test_cases", [])]
    if not cases:
        split = cfg.get("split_source")
        if split:
            payload = json.loads(Path(split).read_text(encoding="utf-8"))
            cases = [str(x) for x in payload.get("test_cases", [])]
    if not cases:
        raise RuntimeError("No test cases found in summary/config")
    return cases, data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="exp_runs/s22_3sd10x5/train_oc_mappo_summary.json")
    ap.add_argument("--output-dir", default="exp_runs/rule60_s22_3sd10x5_test")
    ap.add_argument("--extra-step-budget", type=int, default=40)
    ap.add_argument("--quiet", type=int, default=1)
    args = ap.parse_args()

    summary_path = Path(args.summary)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases, summary = load_test_cases(summary_path)

    rules = [f"{j}_{m}" for j in JOB_RULE_NAMES for m in MACHINE_RULE_NAMES]
    case_rows: List[Dict[str, object]] = []
    ranking_rows: List[Dict[str, object]] = []
    print(f"[rule60] cases={len(cases)} rules={len(rules)} summary={summary_path}", flush=True)

    for r_idx, rule in enumerate(rules, 1):
        rows = []
        for c_idx, case in enumerate(cases, 1):
            row = evaluate_rule_on_case(
                case_path=case,
                rule_name=rule,
                quiet=bool(args.quiet),
                extra_step_budget=int(args.extra_step_budget),
            )
            rows.append(row)
            case_rows.append(row)
        ok = [x for x in rows if not x["error"] and bool(x["done"])]
        mean_makespan = float(np.mean([x["makespan"] for x in ok])) if ok else float("inf")
        std_makespan = float(np.std([x["makespan"] for x in ok])) if ok else float("inf")
        mean_reward = float(np.mean([x["reward"] for x in ok])) if ok else float("nan")
        ranking_rows.append({
            "rule": rule,
            "mean_makespan": mean_makespan,
            "std_makespan": std_makespan,
            "mean_reward": mean_reward,
            "num_cases": len(cases),
            "num_success": len(ok),
            "num_errors": len(rows) - len(ok),
            "success_rate": len(ok) / max(1, len(rows)),
        })
        print(f"[{r_idx:02d}/{len(rules)}] {rule} mean_makespan={mean_makespan:.4f} success={len(ok)}/{len(rows)}", flush=True)

    ranking_rows.sort(key=lambda x: (float(x["mean_makespan"]), -float(x["success_rate"]), str(x["rule"])))
    for rank, row in enumerate(ranking_rows, 1):
        row["rank"] = rank

    with (out_dir / "rule60_case_rows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(case_rows[0].keys()))
        writer.writeheader(); writer.writerows(case_rows)
    with (out_dir / "rule60_ranking.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ranking_rows[0].keys()))
        writer.writeheader(); writer.writerows(ranking_rows)

    model_test = summary.get("test_eval", {}) if isinstance(summary, dict) else {}
    report = {
        "summary_path": str(summary_path),
        "split_source": summary.get("config", {}).get("split_source", ""),
        "num_cases": len(cases),
        "num_rules": len(rules),
        "best_rule": ranking_rows[0],
        "rule_mean_makespan_mean": float(np.mean([float(x["mean_makespan"]) for x in ranking_rows])),
        "rule_mean_makespan_std": float(np.std([float(x["mean_makespan"]) for x in ranking_rows])),
        "model_test_mean_makespan": model_test.get("mean_makespan"),
        "model_minus_best_rule": (float(model_test.get("mean_makespan")) - float(ranking_rows[0]["mean_makespan"]) if model_test.get("mean_makespan") is not None else None),
        "top10_rules": ranking_rows[:10],
    }
    (out_dir / "rule60_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

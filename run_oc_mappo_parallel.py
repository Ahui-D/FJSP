from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class ExperimentSpec:
    name: str
    seed: int
    o_topk: int
    clip_ratio: float
    c_lr: float
    o_lr: float
    critic_lr: float


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _parse_int_grid(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _parse_float_grid(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def build_default_experiments(args) -> List[ExperimentSpec]:
    specs: List[ExperimentSpec] = []
    seeds = _parse_int_grid(args.seeds)
    o_topk_grid = _parse_int_grid(args.o_topk_grid)
    clip_ratio_grid = _parse_float_grid(args.clip_ratio_grid)

    for seed in seeds:
        for o_topk in o_topk_grid:
            for clip_ratio in clip_ratio_grid:
                name = f"s{seed}_k{o_topk}_clip{str(clip_ratio).replace('.', 'p')}"
                specs.append(
                    ExperimentSpec(
                        name=name,
                        seed=seed,
                        o_topk=o_topk,
                        clip_ratio=clip_ratio,
                        c_lr=float(args.c_lr),
                        o_lr=float(args.o_lr),
                        critic_lr=float(args.critic_lr),
                    )
                )
    return specs


def start_process(
    repo_root: Path,
    spec: ExperimentSpec,
    gpu_id: int,
    run_dir: Path,
    args,
) -> subprocess.Popen:
    save_dir = run_dir / spec.name
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "train.log"

    cmd = [
        sys.executable,
        "-u",
        "Train_OC_MAPPO.py",
        "--case-dir",
        str(args.case_dir),
        "--pattern",
        str(args.pattern),
        "--train-count",
        str(args.train_count),
        "--val-count",
        str(args.val_count),
        "--seed",
        str(spec.seed),
        "--device",
        "cuda",
        "--epochs",
        str(args.epochs),
        "--episodes-per-update",
        str(args.episodes_per_update),
        "--eval-interval",
        str(args.eval_interval),
        "--extra-step-budget",
        str(args.extra_step_budget),
        "--save-dir",
        str(save_dir),
        "--o-topk",
        str(spec.o_topk),
        "--c-lr",
        str(spec.c_lr),
        "--o-lr",
        str(spec.o_lr),
        "--critic-lr",
        str(spec.critic_lr),
        "--clip-ratio",
        str(spec.clip_ratio),
        "--ppo-epochs",
        str(args.ppo_epochs),
        "--minibatch-size",
        str(args.minibatch_size),
        "--target-kl-c",
        str(args.target_kl_c),
        "--target-kl-o",
        str(args.target_kl_o),
        "--o-reward-alpha-env",
        str(args.o_reward_alpha_env),
        "--o-reward-beta-shape",
        str(args.o_reward_beta_shape),
        "--o-reward-clip-abs",
        str(args.o_reward_clip_abs),
        "--reward-retain-pos",
        str(args.reward_retain_pos),
        "--reward-retain-neg",
        str(args.reward_retain_neg),
        "--reward-quality-coef",
        str(args.reward_quality_coef),
        "--reward-quality-clip",
        str(args.reward_quality_clip),
        "--reward-quality-hard-threshold",
        str(args.reward_quality_hard_threshold),
        "--reward-quality-hard-penalty",
        str(args.reward_quality_hard_penalty),
        "--reward-mismatch-penalty",
        str(args.reward_mismatch_penalty),
        "--reward-makespan-terminal-coef",
        str(args.reward_makespan_terminal_coef),
        "--o-topk-min",
        str(args.o_topk_min),
        "--o-topk-max",
        str(args.o_topk_max),
        "--o-topk-entropy-gain",
        str(args.o_topk_entropy_gain),
        "--o-entropy-fallback-threshold",
        str(args.o_entropy_fallback_threshold),
        "--o-entropy-fallback-extra-ops",
        str(args.o_entropy_fallback_extra_ops),
        "--o-reward-fallback-scale",
        str(args.o_reward_fallback_scale),
        "--o-reference-c-deterministic",
        str(int(args.o_reference_c_deterministic)),
    ]

    if args.split_json:
        cmd += ["--split-json", str(args.split_json)]

    if args.init_c_checkpoint_path:
        cmd += ["--init-c-checkpoint-path", str(args.init_c_checkpoint_path)]
    if args.init_o_checkpoint_path:
        cmd += ["--init-o-checkpoint-path", str(args.init_o_checkpoint_path)]
    if args.quiet_env:
        cmd += ["--quiet-env"]
    if not bool(args.enable_ppo_early_stop):
        cmd += ["--disable-ppo-early-stop"]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"

    return subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        env=env,
        stdout=open(log_path, "w", encoding="utf-8", buffering=1),
        stderr=subprocess.STDOUT,
    )


def collect_results(run_dir: Path, specs: List[ExperimentSpec]) -> Dict[str, object]:
    rows = []
    for spec in specs:
        summary_path = run_dir / spec.name / "train_oc_mappo_summary.json"
        if not summary_path.exists():
            rows.append(
                {
                    "name": spec.name,
                    "seed": spec.seed,
                    "o_topk": spec.o_topk,
                    "clip_ratio": spec.clip_ratio,
                    "status": "missing_summary",
                }
            )
            continue

        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "name": spec.name,
                    "seed": spec.seed,
                    "o_topk": spec.o_topk,
                    "clip_ratio": spec.clip_ratio,
                    "status": "ok",
                    "best_epoch": summary.get("best_epoch", -1),
                    "best_val_makespan": summary.get("best_val_makespan", None),
                    "test_mean_makespan": summary.get("test_eval", {}).get("mean_makespan", None),
                    "test_mean_reward": summary.get("test_eval", {}).get("mean_reward", None),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "name": spec.name,
                    "seed": spec.seed,
                    "o_topk": spec.o_topk,
                    "clip_ratio": spec.clip_ratio,
                    "status": f"bad_summary: {repr(exc)}",
                }
            )

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    aggregate = {
        "run_dir": str(run_dir),
        "num_experiments": len(specs),
        "num_ok": len(ok_rows),
        "num_failed": len(specs) - len(ok_rows),
        "best_by_test_makespan": None,
        "rows": rows,
    }

    if len(ok_rows) > 0:
        best = min(ok_rows, key=lambda r: float(r.get("test_mean_makespan", float("inf"))))
        aggregate["best_by_test_makespan"] = best

    json_path = run_dir / "parallel_results_summary.json"
    json_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = run_dir / "parallel_results_summary.csv"
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return aggregate


def parse_args(args=None):
    p = argparse.ArgumentParser(description="Run O+C MAPPO experiments in parallel on multi-GPU")
    p.add_argument("--repo-root", type=str, default=".")
    p.add_argument("--case-dir", type=str, default="1_Brandimarte")
    p.add_argument("--pattern", type=str, default="BrandimarteMk*.fjs")
    p.add_argument("--split-json", type=str, default="")
    p.add_argument("--train-count", type=int, default=10)
    p.add_argument("--val-count", type=int, default=2)

    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--episodes-per-update", type=int, default=8)
    p.add_argument("--eval-interval", type=int, default=1)
    p.add_argument("--extra-step-budget", type=int, default=40)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=256)

    p.add_argument("--target-kl-c", type=float, default=0.02)
    p.add_argument("--target-kl-o", type=float, default=0.01)
    p.add_argument("--o-reward-alpha-env", type=float, default=0.30)
    p.add_argument("--o-reward-beta-shape", type=float, default=1.00)
    p.add_argument("--o-reward-clip-abs", type=float, default=1.0)
    p.add_argument("--reward-retain-pos", type=float, default=0.20)
    p.add_argument("--reward-retain-neg", type=float, default=-0.45)
    p.add_argument("--reward-quality-coef", type=float, default=0.30)
    p.add_argument("--reward-quality-clip", type=float, default=0.30)
    p.add_argument("--reward-quality-hard-threshold", type=float, default=0.08)
    p.add_argument("--reward-quality-hard-penalty", type=float, default=0.18)
    p.add_argument("--reward-mismatch-penalty", type=float, default=0.08)
    p.add_argument("--reward-makespan-terminal-coef", type=float, default=0.01)
    p.add_argument("--o-topk-min", type=int, default=2)
    p.add_argument("--o-topk-max", type=int, default=8)
    p.add_argument("--o-topk-entropy-gain", type=float, default=1.0)
    p.add_argument("--o-entropy-fallback-threshold", type=float, default=1.0)
    p.add_argument("--o-entropy-fallback-extra-ops", type=int, default=1)
    p.add_argument("--o-reward-fallback-scale", type=float, default=0.7)
    p.add_argument("--o-reference-c-deterministic", type=int, default=1)

    p.add_argument("--seeds", type=str, default="11,22,33,44")
    p.add_argument("--o-topk-grid", type=str, default="4,6")
    p.add_argument("--clip-ratio-grid", type=str, default="0.15,0.20")
    p.add_argument("--c-lr", type=float, default=2e-4)
    p.add_argument("--o-lr", type=float, default=1e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)

    p.add_argument("--gpus", type=str, default="0,1")
    p.add_argument("--slots-per-gpu", type=int, default=8)
    p.add_argument("--poll-seconds", type=float, default=10.0)
    p.add_argument("--enable-ppo-early-stop", action="store_true", default=False)

    p.add_argument("--init-c-checkpoint-path", type=str, default="")
    p.add_argument("--init-o-checkpoint-path", type=str, default="")

    p.add_argument("--output-root", type=str, default="checkpoints_oc_mappo_parallel")
    p.add_argument("--quiet-env", action="store_true", default=False)
    return p.parse_args(args=args)


def main(args=None):
    args = parse_args(args=args)
    repo_root = Path(args.repo_root).resolve()

    gpu_ids = [int(x.strip()) for x in str(args.gpus).split(",") if x.strip()]
    if len(gpu_ids) == 0:
        raise ValueError("No GPU ids provided")

    specs = build_default_experiments(args)
    output_root = (repo_root / args.output_root).resolve()
    run_dir = output_root / f"run_{_now_tag()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    queue = list(specs)
    active: Dict[int, List[Dict[str, object]]] = {gid: [] for gid in gpu_ids}
    finished = []

    print(
        f"[parallel] start {len(specs)} experiments on GPUs={gpu_ids}, slots_per_gpu={args.slots_per_gpu}, run_dir={run_dir}"
    )

    while len(queue) > 0 or any(len(v) > 0 for v in active.values()):
        for gid in gpu_ids:
            while len(queue) > 0 and len(active[gid]) < int(args.slots_per_gpu):
                spec = queue.pop(0)
                proc = start_process(repo_root=repo_root, spec=spec, gpu_id=gid, run_dir=run_dir, args=args)
                active[gid].append({"spec": spec, "proc": proc, "start_time": time.time()})
                print(f"[launch] gpu={gid} name={spec.name} pid={proc.pid}")

        time.sleep(float(args.poll_seconds))
        for gid in gpu_ids:
            remaining = []
            for item in active[gid]:
                proc: subprocess.Popen = item["proc"]
                ret = proc.poll()
                if ret is None:
                    remaining.append(item)
                    continue

                spec: ExperimentSpec = item["spec"]
                elapsed = time.time() - float(item["start_time"])
                status = "ok" if ret == 0 else f"exit_{ret}"
                print(f"[finish] gpu={gid} name={spec.name} status={status} elapsed_sec={elapsed:.1f}")
                finished.append({"name": spec.name, "status": status, "elapsed_sec": elapsed})

            active[gid] = remaining

    aggregate = collect_results(run_dir=run_dir, specs=specs)
    best = aggregate.get("best_by_test_makespan")
    if best:
        print(
            "[parallel-done] "
            f"ok={aggregate['num_ok']}/{aggregate['num_experiments']} "
            f"best={best['name']} test_ms={best.get('test_mean_makespan')}"
        )
    else:
        print(f"[parallel-done] ok={aggregate['num_ok']}/{aggregate['num_experiments']} no successful runs")


if __name__ == "__main__":
    main()

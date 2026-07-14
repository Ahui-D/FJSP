import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str, fh) -> None:
    line = f"[{_ts()}] {msg}"
    print(line)
    fh.write(line + "\n")
    fh.flush()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce best D1 run for 20 epochs (stable launcher)")
    p.add_argument("--repo-root", type=str, default=".")
    p.add_argument(
        "--source-summary",
        type=str,
        default="checkpoints_oc_mappo_top2_plan/phaseD1_seed33/run_20260403_122442/phaseD1/D1_R1_L8/train_oc_mappo_summary.json",
    )
    p.add_argument("--run-root", type=str, default="checkpoints_oc_mappo_top2_repro20e_bestD1_baseline")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--gpus", type=str, default="0,1")
    p.add_argument("--timestamp", type=str, default="")
    p.add_argument("--sequential", type=int, default=1)
    return p.parse_args()


def _build_cmd(cfg: Dict, split_json: str, save_dir: str, epochs: int) -> List[str]:
    return [
        sys.executable,
        "-u",
        "Train_OC_MAPPO.py",
        "--split-json",
        split_json,
        "--seed",
        str(int(cfg["seed"])),
        "--device",
        str(cfg["device"]),
        "--epochs",
        str(int(epochs)),
        "--episodes-per-update",
        str(int(cfg["episodes_per_update"])),
        "--eval-interval",
        str(int(cfg["eval_interval"])),
        "--extra-step-budget",
        str(int(cfg["extra_step_budget"])),
        "--save-dir",
        save_dir,
        "--o-topk",
        str(int(cfg["o_topk"])),
        "--o-model-type",
        str(cfg.get("o_model_type", "mlp")),
        "--c-lr",
        str(float(cfg["c_lr"])),
        "--o-lr",
        str(float(cfg["o_lr"])),
        "--critic-lr",
        str(float(cfg["critic_lr"])),
        "--clip-ratio-c",
        str(float(cfg["clip_ratio_c"])),
        "--clip-ratio-o",
        str(float(cfg["clip_ratio_o"])),
        "--ppo-epochs",
        str(int(cfg["ppo_epochs"])),
        "--minibatch-size",
        str(int(cfg["minibatch_size"])),
        "--target-kl-c",
        str(float(cfg["target_kl_c"])),
        "--target-kl-o",
        str(float(cfg["target_kl_o"])),
        "--o-reward-alpha-env",
        str(float(cfg["o_reward_alpha_env"])),
        "--o-reward-alpha-env-end",
        str(float(cfg["o_reward_alpha_env_end"])),
        "--o-reward-beta-shape",
        str(float(cfg["o_reward_beta_shape"])),
        "--o-reward-beta-shape-end",
        str(float(cfg["o_reward_beta_shape_end"])),
        "--o-reward-anneal-epochs",
        str(int(cfg["o_reward_anneal_epochs"])),
        "--o-reward-clip-abs",
        str(float(cfg["o_reward_clip_abs"])),
        "--reward-retain-pos",
        str(float(cfg["reward_retain_pos"])),
        "--reward-retain-neg",
        str(float(cfg["reward_retain_neg"])),
        "--reward-quality-coef",
        str(float(cfg["reward_quality_coef"])),
        "--reward-quality-clip",
        str(float(cfg["reward_quality_clip"])),
        "--reward-quality-hard-threshold",
        str(float(cfg["reward_quality_hard_threshold"])),
        "--reward-quality-hard-penalty",
        str(float(cfg["reward_quality_hard_penalty"])),
        "--reward-mismatch-penalty",
        str(float(cfg["reward_mismatch_penalty"])),
        "--reward-mismatch-only-if-not-retained",
        str(int(bool(cfg["reward_mismatch_only_if_not_retained"]))),
        "--reward-makespan-terminal-coef",
        str(float(cfg["reward_makespan_terminal_coef"])),
        "--o-topk-min",
        str(int(cfg["o_topk_min"])),
        "--o-topk-max",
        str(int(cfg["o_topk_max"])),
        "--o-topk-entropy-gain",
        str(float(cfg["o_topk_entropy_gain"])),
        "--o-entropy-fallback-threshold",
        str(float(cfg["o_entropy_fallback_threshold"])),
        "--o-entropy-fallback-extra-ops",
        str(int(cfg["o_entropy_fallback_extra_ops"])),
        "--o-reward-fallback-scale",
        str(float(cfg["o_reward_fallback_scale"])),
        "--o-reference-c-deterministic",
        str(int(bool(cfg["o_reference_c_deterministic"]))),
        "--value-coef-c",
        str(float(cfg["value_coef_c"])),
        "--value-coef-o",
        str(float(cfg["value_coef_o"])),
        "--use-huber-value-loss",
        str(int(bool(cfg["use_huber_value_loss"]))),
        "--value-huber-delta",
        str(float(cfg["value_huber_delta"])),
        "--value-clip-range",
        str(float(cfg["value_clip_range"])),
        "--c-use-graph-encoder",
        "0",
        "--critic-rich-state",
        "0",
        "--o-reward-schedule",
        "linear",
        "--reward-quality-hard-smooth",
        "0",
        "--disable-ppo-early-stop",
    ]


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    source_summary = (repo_root / args.source_summary).resolve()

    payload = json.loads(source_summary.read_text(encoding="utf-8"))
    cfg = payload["config"]
    split_json = str(Path(cfg["split_source"]).resolve())

    ts = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (repo_root / args.run_root / f"run_{ts}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids = [int(x.strip()) for x in str(args.gpus).split(",") if x.strip()]
    if not gpu_ids:
        gpu_ids = [0]

    pipeline_log = run_dir / "pipeline.log"
    summary_json = run_dir / "summary.json"
    results = []

    with open(pipeline_log, "w", encoding="utf-8") as log_fh:
        _log(f"run_dir={run_dir}", log_fh)
        _log(f"source_summary={source_summary}", log_fh)
        _log("mode=baseline_only (E1/E2/E3 switches hard-off)", log_fh)

        if int(args.sequential) != 1:
            _log("Only sequential mode is supported in this stable launcher; forcing sequential=1.", log_fh)

        for i in range(1, int(args.repeats) + 1):
            rep_name = f"rep{i}"
            rep_dir = run_dir / rep_name
            rep_dir.mkdir(parents=True, exist_ok=True)
            log_path = rep_dir / "train.log"

            gpu = gpu_ids[(i - 1) % len(gpu_ids)]
            cmd = _build_cmd(cfg=cfg, split_json=split_json, save_dir=str(rep_dir), epochs=int(args.epochs))
            env = os.environ.copy()
            env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONUNBUFFERED"] = "1"

            _log(f"launch rep={rep_name} gpu={gpu}", log_fh)
            with open(log_path, "w", encoding="utf-8", buffering=1) as out_fh:
                proc = subprocess.Popen(cmd, cwd=str(repo_root), env=env, stdout=out_fh, stderr=subprocess.STDOUT)
                rc = proc.wait()

            _log(f"done rep={rep_name} gpu={gpu} rc={rc}", log_fh)
            results.append(
                {
                    "rep": rep_name,
                    "gpu": gpu,
                    "returncode": int(rc),
                    "save_dir": str(rep_dir),
                    "log_path": str(log_path),
                }
            )

    summary = {
        "run_dir": str(run_dir),
        "source_summary": str(source_summary),
        "repeats": int(args.repeats),
        "epochs": int(args.epochs),
        "results": results,
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

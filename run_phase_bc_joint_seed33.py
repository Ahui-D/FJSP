import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List


@dataclass
class ExpSpec:
    phase: str
    name: str
    extra_args: List[str]


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str, fh) -> None:
    line = f"[{_ts()}] {msg}"
    print(line)
    fh.write(line + "\n")
    fh.flush()


def _build_specs() -> List[ExpSpec]:
    b_specs = [
        ExpSpec("phaseB", "B_R1_default", ["--o-reward-alpha-env-end", "0.70", "--o-reward-beta-shape-end", "0.35", "--o-reward-anneal-epochs", "180", "--reward-mismatch-only-if-not-retained", "1"]),
        ExpSpec("phaseB", "B_R2_env_tail", ["--o-reward-alpha-env-end", "0.85", "--o-reward-beta-shape-end", "0.20", "--o-reward-anneal-epochs", "180", "--reward-mismatch-only-if-not-retained", "1"]),
        ExpSpec("phaseB", "B_R3_slow_anneal", ["--o-reward-alpha-env-end", "0.70", "--o-reward-beta-shape-end", "0.35", "--o-reward-anneal-epochs", "240", "--reward-mismatch-only-if-not-retained", "1"]),
        ExpSpec("phaseB", "B_R4_fast_anneal", ["--o-reward-alpha-env-end", "0.70", "--o-reward-beta-shape-end", "0.35", "--o-reward-anneal-epochs", "120", "--reward-mismatch-only-if-not-retained", "1"]),
        ExpSpec("phaseB", "B_R5_gate_off", ["--o-reward-alpha-env-end", "0.70", "--o-reward-beta-shape-end", "0.35", "--o-reward-anneal-epochs", "180", "--reward-mismatch-only-if-not-retained", "0"]),
        ExpSpec("phaseB", "B_R6_balanced_tail", ["--o-reward-alpha-env-end", "0.60", "--o-reward-beta-shape-end", "0.50", "--o-reward-anneal-epochs", "180", "--reward-mismatch-only-if-not-retained", "1"]),
        ExpSpec("phaseB", "B_R7_env_slow", ["--o-reward-alpha-env-end", "0.80", "--o-reward-beta-shape-end", "0.20", "--o-reward-anneal-epochs", "240", "--reward-mismatch-only-if-not-retained", "1"]),
        ExpSpec("phaseB", "B_R8_env_fast", ["--o-reward-alpha-env-end", "0.80", "--o-reward-beta-shape-end", "0.20", "--o-reward-anneal-epochs", "120", "--reward-mismatch-only-if-not-retained", "1"]),
    ]

    c_specs = [
        ExpSpec("phaseC", "C_L1_default", ["--use-huber-value-loss", "1", "--value-huber-delta", "1.0", "--value-clip-range", "0.2", "--value-coef-c", "0.6", "--value-coef-o", "0.6", "--clip-ratio-c", "0.20", "--clip-ratio-o", "0.20"]),
        ExpSpec("phaseC", "C_L2_no_vclip", ["--use-huber-value-loss", "1", "--value-huber-delta", "1.0", "--value-clip-range", "0.0", "--value-coef-c", "0.6", "--value-coef-o", "0.6", "--clip-ratio-c", "0.20", "--clip-ratio-o", "0.20"]),
        ExpSpec("phaseC", "C_L3_no_huber", ["--use-huber-value-loss", "0", "--value-huber-delta", "1.0", "--value-clip-range", "0.2", "--value-coef-c", "0.6", "--value-coef-o", "0.6", "--clip-ratio-c", "0.20", "--clip-ratio-o", "0.20"]),
        ExpSpec("phaseC", "C_L4_low_o_value", ["--use-huber-value-loss", "1", "--value-huber-delta", "1.0", "--value-clip-range", "0.2", "--value-coef-c", "0.6", "--value-coef-o", "0.4", "--clip-ratio-c", "0.20", "--clip-ratio-o", "0.20"]),
        ExpSpec("phaseC", "C_L5_high_o_value", ["--use-huber-value-loss", "1", "--value-huber-delta", "1.0", "--value-clip-range", "0.2", "--value-coef-c", "0.6", "--value-coef-o", "0.8", "--clip-ratio-c", "0.20", "--clip-ratio-o", "0.20"]),
        ExpSpec("phaseC", "C_L6_clip_c018", ["--use-huber-value-loss", "1", "--value-huber-delta", "1.0", "--value-clip-range", "0.2", "--value-coef-c", "0.6", "--value-coef-o", "0.6", "--clip-ratio-c", "0.18", "--clip-ratio-o", "0.20"]),
        ExpSpec("phaseC", "C_L7_clip_o015", ["--use-huber-value-loss", "1", "--value-huber-delta", "1.0", "--value-clip-range", "0.2", "--value-coef-c", "0.6", "--value-coef-o", "0.6", "--clip-ratio-c", "0.20", "--clip-ratio-o", "0.15"]),
        ExpSpec("phaseC", "C_L8_value_clip01", ["--use-huber-value-loss", "1", "--value-huber-delta", "1.0", "--value-clip-range", "0.1", "--value-coef-c", "0.6", "--value-coef-o", "0.6", "--clip-ratio-c", "0.20", "--clip-ratio-o", "0.20"]),
    ]

    mixed: List[ExpSpec] = []
    for i in range(max(len(b_specs), len(c_specs))):
        if i < len(b_specs):
            mixed.append(b_specs[i])
        if i < len(c_specs):
            mixed.append(c_specs[i])
    return mixed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Phase B and C together (seed33)")
    p.add_argument("--repo-root", type=str, default=".")
    p.add_argument("--phase-root", type=str, default="checkpoints_oc_mappo_top2_plan")
    p.add_argument("--split-json", type=str, default="splits/brandimarte_seed42_v330_paths.json")
    p.add_argument("--seed", type=int, default=33)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--timestamp", type=str, default="")
    p.add_argument("--gpus", type=str, default="0,1")
    p.add_argument("--slots-per-gpu", type=int, default=4)
    p.add_argument("--poll-seconds", type=float, default=20.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    split_json = (repo_root / args.split_json).resolve()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    run_root = repo_root / args.phase_root / "phaseBC_joint_seed33" / f"run_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)

    log_path = run_root / "pipeline.log"
    summary_path = run_root / "summary.json"

    gpu_ids = [int(x.strip()) for x in str(args.gpus).split(",") if x.strip()]
    capacity = {gpu: int(args.slots_per_gpu) for gpu in gpu_ids}
    running: Dict[subprocess.Popen, Dict] = {}

    specs = _build_specs()

    base_cmd = [
        sys.executable,
        "-u",
        "Train_OC_MAPPO.py",
        "--split-json", str(split_json),
        "--seed", str(args.seed),
        "--device", "cuda",
        "--epochs", str(args.epochs),
        "--episodes-per-update", "8",
        "--eval-interval", "1",
        "--o-topk", "6",
        "--c-lr", "2e-4",
        "--o-lr", "1e-4",
        "--critic-lr", "3e-4",
        "--clip-ratio-c", "0.20",
        "--clip-ratio-o", "0.20",
        "--o-reward-alpha-env", "0.30",
        "--o-reward-beta-shape", "1.00",
        "--o-reward-clip-abs", "1.5",
        "--disable-ppo-early-stop",
    ]

    results = []

    with open(log_path, "w", encoding="utf-8") as log_fh:
        _log(f"run_root={run_root}", log_fh)
        _log(f"tasks={len(specs)} gpus={gpu_ids} slots_per_gpu={args.slots_per_gpu}", log_fh)

        queue = list(specs)

        while queue or running:
            # Reap finished
            finished = []
            for proc, info in running.items():
                rc = proc.poll()
                if rc is not None:
                    finished.append((proc, info, rc))

            for proc, info, rc in finished:
                running.pop(proc, None)
                capacity[info["gpu"]] += 1
                results.append({
                    "phase": info["phase"],
                    "name": info["name"],
                    "gpu": info["gpu"],
                    "returncode": rc,
                    "save_dir": info["save_dir"],
                })
                _log(f"done phase={info['phase']} name={info['name']} gpu={info['gpu']} rc={rc}", log_fh)

            # Launch new tasks while slots available
            for gpu in gpu_ids:
                while capacity[gpu] > 0 and queue:
                    spec = queue.pop(0)
                    save_dir = run_root / spec.phase / spec.name
                    save_dir.mkdir(parents=True, exist_ok=True)
                    out_path = save_dir / "launcher_stdout.log"

                    cmd = list(base_cmd)
                    cmd.extend(spec.extra_args)
                    cmd.extend(["--save-dir", str(save_dir)])

                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
                    env["PYTHONUNBUFFERED"] = "1"

                    out_fh = open(out_path, "w", encoding="utf-8", buffering=1)
                    proc = subprocess.Popen(cmd, cwd=str(repo_root), env=env, stdout=out_fh, stderr=subprocess.STDOUT)
                    running[proc] = {
                        "phase": spec.phase,
                        "name": spec.name,
                        "gpu": gpu,
                        "save_dir": str(save_dir),
                        "stdout": out_fh,
                    }
                    capacity[gpu] -= 1
                    _log(f"launch phase={spec.phase} name={spec.name} gpu={gpu}", log_fh)

            _log(
                "status queued={} running={} free_slots={}".format(
                    len(queue),
                    len(running),
                    {k: v for k, v in capacity.items()},
                ),
                log_fh,
            )
            time.sleep(float(args.poll_seconds))

        # close output files
        for info in running.values():
            try:
                info["stdout"].close()
            except Exception:
                pass

        ok = sum(1 for r in results if r["returncode"] == 0)
        _log(f"all_done total={len(results)} ok={ok}", log_fh)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"run_root": str(run_root), "results": results}, f, ensure_ascii=False, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

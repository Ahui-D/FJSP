from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class ScaleConfig:
    scale: str
    split_json: str
    seed: int
    num_envs: int
    rollout_workers: int
    episodes_per_update: int
    c_lr: float
    o_lr: float
    clip_ratio_c: float
    o_topk: int
    val_extra_samples: int


@dataclass
class ActiveJob:
    scale: str
    gpu: int
    proc: subprocess.Popen
    save_dir: Path
    log_file: Path
    start_time: float


SCALE_PRESETS: Dict[str, ScaleConfig] = {
    "10x5": ScaleConfig(
        scale="10x5",
        split_json="splits/3sd_10x5_single-scale_seed42_paths.json",
        seed=22,
        num_envs=24,
        rollout_workers=18,
        episodes_per_update=24,
        c_lr=1.5e-4,
        o_lr=1.0e-4,
        clip_ratio_c=0.18,
        o_topk=6,
        val_extra_samples=24,
    ),
    "15x10": ScaleConfig(
        scale="15x10",
        split_json="splits/3sd_15x10_single-scale_seed42_paths.json",
        seed=22,
        num_envs=20,
        rollout_workers=15,
        episodes_per_update=20,
        c_lr=1.0e-4,
        o_lr=1.0e-4,
        clip_ratio_c=0.15,
        o_topk=6,
        val_extra_samples=20,
    ),
    "20x5": ScaleConfig(
        scale="20x5",
        split_json="splits/3sd_20x5_single-scale_seed42_paths.json",
        seed=33,
        num_envs=20,
        rollout_workers=15,
        episodes_per_update=20,
        c_lr=1.0e-4,
        o_lr=1.0e-4,
        clip_ratio_c=0.15,
        o_topk=6,
        val_extra_samples=20,
    ),
    "20x10": ScaleConfig(
        scale="20x10",
        split_json="splits/3sd_20x10_single-scale_seed42_paths.json",
        seed=22,
        num_envs=16,
        rollout_workers=12,
        episodes_per_update=16,
        c_lr=8.0e-5,
        o_lr=8.0e-5,
        clip_ratio_c=0.12,
        o_topk=6,
        val_extra_samples=20,
    ),
    "30x10": ScaleConfig(
        scale="30x10",
        split_json="splits/3sd_30x10_single-scale_seed22_paths.json",
        seed=22,
        num_envs=12,
        rollout_workers=10,
        episodes_per_update=12,
        c_lr=8.0e-5,
        o_lr=8.0e-5,
        clip_ratio_c=0.12,
        o_topk=8,
        val_extra_samples=16,
    ),
}


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _parse_gpus(text: str) -> List[int]:
    out = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            out.append(int(token))
    if not out:
        raise ValueError("No GPU ids were provided")
    return out


def _parse_scales(text: str) -> List[str]:
    items = []
    for token in str(text).split(","):
        name = token.strip()
        if name:
            items.append(name)
    if not items:
        raise ValueError("No scales were provided")
    unsupported = [name for name in items if name not in SCALE_PRESETS]
    if unsupported:
        raise ValueError(f"Unsupported scales: {unsupported}; supported={sorted(SCALE_PRESETS)}")
    return items


def _safe_run(cmd: List[str]) -> str:
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if res.returncode != 0:
        return ""
    return (res.stdout or "").strip()


def _query_gpu_status() -> Tuple[Dict[int, Dict[str, float]], Dict[int, int]]:
    gpu_info: Dict[int, Dict[str, float]] = {}
    app_count: Dict[int, int] = {}

    gpu_text = _safe_run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if not gpu_text:
        return gpu_info, app_count

    uuid_to_index: Dict[str, int] = {}
    for line in gpu_text.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 5:
            continue
        idx = int(parts[0])
        uuid = parts[1]
        mem_used = float(parts[2])
        mem_total = float(parts[3])
        util = float(parts[4])
        gpu_info[idx] = {
            "mem_used": mem_used,
            "mem_total": mem_total,
            "util": util,
        }
        app_count[idx] = 0
        uuid_to_index[uuid] = idx

    app_text = _safe_run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if not app_text:
        return gpu_info, app_count

    for line in app_text.splitlines():
        line = line.strip()
        if not line or "No running processes found" in line:
            continue
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 2:
            continue
        uuid = parts[0]
        idx = uuid_to_index.get(uuid)
        if idx is None:
            continue
        app_count[idx] = app_count.get(idx, 0) + 1

    return gpu_info, app_count


def _running_scales_from_ps(repo_root: Path) -> Dict[str, int]:
    cmd = ["ps", "-eo", "cmd"]
    out = _safe_run(cmd)
    found: Dict[str, int] = {}
    if not out:
        return found

    pat = re.compile(r"3sd_(\d+x\d+)_single-scale_seed\d+_paths\.json")
    for line in out.splitlines():
        if "Train_OC_MAPPO.py" not in line:
            continue
        if str(repo_root) not in line and "splits/" not in line:
            continue
        m = pat.search(line)
        if not m:
            continue
        scale = m.group(1)
        found[scale] = found.get(scale, 0) + 1
    return found


def _build_train_cmd(cfg: ScaleConfig, save_dir: Path, epochs: int) -> List[str]:
    return [
        "Train_OC_MAPPO.py",
        "--split-json",
        cfg.split_json,
        "--seed",
        str(cfg.seed),
        "--device",
        "cuda",
        "--epochs",
        str(epochs),
        "--episodes-per-update",
        str(cfg.episodes_per_update),
        "--num-envs",
        str(cfg.num_envs),
        "--eval-interval",
        "1",
        "--rollout-backend",
        "mp",
        "--rollout-workers",
        str(cfg.rollout_workers),
        "--rollout-worker-device",
        "cpu",
        "--rollout-mp-start-method",
        "fork",
        "--rollout-worker-policy-mode",
        "train",
        "--rollout-worker-reseed",
        "1",
        "--train-eval-force-deterministic",
        "1",
        "--val-only-best-strategy",
        "1",
        "--val-only-hard-case-topk",
        "3",
        "--val-only-extra-samples-per-case",
        str(cfg.val_extra_samples),
        "--val-only-adaptive-extra-sampling",
        "1",
        "--c-auto-max-candidates",
        "1",
        "--agent-c-topk-jobs",
        "3",
        "--agent-c-topk-machines",
        "3",
        "--agent-c-pairs-per-rule",
        "5",
        "--agent-c-extra-explore-pairs",
        "4",
        "--c-candidate-safety-margin",
        "4",
        "--c-candidate-safety-margin-ratio",
        "0.10",
        "--c-min-max-candidates",
        "16",
        "--c-round-max-candidates-to-power-of-two",
        "0",
        "--o-topk",
        str(cfg.o_topk),
        "--o-topk-min",
        "6",
        "--o-topk-max",
        "12",
        "--o-redundancy-target-ratio",
        "0.50",
        "--o-redundancy-penalty-coef",
        "0.01",
        "--o-entropy-fallback-threshold",
        "0.7",
        "--o-entropy-fallback-extra-ops",
        "2",
        "--o-reward-fallback-scale",
        "0.95",
        "--m-reward-overprune-target-keep-ratio",
        "0.45",
        "--m-reward-overprune-coef",
        "0.15",
        "--p1-fast-skip-c-o-ref",
        "0",
        "--c-lr",
        str(cfg.c_lr),
        "--o-lr",
        str(cfg.o_lr),
        "--clip-ratio-c",
        str(cfg.clip_ratio_c),
        "--target-kl-c",
        "0.01",
        "--m-lr",
        "1e-4",
        "--clip-ratio-m",
        "0.10",
        "--target-kl-m",
        "0.01",
        "--m-ent-coef",
        "0.012",
        "--m-update-interval",
        "2",
        "--kl-penalty-coef-c",
        "0.02",
        "--kl-penalty-coef-o",
        "0.03",
        "--kl-penalty-coef-m",
        "0.02",
        "--reward-version",
        "role_aligned_v1",
        "--reward-schedule-anneal-epochs",
        "30",
        "--o-reward-beta-shape-end",
        "0.04",
        "--o-teacher-coef-end",
        "0.005",
        "--o-teacher-coef-min",
        "0.005",
        "--o-consensus-coef-end",
        "0.003",
        "--o-consensus-coef-min",
        "0.003",
        "--o-set-aux-coef",
        "0.02",
        "--m-reward-beta-shape-end",
        "0.03",
        "--m-teacher-coef-end",
        "0.01",
        "--m-teacher-coef-min",
        "0.01",
        "--m-consensus-coef-end",
        "0.002",
        "--m-consensus-coef-min",
        "0.002",
        "--m-hard-keep-requires-teacher",
        "0",
        "--m-hard-keep-entropy-trigger",
        "1",
        "--m-hard-keep-entropy-threshold",
        "0.20",
        "--m-expand-pairs-for-safety",
        "1",
        "--m-safety-min-total-pairs",
        "3",
        "--m-safety-min-machines-per-op",
        "3",
        "--m-entropy-warn-threshold",
        "0.45",
        "--m-entropy-hard-threshold",
        "0.35",
        "--m-keep-per-o-warn-threshold",
        "1.5",
        "--m-auto-lr-decay-on-collapse",
        "1",
        "--m-auto-lr-decay-factor",
        "0.8",
        "--m-auto-lr-min-scale",
        "0.25",
        "--quiet-env",
        "--save-dir",
        str(save_dir),
    ]


def _pick_gpu(
    gpus: List[int],
    gpu_info: Dict[int, Dict[str, float]],
    app_count: Dict[int, int],
    active: Dict[int, List[ActiveJob]],
    max_procs_per_gpu: int,
) -> Optional[int]:
    candidates: List[Tuple[int, int, float]] = []
    for gid in gpus:
        system_busy = app_count.get(gid, 0)
        local_busy = len(active.get(gid, []))
        busy = max(system_busy, local_busy)
        available = max(0, int(max_procs_per_gpu) - busy)
        if available <= 0:
            continue
        mem_used = gpu_info.get(gid, {}).get("mem_used", 0.0)
        candidates.append((gid, available, mem_used))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (-x[1], x[2], x[0]))
    return candidates[0][0]


def _write_launch_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = ["scale", "seed", "gpu", "pid", "save_dir", "log_file", "status", "message"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args(args=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adaptive multi-scale launcher for OC-MAPPO on GPU0/1")
    p.add_argument("--repo-root", type=str, default=".")
    p.add_argument("--python-exe", type=str, default="/home/jinglei/miniconda3/bin/python")
    p.add_argument("--scales", type=str, default="10x5,15x10,20x5,20x10,30x10")
    p.add_argument("--gpus", type=str, default="0,1")
    p.add_argument("--max-procs-per-gpu", type=int, default=3)
    p.add_argument("--poll-seconds", type=float, default=20.0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--output-root", type=str, default="checkpoints_oc_mappo_top2_plan/multiscale_5scales_adaptive")
    p.add_argument("--run-tag", type=str, default="")
    p.add_argument("--allow-duplicate-scales", action="store_true", default=False)
    p.add_argument("--dry-run", action="store_true", default=False)
    return p.parse_args(args=args)


def main(args=None) -> int:
    cli = parse_args(args=args)
    repo_root = Path(cli.repo_root).resolve()
    gpus = _parse_gpus(cli.gpus)
    scales = _parse_scales(cli.scales)
    run_tag = str(cli.run_tag).strip() or _now_tag()

    run_root = (repo_root / cli.output_root / f"run_{run_tag}").resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    launch_csv = run_root / "launch_table.csv"

    running_scales = _running_scales_from_ps(repo_root=repo_root)
    queue: List[ScaleConfig] = []
    launch_rows: List[Dict[str, object]] = []

    for scale in scales:
        cfg = SCALE_PRESETS[scale]
        split_abs = (repo_root / cfg.split_json).resolve()
        if not split_abs.exists():
            launch_rows.append(
                {
                    "scale": scale,
                    "seed": cfg.seed,
                    "gpu": "",
                    "pid": "",
                    "save_dir": "",
                    "log_file": "",
                    "status": "skip",
                    "message": f"split missing: {split_abs}",
                }
            )
            continue

        if not cli.allow_duplicate_scales and running_scales.get(scale, 0) > 0:
            launch_rows.append(
                {
                    "scale": scale,
                    "seed": cfg.seed,
                    "gpu": "",
                    "pid": "",
                    "save_dir": "",
                    "log_file": "",
                    "status": "skip",
                    "message": f"already running ({running_scales[scale]} proc lines detected)",
                }
            )
            continue

        queue.append(cfg)

    active: Dict[int, List[ActiveJob]] = {gid: [] for gid in gpus}

    print(f"[run-root] {run_root}")
    print(f"[queue-init] {[cfg.scale for cfg in queue]}")

    while queue or any(active[gid] for gid in gpus):
        gpu_info, app_count = _query_gpu_status()

        while queue:
            gid = _pick_gpu(
                gpus=gpus,
                gpu_info=gpu_info,
                app_count=app_count,
                active=active,
                max_procs_per_gpu=int(cli.max_procs_per_gpu),
            )
            if gid is None:
                break

            cfg = queue.pop(0)
            save_dir = run_root / f"{cfg.scale}_seed{cfg.seed}_gpu{gid}"
            save_dir.mkdir(parents=True, exist_ok=True)
            log_file = save_dir / "train.log"

            cmd = [str(cli.python_exe), "-u"] + _build_train_cmd(cfg=cfg, save_dir=save_dir, epochs=int(cli.epochs))
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gid)
            env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
            env["PYTHONUNBUFFERED"] = "1"

            if cli.dry_run:
                print(f"[dry-launch] gpu={gid} scale={cfg.scale} cmd={' '.join(cmd)}")
                launch_rows.append(
                    {
                        "scale": cfg.scale,
                        "seed": cfg.seed,
                        "gpu": gid,
                        "pid": "dry-run",
                        "save_dir": str(save_dir),
                        "log_file": str(log_file),
                        "status": "dry-run",
                        "message": "not started",
                    }
                )
                continue

            fout = open(log_file, "w", encoding="utf-8", buffering=1)
            proc = subprocess.Popen(cmd, cwd=str(repo_root), env=env, stdout=fout, stderr=subprocess.STDOUT)
            active[gid].append(
                ActiveJob(
                    scale=cfg.scale,
                    gpu=gid,
                    proc=proc,
                    save_dir=save_dir,
                    log_file=log_file,
                    start_time=time.time(),
                )
            )

            launch_rows.append(
                {
                    "scale": cfg.scale,
                    "seed": cfg.seed,
                    "gpu": gid,
                    "pid": proc.pid,
                    "save_dir": str(save_dir),
                    "log_file": str(log_file),
                    "status": "launched",
                    "message": "",
                }
            )
            print(f"[launch] gpu={gid} scale={cfg.scale} pid={proc.pid} save_dir={save_dir}")

        for gid in gpus:
            still: List[ActiveJob] = []
            for job in active[gid]:
                ret = job.proc.poll()
                if ret is None:
                    still.append(job)
                    continue
                elapsed = time.time() - job.start_time
                print(
                    f"[finish] gpu={gid} scale={job.scale} pid={job.proc.pid} "
                    f"status={'ok' if ret == 0 else f'exit_{ret}'} elapsed_sec={elapsed:.1f}"
                )
            active[gid] = still

        _write_launch_csv(path=launch_csv, rows=launch_rows)

        if queue or any(active[gid] for gid in gpus):
            time.sleep(float(cli.poll_seconds))

    print(f"[done] launch_table={launch_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def _parse_args(args=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a reproducible 3_SD split and launch Train_OC_MAPPO with configurable num_envs."
        )
    )
    p.add_argument("--repo-root", type=str, default=".")
    p.add_argument("--dataset-root", type=str, default="3_SD")

    p.add_argument("--scale", type=str, default="all", help="Use one bucket like 20x10, or 'all'")
    p.add_argument(
        "--split-policy",
        type=str,
        default="stratified",
        choices=["stratified", "single-scale", "leave-one-scale-out"],
        help=(
            "stratified: split each selected scale separately then merge; "
            "single-scale: split only one scale bucket; "
            "leave-one-scale-out: test on holdout scale, train/val on remaining scales"
        ),
    )
    p.add_argument("--holdout-scale", type=str, default="", help="Required for leave-one-scale-out")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument(
        "--train-count-per-scale",
        type=int,
        default=0,
        help="If >0, override train ratio per scale in stratified/single-scale mode",
    )
    p.add_argument(
        "--val-count-per-scale",
        type=int,
        default=0,
        help="If >0, override val ratio per scale in stratified/single-scale mode",
    )

    p.add_argument("--split-json-out", type=str, default="")
    p.add_argument("--save-dir", type=str, default="")

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--num-envs", type=int, default=10)
    p.add_argument("--eval-interval", type=int, default=1)
    p.add_argument("--extra-step-budget", type=int, default=40)
    p.add_argument(
        "--rollout-backend",
        type=str,
        default="serial",
        choices=["serial", "mp"],
        help="Rollout collection backend used inside Train_OC_MAPPO.",
    )
    p.add_argument(
        "--rollout-workers",
        type=int,
        default=0,
        help="Number of multiprocessing rollout workers (0 means auto in Train_OC_MAPPO).",
    )
    p.add_argument(
        "--rollout-worker-device",
        type=str,
        default="cpu",
        help="Device used by rollout workers. Recommend cpu for multiprocessing.",
    )
    p.add_argument(
        "--rollout-mp-start-method",
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="Python multiprocessing start method for mp rollout backend.",
    )

    p.add_argument("--quiet-env", type=int, default=1)
    p.add_argument("--disable-ppo-early-stop", type=int, default=1)
    p.add_argument("--dry-run", action="store_true", default=False)
    return p.parse_args(args=args)


def _list_scales(dataset_dir: Path) -> List[str]:
    out: List[str] = []
    for p in sorted(dataset_dir.iterdir()):
        if p.is_dir() and any(p.glob("*.fjs")):
            out.append(p.name)
    return out


def _collect_scale_cases(dataset_dir: Path, scale: str) -> List[Path]:
    return sorted((dataset_dir / scale).glob("*.fjs"))


def _resolve_scales(dataset_dir: Path, scale_arg: str) -> List[str]:
    if str(scale_arg).lower() == "all":
        scales = _list_scales(dataset_dir)
        if len(scales) == 0:
            raise FileNotFoundError(f"No scale folders with .fjs found under {dataset_dir}")
        return scales
    candidate = str(scale_arg)
    if not (dataset_dir / candidate).exists():
        raise FileNotFoundError(f"Scale folder not found: {dataset_dir / candidate}")
    return [candidate]


def _split_one_group(
    cases: Sequence[Path],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    train_count: int,
    val_count: int,
) -> Tuple[List[Path], List[Path], List[Path]]:
    if len(cases) < 3:
        raise ValueError("Each split group needs at least 3 cases")

    items = list(cases)
    random.Random(int(seed)).shuffle(items)
    n = len(items)

    if int(train_count) > 0 or int(val_count) > 0:
        train_n = int(train_count)
        val_n = int(val_count)
    else:
        train_n = int(round(float(train_ratio) * n))
        val_n = int(round(float(val_ratio) * n))

    train_n = max(1, min(train_n, n - 2))
    val_n = max(1, min(val_n, n - train_n - 1))
    test_n = n - train_n - val_n
    if test_n <= 0:
        val_n = max(1, n - train_n - 1)
        test_n = n - train_n - val_n
    if test_n <= 0:
        raise ValueError("Invalid split sizing: could not keep non-empty test set")

    train = items[:train_n]
    val = items[train_n : train_n + val_n]
    test = items[train_n + val_n :]
    return train, val, test


def _build_split(cli: argparse.Namespace, dataset_dir: Path) -> Tuple[Dict[str, object], Dict[str, Dict[str, int]]]:
    policy = str(cli.split_policy)
    scales_all = _list_scales(dataset_dir)
    if len(scales_all) == 0:
        raise FileNotFoundError(f"No valid 3_SD scales found under {dataset_dir}")

    selected_scales = _resolve_scales(dataset_dir=dataset_dir, scale_arg=cli.scale)

    train_cases: List[Path] = []
    val_cases: List[Path] = []
    test_cases: List[Path] = []
    per_scale_counts: Dict[str, Dict[str, int]] = {}

    if policy == "single-scale":
        if len(selected_scales) != 1:
            raise ValueError("single-scale policy requires --scale to be one specific scale")
        sc = selected_scales[0]
        cases = _collect_scale_cases(dataset_dir, sc)
        tr, va, te = _split_one_group(
            cases=cases,
            seed=int(cli.seed),
            train_ratio=float(cli.train_ratio),
            val_ratio=float(cli.val_ratio),
            train_count=int(cli.train_count_per_scale),
            val_count=int(cli.val_count_per_scale),
        )
        train_cases.extend(tr)
        val_cases.extend(va)
        test_cases.extend(te)
        per_scale_counts[sc] = {"train": len(tr), "val": len(va), "test": len(te)}

    elif policy == "leave-one-scale-out":
        holdout = str(cli.holdout_scale).strip()
        if not holdout:
            raise ValueError("leave-one-scale-out policy requires --holdout-scale")
        if holdout not in scales_all:
            raise ValueError(f"Holdout scale not found: {holdout}; available={scales_all}")

        holdout_cases = _collect_scale_cases(dataset_dir, holdout)
        if len(holdout_cases) == 0:
            raise ValueError(f"No cases found in holdout scale {holdout}")
        test_cases.extend(holdout_cases)
        per_scale_counts[holdout] = {"train": 0, "val": 0, "test": len(holdout_cases)}

        train_scales = [s for s in scales_all if s != holdout]
        if len(train_scales) == 0:
            raise ValueError("No remaining scales for train/val after holdout selection")

        for i, sc in enumerate(train_scales):
            cases = _collect_scale_cases(dataset_dir, sc)
            tr, va, te = _split_one_group(
                cases=cases,
                seed=int(cli.seed) + 1009 * (i + 1),
                train_ratio=float(cli.train_ratio),
                val_ratio=float(cli.val_ratio),
                train_count=int(cli.train_count_per_scale),
                val_count=int(cli.val_count_per_scale),
            )
            train_cases.extend(tr)
            val_cases.extend(va)
            per_scale_counts[sc] = {"train": len(tr), "val": len(va), "test": 0, "discarded_test": len(te)}

    else:
        for i, sc in enumerate(selected_scales):
            cases = _collect_scale_cases(dataset_dir, sc)
            tr, va, te = _split_one_group(
                cases=cases,
                seed=int(cli.seed) + 1009 * (i + 1),
                train_ratio=float(cli.train_ratio),
                val_ratio=float(cli.val_ratio),
                train_count=int(cli.train_count_per_scale),
                val_count=int(cli.val_count_per_scale),
            )
            train_cases.extend(tr)
            val_cases.extend(va)
            test_cases.extend(te)
            per_scale_counts[sc] = {"train": len(tr), "val": len(va), "test": len(te)}

    if len(train_cases) == 0 or len(val_cases) == 0 or len(test_cases) == 0:
        raise ValueError("Split must contain non-empty train/val/test sets")

    random.Random(int(cli.seed) + 1).shuffle(train_cases)
    random.Random(int(cli.seed) + 2).shuffle(val_cases)
    random.Random(int(cli.seed) + 3).shuffle(test_cases)

    payload: Dict[str, object] = {
        "meta": {
            "dataset_root": str(dataset_dir.resolve()),
            "split_policy": policy,
            "scale": str(cli.scale),
            "holdout_scale": str(cli.holdout_scale),
            "seed": int(cli.seed),
            "train_ratio": float(cli.train_ratio),
            "val_ratio": float(cli.val_ratio),
            "train_count_per_scale": int(cli.train_count_per_scale),
            "val_count_per_scale": int(cli.val_count_per_scale),
            "selected_scales": selected_scales,
        },
        "train_cases": [str(p.resolve()) for p in train_cases],
        "val_cases": [str(p.resolve()) for p in val_cases],
        "test_cases": [str(p.resolve()) for p in test_cases],
        "per_scale_counts": per_scale_counts,
    }
    return payload, per_scale_counts


def _default_split_path(repo_root: Path, scale: str, policy: str, seed: int) -> Path:
    safe_scale = str(scale).replace("/", "_")
    safe_policy = str(policy).replace("_", "-")
    return (repo_root / "splits" / f"3sd_{safe_scale}_{safe_policy}_seed{int(seed)}_paths.json").resolve()


def _default_save_dir(repo_root: Path, scale: str, policy: str, seed: int, num_envs: int) -> Path:
    tag = time.strftime("%Y%m%d_%H%M%S")
    safe_scale = str(scale).replace("/", "_")
    safe_policy = str(policy).replace("_", "-")
    return (
        repo_root
        / "checkpoints_oc_mappo_3sd"
        / f"{safe_scale}_{safe_policy}_seed{int(seed)}_env{int(num_envs)}"
        / f"run_{tag}"
    ).resolve()


def main(args=None) -> int:
    cli = _parse_args(args=args)
    repo_root = Path(cli.repo_root).resolve()
    dataset_dir = (repo_root / cli.dataset_root).resolve()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    split_payload, per_scale_counts = _build_split(cli=cli, dataset_dir=dataset_dir)

    split_path = (
        Path(cli.split_json_out).resolve()
        if str(cli.split_json_out).strip()
        else _default_split_path(
            repo_root=repo_root,
            scale=str(cli.scale),
            policy=str(cli.split_policy),
            seed=int(cli.seed),
        )
    )
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    save_dir = (
        Path(cli.save_dir).resolve()
        if str(cli.save_dir).strip()
        else _default_save_dir(
            repo_root=repo_root,
            scale=str(cli.scale),
            policy=str(cli.split_policy),
            seed=int(cli.seed),
            num_envs=int(cli.num_envs),
        )
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        "Train_OC_MAPPO.py",
        "--split-json",
        str(split_path),
        "--seed",
        str(int(cli.seed)),
        "--device",
        str(cli.device),
        "--epochs",
        str(int(cli.epochs)),
        "--num-envs",
        str(int(cli.num_envs)),
        "--rollout-backend",
        str(cli.rollout_backend),
        "--rollout-workers",
        str(int(cli.rollout_workers)),
        "--rollout-worker-device",
        str(cli.rollout_worker_device),
        "--rollout-mp-start-method",
        str(cli.rollout_mp_start_method),
        "--eval-interval",
        str(int(cli.eval_interval)),
        "--extra-step-budget",
        str(int(cli.extra_step_budget)),
        "--save-dir",
        str(save_dir),
    ]

    if bool(int(cli.quiet_env)):
        cmd.append("--quiet-env")
    if bool(int(cli.disable_ppo_early_stop)):
        cmd.append("--disable-ppo-early-stop")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"

    train_n = len(split_payload["train_cases"])
    val_n = len(split_payload["val_cases"])
    test_n = len(split_payload["test_cases"])
    print("[3sd-split]", split_path)
    print(f"[3sd-counts] train={train_n} val={val_n} test={test_n}")
    print("[3sd-policy]", cli.split_policy)
    print("[3sd-per-scale]", json.dumps(per_scale_counts, ensure_ascii=False))
    print("[save-dir]", save_dir)
    print("[launch-cmd]", " ".join(cmd))

    if bool(cli.dry_run):
        print("[dry-run] command not executed")
        return 0

    proc = subprocess.run(cmd, cwd=str(repo_root), env=env, check=False)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from pathlib import Path
import math
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

def _history_numeric(history: List[Dict[str, object]], key: str):
    xs, ys = [], []
    for item in history:
        if key in item and isinstance(item[key], (int, float)) and np.isfinite(item[key]):
            xs.append(item.get("global_update", item.get("epoch", 0)))
            ys.append(float(item[key]))
    return xs, ys


def _plot_readable_curve(ax, xs, ys, label, color):
    if not xs:
        return
    n = len(xs)
    markevery = max(1, n // 24)
    ax.plot(
        xs,
        ys,
        color=color,
        linewidth=2.0,
        alpha=0.9,
        label=label,
        marker="o",
        markersize=3.2,
        markevery=markevery,
    )

def plot_training_and_baselines(summary, save_dir=None, show=True):
    history = summary["history"]
    final_baseline_eval = summary["baseline_eval"]["test"] if "test" in summary["baseline_eval"] else {}
    final_agent_eval = summary.get("final_test_eval", {})

    if save_dir is None:
        metrics_path = Path(summary["metrics"])
        save_dir = metrics_path.parent
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 7), dpi=160)
    ax = fig.add_subplot(1, 1, 1)

    val_xs, val_ys = _history_numeric(history, "val_mean_makespan")
    train_xs, train_ys = _history_numeric(history, "train_full_mean_makespan")
    if not train_xs:
        train_xs, train_ys = _history_numeric(history, "train_mean_makespan")

    _plot_readable_curve(ax, val_xs, val_ys, "Val_mean_makespan", "#1f77b4")
    _plot_readable_curve(ax, train_xs, train_ys, "train_mean_makespan", "#ff7f0e")

    all_x = (val_xs or []) + (train_xs or [])
    if all_x:
        x_min = int(min(all_x))
        x_max = int(max(all_x))
        ax.set_xlim(x_min, x_max)
        span = max(1, x_max - x_min)
        tick_step = int(math.ceil(span / 8 / 100.0) * 100)
        ax.set_xticks(np.arange(x_min, x_max + 1, max(100, tick_step)))

    ax.set_title("Core KPI: Val vs Train Makespan", fontsize=14, pad=10)
    ax.set_xlabel("Global Update", fontsize=12)
    ax.set_ylabel("Makespan", fontsize=12)
    ax.grid(True, which="major", linestyle="--", alpha=0.35)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", alpha=0.15)
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    train_curve_path = save_dir / "agent_c_training_curves_key_metrics.png"
    fig.savefig(train_curve_path, dpi=260, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    labels = []
    values = []

    baseline_items = []
    for rule_name, rule_result in final_baseline_eval.items():
        baseline_items.append((rule_name, float(rule_result["mean_makespan"])))
    baseline_items = sorted(baseline_items, key=lambda x: x[1])[:10]

    if final_agent_eval:
        labels.append("Agent_C")
        values.append(float(final_agent_eval["mean_makespan"]))

    for rule_name, v in baseline_items:
        labels.append(rule_name)
        values.append(v)

    baseline_path = None
    if labels:
        fig2 = plt.figure(figsize=(10, 6))
        ax = fig2.add_subplot(1, 1, 1)
        bars = ax.bar(labels, values)
        ax.set_title("Final Test: Agent_C vs Top-10 Rule Baselines")
        ax.set_ylabel("Mean Makespan")
        ax.grid(True, axis="y", alpha=0.3)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.2f}", ha="center", va="bottom")
        fig2.tight_layout()
        baseline_path = save_dir / "agent_c_vs_rule_baselines_test_top10.png"
        fig2.savefig(baseline_path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig2)

    return {
        "train_curve_path": str(train_curve_path),
        "baseline_curve_path": str(baseline_path) if baseline_path is not None else None,
    }


from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List


def _rows_by_prefix(rows: List[Dict[str, object]], prefix: str) -> List[Dict[str, object]]:
    return [r for r in rows if str(r.get("name", "")).startswith(prefix)]


def _avg_metric(rows: List[Dict[str, object]], key: str) -> float:
    vals = [float(r.get(key, float("inf"))) for r in rows]
    return mean(vals) if vals else float("inf")


def _std_metric(rows: List[Dict[str, object]], key: str) -> float:
    vals = [float(r.get(key, 0.0)) for r in rows]
    return pstdev(vals) if len(vals) >= 2 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze formal AgentM suite report")
    parser.add_argument("--report-json", type=str, required=True)
    args = parser.parse_args()

    report_path = Path(args.report_json)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = list(report.get("rows", []))
    ranked = list(report.get("ranked", []))

    baseline = _rows_by_prefix(rows, "A_stage2_stable_")
    best = ranked[0] if ranked else {}

    baseline_mean = _avg_metric(baseline, "avg_m_agentc")
    baseline_std = _std_metric(baseline, "avg_m_agentc")

    best_mean = float(best.get("avg_m_agentc", float("inf")))
    improve_pct = 100.0 * (baseline_mean - best_mean) / baseline_mean if baseline_mean > 0 else 0.0

    # Robustness check for best profile across seeds if available.
    best_profile = str(best.get("profile", ""))
    best_profile_rows = [r for r in rows if str(r.get("profile", "")) == best_profile and str(r.get("name", "")).endswith(("s42", "s55", "s77"))]
    best_profile_mean = _avg_metric(best_profile_rows, "avg_m_agentc")
    best_profile_std = _std_metric(best_profile_rows, "avg_m_agentc")

    pass_lvl1 = bool(improve_pct >= 1.5)
    pass_lvl3 = bool(best_profile_std <= 1.1 * baseline_std) if baseline_std > 0 else True

    analysis = {
        "baseline_mean_agentm_agentc": baseline_mean,
        "baseline_std_agentm_agentc": baseline_std,
        "best_run_name": best.get("name"),
        "best_run_profile": best_profile,
        "best_run_seed": best.get("seed"),
        "best_run_avg_agentm_agentc": best_mean,
        "improvement_pct_vs_baseline_mean": improve_pct,
        "best_profile_mean_agentm_agentc": best_profile_mean,
        "best_profile_std_agentm_agentc": best_profile_std,
        "pass_level1_improve_ge_1p5pct": pass_lvl1,
        "pass_level3_std_le_1p1x_baseline": pass_lvl3,
        "rank_top5": [
            {
                "name": r.get("name"),
                "profile": r.get("profile"),
                "seed": r.get("seed"),
                "avg_m_agentc": r.get("avg_m_agentc"),
                "avg_m_o_agentc": r.get("avg_m_o_agentc"),
                "best_val_ms": r.get("best_val_ms"),
                "wtl_vs_baseline": r.get("vs_baseline_casewise", {}),
            }
            for r in ranked[:5]
        ],
    }

    out_json = report_path.with_name("formal_suite_analysis.json")
    out_md = report_path.with_name("formal_suite_analysis.md")

    out_json.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append("# AgentM Formal Suite Analysis")
    lines.append("")
    lines.append(f"- baseline mean (agentm+agentc): {baseline_mean:.4f}")
    lines.append(f"- baseline std (agentm+agentc): {baseline_std:.4f}")
    lines.append(f"- best run: {analysis['best_run_name']} / profile={best_profile} / seed={analysis['best_run_seed']}")
    lines.append(f"- best run avg (agentm+agentc): {best_mean:.4f}")
    lines.append(f"- improve vs baseline mean: {improve_pct:.2f}%")
    lines.append(f"- level1 pass (>=1.5%): {pass_lvl1}")
    lines.append(f"- level3 pass (std <= 1.1x baseline): {pass_lvl3}")
    lines.append("")
    lines.append("## Top 5")
    lines.append("| rank | name | profile | seed | avg_m+ac | avg_m+o+ac | best_val_ms | w/t/l |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---|")
    for i, r in enumerate(analysis["rank_top5"], start=1):
        wtl = r.get("wtl_vs_baseline", {})
        lines.append(
            f"| {i} | {r['name']} | {r['profile']} | {r['seed']} | {float(r['avg_m_agentc']):.4f} | {float(r['avg_m_o_agentc']):.4f} | {float(r['best_val_ms']):.4f} | {wtl.get('wins',0)}/{wtl.get('ties',0)}/{wtl.get('losses',0)} |"
        )

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"analysis_json": str(out_json), "analysis_md": str(out_md)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

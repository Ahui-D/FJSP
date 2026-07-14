#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

TS="${1:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="checkpoints_oc_mappo_top2_plan/critic3_300e_dual/run_${TS}"
mkdir -p "$RUN_ROOT"

SPLIT_JSON="$REPO_ROOT/splits/brandimarte_seed42_v330_paths.json"

PY_CMD=(
  /home/jinglei/miniconda3/bin/conda run -p /home/jinglei/miniconda3 --no-capture-output python
  /home/jinglei/.vscode-server/extensions/ms-python.python-2026.4.0-linux-x64/python_files/get_output_via_markers.py
)

COMMON_ARGS=(
  --split-json "$SPLIT_JSON"
  --seed 33
  --device cuda
  --epochs 300
  --episodes-per-update 8
  --eval-interval 1
  --save-dir PLACEHOLDER
  --o-topk 6
  --c-lr 2e-4
  --o-lr 1e-4
  --clip-ratio-c 0.20
  --clip-ratio-o 0.20
  --value-coef-c 0.6
  --use-huber-value-loss 1
  --value-huber-delta 1.0
  --o-reward-alpha-env 0.30
  --o-reward-alpha-env-end 0.70
  --o-reward-beta-shape 1.00
  --o-reward-beta-shape-end 0.35
  --o-reward-anneal-epochs 180
  --o-reward-clip-abs 1.5
  --reward-mismatch-only-if-not-retained 1
  --reward-balance-coef 0.25
  --reward-wait-coef 0.0
  --disable-ppo-early-stop
  --quiet-env
)

launch_one() {
  local name="$1"
  local gpu="$2"
  shift 2

  local save_dir="$RUN_ROOT/$name"
  local log_file="$save_dir/train.log"
  mkdir -p "$save_dir"

  local args=("${COMMON_ARGS[@]}")
  for i in "${!args[@]}"; do
    if [[ "${args[$i]}" == "PLACEHOLDER" ]]; then
      args[$i]="$save_dir"
    fi
  done

  echo "[$(date '+%F %T')] START $name gpu=$gpu" | tee -a "$RUN_ROOT/pipeline.log"
  echo "command: CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$REPO_ROOT ${PY_CMD[*]} Train_OC_MAPPO.py ${args[*]} $*" >> "$RUN_ROOT/pipeline.log"

  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$REPO_ROOT" \
    "${PY_CMD[@]}" Train_OC_MAPPO.py "${args[@]}" "$@" > "$log_file" 2>&1 &
  local pid=$!

  echo "[$(date '+%F %T')] PID   $name pid=$pid" | tee -a "$RUN_ROOT/pipeline.log"
  echo "$name,$gpu,$pid,$log_file" >> "$RUN_ROOT/pids.csv"
}

# Three groups run independently from scratch (no init checkpoints).
launch_one G1_critic_loss_tune 0 \
  --critic-lr 2e-4 \
  --value-clip-range 0.1 \
  --value-coef-o 0.8

launch_one G2_critic_rich_state 1 \
  --critic-lr 2e-4 \
  --value-clip-range 0.1 \
  --value-coef-o 0.8 \
  --critic-rich-state 1

launch_one G3_critic_split_tower 0 \
  --critic-lr 2e-4 \
  --value-clip-range 0.1 \
  --value-coef-o 0.8 \
  --critic-rich-state 1 \
  --critic-split-tower 1 \
  --critic-tower-hidden-dim 256

while IFS=, read -r name gpu pid log_file; do
  if [[ -z "$name" ]]; then
    continue
  fi
  if wait "$pid"; then
    rc=0
  else
    rc=$?
  fi
  echo "[$(date '+%F %T')] DONE  $name gpu=$gpu pid=$pid rc=$rc" | tee -a "$RUN_ROOT/pipeline.log"
done < "$RUN_ROOT/pids.csv"

export RUN_ROOT
/home/jinglei/miniconda3/bin/python - <<'PY'
import json
import os
from pathlib import Path

run_root = Path(os.environ["RUN_ROOT"])
rows = []
for name in [
    "G1_critic_loss_tune",
    "G2_critic_rich_state",
    "G3_critic_split_tower",
]:
    p = run_root / name / "train_oc_mappo_summary.json"
    if not p.exists():
        rows.append({"name": name, "status": "missing_summary"})
        continue

    data = json.loads(p.read_text(encoding="utf-8"))
    history = data.get("history", [])
    tail50 = history[-50:] if len(history) >= 50 else history

    def _avg(key):
        vals = [float(x.get(key, 0.0)) for x in tail50 if key in x]
        return sum(vals) / len(vals) if vals else None

    rows.append(
        {
            "name": name,
            "status": "ok",
            "best_val_makespan": float(data.get("best_val_makespan", float("inf"))),
            "best_epoch": int(data.get("best_epoch", -1)),
            "test_mean_makespan": float(data.get("test_eval", {}).get("mean_makespan", float("inf"))),
            "last50_val_ms_mean": _avg("val_mean_makespan"),
            "last50_vloss_o_mean": _avg("update_value_loss_o"),
            "last50_kl_c_mean": _avg("update_kl_c"),
            "last50_kl_o_mean": _avg("update_kl_o"),
        }
    )

rows_ok = [r for r in rows if r.get("status") == "ok"]
rows_ok_sorted = sorted(rows_ok, key=lambda r: (r["last50_val_ms_mean"] if r["last50_val_ms_mean"] is not None else float("inf"), r["test_mean_makespan"]))
report = {
    "run_root": str(run_root),
    "rows": rows,
    "rank_by_last50_val_ms_then_test": rows_ok_sorted,
}

out = run_root / "critic3_compare_summary.json"
out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
PY

echo "[DONE] run_root=$RUN_ROOT" | tee -a "$RUN_ROOT/pipeline.log"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SCALE="20x10"
SPLIT_SRC="${SPLIT_SRC:-splits/3sd_20x10_seed32_split_8_1_1.json}"
SEED="${SEED:-32}"
GPU="${GPU:-5}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-exp_runs/s32_3sd20x10_resume_e500_${RUN_TAG}}"
RESUME_STATE_PATH="${RESUME_STATE_PATH:-exp_runs/s32_3sd20x10/train_oc_mappo_state_latest.pt}"
RUN_BACKGROUND="${RUN_BACKGROUND:-1}"

mkdir -p "$RUN_ROOT"

SPLIT_JSON="$RUN_ROOT/split_paths_local.json"
"$PYTHON_BIN" - "$SPLIT_SRC" "$SPLIT_JSON" "$REPO_ROOT" <<'PY'
import sys
from pathlib import Path

src, dst, root = map(Path, sys.argv[1:4])
text = src.read_text(encoding="utf-8")
for old in [
    "/home/jinglei/RAMS/Ahui_fjsp_main_5.10",
    "/home/jinglei/RAMS/Ahui_fjsp_github_clean",
    "/u01/xuwenhui/Run_Code/FJSP",
]:
    text = text.replace(old, str(root))
dst.write_text(text, encoding="utf-8")
PY

COMMON_ARGS=(
  --split-json "$SPLIT_JSON"
  --epochs 500
  --seed "$SEED"
  --save-dir "$RUN_ROOT"
  --resume-state-path "$RESUME_STATE_PATH"
  --active-job-rules ""
  --active-machine-rules ""
  --agent-c-topk-jobs 4
  --agent-c-topk-machines 4
  --agent-c-pairs-per-rule 7
  --agent-c-extra-explore-pairs 6
  --agent-c-refine-max-pairs-per-op 3
  --agent-c-refine-global-reserve-size 5
  --agent-c-refine-diversity-min-machines 3
  --agent-c-refine-keep-explore-min 1
  --c-min-max-candidates 40
  --c-candidate-safety-margin 10
  --c-candidate-safety-margin-ratio 0.12
  --o-op-safety-margin 5
  --o-topk-min 3
  --o-topk-max 9
  --m-safety-min-total-pairs 4
  --m-safety-min-machines-per-op 3
  --full-soft-widen-enabled 1
  --full-soft-widen-c-extra 2
  --full-soft-widen-o-extra 1
  --full-soft-widen-refine-pairs-per-op-extra 1
  --ocm-parallel-om-enabled 1
  --ocm-parallel-fusion-mode union
  --ocm-parallel-min-final-pairs 4
  --ocm-parallel-budget-extra-pairs 2
  --ocm-parallel-max-final-pairs 0
  --ocm-parallel-m-teacher-mode full_ref
  --ocm-parallel-rescue-ops 2
  --ocm-parallel-rescue-pairs-per-op 2
  --reward-version role_aligned_v2
  --c-lr 9e-5
  --o-lr 1e-5
  --m-lr 1e-4
  --clip-ratio-c 0.14
  --clip-ratio-o 0.10
  --clip-ratio-m 0.12
  --ppo-epochs 3
  --target-kl-c 0.011
  --target-kl-o 0.05
  --minibatch-kl-guard-mult-o 5
  --o-update-interval 3
  --o-smooth-update-enabled 1
  --o-smooth-update-min-weight 0.35
  --o-batch-protect-enabled 1
  --o-ratio-batch-soft-limit 0.35
  --o-ratio-batch-hard-limit 0.70
  --o-ratio-batch-soft-scale 0.40
  --o-instability-monitor-enabled 0
  --o-reward-beta-shape 0.04
  --o-reward-beta-shape-end 0.015
  --o-teacher-coef 0.0
  --o-teacher-coef-end 0.0
  --o-teacher-coef-min 0.0
  --o-consensus-coef 0.0
  --o-consensus-coef-end 0.0
  --o-consensus-coef-min 0.0
  --o-set-aux-coef 0.001
  --o-set-ppo-mix 0.02
  --m-update-interval 1
  --m-ent-coef 0.014
  --m-ent-coef-end 0.011
  --m-ent-coef-anneal-updates 100
  --m-reward-beta-shape 0.07
  --m-reward-beta-shape-end 0.05
  --m-teacher-coef 0.04
  --m-teacher-coef-end 0.01
  --m-teacher-coef-min 0.01
  --m-consensus-coef 0
  --m-consensus-coef-end 0
  --m-consensus-coef-min 0
  --val-plateau-patience 3
  --val-plateau-decay 0.95
  --val-plateau-min-scale 0.30
  --val-plateau-apply-c 1
  --val-plateau-apply-o 1
  --val-plateau-apply-m 0
  --val-plateau-restore-best 1
  --val-plateau-restore-min-epoch 8
  --val-plateau-restore-cooldown 3
  --val-plateau-restore-rel-gap 0.05
)

LOG_FILE="$RUN_ROOT/train.log"
echo "[start] scale=$SCALE seed=$SEED gpu=$GPU run_root=$RUN_ROOT"
echo "[cmd] CUDA_VISIBLE_DEVICES=$GPU $PYTHON_BIN -u Train_OC_MAPPO.py ${COMMON_ARGS[*]}" > "$RUN_ROOT/command.txt"

if [[ "$RUN_BACKGROUND" == "1" ]]; then
  nohup env CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" -u Train_OC_MAPPO.py "${COMMON_ARGS[@]}" > "$LOG_FILE" 2>&1 &
  echo "$!" > "$RUN_ROOT/pid.txt"
  echo "[launched] pid=$(cat "$RUN_ROOT/pid.txt") log=$LOG_FILE"
else
  env CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" -u Train_OC_MAPPO.py "${COMMON_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
fi

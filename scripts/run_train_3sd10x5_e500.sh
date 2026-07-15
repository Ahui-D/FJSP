#!/usr/bin/env bash
# 3_SD 10x5, seed=32, 8:1:1, train from scratch for 500 epochs.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ARGS=()
if [[ "${GPU_ID:-}" != "" ]]; then
  GPU_ARGS=(--gpu-id "$GPU_ID")
fi
cd "$REPO_ROOT"
exec "$PYTHON_BIN" Train_OC_MAPPO.py \
  "${GPU_ARGS[@]}" \
  --split-json splits/3sd_10x5_seed32_split_8_1_1.json \
  --epochs 500 \
  --seed 32 \
  --save-dir exp_runs/s22_3sd10x5_e500 \
  --active-job-rules "" \
  --active-machine-rules "" \
  --agent-c-topk-jobs 4 \
  --agent-c-topk-machines 4 \
  --agent-c-pairs-per-rule 7 \
  --agent-c-extra-explore-pairs 6 \
  --agent-c-refine-max-pairs-per-op 3 \
  --agent-c-refine-global-reserve-size 5 \
  --agent-c-refine-diversity-min-machines 3 \
  --agent-c-refine-keep-explore-min 1 \
  --c-min-max-candidates 40 \
  --c-candidate-safety-margin 10 \
  --c-candidate-safety-margin-ratio 0.12 \
  --o-op-safety-margin 5 \
  --o-topk-min 3 \
  --o-topk-max 9 \
  --m-safety-min-total-pairs 4 \
  --m-safety-min-machines-per-op 3 \
  --full-soft-widen-enabled 1 \
  --full-soft-widen-c-extra 2 \
  --full-soft-widen-o-extra 1 \
  --full-soft-widen-refine-pairs-per-op-extra 1 \
  --ocm-parallel-om-enabled 1 \
  --ocm-parallel-fusion-mode union \
  --ocm-parallel-min-final-pairs 4 \
  --ocm-parallel-budget-extra-pairs 2 \
  --ocm-parallel-max-final-pairs 0 \
  --ocm-parallel-m-teacher-mode full_ref \
  --ocm-parallel-rescue-ops 2 \
  --ocm-parallel-rescue-pairs-per-op 2 \
  --reward-version role_aligned_v2 \
  --c-lr 9e-5 \
  --o-lr 1e-5 \
  --m-lr 1e-4 \
  --ppo-epochs 3 \
  --clip-ratio-c 0.14 \
  --clip-ratio-o 0.10 \
  --clip-ratio-m 0.12 \
  --target-kl-c 0.011 \
  --target-kl-o 0.05 \
  --minibatch-kl-guard-mult-o 5 \
  --o-update-interval 3 \
  --o-smooth-update-enabled 1 \
  --o-smooth-update-min-weight 0.35 \
  --o-batch-protect-enabled 1 \
  --o-ratio-batch-soft-limit 0.35 \
  --o-ratio-batch-hard-limit 0.70 \
  --o-ratio-batch-soft-scale 0.40 \
  --o-instability-monitor-enabled 0 \
  --o-reward-beta-shape 0.04 \
  --o-reward-beta-shape-end 0.015 \
  --o-teacher-coef 0.0 \
  --o-teacher-coef-end 0.0 \
  --o-teacher-coef-min 0.0 \
  --o-consensus-coef 0.0 \
  --o-consensus-coef-end 0.0 \
  --o-consensus-coef-min 0.0 \
  --o-set-aux-coef 0.001 \
  --o-set-ppo-mix 0.02 \
  --m-update-interval 1 \
  --m-ent-coef 0.014 \
  --m-ent-coef-end 0.011 \
  --m-ent-coef-anneal-updates 100 \
  --m-reward-beta-shape 0.07 \
  --m-reward-beta-shape-end 0.05 \
  --m-teacher-coef 0.04 \
  --m-teacher-coef-end 0.01 \
  --m-teacher-coef-min 0.01 \
  --m-consensus-coef 0 \
  --m-consensus-coef-end 0 \
  --m-consensus-coef-min 0 \
  --val-plateau-patience 3 \
  --val-plateau-decay 0.95 \
  --val-plateau-min-scale 0.30 \
  --val-plateau-apply-c 1 \
  --val-plateau-apply-o 1 \
  --val-plateau-apply-m 0 \
  --val-plateau-restore-best 1 \
  --val-plateau-restore-min-epoch 8 \
  --val-plateau-restore-cooldown 3 \
  --val-plateau-restore-rel-gap 0.05

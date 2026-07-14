#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/jinglei/RAMS/Dispatching-rules-for-FJSP-main_4.5_v2"
CONDA_PY=(/home/jinglei/miniconda3/bin/python)
SPLIT_JSON="$REPO_ROOT/splits/3sd_15x10_single-scale_seed42_paths.json"
RUN_TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="$REPO_ROOT/checkpoints_oc_mappo_top2_plan/phaseD1_E16_B4_A240_M1_15x10_env20_w15_mc2_4seeds/run_${RUN_TAG}"

mkdir -p "$RUN_ROOT"

if [[ ! -f "$SPLIT_JSON" ]]; then
  echo "[prep] split not found, generating fixed 15x10 split: $SPLIT_JSON"
  (
    cd "$REPO_ROOT"
    PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 "${CONDA_PY[@]}" run_oc_mappo_3sd_parallel10.py \
      --repo-root . \
      --dataset-root 3_SD \
      --scale 15x10 \
      --split-policy single-scale \
      --seed 42 \
      --split-json-out "$SPLIT_JSON" \
      --num-envs 20 \
      --epochs 1000 \
      --dry-run
  )
fi

SEEDS=(22 33 44 55)
GPUS=(0 1 0 1)

LAUNCH_CSV="$RUN_ROOT/launch_table.csv"
echo "seed,gpu,pid,save_dir,log_file" > "$LAUNCH_CSV"

for idx in "${!SEEDS[@]}"; do
  seed="${SEEDS[$idx]}"
  gpu="${GPUS[$idx]}"
  save_dir="$RUN_ROOT/seed_${seed}"
  log_file="$save_dir/train.log"
  mkdir -p "$save_dir"

  (
    cd "$REPO_ROOT"
    nohup env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" \
      PYTHONUNBUFFERED=1 \
      "${CONDA_PY[@]}" -u Train_OC_MAPPO.py \
        --split-json "$SPLIT_JSON" \
        --seed "$seed" \
        --device cuda \
        --epochs 1000 \
        --episodes-per-update 8 \
        --eval-interval 1 \
        --num-envs 20 \
        --extra-step-budget 40 \
        --rollout-backend mp \
        --rollout-workers 15 \
        --rollout-min-cases-per-worker 2 \
        --rollout-worker-device cpu \
        --rollout-mp-start-method fork \
        --rollout-worker-policy-mode train \
        --rollout-worker-reseed 1 \
        --o-topk 6 \
        --c-lr 2e-4 \
        --o-lr 1e-4 \
        --critic-lr 3e-4 \
        --clip-ratio-c 0.20 \
        --clip-ratio-o 0.20 \
        --value-coef-c 0.6 \
        --value-coef-o 0.6 \
        --use-huber-value-loss 0 \
        --value-huber-delta 1.0 \
        --value-clip-range 0.2 \
        --target-kl-c 0.02 \
        --target-kl-o 0.01 \
        --o-reward-alpha-env 0.30 \
        --o-reward-beta-shape 1.00 \
        --o-reward-alpha-env-end 0.60 \
        --o-reward-beta-shape-end 0.50 \
        --o-reward-anneal-epochs 240 \
        --o-reward-clip-abs 1.5 \
        --reward-mismatch-only-if-not-retained 1 \
        --quiet-env \
        --disable-ppo-early-stop \
        --save-dir "$save_dir" > "$log_file" 2>&1 &
    pid=$!
    echo "[launch] seed=$seed gpu=$gpu pid=$pid save_dir=$save_dir"
    echo "$seed,$gpu,$pid,$save_dir,$log_file" >> "$LAUNCH_CSV"
  )
done

echo "[done] run_root=$RUN_ROOT"
echo "[done] launch_csv=$LAUNCH_CSV"
from __future__ import annotations

import json
import sys
from pathlib import Path

_ORIG_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]

from Params import configs as env_configs
from Train_C import TrainConfig, train_agent_c_parallel

sys.argv = _ORIG_ARGV

SPLIT_JSON = Path('splits/3sd_10x5_seed32_split_8_1_1.json')
OUT_DIR = Path('exp_runs/ablation_c_only_s22_3sd10x5_e100')

payload = json.loads(SPLIT_JSON.read_text(encoding='utf-8'))
train_cases = [str(x) for x in payload.get('train_cases', [])]
val_cases = [str(x) for x in payload.get('val_cases', [])]
test_cases = [str(x) for x in payload.get('test_cases', [])]

# Match the candidate-generation width used by s22_3sd10x5 OURS as closely as Train_C allows.
env_configs.agent_c_active_job_rules = ''
env_configs.agent_c_active_machine_rules = ''
env_configs.agent_c_topk_jobs = 4
env_configs.agent_c_topk_machines = 4
env_configs.agent_c_pairs_per_rule = 7
env_configs.agent_c_extra_explore_pairs = 6
env_configs.agent_c_two_stage_refine_enabled = True
env_configs.agent_c_refine_max_pairs_per_op = 4  # s22 uses 3 + full_soft_widen_refine_pairs_per_op_extra=1
env_configs.agent_c_refine_keep_global_best_ect = 1
env_configs.agent_c_refine_keep_global_best_pt = 1
env_configs.agent_c_refine_score_w_eet = 1.0
env_configs.agent_c_refine_score_w_pt = 0.3
env_configs.agent_c_refine_score_w_queue = 0.2
env_configs.agent_c_refine_score_w_support = 0.5
env_configs.agent_c_refine_explore_bonus = 0.1
env_configs.agent_c_candidate_v2_enabled = True
env_configs.agent_c_critical_pairs_enabled = True
env_configs.agent_c_critical_pairs_per_op = 2
env_configs.agent_c_refine_max_pairs_per_machine = 0
env_configs.agent_c_refine_global_reserve_size = 5
env_configs.agent_c_refine_diversity_min_machines = 3
env_configs.agent_c_refine_keep_explore_min = 1
env_configs.agent_c_candidate_debug_print = False

cfg = TrainConfig(
    train_cases=train_cases,
    val_cases=val_cases,
    test_cases=test_cases,
    dataset_name='3SD_10x5_s22_split32_c_only_ablation',
    seed=32,
    device='cuda',
    reset_rule='FIFO_SPT',
    epochs=100,
    episodes_per_update=8,
    eval_interval=1,
    run_test_on_every_eval=False,
    save_dir=str(OUT_DIR),
    checkpoint_name='agent_c_only_best.pt',
    quiet_env=True,
    deterministic_eval=True,
    extra_step_budget=40,
    reward_normalize=True,
    use_candidate_set_feat=True,
    hidden_dim=128,
    attn_heads=4,
    attn_layers=1,
    dropout=0.0,
    lr=9e-5,
    gamma=0.99,
    gae_lambda=0.95,
    clip_ratio=0.14,
    ent_coef=0.005,
    value_coef=0.6,
    max_grad_norm=0.5,
    ppo_epochs=3,
    minibatch_size=256,
    target_kl=0.011,
    max_candidates=40,
    auto_max_candidates=True,
    candidate_scan_rule='FIFO_SPT',
    candidate_safety_margin=10,
    round_max_candidates_to_power_of_two=False,
    train_full_eval_interval=5,
    baseline_cache_enabled=False,
    evaluate_rule_baselines_on_cache_miss=False,
    rollout_parallel_envs=8,
    eval_parallel_envs=8,
    val_patience=0,
    best_last_n_epochs=20,
    use_graph_encoder=False,
    use_hetero_lite=False,
    use_edge_rule_msg=True,
    use_edge_opmch_msg=True,
)

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / 'run_config.json').write_text(json.dumps({
    'split_json': str(SPLIT_JSON),
    'ablation': 'C-only: Train_C Agent_C PPO only; no O/M hierarchy; no O/M local rewards',
    'train_cases': train_cases,
    'val_cases': val_cases,
    'test_cases': test_cases,
    'epochs': cfg.epochs,
    'seed': cfg.seed,
    'candidate_params': {
        'agent_c_topk_jobs': env_configs.agent_c_topk_jobs,
        'agent_c_topk_machines': env_configs.agent_c_topk_machines,
        'agent_c_pairs_per_rule': env_configs.agent_c_pairs_per_rule,
        'agent_c_extra_explore_pairs': env_configs.agent_c_extra_explore_pairs,
        'agent_c_refine_max_pairs_per_op': env_configs.agent_c_refine_max_pairs_per_op,
        'agent_c_refine_global_reserve_size': env_configs.agent_c_refine_global_reserve_size,
    },
}, ensure_ascii=False, indent=2), encoding='utf-8')

summary = train_agent_c_parallel(cfg)
(OUT_DIR / 'c_only_train_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({
    'save_dir': str(OUT_DIR),
    'best_val_makespan': summary.get('best_val_makespan'),
    'selected_test_mean_makespan': summary.get('selected_test_eval', {}).get('mean_makespan'),
    'final_test_mean_makespan': summary.get('final_test_eval', {}).get('mean_makespan'),
}, ensure_ascii=False, indent=2), flush=True)

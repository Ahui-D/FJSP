
from __future__ import annotations

import json
import sys
from pathlib import Path

_ORIG_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]
from Params import configs as env_configs
from Train_OC_MAPPO import OCMAPPOConfig, train_oc_mappo
sys.argv = _ORIG_ARGV

SPLIT_JSON = Path('splits/3sd_10x5_seed32_split_8_1_1.json')
OUT_DIR = Path('exp_runs/ablation_ocmappo_no_rule_s22_3sd10x5_e100')
payload = json.loads(SPLIT_JSON.read_text(encoding='utf-8'))
train_cases = [str(x) for x in payload.get('train_cases', [])]
val_cases = [str(x) for x in payload.get('val_cases', [])]
test_cases = [str(x) for x in payload.get('test_cases', [])]

# Ablation: keep OC-MAPPO O/M/C unchanged, disable dispatching-rule-guided candidate recall.
env_configs.agent_c_candidate_generation_mode = 'no_rule'
env_configs.agent_c_active_job_rules = ''
env_configs.agent_c_active_machine_rules = ''

cfg = OCMAPPOConfig(
    train_cases=train_cases,
    val_cases=val_cases,
    test_cases=test_cases,
    split_source=str(SPLIT_JSON),
    seed=32,
    device='cuda',
    epochs=100,
    episodes_per_update=8,
    num_envs=8,
    eval_interval=1,
    eval_deterministic=True,
    eval_sample_times=1,
    eval_sample_reduce='best',
    deterministic_refine_topk=1,
    deterministic_refine_logit_gap=0.05,
    deterministic_refine_min_ect_gain=0.0,
    train_eval_force_deterministic=True,
    save_dir=str(OUT_DIR),
    active_job_rules='',
    active_machine_rules='',
    agent_c_topk_jobs=4,
    agent_c_topk_machines=4,
    agent_c_pairs_per_rule=7,
    agent_c_extra_explore_pairs=6,
    agent_c_refine_max_pairs_per_op=3,
    agent_c_refine_global_reserve_size=5,
    agent_c_refine_diversity_min_machines=3,
    agent_c_refine_keep_explore_min=1,
    c_min_max_candidates=40,
    c_candidate_safety_margin=10,
    c_candidate_safety_margin_ratio=0.12,
    o_op_safety_margin=5,
    o_topk_min=3,
    o_topk_max=9,
    m_safety_min_total_pairs=4,
    m_safety_min_machines_per_op=3,
    full_soft_widen_enabled=True,
    full_soft_widen_c_extra=2,
    full_soft_widen_o_extra=1,
    full_soft_widen_refine_pairs_per_op_extra=1,
    ocm_parallel_om_enabled=True,
    ocm_parallel_fusion_mode='union',
    ocm_parallel_min_final_pairs=4,
    ocm_parallel_budget_extra_pairs=2,
    ocm_parallel_max_final_pairs=0,
    ocm_parallel_m_teacher_mode='full_ref',
    ocm_parallel_rescue_ops=2,
    ocm_parallel_rescue_pairs_per_op=2,
    reward_version='role_aligned_v2',
    c_lr=9e-5,
    o_lr=1e-5,
    m_lr=1e-4,
    ppo_epochs=3,
    clip_ratio_c=0.14,
    clip_ratio_o=0.10,
    clip_ratio_m=0.12,
    target_kl_c=0.011,
    target_kl_o=0.05,
    minibatch_kl_guard_mult_o=5,
    o_update_interval=3,
    o_smooth_update_enabled=True,
    o_smooth_update_min_weight=0.35,
    o_batch_protect_enabled=True,
    o_ratio_batch_soft_limit=0.35,
    o_ratio_batch_hard_limit=0.70,
    o_ratio_batch_soft_scale=0.40,
    o_instability_monitor_enabled=False,
    o_reward_beta_shape=0.04,
    o_reward_beta_shape_end=0.015,
    o_teacher_coef=0.0,
    o_teacher_coef_end=0.0,
    o_teacher_coef_min=0.0,
    o_consensus_coef=0.0,
    o_consensus_coef_end=0.0,
    o_consensus_coef_min=0.0,
    o_set_aux_coef=0.001,
    o_set_ppo_mix=0.02,
    m_update_interval=1,
    m_ent_coef=0.014,
    m_ent_coef_end=0.011,
    m_ent_coef_anneal_updates=100,
    m_reward_beta_shape=0.07,
    m_reward_beta_shape_end=0.05,
    m_teacher_coef=0.04,
    m_teacher_coef_end=0.01,
    m_teacher_coef_min=0.01,
    m_consensus_coef=0.0,
    m_consensus_coef_end=0.0,
    m_consensus_coef_min=0.0,
    val_plateau_patience=3,
    val_plateau_decay=0.95,
    val_plateau_min_scale=0.30,
    val_plateau_apply_c=True,
    val_plateau_apply_o=True,
    val_plateau_apply_m=False,
    val_plateau_restore_best=True,
    val_plateau_restore_min_epoch=8,
    val_plateau_restore_cooldown=3,
    val_plateau_restore_rel_gap=0.05,
    rollout_backend='serial',
    quiet_env=True,
)

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / 'run_config.json').write_text(json.dumps({
    'ablation': 'OC-MAPPO unchanged; no dispatching-rule-guided candidate recall',
    'candidate_generation_mode': 'no_rule',
    'split_json': str(SPLIT_JSON),
    'epochs': cfg.epochs,
    'seed': cfg.seed,
    'rollout_backend': cfg.rollout_backend,
    'train_cases': train_cases,
    'val_cases': val_cases,
    'test_cases': test_cases,
}, ensure_ascii=False, indent=2), encoding='utf-8')
summary = train_oc_mappo(cfg)
(OUT_DIR / 'no_rule_train_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'save_dir': str(OUT_DIR), 'summary_type': type(summary).__name__}, ensure_ascii=False), flush=True)

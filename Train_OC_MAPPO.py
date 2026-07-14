from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import math
import multiprocessing as mp
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Agent_C import AgentCConfig, Agent_C, _flatten_f32
from Agent_M_learn import AgentMLearnConfig, Agent_M_Learn
from Agent_O_frozenC import AgentOConfig, Agent_O

# Some legacy modules parse CLI arguments at import time; shield this script's CLI args.
_ORIG_ARGV = sys.argv[:]
try:
    sys.argv = [sys.argv[0]]
    from Params import configs as env_configs
    from Train_C import build_env_for_case, infer_training_dims, make_case_split, seed_everything
    from Train_O_frozenC import filter_obs_by_selected_ops
finally:
    sys.argv = _ORIG_ARGV


def _torch_load_compat(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


@dataclass
class OCMAPPOConfig:
    train_cases: List[str]
    val_cases: List[str]
    test_cases: List[str]
    split_source: str = ""
    # Which split groups are used for feature-dimension scanning.
    # Comma-separated values from: train,val,test,all
    scan_case_splits: str = "train,val"
    # Which split groups are evaluated for final makespan summary.
    # Comma-separated values from: train,val,test,all
    eval_case_splits: str = "val,test"

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    reset_rule: str = "FIFO_SPT"

    epochs: int = 80
    episodes_per_update: int = 8
    # Effective number of rollout environments processed per update chunk.
    # Kept separate from episodes_per_update for clearer CLI semantics.
    num_envs: int = 8
    eval_interval: int = 1
    eval_deterministic: bool = True
    eval_sample_times: int = 1
    eval_sample_reduce: str = "best"  # best | mean
    # In deterministic evaluation, optionally re-rank C top-k logits by a
    # deterministic ECT/PT heuristic to reduce myopic argmax errors.
    deterministic_refine_topk: int = 1
    # Only refine when top-1 and alternatives are close enough in policy logits.
    deterministic_refine_logit_gap: float = 0.05
    # Only switch away from argmax when ECT improvement reaches this threshold.
    deterministic_refine_min_ect_gain: float = 0.0
    # Keep train-phase validation deterministic unless explicitly disabled.
    train_eval_force_deterministic: bool = True
    # In pure val-only deterministic evaluation, optionally evaluate extra
    # deterministic refine gap variants per case and keep the best makespan.
    # Example: "argmax,0.024,0.033"
    deterministic_val_only_gap_candidates: str = ""
    # Only affects pure evaluation mode (epochs==0 and test set empty).
    val_only_best_strategy: bool = True
    val_only_add_deterministic_baseline: bool = True
    val_only_extra_samples_per_case: int = 20
    val_only_hard_case_topk: int = 3
    val_only_seed_jitter: bool = True
    # Adaptive extra sampling for val-only best strategy.
    # Score = hardness + var_bonus * uncertainty + ucb_bonus * exploration.
    val_only_adaptive_extra_sampling: bool = True
    val_only_adaptive_var_bonus: float = 0.35
    val_only_adaptive_ucb_bonus: float = 0.15
    extra_step_budget: int = 40
    quiet_env: bool = True
    profile_timing: bool = False
    profile_rule_stats: bool = False
    profile_rule_topk: int = 12
    rollout_backend: str = "serial"  # "serial" or "mp"
    rollout_workers: int = 0  # 0 means auto=min(num_envs, cpu_count)
    rollout_min_cases_per_worker: int = 2
    rollout_worker_device: str = "cpu"
    rollout_mp_start_method: str = "spawn"
    rollout_worker_policy_mode: str = "train"  # "train" or "eval"
    rollout_worker_reseed: bool = True

    save_dir: str = "checkpoints_oc_mappo"
    c_checkpoint_name: str = "agent_c_mappo_best.pt"
    o_checkpoint_name: str = "agent_o_mappo_best.pt"
    m_checkpoint_name: str = "agent_m_mappo_best.pt"
    critic_checkpoint_name: str = "critic_mappo_best.pt"
    c_latest_checkpoint_name: str = "agent_c_mappo_latest.pt"
    o_latest_checkpoint_name: str = "agent_o_mappo_latest.pt"
    m_latest_checkpoint_name: str = "agent_m_mappo_latest.pt"
    critic_latest_checkpoint_name: str = "critic_mappo_latest.pt"
    c_final_checkpoint_name: str = "agent_c_mappo_final.pt"
    o_final_checkpoint_name: str = "agent_o_mappo_final.pt"
    m_final_checkpoint_name: str = "agent_m_mappo_final.pt"
    critic_final_checkpoint_name: str = "critic_mappo_final.pt"
    progress_json_name: str = "train_oc_mappo_progress.json"
    history_jsonl_name: str = "train_oc_mappo_history.jsonl"
    train_state_name: str = "train_oc_mappo_state_latest.pt"
    save_latest_every_epoch: bool = True
    save_progress_every_epoch: bool = True
    save_history_jsonl: bool = True

    init_c_checkpoint_path: str = ""
    init_o_checkpoint_path: str = ""
    init_m_checkpoint_path: str = ""
    init_critic_checkpoint_path: str = ""
    resume_state_path: str = ""

    # C actor
    c_hidden_dim: int = 128
    c_attn_heads: int = 4
    c_attn_layers: int = 1
    c_dropout: float = 0.0
    c_lr: float = 2e-4
    c_max_candidates: int = 16
    c_auto_max_candidates: bool = True
    c_candidate_safety_margin: int = 4
    # Additional ratio-based margin to reduce underestimation when scaling up datasets.
    c_candidate_safety_margin_ratio: float = 0.10
    c_min_max_candidates: int = 16
    c_round_max_candidates_to_power_of_two: bool = False
    c_use_graph_encoder: bool = False
    c_use_edge_rule_msg: bool = True
    c_use_edge_opmch_msg: bool = True
    c_use_adaptive_edge_gates: bool = False
    # Active rule subset used by environment candidate generation.
    # Empty string means using all registered rules.
    active_job_rules: str = "FIFO,CRJ,MWKR"
    active_machine_rules: str = "SPT,EET,EETLQ,EETD"
    # Environment-side candidate-pool width controls.
    agent_c_topk_jobs: int = 3
    agent_c_topk_machines: int = 3
    agent_c_pairs_per_rule: int = 5
    agent_c_extra_explore_pairs: int = 4
    agent_c_two_stage_refine_enabled: bool = True
    agent_c_refine_max_pairs_per_op: int = 2
    agent_c_refine_keep_global_best_ect: int = 1
    agent_c_refine_keep_global_best_pt: int = 1
    agent_c_refine_score_w_eet: float = 1.0
    agent_c_refine_score_w_pt: float = 0.3
    agent_c_refine_score_w_queue: float = 0.2
    agent_c_refine_score_w_support: float = 0.5
    agent_c_refine_explore_bonus: float = 0.1
    agent_c_candidate_v2_enabled: bool = True
    agent_c_critical_pairs_enabled: bool = True
    agent_c_critical_pairs_per_op: int = 2
    agent_c_refine_max_pairs_per_machine: int = 0
    agent_c_refine_global_reserve_size: int = 2
    agent_c_refine_diversity_min_machines: int = 2
    agent_c_refine_keep_explore_min: int = 1
    agent_c_candidate_debug_print: bool = False

    # Full配置的保守放宽（仅用于full-like配置，避免候选空间过窄）
    full_soft_widen_enabled: bool = True
    full_soft_widen_c_extra: int = 2
    full_soft_widen_o_extra: int = 1
    full_soft_widen_refine_pairs_per_op_extra: int = 1
    # Experimental parallel O/M path for validating whether serial O->M
    # over-compresses M's visible space and induces entropy collapse.
    ocm_parallel_om_enabled: bool = False
    ocm_parallel_fusion_mode: str = "o_plus_m_backup"  # intersection | union | o_plus_m_backup
    ocm_parallel_min_final_pairs: int = 3
    ocm_parallel_budget_extra_pairs: int = 2
    ocm_parallel_max_final_pairs: int = 0
    ocm_parallel_debug_print: bool = False
    ocm_parallel_m_teacher_mode: str = "full_ref"  # full_ref | disabled
    ocm_parallel_rescue_ops: int = 2
    ocm_parallel_rescue_pairs_per_op: int = 2
    ocm_parallel_rescue_include_c_full: bool = True
    ocm_parallel_rescue_include_global_best: bool = True

    # O actor
    o_hidden_dim: int = 128
    o_dropout: float = 0.05
    o_lr: float = 2e-4
    o_topk: int = 5
    o_max_ops: int = 32
    o_auto_max_ops: bool = True
    o_op_safety_margin: int = 2
    o_model_type: str = "mlp"

    # M actor
    m_hidden_dim: int = 128
    m_dropout: float = 0.05
    m_lr: float = 1e-4
    m_weight_decay: float = 1e-4
    m_feature_mode: str = "machine_heavy"
    m_strategy_mode: str = "ppo"
    m_enable_entropy_backup: bool = True
    m_backup_entropy_threshold: float = 0.75
    m_backup_max_extra_pairs: int = 1
    m_keep_c_full_top1: bool = True
    m_keep_c_o_top1: bool = True
    # If enabled, hard-keep protection is only active when teacher coef is above threshold.
    # Keep disabled by default to avoid removing structural protection late in training.
    m_hard_keep_requires_teacher: bool = False
    m_hard_keep_teacher_min_coef: float = 0.0
    # Optional safety trigger: even when teacher is inactive, allow hard-keep
    # if M entropy is too low (to avoid late-stage over-pruning collapse).
    m_hard_keep_entropy_trigger: bool = False
    m_hard_keep_entropy_threshold: float = 0.20
    # Pair-level safety expansion before sending candidates to C.
    m_expand_pairs_for_safety: bool = True
    m_safety_min_total_pairs: int = 3
    m_safety_min_machines_per_op: int = 3
    # M-collapse monitoring and optional automatic LR decay.
    m_entropy_warn_threshold: float = 0.45
    m_entropy_hard_threshold: float = 0.35
    m_keep_per_o_warn_threshold: float = 1.5
    m_auto_lr_decay_on_collapse: bool = False
    m_auto_lr_decay_factor: float = 0.8
    m_auto_lr_min_scale: float = 0.25
    # P1 speed optimization: skip the intermediate C(o-filtered) forward pass
    # and use an ECT-based proxy reference from O-filtered candidates.
    p1_fast_skip_c_o_ref: bool = True
    # Optional aggressive mode: skip the initial C(full) reference pass.
    # This reduces C forwards to one per step when combined with p1_fast_skip_c_o_ref.
    p1_skip_c_full_ref: bool = False

    # MAPPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    clip_ratio_c: float = 0.2
    clip_ratio_o: float = 0.2
    clip_ratio_m: float = 0.1
    ppo_epochs: int = 4
    minibatch_size: int = 256
    value_coef: float = 0.6
    value_coef_c: float = 0.6
    value_coef_o: float = 0.6
    value_coef_m: float = 0.6
    use_huber_value_loss: bool = True
    value_huber_delta: float = 1.0
    value_clip_range: float = 0.2
    c_ent_coef: float = 0.005
    o_ent_coef: float = 0.005
    m_ent_coef: float = 0.02
    # M entropy annealing: keep higher exploration early, then gradually tighten.
    m_ent_coef_end: float = 0.012
    m_ent_coef_anneal_updates: int = 120
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    target_kl_c: float = 0.02
    target_kl_o: float = 0.01
    target_kl_m: float = 0.01
    kl_penalty_coef_c: float = 0.0
    kl_penalty_coef_o: float = 0.0
    kl_penalty_coef_m: float = 0.0
    ppo_early_stop: bool = True
    minibatch_kl_guard_enabled: bool = True
    minibatch_kl_guard_mult_c: float = 2.0
    minibatch_kl_guard_mult_o: float = 1.5
    minibatch_kl_guard_mult_m: float = 2.0
    # Actor update frequency controls (1 means update every optimizer step).
    o_update_interval: int = 1
    m_update_interval: int = 1
    # P0: smooth O updates - update every chunk with reduced per-step strength.
    o_smooth_update_enabled: bool = True
    o_smooth_update_min_weight: float = 0.20
    # Additional O minibatch KL protection (beyond global guard/early-stop).
    o_kl_batch_soft_limit_mult: float = 1.5
    o_kl_batch_hard_limit_mult: float = 2.5
    o_kl_batch_soft_scale: float = 0.5
    # P1: stronger minibatch protection for O using ratio+KL thresholds.
    o_batch_protect_enabled: bool = True
    o_ratio_batch_soft_limit: float = 0.40
    o_ratio_batch_hard_limit: float = 0.80
    o_ratio_batch_soft_scale: float = 0.5

    critic_hidden_dim: int = 256
    critic_lr: float = 3e-4
    critic_rich_state: bool = False

    # Val plateau LR decay (stability)
    val_plateau_patience: int = 2
    val_plateau_decay: float = 0.7
    val_plateau_min_scale: float = 0.2
    val_plateau_apply_c: bool = True
    val_plateau_apply_o: bool = True
    val_plateau_apply_m: bool = True
    # Optionally rollback actors/critic to current best checkpoint when plateau triggers.
    val_plateau_restore_best: bool = True
    val_plateau_restore_min_epoch: int = 3
    val_plateau_restore_cooldown: int = 2
    val_plateau_restore_rel_gap: float = 0.03
    val_plateau_restore_abs_gap: float = 0.0
    critic_split_tower: bool = False
    critic_tower_hidden_dim: int = 256
    critic_use_gnn_branch: bool = False
    critic_use_gate_fusion: bool = False
    critic_gnn_hidden_dim: int = 128
    critic_gnn_heads: int = 4
    critic_freeze_gnn_epochs: int = 0

    # Reward versioning
    reward_version: str = "role_aligned_v1"  # legacy | role_aligned_v1 | role_aligned_v2
    reward_schedule_anneal_epochs: int = 100

    # O-specific reward shaping and robust top-k control
    o_reward_alpha_env: float = 1.00
    o_reward_alpha_env_end: float = 1.00
    o_reward_beta_shape: float = 0.10
    o_reward_beta_shape_end: float = 0.04
    o_teacher_coef: float = 0.03
    o_teacher_coef_end: float = 0.005
    o_teacher_coef_min: float = 0.005
    o_consensus_coef: float = 0.02
    o_consensus_coef_end: float = 0.003
    o_consensus_coef_min: float = 0.003
    o_reward_anneal_epochs: int = 180
    o_reward_schedule: str = "linear"
    o_reward_mid_progress: float = 0.55
    o_reward_mid_ratio: float = 0.35
    o_reward_clip_abs: float = 1.5
    reward_retain_pos: float = 0.20
    reward_retain_neg: float = -0.45
    reward_quality_coef: float = 0.30
    reward_quality_clip: float = 0.30
    reward_quality_hard_threshold: float = 0.08
    reward_quality_hard_penalty: float = 0.18
    reward_quality_hard_smooth: bool = False
    reward_quality_hard_smooth_width: float = 0.02
    reward_mismatch_penalty: float = 0.08
    reward_mismatch_only_if_not_retained: bool = True
    reward_makespan_terminal_coef: float = 0.01
    o_topk_min: int = 2
    o_topk_max: int = 8
    # Scale-aware floor: keep at least ceil(num_ops / divisor) ops when divisor > 0.
    o_scale_topk_floor_divisor: int = 10
    o_topk_entropy_gain: float = 1.0
    o_entropy_fallback_threshold: float = 1.0
    o_entropy_fallback_extra_ops: int = 1
    o_entropy_low_fallback_threshold: float = 1.5
    o_entropy_low_fallback_extra_ops: int = 1
    o_reward_fallback_scale: float = 0.85
    o_reference_c_deterministic: bool = True
    o_redundancy_target_ratio: float = 0.35
    o_redundancy_penalty_coef: float = 0.05
    o_coverage_bonus_coef: float = 0.02
    # Auxiliary BCE supervision on O-selected op set (align top-k behavior and loss).
    o_set_aux_coef: float = 0.02
    # Blend PPO's O-branch action log-prob with a set-aligned surrogate log-prob
    # computed from the selected op mask, so training better matches rollout behavior.
    o_set_ppo_mix: float = 0.35
    # Adaptive entropy regularization for O branch to avoid low-entropy over-confidence.
    o_ent_adaptive_target: float = 1.8
    o_ent_adaptive_gain: float = 0.6
    o_ent_adaptive_max_scale: float = 2.5
    # Automatically tighten O-branch training pressure when KL/guard signals
    # indicate instability, so we do not rely only on manual hyper-parameter tuning.
    o_instability_monitor_enabled: bool = True
    o_instability_kl_threshold: float = 0.04
    o_instability_guard_threshold: float = 0.10
    o_instability_entropy_threshold: float = 1.75
    o_auto_lr_decay_on_instability: bool = True
    o_auto_lr_decay_factor: float = 0.85
    o_auto_lr_min_scale: float = 0.25
    o_auto_update_interval_on_instability: bool = True
    o_auto_update_interval_max: int = 6
    o_auto_aux_scale_on_instability: bool = True
    o_auto_aux_scale_factor: float = 0.85
    o_auto_aux_min_scale: float = 0.25
    o_auto_reward_scale_on_instability: bool = True
    o_auto_reward_scale_factor: float = 0.85
    o_auto_reward_min_scale: float = 0.35

    # M-specific reward shaping
    m_reward_alpha_env: float = 1.00
    m_reward_alpha_env_end: float = 1.00
    m_reward_beta_shape: float = 0.08
    m_reward_beta_shape_end: float = 0.03
    m_teacher_coef: float = 0.05
    m_teacher_coef_end: float = 0.01
    m_teacher_coef_min: float = 0.01
    m_consensus_coef: float = 0.006
    m_consensus_coef_end: float = 0.0
    m_consensus_coef_min: float = 0.0
    m_reward_clip_abs: float = 1.5
    m_reward_fallback_scale: float = 0.90
    m_reward_retain_pos: float = 0.20
    m_reward_retain_neg: float = -0.30
    m_reward_quality_coef: float = 0.35
    m_reward_quality_hard_threshold: float = 0.08
    m_reward_quality_hard_penalty: float = 0.10
    m_reward_overprune_target_keep_ratio: float = 0.30
    m_reward_overprune_coef: float = 0.10
    m_reward_mismatch_penalty: float = 0.08
    m_reward_terminal_gap_coef: float = 0.01

    # Candidate-aware M奖励缩放（候选越窄，M额外引导越弱）
    m_candidate_aware_scaling_enabled: bool = True
    # 宽度区间缩放：low->最弱引导，high->接近原强度
    m_candidate_aware_c_low: int = 22
    m_candidate_aware_c_high: int = 34
    m_candidate_aware_o_low: int = 9
    m_candidate_aware_o_high: int = 12
    m_candidate_aware_min_scale: float = 0.35
    m_candidate_aware_gamma: float = 1.30
    m_candidate_aware_apply_to_beta_shape: bool = True
    m_candidate_aware_apply_to_teacher: bool = True
    m_candidate_aware_apply_to_consensus: bool = True
    # Entropy-aware M reward downscaling: when previous-epoch M entropy is low,
    # reduce teacher/consensus pressure first, and only lightly soften shape.
    m_entropy_feedback_enabled: bool = True
    m_entropy_feedback_threshold: float = 0.50
    m_entropy_feedback_power: float = 1.25
    m_entropy_feedback_shape_min_scale: float = 0.90
    m_entropy_feedback_teacher_min_scale: float = 0.20
    m_entropy_feedback_consensus_min_scale: float = 0.35

    # C reward (environment-side) passthrough
    reward_balance_coef: float = 0.0
    reward_wait_coef: float = 0.0


class CentralCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        split_tower: bool = False,
        tower_hidden_dim: int = 256,
        c_pair_dim: int = 0,
        o_op_dim: int = 0,
        m_pair_dim: int = 0,
        use_gnn_branch: bool = False,
        use_gate_fusion: bool = False,
        gnn_hidden_dim: int = 128,
        gnn_heads: int = 4,
    ):
        super().__init__()
        self.split_tower = bool(split_tower)
        self.use_gnn_branch = bool(use_gnn_branch)
        self.use_gate_fusion = bool(use_gate_fusion) and self.use_gnn_branch
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        if self.use_gnn_branch:
            gnn_h = int(max(16, gnn_hidden_dim))
            heads = int(max(1, gnn_heads))
            while heads > 1 and (gnn_h % heads) != 0:
                heads -= 1

            self.c_pair_proj = nn.Sequential(
                nn.Linear(int(c_pair_dim), gnn_h),
                nn.GELU(),
                nn.LayerNorm(gnn_h),
            )
            self.o_op_proj = nn.Sequential(
                nn.Linear(int(o_op_dim), gnn_h),
                nn.GELU(),
                nn.LayerNorm(gnn_h),
            )
            self.m_pair_proj = nn.Sequential(
                nn.Linear(int(m_pair_dim), gnn_h),
                nn.GELU(),
                nn.LayerNorm(gnn_h),
            )
            self.gnn_attn = nn.MultiheadAttention(embed_dim=gnn_h, num_heads=heads, batch_first=True)
            self.gnn_norm = nn.LayerNorm(gnn_h)
            self.gnn_ffn = nn.Sequential(
                nn.Linear(gnn_h, gnn_h),
                nn.GELU(),
                nn.LayerNorm(gnn_h),
            )
            self.gnn_to_hidden = nn.Sequential(
                nn.Linear(gnn_h, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            if self.use_gate_fusion:
                self.fusion_gate = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Sigmoid(),
                )
            self.fusion_proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )

            self._gnn_modules = [
                self.c_pair_proj,
                self.o_op_proj,
                self.m_pair_proj,
                self.gnn_attn,
                self.gnn_norm,
                self.gnn_ffn,
                self.gnn_to_hidden,
                self.fusion_proj,
            ]
            if self.use_gate_fusion:
                self._gnn_modules.append(self.fusion_gate)
        else:
            self._gnn_modules = []

        if self.split_tower:
            tower_h = int(max(8, tower_hidden_dim))
            self.c_tower = nn.Sequential(
                nn.Linear(hidden_dim, tower_h),
                nn.GELU(),
                nn.LayerNorm(tower_h),
            )
            self.o_tower = nn.Sequential(
                nn.Linear(hidden_dim, tower_h),
                nn.GELU(),
                nn.LayerNorm(tower_h),
            )
            self.m_tower = nn.Sequential(
                nn.Linear(hidden_dim, tower_h),
                nn.GELU(),
                nn.LayerNorm(tower_h),
            )
            self.v_c = nn.Linear(tower_h, 1)
            self.v_o = nn.Linear(tower_h, 1)
            self.v_m = nn.Linear(tower_h, 1)
        else:
            self.v_c = nn.Linear(hidden_dim, 1)
            self.v_o = nn.Linear(hidden_dim, 1)
            self.v_m = nn.Linear(hidden_dim, 1)

    def set_gnn_trainable(self, trainable: bool) -> None:
        if not self.use_gnn_branch:
            return
        for module in self._gnn_modules:
            for p in module.parameters():
                p.requires_grad = bool(trainable)

    def _encode_relational(
        self,
        c_pair: torch.Tensor,
        c_mask: torch.Tensor,
        o_op: torch.Tensor,
        o_mask: torch.Tensor,
        m_pair: torch.Tensor,
        m_mask: torch.Tensor,
    ) -> torch.Tensor:
        c_tok = self.c_pair_proj(c_pair)
        o_tok = self.o_op_proj(o_op)
        m_tok = self.m_pair_proj(m_pair)
        tok = torch.cat([c_tok, o_tok, m_tok], dim=1)

        mask = torch.cat([c_mask, o_mask, m_mask], dim=1).to(dtype=torch.bool)
        safe_mask = mask.clone()
        all_invalid = ~safe_mask.any(dim=1)
        if bool(all_invalid.any()):
            safe_mask[all_invalid, 0] = True

        tok = tok * safe_mask.unsqueeze(-1).to(dtype=tok.dtype)
        attn_out, _ = self.gnn_attn(tok, tok, tok, key_padding_mask=(~safe_mask), need_weights=False)
        tok = self.gnn_norm(tok + attn_out)
        tok = self.gnn_ffn(tok)
        denom = safe_mask.sum(dim=1, keepdim=True).clamp(min=1).to(dtype=tok.dtype)
        return (tok * safe_mask.unsqueeze(-1).to(dtype=tok.dtype)).sum(dim=1) / denom

    def forward(
        self,
        x: torch.Tensor,
        c_pair: Optional[torch.Tensor] = None,
        c_mask: Optional[torch.Tensor] = None,
        o_op: Optional[torch.Tensor] = None,
        o_mask: Optional[torch.Tensor] = None,
        m_pair: Optional[torch.Tensor] = None,
        m_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(x)

        if self.use_gnn_branch and c_pair is not None and c_mask is not None and o_op is not None and o_mask is not None and m_pair is not None and m_mask is not None:
            g_rel = self._encode_relational(
                c_pair=c_pair,
                c_mask=c_mask,
                o_op=o_op,
                o_mask=o_mask,
                m_pair=m_pair,
                m_mask=m_mask,
            )
            g_rel = self.gnn_to_hidden(g_rel)
            if self.use_gate_fusion:
                gate = self.fusion_gate(torch.cat([h, g_rel], dim=-1))
                g_rel = gate * g_rel + (1.0 - gate) * h
            h = self.fusion_proj(torch.cat([h, g_rel], dim=-1))

        if self.split_tower:
            return self.v_c(self.c_tower(h)), self.v_o(self.o_tower(h)), self.v_m(self.m_tower(h))
        return self.v_c(h), self.v_o(h), self.v_m(h)


@dataclass
class Transition:
    c_global: np.ndarray
    c_pair: np.ndarray
    c_mask: np.ndarray
    c_action_idx: int
    c_old_log_prob: float

    o_global: np.ndarray
    o_op: np.ndarray
    o_mask: np.ndarray
    o_selected_mask: np.ndarray
    o_action_idx: int
    o_old_log_prob: float
    o_old_set_log_prob: float

    m_global: np.ndarray
    m_pair: np.ndarray
    m_mask: np.ndarray
    m_pair_op_idx: np.ndarray
    m_op_order: List[int]
    m_action_pair_indices: np.ndarray
    m_old_log_prob: float

    state_vec: np.ndarray
    value_c: float
    value_o: float
    value_m: float
    reward_c: float
    reward_o: float
    reward_m: float
    done: bool


@dataclass
class EpisodeRollout:
    case_name: str
    transitions: List[Transition]
    episode_reward: float
    episode_o_reward: float
    episode_m_reward: float
    makespan: float
    steps: int
    done: bool
    o_avg_selected_ops: float = 0.0
    o_fallback_rate: float = 0.0
    o_invalid_op_ratio: float = 0.0
    m_avg_keep_pairs: float = 0.0
    m_adopt_rate: float = 0.0
    c_forward_per_step: float = 0.0
    parallel_avg_m_scope_pairs: float = 0.0
    parallel_avg_final_pairs: float = 0.0
    parallel_avg_rescue_pairs: float = 0.0
    o_avg_shape_term: float = 0.0
    o_avg_teacher_term: float = 0.0
    o_avg_consensus_term: float = 0.0


def infer_o_dims(
    case_paths: List[str],
    reset_rule: str,
    quiet: bool = True,
    use_candidate_set_feat: bool = True,
) -> Tuple[int, int, int]:
    max_global_dim = 1
    op_feat_dim = 1
    observed_max_ops = 1

    for case_path in case_paths:
        env, _ = build_env_for_case(case_path, reset_rule=reset_rule, quiet=quiet)
        obs = env.get_agent_c_obs(batch_idx=0)

        global_feat = _flatten_f32(obs["global_feat"])
        if use_candidate_set_feat and "candidate_set_feat" in obs:
            global_feat = np.concatenate([global_feat, _flatten_f32(obs["candidate_set_feat"])], axis=0)

        op_base_dim = int(np.asarray(obs["op_node_feat"], dtype=np.float32).shape[1])
        # preprocess_obs_o builds 10 handcrafted op-level augment features.
        op_feat_dim = max(op_feat_dim, op_base_dim + 10)
        max_global_dim = max(max_global_dim, int(global_feat.shape[0]))
        observed_max_ops = max(observed_max_ops, len({int(op) for op, _ in obs.get("candidate_pairs", [])}))

    return int(max_global_dim), int(op_feat_dim), int(observed_max_ops)


def infer_m_dims(
    case_paths: List[str],
    reset_rule: str,
    quiet: bool = True,
) -> Tuple[int, int]:
    max_global_dim = 1
    pair_feat_dim = 1
    for case_path in case_paths:
        env, _ = build_env_for_case(case_path, reset_rule=reset_rule, quiet=quiet)
        obs = env.get_agent_c_obs(batch_idx=0)
        global_feat = _flatten_f32(obs.get("global_feat", np.zeros((0,), dtype=np.float32)))
        pair_feat = np.asarray(obs.get("pair_feat", []), dtype=np.float32)
        max_global_dim = max(max_global_dim, int(global_feat.shape[0]))
        if pair_feat.ndim == 2 and pair_feat.shape[1] > 0:
            pair_feat_dim = max(pair_feat_dim, int(pair_feat.shape[1]))
    return int(max_global_dim), int(pair_feat_dim)


def _resolve_pair_sources(obs: Dict[str, object]) -> List[List[str]]:
    pairs = list(obs.get("candidate_pairs", []))
    src = list(obs.get("pair_sources", []))
    if len(src) < len(pairs):
        src = src + [[] for _ in range(len(pairs) - len(src))]
    return src[: len(pairs)]


def _slice_candidate_obs(obs: Dict[str, object], idx_keep: List[int]) -> Dict[str, object]:
    idx_keep = [int(i) for i in idx_keep]
    out = dict(obs)

    pairs = list(obs["candidate_pairs"])
    src = _resolve_pair_sources(obs)
    out["candidate_pairs"] = [pairs[i] for i in idx_keep]
    out["pair_sources"] = [src[i] for i in idx_keep]

    pair_feat = np.asarray(obs["pair_feat"], dtype=np.float32)
    out["pair_feat"] = pair_feat[idx_keep]
    out["pair_node_feat"] = np.asarray(out["pair_feat"], dtype=np.float32)

    if "pair_op_idx" in obs:
        out["pair_op_idx"] = np.asarray(obs["pair_op_idx"], dtype=np.int64)[idx_keep]
    else:
        out["pair_op_idx"] = np.asarray([int(a) for a, _ in out["candidate_pairs"]], dtype=np.int64)

    if "pair_mch_idx" in obs:
        out["pair_mch_idx"] = np.asarray(obs["pair_mch_idx"], dtype=np.int64)[idx_keep]
    else:
        out["pair_mch_idx"] = np.asarray([int(b) for _, b in out["candidate_pairs"]], dtype=np.int64)

    out["edge_op_to_pair"] = np.asarray(out["pair_op_idx"], dtype=np.int64)
    out["edge_mch_to_pair"] = np.asarray(out["pair_mch_idx"], dtype=np.int64)

    if "edge_rule_to_pair_feat" in obs:
        out["edge_rule_to_pair_feat"] = np.asarray(obs["edge_rule_to_pair_feat"], dtype=np.float32)[idx_keep]
    if "edge_opmch_to_pair_feat" in obs:
        out["edge_opmch_to_pair_feat"] = np.asarray(obs["edge_opmch_to_pair_feat"], dtype=np.float32)[idx_keep]

    out["pair_mask"] = np.ones((len(out["candidate_pairs"]),), dtype=bool)
    out["candidate_set_feat"] = np.asarray(
        [
            float(len(out["candidate_pairs"])),
            float(len({int(a) for a, _ in out["candidate_pairs"]})),
            float(len({int(b) for _, b in out["candidate_pairs"]})),
        ],
        dtype=np.float32,
    )
    out["legal_pairs_set"] = {(int(a), int(b)) for a, b in out["candidate_pairs"]}
    return out


def _selected_pair_indices_for_ops(obs: Dict[str, object], selected_ops: List[int]) -> List[int]:
    pairs = list(obs.get("candidate_pairs", []))
    if len(pairs) == 0:
        return []

    selected_set = {int(x) for x in selected_ops}
    idx_keep = [i for i, (op_id, _) in enumerate(pairs) if int(op_id) in selected_set]
    if len(idx_keep) > 0:
        return idx_keep

    pair_feat = np.asarray(obs.get("pair_feat", []), dtype=np.float32)
    if pair_feat.ndim == 2 and pair_feat.shape[0] > 0 and pair_feat.shape[1] > 4:
        return [int(np.argmin(pair_feat[:, 4]))]
    return [0]


def _find_pair_index(obs: Dict[str, object], op: int, mch: int) -> int:
    pairs = list(obs.get("candidate_pairs", []))
    for i, (a, b) in enumerate(pairs):
        if int(a) == int(op) and int(b) == int(mch):
            return int(i)
    return -1


def _all_op_ids_from_obs(obs: Dict[str, object]) -> List[int]:
    return list(dict.fromkeys(int(op_id) for op_id, _ in obs.get("candidate_pairs", [])))


def _pair_priority_key(full_obs: Dict[str, object], idx: int) -> Tuple[float, float, float, float, int]:
    pair_feat = np.asarray(full_obs.get("pair_feat", []), dtype=np.float32)
    pair_sources = _resolve_pair_sources(full_obs)

    ect = float("inf")
    pt = float("inf")
    queue = float("inf")
    if pair_feat.ndim == 2 and 0 <= int(idx) < pair_feat.shape[0]:
        if pair_feat.shape[1] > 4:
            ect = float(pair_feat[int(idx), 4])
        if pair_feat.shape[1] > 0:
            pt = float(pair_feat[int(idx), 0])
        if pair_feat.shape[1] > 19:
            queue = float(pair_feat[int(idx), 19])

    support = 0.0
    if 0 <= int(idx) < len(pair_sources):
        support = float(_source_rule_support_count(pair_sources[int(idx)]))

    return (ect, pt, queue, -support, int(idx))


def _rank_pair_indices(full_obs: Dict[str, object], indices: List[int]) -> List[int]:
    uniq = sorted(
        {
            int(i)
            for i in indices
            if int(i) >= 0
        }
    )
    return sorted(uniq, key=lambda i: _pair_priority_key(full_obs, int(i)))


def _build_parallel_m_scope(
    full_obs: Dict[str, object],
    selected_ops: List[int],
    selected_pair_indices: List[int],
    c_full_pair: Tuple[int, int],
    cfg: OCMAPPOConfig,
) -> Dict[str, object]:
    pairs = list(full_obs.get("candidate_pairs", []))
    pair_n = int(len(pairs))
    if pair_n <= 0:
        return {
            "selected_ops": [],
            "pair_indices": [],
            "rescue_ops": [],
            "rescue_pair_indices": [],
        }

    selected_ops_order = list(dict.fromkeys(int(x) for x in selected_ops))
    selected_set = set(selected_ops_order)
    selected_pair_keep = _rank_pair_indices(full_obs, list(selected_pair_indices))

    rescue_ops_limit = int(max(0, int(getattr(cfg, "ocm_parallel_rescue_ops", 2))))
    rescue_pairs_per_op = int(max(1, int(getattr(cfg, "ocm_parallel_rescue_pairs_per_op", 2))))
    rescue_ops: List[int] = []

    def _maybe_add_rescue_op(op_id: int) -> None:
        op_i = int(op_id)
        if op_i in selected_set or op_i in rescue_ops:
            return
        if len(rescue_ops) >= rescue_ops_limit:
            return
        rescue_ops.append(op_i)

    if bool(getattr(cfg, "ocm_parallel_rescue_include_c_full", True)):
        _maybe_add_rescue_op(int(c_full_pair[0]))

    if bool(getattr(cfg, "ocm_parallel_rescue_include_global_best", True)):
        best_op, _best_m = _best_pair_by_ect(full_obs)
        if int(best_op) >= 0:
            _maybe_add_rescue_op(int(best_op))

    if len(rescue_ops) < rescue_ops_limit:
        best_map = _op_best_ect_map(full_obs)
        remaining_ops = [int(op) for op in best_map.keys() if int(op) not in selected_set and int(op) not in set(rescue_ops)]
        remaining_ops.sort(key=lambda op: (float(best_map[int(op)]), int(op)))
        for op in remaining_ops:
            _maybe_add_rescue_op(int(op))
            if len(rescue_ops) >= rescue_ops_limit:
                break

    rescue_pair_indices: List[int] = []
    selected_pair_set = set(int(i) for i in selected_pair_keep)
    for op in rescue_ops:
        op_indices = [
            int(i)
            for i, (op_id, _mch) in enumerate(pairs)
            if int(op_id) == int(op) and int(i) not in selected_pair_set
        ]
        ranked = _rank_pair_indices(full_obs, op_indices)
        rescue_pair_indices.extend(ranked[:rescue_pairs_per_op])

    pair_indices = _rank_pair_indices(full_obs, selected_pair_keep + rescue_pair_indices)
    if len(pair_indices) == 0:
        pair_indices = _rank_pair_indices(full_obs, [0])

    return {
        "selected_ops": list(dict.fromkeys(selected_ops_order + rescue_ops)),
        "pair_indices": pair_indices,
        "rescue_ops": rescue_ops,
        "rescue_pair_indices": _rank_pair_indices(full_obs, rescue_pair_indices),
    }


def _parallel_fuse_keep_pair_indices(
    full_obs: Dict[str, object],
    o_keep_pair_indices: List[int],
    m_keep_pair_indices: List[int],
    cfg: OCMAPPOConfig,
) -> Tuple[List[int], Dict[str, object]]:
    pairs = list(full_obs.get("candidate_pairs", []))
    pair_n = int(len(pairs))
    if pair_n <= 0:
        return [], {
            "mode": str(getattr(cfg, "ocm_parallel_fusion_mode", "o_plus_m_backup")),
            "fallback": "no_pairs",
            "o_pairs": 0,
            "m_pairs": 0,
            "intersection_pairs": 0,
            "union_pairs": 0,
            "final_pairs": 0,
        }

    mode = str(getattr(cfg, "ocm_parallel_fusion_mode", "o_plus_m_backup")).strip().lower()
    if mode not in {"intersection", "union", "o_plus_m_backup"}:
        mode = "o_plus_m_backup"
    min_final_pairs = int(max(1, int(getattr(cfg, "ocm_parallel_min_final_pairs", 3))))
    budget_extra_pairs = int(max(0, int(getattr(cfg, "ocm_parallel_budget_extra_pairs", 2))))
    max_final_pairs = int(max(0, int(getattr(cfg, "ocm_parallel_max_final_pairs", 0))))

    o_set = {
        int(i)
        for i in o_keep_pair_indices
        if 0 <= int(i) < pair_n
    }
    m_set = {
        int(i)
        for i in m_keep_pair_indices
        if 0 <= int(i) < pair_n
    }
    inter = sorted(int(i) for i in (o_set & m_set))
    union = sorted(int(i) for i in (o_set | m_set))
    o_only = _rank_pair_indices(full_obs, sorted(int(i) for i in (o_set - m_set)))
    m_only = _rank_pair_indices(full_obs, sorted(int(i) for i in (m_set - o_set)))
    inter_ranked = _rank_pair_indices(full_obs, inter)

    target_final_pairs = int(max(min_final_pairs, len(m_set) + budget_extra_pairs))
    if max_final_pairs > 0:
        target_final_pairs = int(min(target_final_pairs, max_final_pairs))
    if len(union) > 0:
        target_final_pairs = int(min(target_final_pairs, len(union)))
    else:
        target_final_pairs = int(min(target_final_pairs, pair_n))
    target_final_pairs = int(max(min_final_pairs, target_final_pairs))

    fallback = "none"
    final_keep: List[int] = []
    final_keep.extend(inter_ranked[:target_final_pairs])
    existing = set(int(i) for i in final_keep)

    def _append_ranked(indices: List[int], limit: int) -> int:
        added = 0
        for idx in indices:
            idx_i = int(idx)
            if idx_i in existing:
                continue
            final_keep.append(idx_i)
            existing.add(idx_i)
            added += 1
            if added >= int(limit) or len(final_keep) >= target_final_pairs:
                break
        return int(added)

    if mode == "intersection":
        if len(final_keep) == 0 and len(union) > 0:
            fallback = "intersection_empty_budgeted_fill"
        _append_ranked(_rank_pair_indices(full_obs, union), target_final_pairs)
    elif mode == "union":
        o_quota = int(math.ceil(max(0, target_final_pairs - len(final_keep)) / 2.0))
        m_quota = int(max(0, target_final_pairs - len(final_keep) - o_quota))
        added_o = _append_ranked(o_only, o_quota)
        added_m = _append_ranked(m_only, m_quota)
        remaining = int(max(0, target_final_pairs - len(final_keep)))
        if remaining > 0:
            pool = _rank_pair_indices(full_obs, o_only[added_o:] + m_only[added_m:])
            if len(pool) > 0:
                _append_ranked(pool, remaining)
        if len(inter) < min_final_pairs:
            fallback = "budgeted_union_fill"
    else:
        added_o = _append_ranked(o_only, max(0, target_final_pairs - len(final_keep)))
        remaining = int(max(0, target_final_pairs - len(final_keep)))
        if remaining > 0:
            pool = _rank_pair_indices(full_obs, m_only + o_only[added_o:])
            if len(pool) > 0:
                _append_ranked(pool, remaining)
        if len(inter) < min_final_pairs:
            fallback = "o_primary_budgeted_fill"

    if len(final_keep) < min_final_pairs:
        pool = _rank_pair_indices(full_obs, list(union) if len(union) > 0 else list(range(pair_n)))
        before = len(final_keep)
        _append_ranked(pool, int(max(0, min_final_pairs - len(final_keep))))
        if len(final_keep) > before:
            if fallback == "none":
                fallback = "supplement_to_min_pairs"
            else:
                fallback = str(fallback) + "+supplement_to_min_pairs"

    if len(final_keep) == 0:
        ranked_all = _rank_pair_indices(full_obs, list(range(pair_n)))
        final_keep = ranked_all[:1] if len(ranked_all) > 0 else [0]
        fallback = "fallback_best_ect"

    final_keep = _rank_pair_indices(full_obs, [int(i) for i in final_keep if 0 <= int(i) < pair_n])
    info = {
        "mode": str(mode),
        "fallback": str(fallback),
        "o_pairs": int(len(o_set)),
        "m_pairs": int(len(m_set)),
        "o_only_pairs": int(len(o_only)),
        "m_only_pairs": int(len(m_only)),
        "intersection_pairs": int(len(inter)),
        "union_pairs": int(len(union)),
        "target_final_pairs": int(target_final_pairs),
        "final_pairs": int(len(final_keep)),
    }
    return final_keep, info


def _expand_pairs_for_safety(
    full_obs: Dict[str, object],
    selected_ops: List[int],
    keep_pair_indices: List[int],
    min_pairs_total: int,
    per_op_min_machines: int,
) -> List[int]:
    pairs = list(full_obs.get("candidate_pairs", []))
    if len(pairs) == 0:
        return []

    pair_feat = np.asarray(full_obs.get("pair_feat", []), dtype=np.float32)
    edge_rule = np.asarray(full_obs.get("edge_rule_to_pair_feat", []), dtype=np.float32)

    def _ect(idx: int) -> float:
        if pair_feat.ndim == 2 and 0 <= int(idx) < pair_feat.shape[0] and pair_feat.shape[1] > 4:
            return float(pair_feat[int(idx), 4])
        return float("inf")

    def _proc(idx: int) -> float:
        if pair_feat.ndim == 2 and 0 <= int(idx) < pair_feat.shape[0] and pair_feat.shape[1] > 0:
            return float(pair_feat[int(idx), 0])
        return float("inf")

    def _rule_support(idx: int) -> float:
        if edge_rule.ndim == 2 and 0 <= int(idx) < edge_rule.shape[0]:
            return float(np.sum(edge_rule[int(idx)] > 0.0))
        return 0.0

    keep_set = {
        int(i)
        for i in keep_pair_indices
        if 0 <= int(i) < len(pairs)
    }
    selected_unique = list(dict.fromkeys(int(x) for x in selected_ops))

    used_machines = {int(pairs[i][1]) for i in keep_set}
    per_op_target = int(max(1, per_op_min_machines))

    for op in selected_unique:
        idx_op = [i for i, (op_id, _mch) in enumerate(pairs) if int(op_id) == int(op)]
        if len(idx_op) == 0:
            continue

        feasible_machines = {int(pairs[i][1]) for i in idx_op}
        target = int(min(per_op_target, max(1, len(feasible_machines))))
        kept_op_machines = {int(pairs[i][1]) for i in keep_set if int(pairs[i][0]) == int(op)}

        while len(kept_op_machines) < target:
            cand = [i for i in idx_op if i not in keep_set]
            if len(cand) == 0:
                break
            cand.sort(
                key=lambda i: (
                    1 if int(pairs[i][1]) in kept_op_machines else 0,
                    1 if int(pairs[i][1]) in used_machines else 0,
                    _ect(i),
                    -_rule_support(i),
                    _proc(i),
                    int(pairs[i][1]),
                    int(pairs[i][0]),
                )
            )
            best = int(cand[0])
            keep_set.add(best)
            kept_op_machines.add(int(pairs[best][1]))
            used_machines.add(int(pairs[best][1]))

    target_total = int(max(1, min_pairs_total))
    if len(keep_set) < target_total:
        remaining = [i for i in range(len(pairs)) if i not in keep_set]
        remaining.sort(
            key=lambda i: (
                1 if int(pairs[i][1]) in used_machines else 0,
                _ect(i),
                -_rule_support(i),
                _proc(i),
                int(pairs[i][1]),
                int(pairs[i][0]),
            )
        )
        for i in remaining:
            keep_set.add(int(i))
            used_machines.add(int(pairs[int(i)][1]))
            if len(keep_set) >= target_total:
                break

    return sorted(int(i) for i in keep_set)


def _safe_masked_mean(feat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    feat = np.asarray(feat, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if feat.ndim != 2:
        return np.zeros((0,), dtype=np.float32)
    if feat.shape[0] != mask.shape[0] or mask.sum() == 0:
        return np.zeros((feat.shape[1],), dtype=np.float32)
    return feat[mask].mean(axis=0).astype(np.float32)


def _safe_masked_scalar_stats(feat: np.ndarray, mask: np.ndarray, col: int) -> np.ndarray:
    feat = np.asarray(feat, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if feat.ndim != 2 or feat.shape[0] != mask.shape[0] or feat.shape[1] <= int(col):
        return np.zeros((7,), dtype=np.float32)
    vals = feat[mask, int(col)]
    if vals.size == 0:
        return np.zeros((7,), dtype=np.float32)
    q25, q50, q75 = np.percentile(vals, [25, 50, 75]).astype(np.float32)
    return np.asarray(
        [
            float(np.mean(vals)),
            float(np.std(vals)),
            float(np.min(vals)),
            float(np.max(vals)),
            float(q25),
            float(q50),
            float(q75),
        ],
        dtype=np.float32,
    )


def build_central_state(
    packed_c: Dict[str, object],
    packed_o: Dict[str, object],
    packed_m: Dict[str, object],
    selected_ops: List[int],
    use_rich_stats: bool = False,
) -> np.ndarray:
    c_global = np.asarray(packed_c["global_feat"], dtype=np.float32)
    c_pair_mean = _safe_masked_mean(
        np.asarray(packed_c["pair_feat"], dtype=np.float32),
        np.asarray(packed_c["pair_mask"], dtype=bool),
    )
    o_global = np.asarray(packed_o["global_feat"], dtype=np.float32)
    o_op_mean = _safe_masked_mean(
        np.asarray(packed_o["op_feat"], dtype=np.float32),
        np.asarray(packed_o["op_mask"], dtype=bool),
    )
    m_global = np.asarray(packed_m["global_feat"], dtype=np.float32)
    m_pair_mean = _safe_masked_mean(
        np.asarray(packed_m["pair_feat"], dtype=np.float32),
        np.asarray(packed_m["pair_mask"], dtype=bool),
    )

    op_valid = int(np.asarray(packed_o["op_mask"], dtype=bool).sum())
    selected_ratio = float(len(set(int(x) for x in selected_ops))) / float(max(op_valid, 1))
    stats = np.asarray(
        [
            selected_ratio,
            float(np.asarray(packed_c["pair_mask"], dtype=bool).sum()),
            float(op_valid),
            float(np.asarray(packed_c["pair_mask"], dtype=bool).sum()) / float(max(op_valid, 1)),
            float(np.asarray(packed_m["pair_mask"], dtype=bool).sum()),
        ],
        dtype=np.float32,
    )

    if not bool(use_rich_stats):
        return np.concatenate([c_global, c_pair_mean, o_global, o_op_mean, m_global, m_pair_mean, stats], axis=0).astype(np.float32)

    c_pair = np.asarray(packed_c["pair_feat"], dtype=np.float32)
    c_pair_mask = np.asarray(packed_c["pair_mask"], dtype=bool)
    o_op = np.asarray(packed_o["op_feat"], dtype=np.float32)
    o_op_mask = np.asarray(packed_o["op_mask"], dtype=bool)
    m_pair = np.asarray(packed_m["pair_feat"], dtype=np.float32)
    m_pair_mask = np.asarray(packed_m["pair_mask"], dtype=bool)

    pair_ect_stats = _safe_masked_scalar_stats(c_pair, c_pair_mask, col=4)
    pair_primary_stats = _safe_masked_scalar_stats(c_pair, c_pair_mask, col=0)
    m_pair_ect_stats = _safe_masked_scalar_stats(m_pair, m_pair_mask, col=4)
    op_primary_stats = _safe_masked_scalar_stats(o_op, o_op_mask, col=0)

    selected_unique = len(set(int(x) for x in selected_ops))
    rich_stats = np.asarray(
        [
            float(c_pair_mask.sum()) / float(max(1, c_pair.shape[0])),
            float(o_op_mask.sum()) / float(max(1, o_op.shape[0])),
            float(m_pair_mask.sum()) / float(max(1, m_pair.shape[0])),
            float(selected_unique),
            float(selected_unique) / float(max(op_valid, 1)),
        ],
        dtype=np.float32,
    )

    return np.concatenate(
        [
            c_global,
            c_pair_mean,
            o_global,
            o_op_mean,
            m_global,
            m_pair_mean,
            stats,
            pair_ect_stats,
            pair_primary_stats,
            m_pair_ect_stats,
            op_primary_stats,
            rich_stats,
        ],
        axis=0,
    ).astype(np.float32)


def _best_ect(obs: Dict[str, object]) -> float:
    pair_feat = np.asarray(obs.get("pair_feat", []), dtype=np.float32)
    if pair_feat.ndim == 2 and pair_feat.shape[0] > 0 and pair_feat.shape[1] > 4:
        return float(np.min(pair_feat[:, 4]))
    return 0.0


def _op_best_ect_map(obs: Dict[str, object]) -> Dict[int, float]:
    pairs = list(obs.get("candidate_pairs", []))
    pair_feat = np.asarray(obs.get("pair_feat", []), dtype=np.float32)
    best: Dict[int, float] = {}
    for i, (op_id, _) in enumerate(pairs):
        op = int(op_id)
        ect = float(pair_feat[i, 4]) if pair_feat.ndim == 2 and pair_feat.shape[1] > 4 and i < pair_feat.shape[0] else 0.0
        if op not in best or ect < best[op]:
            best[op] = ect
    return best


def _best_pair_by_ect(obs: Dict[str, object]) -> Tuple[int, int]:
    pairs = list(obs.get("candidate_pairs", []))
    if len(pairs) == 0:
        return -1, -1

    pair_feat = np.asarray(obs.get("pair_feat", []), dtype=np.float32)
    best_idx = 0
    best_key = (float("inf"), float("inf"))

    for i, (_op, _mch) in enumerate(pairs):
        if pair_feat.ndim == 2 and i < pair_feat.shape[0]:
            ect = float(pair_feat[i, 4]) if pair_feat.shape[1] > 4 else float("inf")
            pt = float(pair_feat[i, 0]) if pair_feat.shape[1] > 0 else 0.0
        else:
            ect = float("inf")
            pt = 0.0
        key = (ect, pt)
        if key < best_key:
            best_key = key
            best_idx = int(i)

    op_id, mch_id = pairs[int(best_idx)]
    return int(op_id), int(mch_id)


def _estimate_op_uncertainty_from_ect(obs: Dict[str, object]) -> float:
    best_map = _op_best_ect_map(obs)
    if len(best_map) <= 1:
        return 0.0
    scores = -np.asarray(list(best_map.values()), dtype=np.float32)
    scores = scores - float(np.max(scores))
    probs = np.exp(scores)
    denom = float(np.sum(probs))
    if denom <= 1e-8:
        return 0.0
    probs = probs / denom
    entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0))))
    return float(entropy)


def _compute_dynamic_topk(full_obs: Dict[str, object], c_entropy: float, cfg: OCMAPPOConfig) -> int:
    n_ops = len({int(op) for op, _ in full_obs.get("candidate_pairs", [])})
    if n_ops <= 0:
        return 1
    k = int(cfg.o_topk)
    scale_floor_divisor = int(getattr(cfg, "o_scale_topk_floor_divisor", 0))
    if scale_floor_divisor > 0:
        k = max(k, int(np.ceil(float(n_ops) / float(scale_floor_divisor))))
    if float(c_entropy) > 1.0:
        k += int(np.ceil(float(cfg.o_topk_entropy_gain) * (float(c_entropy) - 1.0)))
    k = max(int(cfg.o_topk_min), k)
    k = min(int(cfg.o_topk_max), k)
    return int(max(1, min(n_ops, k)))


def _expand_ops_for_safety(full_obs: Dict[str, object], selected_ops: List[int], min_keep: int) -> List[int]:
    selected = list(dict.fromkeys([int(x) for x in selected_ops]))
    min_keep = int(max(1, min_keep))
    if len(selected) >= min_keep:
        return selected

    best_map = _op_best_ect_map(full_obs)
    remaining = [op for op in best_map.keys() if op not in set(selected)]
    remaining.sort(key=lambda op: best_map[op])
    need = max(0, min_keep - len(selected))
    selected.extend(remaining[:need])
    return list(dict.fromkeys(selected))


def _maybe_uncertainty_fallback(
    full_obs: Dict[str, object],
    selected_ops: List[int],
    c_entropy_filtered: float,
    cfg: OCMAPPOConfig,
) -> List[int]:
    entropy = float(c_entropy_filtered)
    extra_ops = 0
    if entropy > float(cfg.o_entropy_fallback_threshold):
        extra_ops = max(extra_ops, int(cfg.o_entropy_fallback_extra_ops))
    if entropy < float(cfg.o_entropy_low_fallback_threshold):
        extra_ops = max(extra_ops, int(cfg.o_entropy_low_fallback_extra_ops))
    if extra_ops <= 0:
        return selected_ops
    min_keep = int(len(selected_ops) + max(1, extra_ops))
    return _expand_ops_for_safety(full_obs=full_obs, selected_ops=selected_ops, min_keep=min_keep)


def _compute_o_shape_reward_legacy(
    full_obs: Dict[str, object],
    filtered_obs: Dict[str, object],
    selected_ops: List[int],
    c_full_op: int,
    c_filtered_op: int,
    new_makespan: float,
    initial_lb_makespan: float,
    is_terminal: bool,
    cfg: OCMAPPOConfig,
) -> Dict[str, float]:
    selected_set = {int(x) for x in selected_ops}
    retain_bonus = float(cfg.reward_retain_pos) if int(c_full_op) in selected_set else float(cfg.reward_retain_neg)

    ect_full = _best_ect(full_obs)
    ect_filtered = _best_ect(filtered_obs)
    quality_gap = 0.0
    if abs(ect_full) > 1e-8:
        quality_gap = max(0.0, (ect_filtered - ect_full) / abs(ect_full))
    quality_penalty = float(cfg.reward_quality_coef) * float(np.clip(quality_gap, 0.0, abs(float(cfg.reward_quality_clip))))
    if bool(cfg.reward_quality_hard_smooth):
        width = max(1e-6, float(cfg.reward_quality_hard_smooth_width))
        x = (float(quality_gap) - float(cfg.reward_quality_hard_threshold)) / width
        hard_quality_penalty = float(cfg.reward_quality_hard_penalty) / (1.0 + np.exp(-x))
    else:
        hard_quality_penalty = (
            float(cfg.reward_quality_hard_penalty) if quality_gap > float(cfg.reward_quality_hard_threshold) else 0.0
        )

    if bool(cfg.reward_mismatch_only_if_not_retained):
        mismatch_penalty = (
            float(cfg.reward_mismatch_penalty)
            if (int(c_full_op) not in selected_set and int(c_full_op) != int(c_filtered_op))
            else 0.0
        )
    else:
        mismatch_penalty = float(cfg.reward_mismatch_penalty) if int(c_full_op) != int(c_filtered_op) else 0.0

    terminal_penalty = 0.0
    if bool(is_terminal):
        denom = max(abs(float(initial_lb_makespan)), 1e-6)
        terminal_gap_ratio = max(0.0, (float(new_makespan) - float(initial_lb_makespan)) / denom)
        terminal_penalty = float(cfg.reward_makespan_terminal_coef) * float(terminal_gap_ratio)

    n_ops_total = max(1, len({int(op) for op, _ in full_obs.get("candidate_pairs", [])}))
    selected_ratio = float(len(selected_set)) / float(n_ops_total)
    redundancy_penalty = float(cfg.o_redundancy_penalty_coef) * max(
        0.0,
        selected_ratio - float(cfg.o_redundancy_target_ratio),
    )

    best_map = _op_best_ect_map(full_obs)
    coverage_bonus = 0.0
    if len(best_map) > 0:
        best_op = int(min(best_map.keys(), key=lambda x: best_map[x]))
        if best_op in selected_set:
            coverage_bonus = float(cfg.o_coverage_bonus_coef)

    total = (
        retain_bonus
        + coverage_bonus
        - quality_penalty
        - hard_quality_penalty
        - mismatch_penalty
        - terminal_penalty
        - redundancy_penalty
    )
    return {
        "total": float(total),
        "retain_bonus": float(retain_bonus),
        "coverage_bonus": float(coverage_bonus),
        "quality_penalty": float(quality_penalty),
        "hard_quality_penalty": float(hard_quality_penalty),
        "quality_gap": float(quality_gap),
        "mismatch_penalty": float(mismatch_penalty),
        "terminal_penalty": float(terminal_penalty),
        "selected_ratio": float(selected_ratio),
        "redundancy_penalty": float(redundancy_penalty),
    }


def _compute_m_shape_reward_legacy(
    base_obs: Dict[str, object],
    filtered_obs: Dict[str, object],
    c_ref_pair: Tuple[int, int],
    c_final_pair: Tuple[int, int],
    new_makespan: float,
    initial_lb_makespan: float,
    is_terminal: bool,
    cfg: OCMAPPOConfig,
) -> Dict[str, float]:
    retained = (int(c_ref_pair[0]), int(c_ref_pair[1])) in filtered_obs.get("legal_pairs_set", set())
    retain_term = float(cfg.m_reward_retain_pos) if retained else float(cfg.m_reward_retain_neg)

    ect_base = _best_ect(base_obs)
    ect_filtered = _best_ect(filtered_obs)
    quality_gap = 0.0
    if abs(ect_base) > 1e-8:
        quality_gap = max(0.0, (ect_filtered - ect_base) / abs(ect_base))

    quality_pen = float(cfg.m_reward_quality_coef) * float(quality_gap)
    hard_quality_pen = float(cfg.m_reward_quality_hard_penalty) if quality_gap > float(cfg.m_reward_quality_hard_threshold) else 0.0

    base_n = max(1, len(list(base_obs.get("candidate_pairs", []))))
    keep_n = len(list(filtered_obs.get("candidate_pairs", [])))
    keep_ratio = float(keep_n) / float(base_n)
    overprune_pen = float(cfg.m_reward_overprune_coef) * max(0.0, float(cfg.m_reward_overprune_target_keep_ratio) - keep_ratio)

    mismatch_pen = (
        float(cfg.m_reward_mismatch_penalty)
        if (int(c_ref_pair[0]) != int(c_final_pair[0]) or int(c_ref_pair[1]) != int(c_final_pair[1]))
        else 0.0
    )

    terminal_pen = 0.0
    if bool(is_terminal):
        denom = max(abs(float(initial_lb_makespan)), 1e-6)
        gap = max(0.0, (float(new_makespan) - float(initial_lb_makespan)) / denom)
        terminal_pen = float(cfg.m_reward_terminal_gap_coef) * gap

    total = retain_term - quality_pen - hard_quality_pen - overprune_pen - mismatch_pen - terminal_pen
    return {
        "total": float(total),
        "retain_term": float(retain_term),
        "quality_gap": float(quality_gap),
        "quality_pen": float(quality_pen),
        "hard_quality_pen": float(hard_quality_pen),
        "keep_ratio": float(keep_ratio),
        "overprune_pen": float(overprune_pen),
        "mismatch_pen": float(mismatch_pen),
        "terminal_pen": float(terminal_pen),
    }


def _compute_o_shape_reward_role_aligned_v1(
    full_obs: Dict[str, object],
    filtered_obs: Dict[str, object],
    selected_ops: List[int],
) -> Dict[str, float]:
    selected_set = {int(x) for x in selected_ops}
    full_best_map = _op_best_ect_map(full_obs)
    n_ops = int(max(1, len(full_best_map)))

    q = int(np.clip(np.ceil(0.25 * float(n_ops)), 2, 4))
    q = int(max(1, min(q, n_ops)))

    frontier_sorted = sorted(full_best_map.items(), key=lambda kv: (float(kv[1]), int(kv[0])))
    frontier_ops = [int(op_id) for op_id, _ in frontier_sorted[:q]]
    frontier_hit = len(selected_set.intersection(frontier_ops))
    frontier_coverage = float(frontier_hit) / float(max(1, q))

    ect_full = _best_ect(full_obs)
    ect_filtered = _best_ect(filtered_obs)
    best_ect_gap_raw = max(0.0, (float(ect_filtered) - float(ect_full)) / max(abs(float(ect_full)), 1e-6))
    best_ect_gap = float(np.clip(best_ect_gap_raw / 0.10, 0.0, 1.0))

    selected_n = int(len(selected_set))
    min_keep = int(np.ceil(0.20 * float(n_ops)))
    max_keep = int(np.ceil(0.40 * float(n_ops)))
    max_keep = int(max(min_keep, max_keep))

    if selected_n < min_keep:
        size_violation = float(min_keep - selected_n) / float(max(1, min_keep))
    elif selected_n > max_keep:
        size_violation = float(selected_n - max_keep) / float(max(1, n_ops - max_keep))
    else:
        size_violation = 0.0
    size_violation = float(np.clip(size_violation, 0.0, 1.0))

    total = 0.50 * float(frontier_coverage) - 0.35 * float(best_ect_gap) - 0.15 * float(size_violation)
    return {
        "total": float(total),
        "frontier_q": float(q),
        "frontier_coverage": float(frontier_coverage),
        "best_ect_gap_raw": float(best_ect_gap_raw),
        "best_ect_gap": float(best_ect_gap),
        "size_violation": float(size_violation),
        "selected_ops": float(selected_n),
        "n_ops": float(n_ops),
    }


def _compute_o_shape_reward_role_aligned_v2(
    full_obs: Dict[str, object],
    filtered_obs: Dict[str, object],
    selected_ops: List[int],
) -> Dict[str, float]:
    selected_set = {int(x) for x in selected_ops}
    full_best_map = _op_best_ect_map(full_obs)
    n_ops = int(max(1, len(full_best_map)))

    ranked_ops = [int(op_id) for op_id, _ in sorted(full_best_map.items(), key=lambda kv: (float(kv[1]), int(kv[0])))]
    best_op = int(ranked_ops[0]) if len(ranked_ops) > 0 else -1
    top_q = int(np.clip(np.ceil(0.20 * float(n_ops)), 2, 3))
    top_q = int(max(1, min(top_q, n_ops)))
    frontier_ops = ranked_ops[:top_q]

    best_op_hit = 1.0 if best_op in selected_set else 0.0
    frontier_coverage = float(len(selected_set.intersection(frontier_ops))) / float(max(1, top_q))

    ect_full = _best_ect(full_obs)
    ect_filtered = _best_ect(filtered_obs)
    selected_best_gap_raw = max(0.0, (float(ect_filtered) - float(ect_full)) / max(abs(float(ect_full)), 1e-6))
    selected_best_gap = float(np.clip(selected_best_gap_raw / 0.08, 0.0, 1.0))

    selected_n = int(len(selected_set))
    min_keep = int(max(1, np.ceil(0.15 * float(n_ops))))
    max_keep = int(max(min_keep, np.ceil(0.35 * float(n_ops))))
    if selected_n < min_keep:
        size_violation = float(min_keep - selected_n) / float(max(1, min_keep))
    elif selected_n > max_keep:
        size_violation = float(selected_n - max_keep) / float(max(1, n_ops - max_keep))
    else:
        size_violation = 0.0
    size_violation = float(np.clip(size_violation, 0.0, 1.0))

    total = (
        0.60 * float(best_op_hit)
        + 0.25 * float(frontier_coverage)
        - 0.12 * float(selected_best_gap)
        - 0.03 * float(size_violation)
    )
    return {
        "total": float(total),
        "best_op_hit": float(best_op_hit),
        "frontier_q": float(top_q),
        "frontier_coverage": float(frontier_coverage),
        "selected_best_gap_raw": float(selected_best_gap_raw),
        "selected_best_gap": float(selected_best_gap),
        "size_violation": float(size_violation),
        "selected_ops": float(selected_n),
        "n_ops": float(n_ops),
    }


def _compute_m_shape_reward_role_aligned_v1(
    base_obs: Dict[str, object],
    filtered_obs: Dict[str, object],
    selected_ops: List[int],
    cfg: OCMAPPOConfig,
) -> Dict[str, float]:
    selected_unique = list(dict.fromkeys(int(x) for x in selected_ops))
    denom_ops = int(max(1, len(selected_unique)))

    base_best = _op_best_ect_map(base_obs)
    filtered_best = _op_best_ect_map(filtered_obs)

    gap_vals: List[float] = []
    missing_ops = 0
    for op_id in selected_unique:
        if int(op_id) not in filtered_best:
            missing_ops += 1
            continue

        best_ect_op = float(base_best.get(int(op_id), filtered_best[int(op_id)]))
        kept_ect_op = float(filtered_best[int(op_id)])
        gap_raw = max(0.0, (kept_ect_op - best_ect_op) / max(abs(best_ect_op), 1e-6))
        gap_vals.append(float(np.clip(gap_raw / 0.10, 0.0, 1.0)))

    mean_machine_gap = float(np.mean(gap_vals)) if len(gap_vals) > 0 else 0.0
    missing_op_ratio = float(missing_ops) / float(max(1, denom_ops))
    base_pairs_n = int(max(1, len(list(base_obs.get("candidate_pairs", [])))))
    keep_pairs_n = int(len(list(filtered_obs.get("candidate_pairs", []))))
    keep_ratio = float(keep_pairs_n) / float(base_pairs_n)

    quality_pen = float(cfg.m_reward_quality_coef) * float(mean_machine_gap)
    hard_quality_pen = (
        float(cfg.m_reward_quality_hard_penalty)
        if float(mean_machine_gap) > float(cfg.m_reward_quality_hard_threshold)
        else 0.0
    )
    overprune_pen = float(cfg.m_reward_overprune_coef) * max(
        0.0,
        float(cfg.m_reward_overprune_target_keep_ratio) - float(keep_ratio),
    )

    # Keep a small positive anchor so M-branch rewards are not purely punitive.
    coverage_bonus = 0.25 * max(0.0, 1.0 - float(missing_op_ratio))
    total = float(coverage_bonus) - float(quality_pen) - float(hard_quality_pen) - float(overprune_pen)

    return {
        "total": float(total),
        "coverage_bonus": float(coverage_bonus),
        "quality_pen": float(quality_pen),
        "hard_quality_pen": float(hard_quality_pen),
        "keep_ratio": float(keep_ratio),
        "overprune_pen": float(overprune_pen),
        "mean_machine_gap": float(mean_machine_gap),
        "missing_op_ratio": float(missing_op_ratio),
        "missing_ops": float(missing_ops),
        "selected_ops": float(denom_ops),
    }


def _compute_gae_triple(episodes: List[EpisodeRollout], gamma: float, gae_lambda: float):
    advantages_c: List[float] = []
    returns_c: List[float] = []
    advantages_o: List[float] = []
    returns_o: List[float] = []
    advantages_m: List[float] = []
    returns_m: List[float] = []
    flat: List[Transition] = []

    for ep in episodes:
        if len(ep.transitions) == 0:
            continue

        rewards_c = np.asarray([t.reward_c for t in ep.transitions], dtype=np.float32)
        rewards_o = np.asarray([t.reward_o for t in ep.transitions], dtype=np.float32)
        rewards_m = np.asarray([t.reward_m for t in ep.transitions], dtype=np.float32)
        values_c = np.asarray([t.value_c for t in ep.transitions], dtype=np.float32)
        values_o = np.asarray([t.value_o for t in ep.transitions], dtype=np.float32)
        values_m = np.asarray([t.value_m for t in ep.transitions], dtype=np.float32)
        dones = np.asarray([1.0 if t.done else 0.0 for t in ep.transitions], dtype=np.float32)

        adv_c = np.zeros_like(rewards_c, dtype=np.float32)
        adv_o = np.zeros_like(rewards_o, dtype=np.float32)
        adv_m = np.zeros_like(rewards_m, dtype=np.float32)
        last_adv_c = 0.0
        last_adv_o = 0.0
        last_adv_m = 0.0
        # Bootstrap from the last in-episode value when rollout ends by step budget (non-terminal).
        next_value_c = 0.0 if bool(ep.done) else float(values_c[-1])
        next_value_o = 0.0 if bool(ep.done) else float(values_o[-1])
        next_value_m = 0.0 if bool(ep.done) else float(values_m[-1])

        for t in reversed(range(len(ep.transitions))):
            nonterminal = 1.0 - dones[t]
            delta_c = rewards_c[t] + gamma * next_value_c * nonterminal - values_c[t]
            delta_o = rewards_o[t] + gamma * next_value_o * nonterminal - values_o[t]
            delta_m = rewards_m[t] + gamma * next_value_m * nonterminal - values_m[t]
            last_adv_c = delta_c + gamma * gae_lambda * nonterminal * last_adv_c
            last_adv_o = delta_o + gamma * gae_lambda * nonterminal * last_adv_o
            last_adv_m = delta_m + gamma * gae_lambda * nonterminal * last_adv_m
            adv_c[t] = last_adv_c
            adv_o[t] = last_adv_o
            adv_m[t] = last_adv_m
            next_value_c = values_c[t]
            next_value_o = values_o[t]
            next_value_m = values_m[t]

        ret_c = adv_c + values_c
        ret_o = adv_o + values_o
        ret_m = adv_m + values_m

        flat.extend(ep.transitions)
        advantages_c.extend(adv_c.tolist())
        returns_c.extend(ret_c.tolist())
        advantages_o.extend(adv_o.tolist())
        returns_o.extend(ret_o.tolist())
        advantages_m.extend(adv_m.tolist())
        returns_m.extend(ret_m.tolist())

    return (
        flat,
        np.asarray(advantages_c, dtype=np.float32),
        np.asarray(returns_c, dtype=np.float32),
        np.asarray(advantages_o, dtype=np.float32),
        np.asarray(returns_o, dtype=np.float32),
        np.asarray(advantages_m, dtype=np.float32),
        np.asarray(returns_m, dtype=np.float32),
    )


def _new_timing_meter() -> Dict[str, float]:
    return {
        "episodes": 0.0,
        "step_count": 0.0,
        "episode_total_sec": 0.0,
        "env_obs_sec": 0.0,
        "select_c_full_sec": 0.0,
        "select_o_sec": 0.0,
        "select_c_filtered_sec": 0.0,
        "select_m_sec": 0.0,
        "critic_forward_sec": 0.0,
        "env_step_sec": 0.0,
        "reward_shape_sec": 0.0,
        "scan_c_sec": 0.0,
        "scan_o_sec": 0.0,
        "rollout_total_sec": 0.0,
        "update_total_sec": 0.0,
        "eval_total_sec": 0.0,
        "checkpoint_save_sec": 0.0,
        "final_eval_sec": 0.0,
    }


def _add_timing(meter: Optional[Dict[str, float]], key: str, value: float) -> None:
    if meter is None:
        return
    meter[key] = float(meter.get(key, 0.0) + float(value))


def _new_rule_meter() -> Dict[str, object]:
    return {
        "steps": 0,
        "selected_steps": 0,
        "rules": {},
    }


def _round_up_power_of_two(x: int) -> int:
    x = int(max(1, x))
    return 1 << (x - 1).bit_length()


def _apply_monotonic_decay(scale: float, decay: float, min_scale: float) -> float:
    scale = float(max(0.0, scale))
    decay = float(np.clip(float(decay), 0.0, 1.0))
    min_scale = float(np.clip(float(min_scale), 0.0, 1.0))
    decayed = max(min_scale, scale * decay)
    # Keep runtime interventions monotonic: never let a later safeguard
    # raise a scale that an earlier safeguard already pushed lower.
    return float(min(scale, decayed))


def _mean_log_value(
    logs: List[Dict[str, float]],
    key: str,
    active_key: Optional[str] = None,
    default: float = 0.0,
) -> float:
    if len(logs) == 0:
        return float(default)
    if active_key is None:
        vals = [float(x.get(key, default)) for x in logs]
    else:
        vals = [float(x.get(key, default)) for x in logs if float(x.get(active_key, 0.0)) > 0.0]
    if len(vals) == 0:
        return float(default)
    return float(np.mean(vals))


def _resolve_auto_c_max_candidates(
    observed_max_candidates: int,
    fixed_margin: int,
    margin_ratio: float,
    min_candidates: int,
    round_to_power_of_two: bool,
) -> Tuple[int, Dict[str, int]]:
    observed = int(max(1, observed_max_candidates))
    fixed_margin = int(max(0, fixed_margin))
    ratio_margin = int(max(0, math.ceil(float(observed) * max(0.0, float(margin_ratio)))))
    effective_margin = int(max(fixed_margin, ratio_margin))
    recommended = int(observed + effective_margin)
    recommended = int(max(recommended, int(max(1, min_candidates))))
    if bool(round_to_power_of_two):
        recommended = int(_round_up_power_of_two(recommended))

    detail = {
        "observed": int(observed),
        "fixed_margin": int(fixed_margin),
        "ratio_margin": int(ratio_margin),
        "effective_margin": int(effective_margin),
        "min_candidates": int(max(1, min_candidates)),
    }
    return int(recommended), detail


def _rule_row(meter: Dict[str, object], rule_name: str) -> Dict[str, float]:
    rules = meter["rules"]
    if rule_name not in rules:
        rules[rule_name] = {
            "candidate_step_count": 0,
            "candidate_pair_count": 0,
            "selected_count": 0,
            "selected_weighted_count": 0.0,
            "selected_full_match_count": 0,
            "selected_reward_c_sum": 0.0,
            "selected_reward_o_sum": 0.0,
            "selected_terminal_count": 0,
        }
    return rules[rule_name]


def _normalize_source_set(src: List[object]) -> List[str]:
    names = [str(x).strip() for x in src if str(x).strip()]
    if len(names) == 0:
        return ["UNSPECIFIED"]
    return sorted(set(names))


def _source_rule_support_count(src: object) -> int:
    if isinstance(src, (list, tuple, set)):
        names = _normalize_source_set(list(src))
    elif isinstance(src, str):
        names = _normalize_source_set([src])
    else:
        names = _normalize_source_set([])
    return int(sum(1 for x in names if str(x) != "UNSPECIFIED"))


def _relative_consensus_bonus(selected_mean: float, all_mean: float, all_max: float) -> float:
    denom = max(1e-6, float(all_max) - float(all_mean))
    score = (float(selected_mean) - float(all_mean)) / denom
    return float(np.clip(max(0.0, score), 0.0, 1.0))


def _compute_o_rule_consensus_bonus(
    full_obs: Dict[str, object],
    selected_ops: List[int],
) -> Dict[str, float]:
    pairs = list(full_obs.get("candidate_pairs", []))
    if len(pairs) == 0:
        return {
            "total": 0.0,
            "selected_mean": 0.0,
            "all_mean": 0.0,
            "all_max": 0.0,
            "selected_ops": float(len(set(int(x) for x in selected_ops))),
            "scored_ops": 0.0,
        }

    src = _resolve_pair_sources(full_obs)
    op_to_rules: Dict[int, set] = {}
    for idx, (op_id, _) in enumerate(pairs):
        op_key = int(op_id)
        if op_key not in op_to_rules:
            op_to_rules[op_key] = set()
        src_i = src[idx] if idx < len(src) else []
        src_names = _normalize_source_set(list(src_i) if isinstance(src_i, (list, tuple, set)) else [])
        for name in src_names:
            if str(name) == "UNSPECIFIED":
                continue
            op_to_rules[op_key].add(str(name))

    if len(op_to_rules) == 0:
        return {
            "total": 0.0,
            "selected_mean": 0.0,
            "all_mean": 0.0,
            "all_max": 0.0,
            "selected_ops": float(len(set(int(x) for x in selected_ops))),
            "scored_ops": 0.0,
        }

    op_support = {int(k): float(len(v)) for k, v in op_to_rules.items()}
    all_vals = np.asarray(list(op_support.values()), dtype=np.float32)
    all_mean = float(np.mean(all_vals)) if all_vals.size > 0 else 0.0
    all_max = float(np.max(all_vals)) if all_vals.size > 0 else 0.0

    selected_set = set(int(x) for x in selected_ops)
    if len(selected_set) == 0:
        selected_mean = 0.0
    else:
        selected_vals = np.asarray([float(op_support.get(int(op), 0.0)) for op in selected_set], dtype=np.float32)
        selected_mean = float(np.mean(selected_vals)) if selected_vals.size > 0 else 0.0

    total = _relative_consensus_bonus(selected_mean=selected_mean, all_mean=all_mean, all_max=all_max)
    return {
        "total": float(total),
        "selected_mean": float(selected_mean),
        "all_mean": float(all_mean),
        "all_max": float(all_max),
        "selected_ops": float(len(selected_set)),
        "scored_ops": float(len(op_support)),
    }


def _compute_m_rule_consensus_bonus(
    full_obs: Dict[str, object],
    kept_pair_indices: List[int],
) -> Dict[str, float]:
    pairs = list(full_obs.get("candidate_pairs", []))
    if len(pairs) == 0:
        return {
            "total": 0.0,
            "selected_mean": 0.0,
            "all_mean": 0.0,
            "all_max": 0.0,
            "selected_pairs": 0.0,
            "scored_pairs": 0.0,
        }

    src = _resolve_pair_sources(full_obs)
    pair_support = np.asarray(
        [
            float(_source_rule_support_count(src[idx] if idx < len(src) else []))
            for idx in range(len(pairs))
        ],
        dtype=np.float32,
    )

    all_mean = float(np.mean(pair_support)) if pair_support.size > 0 else 0.0
    all_max = float(np.max(pair_support)) if pair_support.size > 0 else 0.0

    keep_idx = sorted(
        {
            int(i)
            for i in kept_pair_indices
            if 0 <= int(i) < int(pair_support.shape[0])
        }
    )
    if len(keep_idx) == 0:
        selected_mean = 0.0
    else:
        selected_mean = float(np.mean(pair_support[np.asarray(keep_idx, dtype=np.int64)]))

    total = _relative_consensus_bonus(selected_mean=selected_mean, all_mean=all_mean, all_max=all_max)
    return {
        "total": float(total),
        "selected_mean": float(selected_mean),
        "all_mean": float(all_mean),
        "all_max": float(all_max),
        "selected_pairs": float(len(keep_idx)),
        "scored_pairs": float(pair_support.shape[0]),
    }


def _update_rule_meter_step(
    meter: Optional[Dict[str, object]],
    full_obs: Dict[str, object],
    filtered_obs: Dict[str, object],
    c_full_info: Dict[str, object],
    c_info: Dict[str, object],
    reward_c: float,
    reward_o: float,
    done: bool,
) -> None:
    if meter is None:
        return

    meter["steps"] = int(meter.get("steps", 0)) + 1

    full_sources = list(full_obs.get("pair_sources", []))
    appeared_this_step = set()
    for src in full_sources:
        src_rules = _normalize_source_set(list(src) if isinstance(src, (list, tuple, set)) else [])
        for r in src_rules:
            row = _rule_row(meter, r)
            row["candidate_pair_count"] += 1
            appeared_this_step.add(r)
    for r in appeared_this_step:
        row = _rule_row(meter, r)
        row["candidate_step_count"] += 1

    filtered_sources = list(filtered_obs.get("pair_sources", []))
    action_idx = int(c_info.get("action_idx", -1))
    selected_src = []
    if 0 <= action_idx < len(filtered_sources):
        selected_src = list(filtered_sources[action_idx])
    selected_rules = _normalize_source_set(selected_src)

    meter["selected_steps"] = int(meter.get("selected_steps", 0)) + 1
    weight = 1.0 / float(max(1, len(selected_rules)))
    full_match = int(
        int(c_full_info.get("op_id", -1)) == int(c_info.get("op_id", -2))
        and int(c_full_info.get("mch_id", -1)) == int(c_info.get("mch_id", -2))
    )
    for r in selected_rules:
        row = _rule_row(meter, r)
        row["selected_count"] += 1
        row["selected_weighted_count"] += float(weight)
        row["selected_full_match_count"] += int(full_match)
        row["selected_reward_c_sum"] += float(reward_c)
        row["selected_reward_o_sum"] += float(reward_o)
        row["selected_terminal_count"] += int(bool(done))


def _summarize_rule_meter(meter: Optional[Dict[str, object]], topk: int = 12) -> Dict[str, object]:
    if meter is None:
        return {}

    steps = int(meter.get("steps", 0))
    selected_steps = int(meter.get("selected_steps", 0))
    denom_steps = float(max(1, steps))
    denom_selected_steps = float(max(1, selected_steps))

    rows = []
    rules = meter.get("rules", {})
    for rule_name, raw in rules.items():
        candidate_step_count = int(raw.get("candidate_step_count", 0))
        candidate_pair_count = int(raw.get("candidate_pair_count", 0))
        selected_count = int(raw.get("selected_count", 0))
        selected_weighted_count = float(raw.get("selected_weighted_count", 0.0))
        selected_full_match_count = int(raw.get("selected_full_match_count", 0))
        selected_reward_c_sum = float(raw.get("selected_reward_c_sum", 0.0))
        selected_reward_o_sum = float(raw.get("selected_reward_o_sum", 0.0))
        selected_terminal_count = int(raw.get("selected_terminal_count", 0))

        rows.append(
            {
                "rule": str(rule_name),
                "candidate_step_count": candidate_step_count,
                "candidate_pair_count": candidate_pair_count,
                "selected_count": selected_count,
                "selected_weighted_count": selected_weighted_count,
                "candidate_step_ratio": float(candidate_step_count / denom_steps),
                "selected_step_ratio": float(selected_count / denom_selected_steps),
                "selection_given_candidate_step": float(selected_count / max(1, candidate_step_count)),
                "full_match_ratio_when_selected": float(selected_full_match_count / max(1, selected_count)),
                "mean_reward_c_when_selected": float(selected_reward_c_sum / max(1, selected_count)),
                "mean_reward_o_when_selected": float(selected_reward_o_sum / max(1, selected_count)),
                "terminal_ratio_when_selected": float(selected_terminal_count / max(1, selected_count)),
            }
        )

    rows.sort(key=lambda x: (-x["selected_count"], -x["candidate_step_count"], x["rule"]))
    likely_redundant_rules = [
        x["rule"]
        for x in rows
        if int(x["candidate_step_count"]) >= max(20, int(0.05 * max(1, steps))) and int(x["selected_count"]) == 0
    ]

    return {
        "steps": steps,
        "selected_steps": selected_steps,
        "num_rules_observed": int(len(rows)),
        "likely_redundant_rules": likely_redundant_rules,
        "top_rules_by_selected_count": rows[: int(max(1, topk))],
        "all_rules": rows,
    }


def _summarize_timing_profile(
    meter: Optional[Dict[str, float]],
    history: List[Dict[str, object]],
    total_wall_sec: float,
) -> Dict[str, object]:
    if meter is None:
        return {}

    episodes = float(max(1.0, meter.get("episodes", 0.0)))
    steps = float(max(1.0, meter.get("step_count", 0.0)))

    epoch_rows = [r for r in history if "epoch_rollout_sec" in r]
    rollout_epoch_avg = float(np.mean([float(r.get("epoch_rollout_sec", 0.0)) for r in epoch_rows])) if epoch_rows else 0.0
    update_epoch_avg = float(np.mean([float(r.get("epoch_update_sec", 0.0)) for r in epoch_rows])) if epoch_rows else 0.0
    eval_epoch_avg = float(np.mean([float(r.get("epoch_eval_sec", 0.0)) for r in epoch_rows])) if epoch_rows else 0.0

    return {
        "total_wall_sec": float(total_wall_sec),
        "scan_c_sec": float(meter.get("scan_c_sec", 0.0)),
        "scan_o_sec": float(meter.get("scan_o_sec", 0.0)),
        "rollout_total_sec": float(meter.get("rollout_total_sec", 0.0)),
        "update_total_sec": float(meter.get("update_total_sec", 0.0)),
        "eval_total_sec": float(meter.get("eval_total_sec", 0.0)),
        "checkpoint_save_sec": float(meter.get("checkpoint_save_sec", 0.0)),
        "final_eval_sec": float(meter.get("final_eval_sec", 0.0)),
        "episodes": int(meter.get("episodes", 0.0)),
        "steps": int(meter.get("step_count", 0.0)),
        "avg_episode_sec": float(meter.get("episode_total_sec", 0.0) / episodes),
        "avg_step_sec": float(meter.get("episode_total_sec", 0.0) / steps),
        "avg_env_obs_ms": float(1000.0 * meter.get("env_obs_sec", 0.0) / steps),
        "avg_select_c_full_ms": float(1000.0 * meter.get("select_c_full_sec", 0.0) / steps),
        "avg_select_o_ms": float(1000.0 * meter.get("select_o_sec", 0.0) / steps),
        "avg_select_c_filtered_ms": float(1000.0 * meter.get("select_c_filtered_sec", 0.0) / steps),
        "avg_select_m_ms": float(1000.0 * meter.get("select_m_sec", 0.0) / steps),
        "avg_critic_forward_ms": float(1000.0 * meter.get("critic_forward_sec", 0.0) / steps),
        "avg_env_step_ms": float(1000.0 * meter.get("env_step_sec", 0.0) / steps),
        "avg_reward_shape_ms": float(1000.0 * meter.get("reward_shape_sec", 0.0) / steps),
        "avg_epoch_rollout_sec": rollout_epoch_avg,
        "avg_epoch_update_sec": update_epoch_avg,
        "avg_epoch_eval_sec": eval_epoch_avg,
    }


def _resolve_o_reward_weights(cfg: OCMAPPOConfig, reward_progress: float) -> Tuple[float, float]:
    p = float(np.clip(reward_progress, 0.0, 1.0))
    schedule = str(getattr(cfg, "o_reward_schedule", "linear")).strip().lower()
    if schedule == "piecewise":
        mid_p = float(np.clip(float(cfg.o_reward_mid_progress), 1e-3, 1.0 - 1e-3))
        mid_ratio = float(np.clip(float(cfg.o_reward_mid_ratio), 0.0, 1.0))

        alpha_start = float(cfg.o_reward_alpha_env)
        alpha_end = float(cfg.o_reward_alpha_env_end)
        alpha_mid = alpha_start + (alpha_end - alpha_start) * mid_ratio

        beta_start = float(cfg.o_reward_beta_shape)
        beta_end = float(cfg.o_reward_beta_shape_end)
        beta_mid = beta_start + (beta_end - beta_start) * mid_ratio

        if p <= mid_p:
            t = p / mid_p
            alpha = alpha_start + (alpha_mid - alpha_start) * t
            beta = beta_start + (beta_mid - beta_start) * t
        else:
            t = (p - mid_p) / (1.0 - mid_p)
            alpha = alpha_mid + (alpha_end - alpha_mid) * t
            beta = beta_mid + (beta_end - beta_mid) * t
    else:
        alpha = float(cfg.o_reward_alpha_env) + (float(cfg.o_reward_alpha_env_end) - float(cfg.o_reward_alpha_env)) * p
        beta = float(cfg.o_reward_beta_shape) + (float(cfg.o_reward_beta_shape_end) - float(cfg.o_reward_beta_shape)) * p
    return float(alpha), float(beta)


def _resolve_reward_progress(cfg: OCMAPPOConfig, epoch: int) -> float:
    version = str(getattr(cfg, "reward_version", "role_aligned_v1")).strip().lower()
    if version == "legacy":
        anneal_epochs = int(max(1, int(getattr(cfg, "o_reward_anneal_epochs", 1))))
    else:
        anneal_epochs = int(max(1, int(getattr(cfg, "reward_schedule_anneal_epochs", 100))))
    return float(min(1.0, max(0.0, (int(epoch) - 1) / float(anneal_epochs))))


def _resolve_reward_coeffs(
    cfg: OCMAPPOConfig,
    reward_progress: float,
    c_max_candidates_effective: Optional[int] = None,
    o_max_ops_effective: Optional[int] = None,
    m_entropy_signal: Optional[float] = None,
    o_reward_term_runtime_scale: float = 1.0,
) -> Dict[str, float]:
    p = float(np.clip(reward_progress, 0.0, 1.0))
    version = str(getattr(cfg, "reward_version", "role_aligned_v1")).strip().lower()
    if version not in {"legacy", "role_aligned_v1", "role_aligned_v2"}:
        version = "role_aligned_v1"

    # Candidate-aware M scaling: narrower candidate space => weaker M-side shaping/teacher/consensus.
    m_candidate_scale = 1.0
    if bool(getattr(cfg, "m_candidate_aware_scaling_enabled", False)):
        c_low = float(max(1, int(getattr(cfg, "m_candidate_aware_c_low", 22))))
        c_high = float(max(int(c_low) + 1, int(getattr(cfg, "m_candidate_aware_c_high", 34))))
        o_low = float(max(1, int(getattr(cfg, "m_candidate_aware_o_low", 9))))
        o_high = float(max(int(o_low) + 1, int(getattr(cfg, "m_candidate_aware_o_high", 12))))

        c_eff = float(max(1, int(c_max_candidates_effective if c_max_candidates_effective is not None else c_low)))
        o_eff = float(max(1, int(o_max_ops_effective if o_max_ops_effective is not None else o_low)))

        c_ratio = float(np.clip((c_eff - c_low) / max(1e-6, (c_high - c_low)), 0.0, 1.0))
        o_ratio = float(np.clip((o_eff - o_low) / max(1e-6, (o_high - o_low)), 0.0, 1.0))
        # 更保守：由更窄的一侧决定缩放上限，避免某一侧过宽掩盖另一侧过窄。
        base = float(min(c_ratio, o_ratio))
        gamma = float(max(1e-6, float(getattr(cfg, "m_candidate_aware_gamma", 1.0))))
        min_scale = float(np.clip(float(getattr(cfg, "m_candidate_aware_min_scale", 0.6)), 0.0, 1.0))
        m_candidate_scale = float(min_scale + (1.0 - min_scale) * (base ** gamma))

    m_shape_entropy_scale = 1.0
    m_teacher_entropy_scale = 1.0
    m_consensus_entropy_scale = 1.0
    if bool(getattr(cfg, "m_entropy_feedback_enabled", False)) and m_entropy_signal is not None:
        ent_thr = float(max(1e-6, float(getattr(cfg, "m_entropy_feedback_threshold", 0.50))))
        ent_pow = float(max(1e-6, float(getattr(cfg, "m_entropy_feedback_power", 1.25))))
        ent_ratio = float(np.clip(float(m_entropy_signal) / ent_thr, 0.0, 1.0))
        ent_base = float(ent_ratio ** ent_pow)

        shape_min = float(np.clip(float(getattr(cfg, "m_entropy_feedback_shape_min_scale", 0.90)), 0.0, 1.0))
        teacher_min = float(np.clip(float(getattr(cfg, "m_entropy_feedback_teacher_min_scale", 0.20)), 0.0, 1.0))
        consensus_min = float(np.clip(float(getattr(cfg, "m_entropy_feedback_consensus_min_scale", 0.35)), 0.0, 1.0))

        m_shape_entropy_scale = float(shape_min + (1.0 - shape_min) * ent_base)
        m_teacher_entropy_scale = float(teacher_min + (1.0 - teacher_min) * ent_base)
        m_consensus_entropy_scale = float(consensus_min + (1.0 - consensus_min) * ent_base)

    if version == "legacy":
        o_alpha_env, o_beta_shape = _resolve_o_reward_weights(cfg=cfg, reward_progress=p)
        o_beta_shape = float(o_beta_shape * float(np.clip(float(o_reward_term_runtime_scale), 0.0, 1.0)))
        m_beta_shape = float(cfg.m_reward_beta_shape)
        if bool(getattr(cfg, "m_candidate_aware_apply_to_beta_shape", True)):
            m_beta_shape = float(m_beta_shape * m_candidate_scale)
        m_beta_shape = float(m_beta_shape * m_shape_entropy_scale)
        return {
            "version": version,
            "o_alpha_env": float(o_alpha_env),
            "o_beta_shape": float(o_beta_shape),
            "o_teacher_coef": 0.0,
            "o_consensus_coef": 0.0,
            "m_alpha_env": float(cfg.m_reward_alpha_env),
            "m_beta_shape": float(m_beta_shape),
            "m_teacher_coef": 0.0,
            "m_consensus_coef": 0.0,
            "m_candidate_scale": float(m_candidate_scale),
            "m_entropy_signal": (None if m_entropy_signal is None else float(m_entropy_signal)),
            "m_shape_entropy_scale": float(m_shape_entropy_scale),
            "m_teacher_entropy_scale": float(m_teacher_entropy_scale),
            "m_consensus_entropy_scale": float(m_consensus_entropy_scale),
        }

    o_alpha_env = float(cfg.o_reward_alpha_env) + (float(cfg.o_reward_alpha_env_end) - float(cfg.o_reward_alpha_env)) * p
    o_beta_shape = float(cfg.o_reward_beta_shape) + (float(cfg.o_reward_beta_shape_end) - float(cfg.o_reward_beta_shape)) * p
    o_teacher_coef = float(cfg.o_teacher_coef) + (float(cfg.o_teacher_coef_end) - float(cfg.o_teacher_coef)) * p
    o_consensus_coef = float(cfg.o_consensus_coef) + (float(cfg.o_consensus_coef_end) - float(cfg.o_consensus_coef)) * p
    m_alpha_env = float(cfg.m_reward_alpha_env) + (float(cfg.m_reward_alpha_env_end) - float(cfg.m_reward_alpha_env)) * p
    m_beta_shape = float(cfg.m_reward_beta_shape) + (float(cfg.m_reward_beta_shape_end) - float(cfg.m_reward_beta_shape)) * p
    m_teacher_coef = float(cfg.m_teacher_coef) + (float(cfg.m_teacher_coef_end) - float(cfg.m_teacher_coef)) * p
    m_consensus_coef = float(cfg.m_consensus_coef) + (float(cfg.m_consensus_coef_end) - float(cfg.m_consensus_coef)) * p

    o_teacher_min = float(max(0.0, float(getattr(cfg, "o_teacher_coef_min", 0.0))))
    o_consensus_min = float(max(0.0, float(getattr(cfg, "o_consensus_coef_min", 0.0))))
    m_teacher_min = float(max(0.0, float(getattr(cfg, "m_teacher_coef_min", 0.0))))
    m_consensus_min = float(max(0.0, float(getattr(cfg, "m_consensus_coef_min", 0.0))))

    o_teacher_coef = max(o_teacher_min, float(o_teacher_coef))
    o_consensus_coef = max(o_consensus_min, float(o_consensus_coef))
    m_teacher_coef = max(m_teacher_min, float(m_teacher_coef))
    m_consensus_coef = max(m_consensus_min, float(m_consensus_coef))

    if version == "role_aligned_v2":
        # For O reward v2, rely on a stronger shape term and remove the moving-target
        # teacher/consensus bonuses that can over-constrain O into mirroring C(full).
        o_teacher_coef = 0.0
        o_consensus_coef = 0.0

    o_runtime_scale = float(np.clip(float(o_reward_term_runtime_scale), 0.0, 1.0))
    o_beta_shape = float(o_beta_shape * o_runtime_scale)
    o_teacher_coef = float(o_teacher_coef * o_runtime_scale)
    o_consensus_coef = float(o_consensus_coef * o_runtime_scale)

    if bool(getattr(cfg, "m_candidate_aware_apply_to_beta_shape", True)):
        m_beta_shape = float(m_beta_shape * m_candidate_scale)
    if bool(getattr(cfg, "m_candidate_aware_apply_to_teacher", True)):
        m_teacher_coef = float(m_teacher_coef * m_candidate_scale)
    if bool(getattr(cfg, "m_candidate_aware_apply_to_consensus", True)):
        m_consensus_coef = float(m_consensus_coef * m_candidate_scale)

    m_beta_shape = float(m_beta_shape * m_shape_entropy_scale)
    m_teacher_coef = float(m_teacher_coef * m_teacher_entropy_scale)
    m_consensus_coef = float(m_consensus_coef * m_consensus_entropy_scale)

    return {
        "version": version,
        "o_alpha_env": float(o_alpha_env),
        "o_beta_shape": float(o_beta_shape),
        "o_teacher_coef": float(max(0.0, o_teacher_coef)),
        "o_consensus_coef": float(max(0.0, o_consensus_coef)),
        "m_alpha_env": float(m_alpha_env),
        "m_beta_shape": float(m_beta_shape),
        "m_teacher_coef": float(max(0.0, m_teacher_coef)),
        "m_consensus_coef": float(max(0.0, m_consensus_coef)),
        "m_candidate_scale": float(m_candidate_scale),
        "m_entropy_signal": (None if m_entropy_signal is None else float(m_entropy_signal)),
        "m_shape_entropy_scale": float(m_shape_entropy_scale),
        "m_teacher_entropy_scale": float(m_teacher_entropy_scale),
        "m_consensus_entropy_scale": float(m_consensus_entropy_scale),
    }


def _pointwise_value_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    use_huber: bool,
    huber_delta: float,
) -> torch.Tensor:
    if not bool(use_huber):
        return (pred - target).pow(2)
    delta = float(max(1e-6, huber_delta))
    diff = pred - target
    abs_diff = diff.abs()
    quadratic = torch.minimum(abs_diff, torch.full_like(abs_diff, delta))
    linear = abs_diff - quadratic
    return 0.5 * quadratic.pow(2) / delta + linear


def _masked_binary_log_prob_np(
    logits: np.ndarray,
    target_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> float:
    logits = np.asarray(logits, dtype=np.float32)
    target = np.asarray(target_mask, dtype=bool)
    valid = np.asarray(valid_mask, dtype=bool)
    if logits.shape != target.shape or logits.shape != valid.shape:
        raise ValueError("logits/target_mask/valid_mask must share the same shape")
    if not bool(np.any(valid)):
        return 0.0
    x = logits[valid].astype(np.float32)
    y = target[valid].astype(np.float32)
    log_pos = -np.logaddexp(0.0, -x)
    log_neg = -np.logaddexp(0.0, x)
    ll = y * log_pos + (1.0 - y) * log_neg
    return float(np.mean(ll, dtype=np.float32))


def _masked_binary_log_prob_t(
    logits: torch.Tensor,
    target_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    target = target_mask.to(dtype=logits.dtype)
    valid = valid_mask.to(dtype=logits.dtype)
    ll = target * F.logsigmoid(logits) + (1.0 - target) * F.logsigmoid(-logits)
    ll = ll * valid
    denom = valid.sum(dim=-1).clamp(min=1.0)
    return ll.sum(dim=-1) / denom


def _stack_batch(flat: List[Transition], device: torch.device):
    c_global = torch.tensor(np.stack([t.c_global for t in flat]), dtype=torch.float32, device=device)

    c_pair_arrs = [np.asarray(t.c_pair, dtype=np.float32) for t in flat]
    c_mask_arrs = [np.asarray(t.c_mask, dtype=bool).reshape(-1) for t in flat]
    c_pair_lens = [int(a.shape[0]) if a.ndim == 2 else 0 for a in c_pair_arrs]
    c_pair_max = max(1, max(c_pair_lens)) if len(c_pair_lens) > 0 else 1
    c_pair_dim = 1
    for a in c_pair_arrs:
        if a.ndim == 2 and a.shape[1] > 0:
            c_pair_dim = max(c_pair_dim, int(a.shape[1]))
    c_pair_np = np.zeros((len(flat), c_pair_max, c_pair_dim), dtype=np.float32)
    c_mask_np = np.zeros((len(flat), c_pair_max), dtype=bool)
    for i, a in enumerate(c_pair_arrs):
        if a.ndim != 2 or a.shape[0] == 0:
            continue
        rows = min(c_pair_max, int(a.shape[0]))
        cols = min(c_pair_dim, int(a.shape[1]))
        c_pair_np[i, :rows, :cols] = a[:rows, :cols]
        if c_mask_arrs[i].shape[0] > 0:
            valid_rows = min(rows, int(c_mask_arrs[i].shape[0]))
            c_mask_np[i, :valid_rows] = c_mask_arrs[i][:valid_rows]
    c_pair = torch.tensor(c_pair_np, dtype=torch.float32, device=device)
    c_mask = torch.tensor(c_mask_np, dtype=torch.bool, device=device)

    c_action = torch.tensor(np.asarray([t.c_action_idx for t in flat]), dtype=torch.long, device=device)
    c_old_logp = torch.tensor(np.asarray([t.c_old_log_prob for t in flat]), dtype=torch.float32, device=device)

    o_global = torch.tensor(np.stack([t.o_global for t in flat]), dtype=torch.float32, device=device)
    o_op = torch.tensor(np.stack([t.o_op for t in flat]), dtype=torch.float32, device=device)
    o_mask = torch.tensor(np.stack([t.o_mask for t in flat]), dtype=torch.bool, device=device)
    o_selected_mask = torch.tensor(np.stack([t.o_selected_mask for t in flat]), dtype=torch.bool, device=device)
    o_action = torch.tensor(np.asarray([t.o_action_idx for t in flat]), dtype=torch.long, device=device)
    o_old_logp = torch.tensor(np.asarray([t.o_old_log_prob for t in flat]), dtype=torch.float32, device=device)
    o_old_set_logp = torch.tensor(np.asarray([t.o_old_set_log_prob for t in flat]), dtype=torch.float32, device=device)

    m_global = torch.tensor(np.stack([t.m_global for t in flat]), dtype=torch.float32, device=device)
    m_old_logp = torch.tensor(np.asarray([t.m_old_log_prob for t in flat]), dtype=torch.float32, device=device)

    m_pair_arrs = [np.asarray(t.m_pair, dtype=np.float32) for t in flat]
    m_pair_lens = [int(a.shape[0]) if a.ndim == 2 else 0 for a in m_pair_arrs]
    m_pair_max = max(1, max(m_pair_lens)) if len(m_pair_lens) > 0 else 1
    m_pair_dim = 1
    for a in m_pair_arrs:
        if a.ndim == 2 and a.shape[1] > 0:
            m_pair_dim = max(m_pair_dim, int(a.shape[1]))
    m_pair_np = np.zeros((len(flat), m_pair_max, m_pair_dim), dtype=np.float32)
    m_mask_np = np.zeros((len(flat), m_pair_max), dtype=bool)
    for i, a in enumerate(m_pair_arrs):
        if a.ndim != 2 or a.shape[0] == 0:
            continue
        cols = min(m_pair_dim, int(a.shape[1]))
        rows = min(m_pair_max, int(a.shape[0]))
        m_pair_np[i, :rows, :cols] = a[:rows, :cols]
        m_mask_np[i, :rows] = True
    m_pair = torch.tensor(m_pair_np, dtype=torch.float32, device=device)
    m_mask = torch.tensor(m_mask_np, dtype=torch.bool, device=device)

    states = torch.tensor(np.stack([t.state_vec for t in flat]), dtype=torch.float32, device=device)
    old_value_c = torch.tensor(np.asarray([t.value_c for t in flat]), dtype=torch.float32, device=device)
    old_value_o = torch.tensor(np.asarray([t.value_o for t in flat]), dtype=torch.float32, device=device)
    old_value_m = torch.tensor(np.asarray([t.value_m for t in flat]), dtype=torch.float32, device=device)

    return {
        "c_global": c_global,
        "c_pair": c_pair,
        "c_mask": c_mask,
        "c_action": c_action,
        "c_old_logp": c_old_logp,
        "o_global": o_global,
        "o_op": o_op,
        "o_mask": o_mask,
        "o_selected_mask": o_selected_mask,
        "o_action": o_action,
        "o_old_logp": o_old_logp,
        "o_old_set_logp": o_old_set_logp,
        "m_global": m_global,
        "m_pair": m_pair,
        "m_mask": m_mask,
        "m_old_logp": m_old_logp,
        "states": states,
        "old_value_c": old_value_c,
        "old_value_o": old_value_o,
        "old_value_m": old_value_m,
    }


def update_mappo(
    agent_c: Agent_C,
    agent_o: Agent_O,
    agent_m: Agent_M_Learn,
    critic: CentralCritic,
    critic_optimizer: optim.Optimizer,
    episodes: List[EpisodeRollout],
    cfg: OCMAPPOConfig,
    update_step: int = 1,
    o_update_interval_override: Optional[int] = None,
    o_aux_coef_scale: float = 1.0,
    o_update_weight: float = 1.0,
) -> Dict[str, float]:
    flat, adv_c_np, ret_c_np, adv_o_np, ret_o_np, adv_m_np, ret_m_np = _compute_gae_triple(
        episodes,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
    )
    if len(flat) == 0:
        return {
            "loss_c": 0.0,
            "loss_o": 0.0,
            "value_loss": 0.0,
            "value_loss_c": 0.0,
            "value_loss_o": 0.0,
            "value_loss_m": 0.0,
            "entropy_c": 0.0,
            "entropy_o": 0.0,
            "entropy_m": 0.0,
            "approx_kl_c": 0.0,
            "approx_kl_o": 0.0,
            "approx_kl_m": 0.0,
            "kl_penalty_c": 0.0,
            "kl_penalty_o": 0.0,
            "kl_penalty_m": 0.0,
            "loss_o_aux": 0.0,
            "o_set_aux_coef_eff": float(getattr(cfg, "o_set_aux_coef", 0.0)) * float(np.clip(float(o_aux_coef_scale), 0.0, 1.0)),
            "o_ent_coef_eff": 0.0,
            "m_ent_coef_eff": float(getattr(cfg, "m_ent_coef", 0.0)),
            "loss_m": 0.0,
            "minibatch_kl_guard_c": 0.0,
            "minibatch_kl_guard_o": 0.0,
            "minibatch_kl_guard_m": 0.0,
            "o_batch_soft_protect": 0.0,
            "o_batch_hard_protect": 0.0,
            "o_batch_update_weight": 1.0,
            "o_metrics_valid": 0.0,
            "m_metrics_valid": 0.0,
        }

    device = agent_c.device
    batch = _stack_batch(flat, device=device)

    adv_c = torch.tensor(adv_c_np, dtype=torch.float32, device=device)
    adv_o = torch.tensor(adv_o_np, dtype=torch.float32, device=device)
    adv_m = torch.tensor(adv_m_np, dtype=torch.float32, device=device)
    adv_c = (adv_c - adv_c.mean()) / adv_c.std(unbiased=False).clamp(min=1e-6)
    adv_o = (adv_o - adv_o.mean()) / adv_o.std(unbiased=False).clamp(min=1e-6)
    adv_m = (adv_m - adv_m.mean()) / adv_m.std(unbiased=False).clamp(min=1e-6)

    ret_c = torch.tensor(ret_c_np, dtype=torch.float32, device=device)
    ret_o = torch.tensor(ret_o_np, dtype=torch.float32, device=device)
    ret_m = torch.tensor(ret_m_np, dtype=torch.float32, device=device)

    n = int(len(flat))
    mb = int(min(max(16, cfg.minibatch_size), n))
    if o_update_interval_override is None:
        o_update_interval = int(max(1, int(getattr(cfg, "o_update_interval", 1))))
    else:
        o_update_interval = int(max(1, int(o_update_interval_override)))
    m_update_interval = int(max(1, int(getattr(cfg, "m_update_interval", 1))))
    should_update_o = (int(update_step) % o_update_interval) == 0
    should_update_m_by_interval = (int(update_step) % m_update_interval) == 0
    o_update_weight_eff = float(np.clip(float(o_update_weight), 0.0, 1.0))
    minibatch_kl_guard_enabled = bool(getattr(cfg, "minibatch_kl_guard_enabled", True))
    target_kl_c = float(max(0.0, float(getattr(cfg, "target_kl_c", 0.0))))
    target_kl_o = float(max(0.0, float(getattr(cfg, "target_kl_o", 0.0))))
    target_kl_m = float(max(0.0, float(getattr(cfg, "target_kl_m", 0.0))))
    kl_guard_thr_c = target_kl_c * float(max(1.0, float(getattr(cfg, "minibatch_kl_guard_mult_c", 2.0))))
    kl_guard_thr_o = target_kl_o * float(max(1.0, float(getattr(cfg, "minibatch_kl_guard_mult_o", 1.5))))
    kl_guard_thr_m = target_kl_m * float(max(1.0, float(getattr(cfg, "minibatch_kl_guard_mult_m", 2.0))))

    # Update-step based M entropy annealing.
    m_ent_start = float(getattr(cfg, "m_ent_coef", 0.0))
    m_ent_end = float(getattr(cfg, "m_ent_coef_end", m_ent_start))
    m_ent_anneal_updates = int(max(1, int(getattr(cfg, "m_ent_coef_anneal_updates", 120))))
    m_ent_p = float(np.clip((int(update_step) - 1) / float(m_ent_anneal_updates), 0.0, 1.0))
    m_ent_coef_eff = float(m_ent_start + (m_ent_end - m_ent_start) * m_ent_p)

    logs = {
        "loss_c": [],
        "loss_o": [],
        "value_loss": [],
        "value_loss_c": [],
        "value_loss_o": [],
        "value_loss_m": [],
        "entropy_c": [],
        "entropy_o": [],
        "entropy_m": [],
        "approx_kl_c": [],
        "approx_kl_o": [],
        "approx_kl_m": [],
        "kl_penalty_c": [],
        "kl_penalty_o": [],
        "kl_penalty_m": [],
        "loss_o_aux": [],
        "o_set_aux_coef_eff": [],
        "o_ent_coef_eff": [],
        "m_ent_coef_eff": [],
        "loss_m": [],
        "minibatch_kl_guard_c": [],
        "minibatch_kl_guard_o": [],
        "minibatch_kl_guard_m": [],
        "o_batch_soft_protect": [],
        "o_batch_hard_protect": [],
        "o_batch_update_weight": [],
    }

    freeze_c_for_update = False
    freeze_o_for_update = False
    freeze_m_for_update = False

    for _ in range(int(cfg.ppo_epochs)):
        perm = torch.randperm(n, device=device)
        epoch_kls_c = []
        epoch_kls_o = []
        epoch_kls_m = []

        for start in range(0, n, mb):
            idx = perm[start : start + mb]

            c_global = batch["c_global"][idx]
            c_pair = batch["c_pair"][idx]
            c_mask = batch["c_mask"][idx]
            c_action = batch["c_action"][idx]
            c_old_logp = batch["c_old_logp"][idx]

            o_global = batch["o_global"][idx]
            o_op = batch["o_op"][idx]
            o_mask = batch["o_mask"][idx]
            o_selected_mask = batch["o_selected_mask"][idx]
            o_action = batch["o_action"][idx]
            o_old_logp = batch["o_old_logp"][idx]
            o_old_set_logp = batch["o_old_set_logp"][idx]

            m_pair = batch["m_pair"][idx]
            m_mask = batch["m_mask"][idx]
            m_old_logp = batch["m_old_logp"][idx]

            states = batch["states"][idx]
            mb_old_value_c = batch["old_value_c"][idx]
            mb_old_value_o = batch["old_value_o"][idx]
            mb_old_value_m = batch["old_value_m"][idx]
            mb_adv_c = adv_c[idx]
            mb_adv_o = adv_o[idx]
            mb_adv_m = adv_m[idx]
            mb_ret_c = ret_c[idx]
            mb_ret_o = ret_o[idx]
            mb_ret_m = ret_m[idx]

            values_c_new, values_o_new, values_m_new = critic(
                states,
                c_pair=c_pair,
                c_mask=c_mask,
                o_op=o_op,
                o_mask=o_mask,
                m_pair=m_pair,
                m_mask=m_mask,
            )
            values_c_new = values_c_new.squeeze(-1)
            values_o_new = values_o_new.squeeze(-1)
            values_m_new = values_m_new.squeeze(-1)
            value_loss_c_unclipped = _pointwise_value_loss(
                values_c_new,
                mb_ret_c,
                use_huber=cfg.use_huber_value_loss,
                huber_delta=cfg.value_huber_delta,
            )
            value_loss_o_unclipped = _pointwise_value_loss(
                values_o_new,
                mb_ret_o,
                use_huber=cfg.use_huber_value_loss,
                huber_delta=cfg.value_huber_delta,
            )
            value_loss_m_unclipped = _pointwise_value_loss(
                values_m_new,
                mb_ret_m,
                use_huber=cfg.use_huber_value_loss,
                huber_delta=cfg.value_huber_delta,
            )

            if float(cfg.value_clip_range) > 0:
                c_clip = mb_old_value_c + torch.clamp(values_c_new - mb_old_value_c, -float(cfg.value_clip_range), float(cfg.value_clip_range))
                o_clip = mb_old_value_o + torch.clamp(values_o_new - mb_old_value_o, -float(cfg.value_clip_range), float(cfg.value_clip_range))
                m_clip = mb_old_value_m + torch.clamp(values_m_new - mb_old_value_m, -float(cfg.value_clip_range), float(cfg.value_clip_range))
                value_loss_c_clipped = _pointwise_value_loss(
                    c_clip,
                    mb_ret_c,
                    use_huber=cfg.use_huber_value_loss,
                    huber_delta=cfg.value_huber_delta,
                )
                value_loss_o_clipped = _pointwise_value_loss(
                    o_clip,
                    mb_ret_o,
                    use_huber=cfg.use_huber_value_loss,
                    huber_delta=cfg.value_huber_delta,
                )
                value_loss_m_clipped = _pointwise_value_loss(
                    m_clip,
                    mb_ret_m,
                    use_huber=cfg.use_huber_value_loss,
                    huber_delta=cfg.value_huber_delta,
                )
                value_loss_c = torch.max(value_loss_c_unclipped, value_loss_c_clipped).mean()
                value_loss_o = torch.max(value_loss_o_unclipped, value_loss_o_clipped).mean()
                value_loss_m = torch.max(value_loss_m_unclipped, value_loss_m_clipped).mean()
            else:
                value_loss_c = value_loss_c_unclipped.mean()
                value_loss_o = value_loss_o_unclipped.mean()
                value_loss_m = value_loss_m_unclipped.mean()

            value_loss = (value_loss_c + value_loss_o + value_loss_m) / 3.0

            logits_c, _, _ = agent_c.policy.forward(c_global, c_pair, c_mask)
            dist_c = torch.distributions.Categorical(logits=logits_c)
            new_logp_c = dist_c.log_prob(c_action)
            entropy_c = dist_c.entropy().mean()
            ratio_c = torch.exp(new_logp_c - c_old_logp)
            ratio_c_clip = torch.clamp(ratio_c, 1.0 - float(cfg.clip_ratio_c), 1.0 + float(cfg.clip_ratio_c))
            kl_penalty_c_t = torch.abs(c_old_logp - new_logp_c).mean()
            loss_c = (
                -torch.min(ratio_c * mb_adv_c, ratio_c_clip * mb_adv_c).mean()
                - cfg.c_ent_coef * entropy_c
                + float(cfg.kl_penalty_coef_c) * kl_penalty_c_t
            )

            logits_o, _ = agent_o.policy.forward(o_global, o_op, o_mask)
            dist_o = torch.distributions.Categorical(logits=logits_o)
            new_logp_o_action = dist_o.log_prob(o_action)
            new_logp_o_set = _masked_binary_log_prob_t(logits_o, o_selected_mask, o_mask)
            o_set_ppo_mix = float(np.clip(float(getattr(cfg, "o_set_ppo_mix", 0.35)), 0.0, 1.0))
            new_logp_o = (1.0 - o_set_ppo_mix) * new_logp_o_action + o_set_ppo_mix * new_logp_o_set
            old_logp_o_mix = (1.0 - o_set_ppo_mix) * o_old_logp + o_set_ppo_mix * o_old_set_logp
            entropy_o = dist_o.entropy().mean()
            ratio_o = torch.exp(new_logp_o - old_logp_o_mix)
            ratio_o_clip = torch.clamp(ratio_o, 1.0 - float(cfg.clip_ratio_o), 1.0 + float(cfg.clip_ratio_o))
            loss_o_ppo = -torch.min(ratio_o * mb_adv_o, ratio_o_clip * mb_adv_o).mean()
            kl_penalty_o_t = torch.abs(old_logp_o_mix - new_logp_o).mean()

            o_valid = o_mask
            if bool(torch.any(o_valid)):
                o_logits_valid = logits_o[o_valid]
                o_target_valid = o_selected_mask[o_valid].to(dtype=torch.float32)
                pos_cnt = float(o_target_valid.sum().item())
                neg_cnt = float(o_target_valid.numel() - pos_cnt)
                if pos_cnt > 0.0 and neg_cnt > 0.0:
                    pos_weight = torch.tensor(neg_cnt / pos_cnt, dtype=torch.float32, device=device)
                    loss_o_aux = F.binary_cross_entropy_with_logits(o_logits_valid, o_target_valid, pos_weight=pos_weight)
                else:
                    loss_o_aux = F.binary_cross_entropy_with_logits(o_logits_valid, o_target_valid)
            else:
                loss_o_aux = torch.tensor(0.0, dtype=torch.float32, device=device)

            o_ent_coef_eff = float(cfg.o_ent_coef)
            if float(getattr(cfg, "o_ent_adaptive_gain", 0.0)) > 0.0:
                entropy_gap = max(0.0, float(getattr(cfg, "o_ent_adaptive_target", 0.0)) - float(entropy_o.item()))
                ent_scale = 1.0 + float(getattr(cfg, "o_ent_adaptive_gain", 0.0)) * entropy_gap
                ent_scale = min(ent_scale, float(getattr(cfg, "o_ent_adaptive_max_scale", 1.0)))
                o_ent_coef_eff = o_ent_coef_eff * ent_scale
            o_set_aux_coef_eff = float(getattr(cfg, "o_set_aux_coef", 0.0)) * float(np.clip(float(o_aux_coef_scale), 0.0, 1.0))

            loss_o = (
                loss_o_ppo
                - o_ent_coef_eff * entropy_o
                + float(o_set_aux_coef_eff) * loss_o_aux
                + float(cfg.kl_penalty_coef_o) * kl_penalty_o_t
            )

            # P1: stronger minibatch O protection using both KL and ratio out-of-range signals.
            o_soft_protect_hit = False
            o_hard_protect_hit = False
            if bool(should_update_o):
                tkl_o = float(max(1e-8, target_kl_o)) if target_kl_o > 0.0 else 1e-8
                soft_mult = float(max(1.0, float(getattr(cfg, "o_kl_batch_soft_limit_mult", 1.5))))
                hard_mult = float(max(soft_mult, float(getattr(cfg, "o_kl_batch_hard_limit_mult", 2.5))))
                soft_scale = float(np.clip(float(getattr(cfg, "o_kl_batch_soft_scale", 0.5)), 0.0, 1.0))

                # KL-based protection
                if tkl_o > 1e-8:
                    kl_ratio_o = float(kl_penalty_o_t.detach().item()) / tkl_o
                    if kl_ratio_o >= hard_mult:
                        o_update_weight_eff = 0.0
                        o_hard_protect_hit = True
                    elif kl_ratio_o >= soft_mult:
                        o_update_weight_eff = min(o_update_weight_eff, soft_scale)
                        o_soft_protect_hit = True

                # Ratio-based protection (fraction of samples clipped by PPO ratio window)
                if bool(getattr(cfg, "o_batch_protect_enabled", True)):
                    ratio_dev = torch.abs(ratio_o - 1.0)
                    clip_lim = float(max(1e-8, float(cfg.clip_ratio_o)))
                    out_ratio = float((ratio_dev > clip_lim).float().mean().item())
                    ratio_soft = float(np.clip(float(getattr(cfg, "o_ratio_batch_soft_limit", 0.40)), 0.0, 1.0))
                    ratio_hard = float(np.clip(float(getattr(cfg, "o_ratio_batch_hard_limit", 0.80)), ratio_soft, 1.0))
                    ratio_soft_scale = float(np.clip(float(getattr(cfg, "o_ratio_batch_soft_scale", 0.5)), 0.0, 1.0))
                    if out_ratio >= ratio_hard:
                        o_update_weight_eff = 0.0
                        o_hard_protect_hit = True
                    elif out_ratio >= ratio_soft:
                        o_update_weight_eff = min(o_update_weight_eff, ratio_soft_scale)
                        o_soft_protect_hit = True

                if o_update_weight_eff < 1.0 - 1e-12:
                    loss_o = float(o_update_weight_eff) * loss_o

            m_policy_losses: List[torch.Tensor] = []
            m_entropies: List[torch.Tensor] = []
            new_logp_m_list: List[torch.Tensor] = []
            old_logp_m_list: List[torch.Tensor] = []
            for local_i, global_i in enumerate(idx.tolist()):
                tr = flat[int(global_i)]
                if len(tr.m_op_order) == 0 or np.asarray(tr.m_action_pair_indices, dtype=np.int64).size == 0:
                    continue
                rec = {
                    "global_feat": np.asarray(tr.m_global, dtype=np.float32),
                    "pair_feat": np.asarray(tr.m_pair, dtype=np.float32),
                    "pair_mask": np.asarray(tr.m_mask, dtype=bool),
                    "pair_op_idx": np.asarray(tr.m_pair_op_idx, dtype=np.int64),
                    "op_order": [int(x) for x in tr.m_op_order],
                    "action_pair_indices": np.asarray(tr.m_action_pair_indices, dtype=np.int64),
                }
                out_m = agent_m.evaluate_action(rec)
                new_lp_m = out_m["log_prob"]
                ent_m = out_m["entropy"]
                if not bool(new_lp_m.requires_grad):
                    continue
                old_lp_m = m_old_logp[local_i]
                ratio_m = torch.exp(new_lp_m - old_lp_m)
                ratio_m_clip = torch.clamp(ratio_m, 1.0 - float(cfg.clip_ratio_m), 1.0 + float(cfg.clip_ratio_m))
                adv_m_i = mb_adv_m[local_i]
                pol_m = -torch.min(ratio_m * adv_m_i, ratio_m_clip * adv_m_i)
                m_policy_losses.append(pol_m)
                m_entropies.append(ent_m)
                new_logp_m_list.append(new_lp_m)
                old_logp_m_list.append(old_lp_m)

            if len(m_policy_losses) > 0:
                m_policy_loss = torch.stack(m_policy_losses).mean()
                entropy_m = torch.stack(m_entropies).mean()
                kl_m = torch.stack([torch.abs(o - n) for o, n in zip(old_logp_m_list, new_logp_m_list)]).mean()
                loss_m = (
                    m_policy_loss
                    - float(m_ent_coef_eff) * entropy_m
                    + float(cfg.kl_penalty_coef_m) * kl_m
                )
                do_m_update = True
            else:
                loss_m = torch.tensor(0.0, dtype=torch.float32, device=device)
                entropy_m = torch.tensor(0.0, dtype=torch.float32, device=device)
                kl_m = torch.tensor(0.0, dtype=torch.float32, device=device)
                do_m_update = False

            kl_c = float((c_old_logp - new_logp_c).abs().mean().item())
            kl_o = float((old_logp_o_mix - new_logp_o).abs().mean().item())
            kl_m_v = float(kl_m.item())

            guard_c_hit = False
            guard_o_hit = False
            guard_m_hit = False
            if bool(minibatch_kl_guard_enabled):
                if (not freeze_c_for_update) and target_kl_c > 0.0 and kl_c > kl_guard_thr_c:
                    freeze_c_for_update = True
                    guard_c_hit = True
                if (
                    (not freeze_o_for_update)
                    and bool(should_update_o)
                    and target_kl_o > 0.0
                    and kl_o > kl_guard_thr_o
                ):
                    freeze_o_for_update = True
                    guard_o_hit = True
                if (
                    (not freeze_m_for_update)
                    and bool(do_m_update)
                    and bool(should_update_m_by_interval)
                    and target_kl_m > 0.0
                    and kl_m_v > kl_guard_thr_m
                ):
                    freeze_m_for_update = True
                    guard_m_hit = True

            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss = (
                float(cfg.value_coef_c) * value_loss_c
                + float(cfg.value_coef_o) * value_loss_o
                + float(cfg.value_coef_m) * value_loss_m
            )
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
            critic_optimizer.step()

            if not bool(freeze_c_for_update):
                agent_c.optimizer.zero_grad(set_to_none=True)
                loss_c.backward()
                nn.utils.clip_grad_norm_(agent_c.policy.parameters(), cfg.max_grad_norm)
                agent_c.optimizer.step()

            if bool(should_update_o) and (not bool(freeze_o_for_update)):
                agent_o.optimizer.zero_grad(set_to_none=True)
                loss_o.backward()
                nn.utils.clip_grad_norm_(agent_o.policy.parameters(), cfg.max_grad_norm)
                agent_o.optimizer.step()

            if bool(do_m_update) and bool(should_update_m_by_interval) and (not bool(freeze_m_for_update)):
                agent_m.optimizer.zero_grad(set_to_none=True)
                loss_m.backward()
                nn.utils.clip_grad_norm_(agent_m.policy.parameters(), cfg.max_grad_norm)
                agent_m.optimizer.step()

            epoch_kls_c.append(kl_c)
            if bool(should_update_o):
                epoch_kls_o.append(kl_o)
            if bool(should_update_m_by_interval):
                epoch_kls_m.append(kl_m_v)

            logs["loss_c"].append(float(loss_c.item()))
            if bool(should_update_o):
                logs["loss_o"].append(float(loss_o.item()))
            logs["loss_m"].append(float(loss_m.item()))
            logs["value_loss"].append(float(value_loss.item()))
            logs["value_loss_c"].append(float(value_loss_c.item()))
            if bool(should_update_o):
                logs["value_loss_o"].append(float(value_loss_o.item()))
            logs["value_loss_m"].append(float(value_loss_m.item()))
            logs["entropy_c"].append(float(entropy_c.item()))
            if bool(should_update_o):
                logs["entropy_o"].append(float(entropy_o.item()))
            if bool(should_update_m_by_interval):
                logs["entropy_m"].append(float(entropy_m.item()))
            logs["approx_kl_c"].append(kl_c)
            if bool(should_update_o):
                logs["approx_kl_o"].append(kl_o)
            if bool(should_update_m_by_interval):
                logs["approx_kl_m"].append(kl_m_v)
            logs["kl_penalty_c"].append(float(kl_penalty_c_t.item()))
            if bool(should_update_o):
                logs["kl_penalty_o"].append(float(kl_penalty_o_t.item()))
                logs["loss_o_aux"].append(float(loss_o_aux.item()))
                logs["o_set_aux_coef_eff"].append(float(o_set_aux_coef_eff))
                logs["o_ent_coef_eff"].append(float(o_ent_coef_eff))
            if bool(should_update_m_by_interval):
                logs["kl_penalty_m"].append(float(kl_m.item()))
                logs["m_ent_coef_eff"].append(float(m_ent_coef_eff))
            logs["minibatch_kl_guard_c"].append(1.0 if guard_c_hit else 0.0)
            if bool(should_update_o):
                logs["minibatch_kl_guard_o"].append(1.0 if guard_o_hit else 0.0)
                logs["o_batch_soft_protect"].append(1.0 if o_soft_protect_hit else 0.0)
                logs["o_batch_hard_protect"].append(1.0 if o_hard_protect_hit else 0.0)
                logs["o_batch_update_weight"].append(float(o_update_weight_eff))
            if bool(should_update_m_by_interval):
                logs["minibatch_kl_guard_m"].append(1.0 if guard_m_hit else 0.0)

        if bool(cfg.ppo_early_stop):
            hit_kl_c = len(epoch_kls_c) > 0 and float(np.mean(epoch_kls_c)) > float(cfg.target_kl_c)
            hit_kl_o = bool(should_update_o) and len(epoch_kls_o) > 0 and float(np.mean(epoch_kls_o)) > float(cfg.target_kl_o)
            hit_kl_m = bool(should_update_m_by_interval) and len(epoch_kls_m) > 0 and float(np.mean(epoch_kls_m)) > float(cfg.target_kl_m)
            if hit_kl_c or hit_kl_o or hit_kl_m:
                break

    summary = {k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in logs.items()}
    summary["o_metrics_valid"] = 1.0 if bool(should_update_o) and len(logs["approx_kl_o"]) > 0 else 0.0
    summary["m_metrics_valid"] = 1.0 if bool(should_update_m_by_interval) and len(logs["approx_kl_m"]) > 0 else 0.0
    return summary


def _load_agent_m_checkpoint(agent_m: Agent_M_Learn, ckpt_path: str, strict: bool = False) -> None:
    payload = _torch_load_compat(str(ckpt_path), map_location=agent_m.device)
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        state_dict = payload
    agent_m.policy.load_state_dict(state_dict, strict=bool(strict))


def _save_latest_train_state(
    train_state_path: Path,
    epoch: int,
    best_val_makespan: float,
    best_epoch: int,
    history: List[Dict[str, object]],
    agent_c: Agent_C,
    agent_o: Agent_O,
    agent_m: Agent_M_Learn,
    critic: CentralCritic,
    critic_optimizer: optim.Optimizer,
) -> None:
    torch.save(
        {
            "epoch": int(epoch),
            "best_val_makespan": float(best_val_makespan),
            "best_epoch": int(best_epoch),
            "history": history,
            "agent_c_state": agent_c.policy.state_dict(),
            "agent_o_state": agent_o.policy.state_dict(),
            "agent_m_state": agent_m.policy.state_dict(),
            "critic_state": critic.state_dict(),
            "optimizer_c_state": agent_c.optimizer.state_dict(),
            "optimizer_o_state": agent_o.optimizer.state_dict(),
            "optimizer_m_state": agent_m.optimizer.state_dict(),
            "optimizer_critic_state": critic_optimizer.state_dict(),
        },
        str(train_state_path),
    )


def _load_latest_train_state(
    train_state_path: Path,
    agent_c: Agent_C,
    agent_o: Agent_O,
    agent_m: Agent_M_Learn,
    critic: CentralCritic,
    critic_optimizer: optim.Optimizer,
) -> Dict[str, object]:
    payload = _torch_load_compat(str(train_state_path), map_location=agent_c.device)
    agent_c.policy.load_state_dict(payload["agent_c_state"], strict=True)
    agent_o.policy.load_state_dict(payload["agent_o_state"], strict=True)
    agent_m.policy.load_state_dict(payload["agent_m_state"], strict=True)
    critic.load_state_dict(payload["critic_state"], strict=True)

    if "optimizer_c_state" in payload:
        agent_c.optimizer.load_state_dict(payload["optimizer_c_state"])
    if "optimizer_o_state" in payload:
        agent_o.optimizer.load_state_dict(payload["optimizer_o_state"])
    if "optimizer_m_state" in payload:
        agent_m.optimizer.load_state_dict(payload["optimizer_m_state"])
    if "optimizer_critic_state" in payload:
        critic_optimizer.load_state_dict(payload["optimizer_critic_state"])

    return {
        "epoch": int(payload.get("epoch", 0)),
        "best_val_makespan": float(payload.get("best_val_makespan", float("inf"))),
        "best_epoch": int(payload.get("best_epoch", 0)),
        "history": list(payload.get("history", [])),
    }


@torch.no_grad()
def _refine_c_action_deterministic_topk(
    agent_c: Agent_C,
    obs: Dict[str, object],
    c_info: Dict[str, object],
    cfg: OCMAPPOConfig,
) -> Dict[str, object]:
    topk = int(max(1, int(getattr(cfg, "deterministic_refine_topk", 1))))
    if topk <= 1:
        return c_info
    logit_gap = float(max(0.0, float(getattr(cfg, "deterministic_refine_logit_gap", 0.05))))
    min_ect_gain = float(max(0.0, float(getattr(cfg, "deterministic_refine_min_ect_gain", 0.0))))

    try:
        packed, global_tensor, pair_tensor, mask_tensor, graph_tensors = agent_c._obs_to_tensors(obs)
        logits, _values, _attn_weight = agent_c.policy.forward(
            global_tensor,
            pair_tensor,
            mask_tensor,
            **graph_tensors,
        )
        logits_1d = logits.squeeze(0)
        valid_idx = torch.where(mask_tensor.squeeze(0))[0]
        if int(valid_idx.numel()) <= 1:
            return c_info

        k = int(min(topk, int(valid_idx.numel())))
        top_local = torch.topk(logits_1d[valid_idx], k=k).indices
        top_idx = valid_idx[top_local].detach().cpu().numpy().astype(np.int64)

        top_scores = logits_1d[torch.tensor(top_idx, dtype=torch.long, device=logits_1d.device)]
        top_best = float(torch.max(top_scores).item())
        gated_idx = [
            int(i)
            for i in top_idx.tolist()
            if (top_best - float(logits_1d[int(i)].item())) <= logit_gap
        ]
        if len(gated_idx) <= 1:
            return c_info

        pair_feat = np.asarray(packed.get("pair_feat", []), dtype=np.float32)
        if pair_feat.ndim != 2 or pair_feat.shape[0] == 0 or pair_feat.shape[1] < 5:
            return c_info

        # pair_feat: [processing_time, ..., estimated_completion_time, ...]
        def _rank_key(idx: int) -> Tuple[float, float, float, int]:
            ect = float(pair_feat[idx, 4])
            pt = float(pair_feat[idx, 0])
            neg_logit = -float(logits_1d[idx].item())
            return (ect, pt, neg_logit, int(idx))

        best_idx = int(min((int(i) for i in gated_idx), key=_rank_key))
        if best_idx == int(c_info.get("action_idx", -1)):
            return c_info

        orig_idx = int(c_info.get("action_idx", -1))
        if 0 <= orig_idx < int(pair_feat.shape[0]) and min_ect_gain > 0.0:
            orig_ect = float(pair_feat[orig_idx, 4])
            best_ect = float(pair_feat[best_idx, 4])
            if (orig_ect - best_ect) < min_ect_gain:
                return c_info

        op_id, mch_id = packed["candidate_pairs"][best_idx]
        dist = torch.distributions.Categorical(logits=logits_1d.unsqueeze(0))
        action_t = torch.tensor([best_idx], dtype=torch.long, device=logits_1d.device)

        refined = dict(c_info)
        refined.update(
            {
                "action_idx": int(best_idx),
                "op_id": int(op_id),
                "mch_id": int(mch_id),
                "log_prob": float(dist.log_prob(action_t).item()),
                "entropy": float(dist.entropy().item()),
                "packed_obs": packed,
                "was_truncated": bool(packed.get("was_truncated", False)),
                "overflow_count": int(packed.get("overflow_count", 0)),
                "original_candidate_count": int(packed.get("original_candidate_count", len(obs.get("candidate_pairs", [])))),
                "used_candidate_count": int(packed.get("used_candidate_count", len(packed.get("candidate_pairs", [])))),
            }
        )
        return refined
    except Exception:
        return c_info


def run_episode_oc(
    agent_c: Agent_C,
    agent_o: Agent_O,
    agent_m: Agent_M_Learn,
    critic: CentralCritic,
    case_path: str,
    cfg: OCMAPPOConfig,
    deterministic: bool,
    collect_transitions: bool,
    reward_progress: float = 1.0,
    m_entropy_signal: Optional[float] = None,
    timing_meter: Optional[Dict[str, float]] = None,
    rule_meter: Optional[Dict[str, object]] = None,
) -> EpisodeRollout:
    ep_t0 = time.perf_counter()
    env, total_tasks = build_env_for_case(case_path, reset_rule=cfg.reset_rule, quiet=cfg.quiet_env)
    initial_lb_makespan = float(env.LBm[0].max())

    transitions: List[Transition] = []
    episode_reward = 0.0
    episode_o_reward = 0.0
    episode_m_reward = 0.0
    done = False
    reward_ctrl = _resolve_reward_coeffs(
        cfg=cfg,
        reward_progress=reward_progress,
        c_max_candidates_effective=int(agent_c.config.max_candidates),
        o_max_ops_effective=int(agent_o.config.max_ops),
        m_entropy_signal=m_entropy_signal,
    )
    reward_version = str(reward_ctrl["version"])
    o_alpha_env = float(reward_ctrl["o_alpha_env"])
    o_beta_shape = float(reward_ctrl["o_beta_shape"])
    o_teacher_coef = float(reward_ctrl["o_teacher_coef"])
    o_consensus_coef = float(reward_ctrl["o_consensus_coef"])
    m_alpha_env = float(reward_ctrl["m_alpha_env"])
    m_beta_shape = float(reward_ctrl["m_beta_shape"])
    m_teacher_coef = float(reward_ctrl["m_teacher_coef"])
    m_consensus_coef = float(reward_ctrl["m_consensus_coef"])

    teacher_active = (
        reward_version == "role_aligned_v1"
        and (o_teacher_coef > 1e-8 or m_teacher_coef > 1e-8)
    )
    use_fast_cascade = bool(getattr(cfg, "p1_fast_skip_c_o_ref", True)) and (not teacher_active)
    skip_c_full_ref = bool(getattr(cfg, "p1_skip_c_full_ref", False)) and (not teacher_active)
    requires_teacher_for_hard_keep = bool(getattr(cfg, "m_hard_keep_requires_teacher", False))
    min_teacher_for_hard_keep = float(max(0.0, float(getattr(cfg, "m_hard_keep_teacher_min_coef", 0.0))))
    hard_keep_entropy_trigger = bool(getattr(cfg, "m_hard_keep_entropy_trigger", False))
    hard_keep_entropy_threshold = float(max(0.0, float(getattr(cfg, "m_hard_keep_entropy_threshold", 0.20))))
    allow_m_hard_keep_base = (not requires_teacher_for_hard_keep) or (m_teacher_coef > min_teacher_for_hard_keep)
    parallel_om_enabled = bool(getattr(cfg, "ocm_parallel_om_enabled", False))
    parallel_m_teacher_mode = str(getattr(cfg, "ocm_parallel_m_teacher_mode", "full_ref")).strip().lower()
    if parallel_m_teacher_mode not in {"full_ref", "disabled"}:
        parallel_m_teacher_mode = "full_ref"

    stat_steps = 0
    c_forward_calls = 0
    o_selected_ops_sum = 0.0
    o_invalid_ops_sum = 0.0
    o_fallback_count = 0
    m_keep_pairs_sum = 0.0
    m_adopt_count = 0
    parallel_m_scope_pairs_sum = 0.0
    parallel_final_pairs_sum = 0.0
    parallel_rescue_pairs_sum = 0.0
    o_shape_term_sum = 0.0
    o_teacher_term_sum = 0.0
    o_consensus_term_sum = 0.0

    for _ in range(int(total_tasks) + int(cfg.extra_step_budget)):
        stat_steps += 1
        t_obs = time.perf_counter()
        full_obs = env.get_agent_c_obs(batch_idx=0)
        _add_timing(timing_meter, "env_obs_sec", time.perf_counter() - t_obs)

        if bool(skip_c_full_ref):
            ref_op, ref_mch = _best_pair_by_ect(full_obs)
            c_full_info = {
                "op_id": int(ref_op),
                "mch_id": int(ref_mch),
                "entropy": float(_estimate_op_uncertainty_from_ect(full_obs)),
            }
            c_entropy_full = float(c_full_info.get("entropy", 0.0))
        else:
            # Use a deterministic C reference by default to reduce reward-shaping noise for O.
            t_c_full = time.perf_counter()
            c_full_info = agent_c.select_action(full_obs, deterministic=bool(cfg.o_reference_c_deterministic))
            _add_timing(timing_meter, "select_c_full_sec", time.perf_counter() - t_c_full)
            c_forward_calls += 1
            c_entropy_full = float(c_full_info.get("entropy", 0.0))

        dynamic_topk = _compute_dynamic_topk(full_obs=full_obs, c_entropy=c_entropy_full, cfg=cfg)
        t_o = time.perf_counter()
        o_info = agent_o.select_action(full_obs, deterministic=deterministic, topk_k=dynamic_topk)
        _add_timing(timing_meter, "select_o_sec", time.perf_counter() - t_o)
        selected_ops = list(dict.fromkeys([int(x) for x in o_info.get("selected_ops", [])]))
        if len(selected_ops) == 0:
            selected_ops = [int(o_info["action_op_id"])]
        legal_op_set = {int(op) for op, _ in full_obs.get("candidate_pairs", [])}
        invalid_ops = [op for op in selected_ops if int(op) not in legal_op_set]
        o_invalid_ops_sum += float(len(invalid_ops))

        min_keep = max(int(cfg.o_topk_min), int(dynamic_topk))
        selected_ops = _expand_ops_for_safety(full_obs=full_obs, selected_ops=selected_ops, min_keep=min_keep)
        o_selected_ops_sum += float(len(selected_ops))

        selected_pair_indices = _selected_pair_indices_for_ops(full_obs, selected_ops)
        o_filtered_obs = _slice_candidate_obs(full_obs, selected_pair_indices)
        if bool(use_fast_cascade):
            c_entropy_filtered = float(_estimate_op_uncertainty_from_ect(o_filtered_obs))
            c_o_pair = _best_pair_by_ect(o_filtered_obs)
            c_o_op = int(c_o_pair[0]) if int(c_o_pair[0]) >= 0 else int(c_full_info["op_id"])
        else:
            t_c_filtered = time.perf_counter()
            c_o_info = agent_c.select_action(o_filtered_obs, deterministic=deterministic)
            _add_timing(timing_meter, "select_c_filtered_sec", time.perf_counter() - t_c_filtered)
            c_entropy_filtered = float(c_o_info.get("entropy", 0.0))
            c_o_pair = (int(c_o_info["op_id"]), int(c_o_info["mch_id"]))
            c_o_op = int(c_o_info["op_id"])

        selected_after_fallback = _maybe_uncertainty_fallback(
            full_obs=full_obs,
            selected_ops=selected_ops,
            c_entropy_filtered=c_entropy_filtered,
            cfg=cfg,
        )
        fallback_triggered = len(selected_after_fallback) > len(selected_ops)
        if fallback_triggered:
            o_fallback_count += 1
            selected_ops = selected_after_fallback
            selected_pair_indices = _selected_pair_indices_for_ops(full_obs, selected_ops)
            o_filtered_obs = _slice_candidate_obs(full_obs, selected_pair_indices)
            if bool(use_fast_cascade):
                c_o_pair = _best_pair_by_ect(o_filtered_obs)
                c_o_op = int(c_o_pair[0]) if int(c_o_pair[0]) >= 0 else int(c_full_info["op_id"])
            else:
                c_o_info = agent_c.select_action(o_filtered_obs, deterministic=deterministic)
                c_o_pair = (int(c_o_info["op_id"]), int(c_o_info["mch_id"]))
                c_o_op = int(c_o_info["op_id"])

        c_full_pair = (int(c_full_info["op_id"]), int(c_full_info["mch_id"]))

        t_m = time.perf_counter()
        parallel_scope_info = None
        if bool(parallel_om_enabled):
            parallel_scope_info = _build_parallel_m_scope(
                full_obs=full_obs,
                selected_ops=selected_ops,
                selected_pair_indices=selected_pair_indices,
                c_full_pair=c_full_pair,
                cfg=cfg,
            )
            m_selected_ops = list(parallel_scope_info["selected_ops"])
            m_pair_indices = list(parallel_scope_info["pair_indices"])
            parallel_m_scope_pairs_sum += float(len(m_pair_indices))
            parallel_rescue_pairs_sum += float(len(parallel_scope_info.get("rescue_pair_indices", [])))
        else:
            m_selected_ops = list(selected_ops)
            m_pair_indices = list(selected_pair_indices)
        pm = agent_m.build_packed_obs(
            full_obs,
            selected_ops=m_selected_ops,
            pair_indices=m_pair_indices,
        )
        m_fallback_triggered = False
        if len(pm["candidate_pairs"]) == 0:
            m_info = {
                "keep_indices": [],
                "action_pair_indices": np.zeros((0,), dtype=np.int64),
                "log_prob": 0.0,
                "entropy": 0.0,
                "value": 0.0,
            }
            keep_full_indices = []
            m_fallback_triggered = True
        else:
            m_info = agent_m.act(pm, deterministic=deterministic)
            local_keep = [int(i) for i in m_info.get("keep_indices", [])]
            full_idx_map = np.asarray(pm["full_pair_indices"], dtype=np.int64)
            keep_full_indices = [int(full_idx_map[i]) for i in local_keep if 0 <= int(i) < full_idx_map.shape[0]]

        allow_m_hard_keep_step = bool(allow_m_hard_keep_base)
        if (not allow_m_hard_keep_step) and bool(hard_keep_entropy_trigger):
            m_entropy_now = float(m_info.get("entropy", 0.0))
            allow_m_hard_keep_step = bool(m_entropy_now <= hard_keep_entropy_threshold)

        if bool(cfg.m_keep_c_full_top1) and bool(allow_m_hard_keep_step):
            c_full_idx = _find_pair_index(full_obs, c_full_pair[0], c_full_pair[1])
            if c_full_idx >= 0:
                keep_full_indices.append(int(c_full_idx))

        if bool(cfg.m_keep_c_o_top1) and bool(allow_m_hard_keep_step):
            c_o_idx = _find_pair_index(full_obs, c_o_pair[0], c_o_pair[1])
            if c_o_idx >= 0:
                keep_full_indices.append(int(c_o_idx))

        if bool(getattr(cfg, "m_expand_pairs_for_safety", True)):
            keep_full_indices = _expand_pairs_for_safety(
                full_obs=full_obs,
                selected_ops=(m_selected_ops if bool(parallel_om_enabled) else selected_ops),
                keep_pair_indices=keep_full_indices,
                min_pairs_total=int(getattr(cfg, "m_safety_min_total_pairs", 3)),
                per_op_min_machines=int(getattr(cfg, "m_safety_min_machines_per_op", 3)),
            )

        if len(keep_full_indices) == 0:
            pf = np.asarray(full_obs.get("pair_feat", []), dtype=np.float32)
            if pf.ndim == 2 and pf.shape[0] > 0 and pf.shape[1] > 4:
                keep_full_indices = [int(np.argmin(pf[:, 4]))]
            else:
                keep_full_indices = [0]
            m_fallback_triggered = True

        keep_full_indices = sorted(set(int(x) for x in keep_full_indices))
        m_keep_pairs_sum += float(len(keep_full_indices))
        m_filtered_obs = _slice_candidate_obs(full_obs, keep_full_indices)
        final_keep_full_indices = list(keep_full_indices)
        fusion_info = {
            "mode": "serial",
            "fallback": "none",
            "o_pairs": int(len(selected_pair_indices)),
            "m_pairs": int(len(keep_full_indices)),
            "intersection_pairs": int(len(keep_full_indices)),
            "union_pairs": int(len(keep_full_indices)),
            "final_pairs": int(len(keep_full_indices)),
        }
        if bool(parallel_om_enabled):
            final_keep_full_indices, fusion_info = _parallel_fuse_keep_pair_indices(
                full_obs=full_obs,
                o_keep_pair_indices=selected_pair_indices,
                m_keep_pair_indices=keep_full_indices,
                cfg=cfg,
            )
        final_filtered_obs = _slice_candidate_obs(full_obs, final_keep_full_indices)
        if bool(parallel_om_enabled):
            parallel_final_pairs_sum += float(len(final_keep_full_indices))
        if bool(parallel_om_enabled) and bool(getattr(cfg, "ocm_parallel_debug_print", False)):
            print(
                f"[ocm-parallel] case={Path(case_path).name} step={int(stat_steps)} "
                f"mode={fusion_info['mode']} o_pairs={int(fusion_info['o_pairs'])} "
                f"m_pairs={int(fusion_info['m_pairs'])} inter={int(fusion_info['intersection_pairs'])} "
                f"union={int(fusion_info['union_pairs'])} target={int(fusion_info.get('target_final_pairs', fusion_info['final_pairs']))} "
                f"final={int(fusion_info['final_pairs'])} "
                f"m_scope={len(m_pair_indices)} rescue={0 if parallel_scope_info is None else len(parallel_scope_info.get('rescue_pair_indices', []))} "
                f"fallback={str(fusion_info['fallback'])}",
                flush=True,
            )
        c_info = agent_c.select_action(final_filtered_obs, deterministic=deterministic)
        c_forward_calls += 1
        if bool(deterministic) and (not bool(collect_transitions)):
            c_info = _refine_c_action_deterministic_topk(
                agent_c=agent_c,
                obs=final_filtered_obs,
                c_info=c_info,
                cfg=cfg,
            )
        _add_timing(timing_meter, "select_m_sec", time.perf_counter() - t_m)

        pc = c_info["packed_obs"]
        po = o_info["packed_obs"]
        pm_for_state = {
            "global_feat": np.asarray(pm["global_feat"], dtype=np.float32),
            "pair_feat": np.asarray(pm["pair_feat"], dtype=np.float32),
            "pair_mask": np.asarray(pm["pair_mask"], dtype=bool),
        }

        state_vec = build_central_state(
            packed_c=c_info["packed_obs"],
            packed_o=o_info["packed_obs"],
            packed_m=pm_for_state,
            selected_ops=(_all_op_ids_from_obs(final_filtered_obs) if bool(parallel_om_enabled) else selected_ops),
            use_rich_stats=bool(cfg.critic_rich_state),
        )

        t_critic = time.perf_counter()
        with torch.no_grad():
            v_c, v_o, v_m = critic(
                torch.tensor(state_vec, dtype=torch.float32, device=agent_c.device).unsqueeze(0),
                c_pair=torch.tensor(np.asarray(pc["pair_feat"], dtype=np.float32), dtype=torch.float32, device=agent_c.device).unsqueeze(0),
                c_mask=torch.tensor(np.asarray(pc["pair_mask"], dtype=bool), dtype=torch.bool, device=agent_c.device).unsqueeze(0),
                o_op=torch.tensor(np.asarray(po["op_feat"], dtype=np.float32), dtype=torch.float32, device=agent_c.device).unsqueeze(0),
                o_mask=torch.tensor(np.asarray(po["op_mask"], dtype=bool), dtype=torch.bool, device=agent_c.device).unsqueeze(0),
                m_pair=torch.tensor(np.asarray(pm_for_state["pair_feat"], dtype=np.float32), dtype=torch.float32, device=agent_c.device).unsqueeze(0),
                m_mask=torch.tensor(np.asarray(pm_for_state["pair_mask"], dtype=bool), dtype=torch.bool, device=agent_c.device).unsqueeze(0),
            )
            value_c = float(v_c.item())
            value_o = float(v_o.item())
            value_m = float(v_m.item())
        _add_timing(timing_meter, "critic_forward_sec", time.perf_counter() - t_critic)

        t_env_step = time.perf_counter()
        out = env.step_with_pair(c_info["op_id"], c_info["mch_id"], batch_idx=0)
        _add_timing(timing_meter, "env_step_sec", time.perf_counter() - t_env_step)
        reward = float(out[2][0])
        done = bool(out[3][0])
        new_makespan = float(env.LBm[0].max())

        t_shape = time.perf_counter()
        o_shape_term = 0.0
        o_teacher_term = 0.0
        o_consensus_term = 0.0
        if reward_version == "legacy":
            o_terms = _compute_o_shape_reward_legacy(
                full_obs=full_obs,
                filtered_obs=o_filtered_obs,
                selected_ops=selected_ops,
                c_full_op=int(c_full_info["op_id"]),
                c_filtered_op=int(c_o_op),
                new_makespan=new_makespan,
                initial_lb_makespan=initial_lb_makespan,
                is_terminal=done,
                cfg=cfg,
            )
            o_shape_term = float(o_beta_shape) * float(o_terms["total"])
            reward_o = float(o_alpha_env) * reward + float(o_shape_term)
        else:
            if reward_version == "role_aligned_v2":
                o_terms = _compute_o_shape_reward_role_aligned_v2(
                    full_obs=full_obs,
                    filtered_obs=o_filtered_obs,
                    selected_ops=selected_ops,
                )
            else:
                o_terms = _compute_o_shape_reward_role_aligned_v1(
                    full_obs=full_obs,
                    filtered_obs=o_filtered_obs,
                    selected_ops=selected_ops,
                )
            o_consensus = _compute_o_rule_consensus_bonus(
                full_obs=full_obs,
                selected_ops=selected_ops,
            )
            teacher_align_o = 1.0 if int(c_full_info["op_id"]) in {int(x) for x in selected_ops} else 0.0
            o_shape_term = float(o_beta_shape) * float(o_terms["total"])
            o_teacher_term = float(o_teacher_coef) * float(teacher_align_o)
            o_consensus_term = float(o_consensus_coef) * float(o_consensus["total"])
            reward_o = (
                float(o_alpha_env) * reward
                + float(o_shape_term)
                + float(o_teacher_term)
                + float(o_consensus_term)
            )
        if fallback_triggered:
            reward_o *= float(cfg.o_reward_fallback_scale)
            o_shape_term *= float(cfg.o_reward_fallback_scale)
            o_teacher_term *= float(cfg.o_reward_fallback_scale)
            o_consensus_term *= float(cfg.o_reward_fallback_scale)
        if float(cfg.o_reward_clip_abs) > 0:
            reward_o = float(np.clip(reward_o, -float(cfg.o_reward_clip_abs), float(cfg.o_reward_clip_abs)))
        o_shape_term_sum += float(o_shape_term)
        o_teacher_term_sum += float(o_teacher_term)
        o_consensus_term_sum += float(o_consensus_term)

        if reward_version == "legacy":
            m_terms = _compute_m_shape_reward_legacy(
                base_obs=(full_obs if bool(parallel_om_enabled) else o_filtered_obs),
                filtered_obs=m_filtered_obs,
                c_ref_pair=(
                    c_full_pair
                    if (bool(parallel_om_enabled) and str(parallel_m_teacher_mode) == "full_ref")
                    else c_o_pair
                ),
                c_final_pair=(int(c_info["op_id"]), int(c_info["mch_id"])),
                new_makespan=new_makespan,
                initial_lb_makespan=initial_lb_makespan,
                is_terminal=done,
                cfg=cfg,
            )
            reward_m = float(m_alpha_env) * reward + float(m_beta_shape) * float(m_terms["total"])
        else:
            m_terms = _compute_m_shape_reward_role_aligned_v1(
                base_obs=(full_obs if bool(parallel_om_enabled) else o_filtered_obs),
                filtered_obs=(final_filtered_obs if bool(parallel_om_enabled) else m_filtered_obs),
                selected_ops=(m_selected_ops if bool(parallel_om_enabled) else selected_ops),
                cfg=cfg,
            )
            m_consensus = _compute_m_rule_consensus_bonus(
                full_obs=full_obs,
                kept_pair_indices=(final_keep_full_indices if bool(parallel_om_enabled) else keep_full_indices),
            )
            legal_pairs_set = set((final_filtered_obs if bool(parallel_om_enabled) else m_filtered_obs).get("legal_pairs_set", set()))
            if bool(parallel_om_enabled):
                if str(parallel_m_teacher_mode) == "disabled":
                    teacher_align_m = 0.0
                else:
                    teacher_align_m = 1.0 if (int(c_full_pair[0]), int(c_full_pair[1])) in legal_pairs_set else 0.0
            else:
                teacher_align_m = 1.0 if (int(c_o_pair[0]), int(c_o_pair[1])) in legal_pairs_set else 0.0
            reward_m = (
                float(m_alpha_env) * reward
                + float(m_beta_shape) * float(m_terms["total"])
                + float(m_teacher_coef) * float(teacher_align_m)
                + float(m_consensus_coef) * float(m_consensus["total"])
            )
        if m_fallback_triggered:
            reward_m *= float(cfg.m_reward_fallback_scale)
        if float(cfg.m_reward_clip_abs) > 0:
            reward_m = float(np.clip(reward_m, -float(cfg.m_reward_clip_abs), float(cfg.m_reward_clip_abs)))
        _add_timing(timing_meter, "reward_shape_sec", time.perf_counter() - t_shape)

        _update_rule_meter_step(
            meter=rule_meter,
            full_obs=full_obs,
            filtered_obs=final_filtered_obs,
            c_full_info=c_full_info,
            c_info=c_info,
            reward_c=reward,
            reward_o=reward_o,
            done=done,
        )

        episode_reward += reward
        episode_o_reward += reward_o
        episode_m_reward += reward_m

        chosen_full_action_idx = np.asarray(m_info.get("action_pair_indices", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
        if chosen_full_action_idx.size > 0:
            full_idx_map = np.asarray(pm.get("full_pair_indices", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
            chosen_full = set(int(full_idx_map[i]) for i in chosen_full_action_idx.tolist() if 0 <= int(i) < full_idx_map.shape[0])
            c_final_idx = _find_pair_index(full_obs, int(c_info["op_id"]), int(c_info["mch_id"]))
            if int(c_final_idx) >= 0 and int(c_final_idx) in chosen_full:
                m_adopt_count += 1

        _add_timing(timing_meter, "step_count", 1.0)

        if collect_transitions:
            o_selected_mask = np.zeros_like(np.asarray(po["op_mask"], dtype=bool), dtype=bool)
            legal_ops = [int(x) for x in po.get("legal_ops", [])]
            op_to_idx = {int(op): int(i) for i, op in enumerate(legal_ops)}
            for op in selected_ops:
                loc = op_to_idx.get(int(op), None)
                if loc is None:
                    continue
                if 0 <= int(loc) < o_selected_mask.shape[0]:
                    o_selected_mask[int(loc)] = True
            if not bool(np.any(o_selected_mask)):
                act_idx = int(o_info.get("action_idx", 0))
                if 0 <= act_idx < o_selected_mask.shape[0]:
                    o_selected_mask[act_idx] = True

            o_logits_full = np.zeros_like(np.asarray(po["op_mask"], dtype=np.float32), dtype=np.float32)
            valid_logits = np.asarray(o_info.get("valid_logits", []), dtype=np.float32).reshape(-1)
            valid_n = int(min(valid_logits.shape[0], o_logits_full.shape[0]))
            if valid_n > 0:
                o_logits_full[:valid_n] = valid_logits[:valid_n]
            o_old_set_log_prob = _masked_binary_log_prob_np(
                logits=o_logits_full,
                target_mask=np.asarray(o_selected_mask, dtype=bool),
                valid_mask=np.asarray(po["op_mask"], dtype=bool),
            )

            transitions.append(
                Transition(
                    c_global=np.asarray(pc["global_feat"], dtype=np.float32),
                    c_pair=np.asarray(pc["pair_feat"], dtype=np.float32),
                    c_mask=np.asarray(pc["pair_mask"], dtype=bool),
                    c_action_idx=int(c_info["action_idx"]),
                    c_old_log_prob=float(c_info["log_prob"]),
                    o_global=np.asarray(po["global_feat"], dtype=np.float32),
                    o_op=np.asarray(po["op_feat"], dtype=np.float32),
                    o_mask=np.asarray(po["op_mask"], dtype=bool),
                    o_selected_mask=np.asarray(o_selected_mask, dtype=bool),
                    o_action_idx=int(o_info["action_idx"]),
                    o_old_log_prob=float(o_info["log_prob"]),
                    o_old_set_log_prob=float(o_old_set_log_prob),
                    m_global=np.asarray(pm["global_feat"], dtype=np.float32),
                    m_pair=np.asarray(pm["pair_feat"], dtype=np.float32),
                    m_mask=np.asarray(pm["pair_mask"], dtype=bool),
                    m_pair_op_idx=np.asarray(pm["pair_op_idx"], dtype=np.int64),
                    m_op_order=[int(x) for x in pm.get("op_order", [])],
                    m_action_pair_indices=np.asarray(m_info.get("action_pair_indices", np.zeros((0,), dtype=np.int64)), dtype=np.int64),
                    m_old_log_prob=float(m_info.get("log_prob", 0.0)),
                    state_vec=np.asarray(state_vec, dtype=np.float32),
                    value_c=value_c,
                    value_o=value_o,
                    value_m=value_m,
                    reward_c=reward,
                    reward_o=reward_o,
                    reward_m=reward_m,
                    done=done,
                )
            )

        if done:
            break

    makespan = float(env.LBm[0].max())
    _add_timing(timing_meter, "episodes", 1.0)
    _add_timing(timing_meter, "episode_total_sec", time.perf_counter() - ep_t0)
    denom_steps = float(max(1, stat_steps))
    return EpisodeRollout(
        case_name=Path(case_path).name,
        transitions=transitions,
        episode_reward=float(episode_reward),
        episode_o_reward=float(episode_o_reward),
        episode_m_reward=float(episode_m_reward),
        makespan=float(makespan),
        steps=int(len(transitions) if collect_transitions else 0),
        done=bool(done),
        o_avg_selected_ops=float(o_selected_ops_sum / denom_steps),
        o_fallback_rate=float(o_fallback_count / denom_steps),
        o_invalid_op_ratio=float(o_invalid_ops_sum / denom_steps),
        m_avg_keep_pairs=float(m_keep_pairs_sum / denom_steps),
        m_adopt_rate=float(m_adopt_count / denom_steps),
        c_forward_per_step=float(c_forward_calls / denom_steps),
        parallel_avg_m_scope_pairs=(float(parallel_m_scope_pairs_sum / denom_steps) if bool(parallel_om_enabled) else 0.0),
        parallel_avg_final_pairs=(float(parallel_final_pairs_sum / denom_steps) if bool(parallel_om_enabled) else 0.0),
        parallel_avg_rescue_pairs=(float(parallel_rescue_pairs_sum / denom_steps) if bool(parallel_om_enabled) else 0.0),
        o_avg_shape_term=float(o_shape_term_sum / denom_steps),
        o_avg_teacher_term=float(o_teacher_term_sum / denom_steps),
        o_avg_consensus_term=float(o_consensus_term_sum / denom_steps),
    )


def evaluate_oc(
    agent_c: Agent_C,
    agent_o: Agent_O,
    agent_m: Agent_M_Learn,
    critic: CentralCritic,
    case_paths: List[str],
    cfg: OCMAPPOConfig,
    eval_deterministic_override: Optional[bool] = None,
    eval_sample_times_override: Optional[int] = None,
    eval_sample_reduce_override: Optional[str] = None,
    val_only_best_strategy_override: Optional[bool] = None,
) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    errors: List[Dict[str, object]] = []

    if eval_deterministic_override is None:
        eval_deterministic = bool(getattr(cfg, "eval_deterministic", True))
    else:
        eval_deterministic = bool(eval_deterministic_override)

    if eval_sample_times_override is None:
        eval_sample_times = int(max(1, int(getattr(cfg, "eval_sample_times", 1))))
    else:
        eval_sample_times = int(max(1, int(eval_sample_times_override)))

    if eval_sample_reduce_override is None:
        eval_sample_reduce = str(getattr(cfg, "eval_sample_reduce", "best")).strip().lower()
    else:
        eval_sample_reduce = str(eval_sample_reduce_override).strip().lower()
    if eval_sample_reduce not in {"best", "mean"}:
        eval_sample_reduce = "best"
    if eval_deterministic:
        eval_sample_times = 1

    is_val_only_eval = (
        int(getattr(cfg, "epochs", 0)) <= 0
        and len(list(getattr(cfg, "test_cases", []))) == 0
        and len(case_paths) == len(list(getattr(cfg, "val_cases", [])))
    )
    if val_only_best_strategy_override is None:
        enable_val_only_best_strategy = bool(getattr(cfg, "val_only_best_strategy", False)) and is_val_only_eval
    else:
        enable_val_only_best_strategy = bool(val_only_best_strategy_override)
    enable_val_only_best_strategy = bool(enable_val_only_best_strategy and (not eval_deterministic))

    add_det_baseline = bool(getattr(cfg, "val_only_add_deterministic_baseline", True))
    extra_samples_per_case = int(max(0, int(getattr(cfg, "val_only_extra_samples_per_case", 0))))
    hard_case_topk = int(max(0, int(getattr(cfg, "val_only_hard_case_topk", 0))))
    seed_jitter = bool(getattr(cfg, "val_only_seed_jitter", True))
    adaptive_extra_sampling = bool(getattr(cfg, "val_only_adaptive_extra_sampling", True))
    adaptive_var_bonus = float(getattr(cfg, "val_only_adaptive_var_bonus", 0.35))
    adaptive_ucb_bonus = float(getattr(cfg, "val_only_adaptive_ucb_bonus", 0.15))

    case_paths_list = [str(x) for x in case_paths]
    case_runs_by_idx: List[List[EpisodeRollout]] = [[] for _ in case_paths_list]

    det_valonly_cfg_variants: List[OCMAPPOConfig] = [cfg]
    if eval_deterministic and is_val_only_eval:
        raw_candidates = str(getattr(cfg, "deterministic_val_only_gap_candidates", "")).strip()
        if len(raw_candidates) > 0:
            variants: List[OCMAPPOConfig] = [cfg]
            for token in [x.strip() for x in raw_candidates.split(",") if len(x.strip()) > 0]:
                t = token.lower()
                if t in {"argmax", "base", "none"}:
                    variants.append(
                        dataclasses.replace(
                            cfg,
                            deterministic_refine_topk=1,
                            deterministic_refine_logit_gap=0.0,
                            deterministic_refine_min_ect_gain=0.0,
                        )
                    )
                    continue
                try:
                    gap_v = float(token)
                except Exception:
                    continue
                if gap_v < 0.0:
                    continue
                variants.append(
                    dataclasses.replace(
                        cfg,
                        deterministic_refine_topk=int(max(1, int(getattr(cfg, "deterministic_refine_topk", 1)))),
                        deterministic_refine_logit_gap=float(gap_v),
                    )
                )

            dedup: List[OCMAPPOConfig] = []
            seen_keys: set = set()
            for v in variants:
                key = (
                    int(max(1, int(getattr(v, "deterministic_refine_topk", 1)))),
                    round(float(getattr(v, "deterministic_refine_logit_gap", 0.0)), 6),
                    round(float(getattr(v, "deterministic_refine_min_ect_gain", 0.0)), 6),
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                dedup.append(v)
            if len(dedup) > 0:
                det_valonly_cfg_variants = dedup

    def _maybe_reseed(case_idx: int, sample_idx: int, deterministic: bool) -> None:
        if not (enable_val_only_best_strategy and seed_jitter):
            return
        # Deterministic, reproducible jitter to diversify sampling trajectories across attempts.
        jitter_seed = int(cfg.seed) + 1000003 * int(case_idx + 1) + 10007 * int(sample_idx + 1) + (1 if deterministic else 0)
        seed_everything(int(jitter_seed))

    def _try_rollout(
        case_idx: int,
        case_path: str,
        deterministic: bool,
        sample_idx: int,
        cfg_override: Optional[OCMAPPOConfig] = None,
    ) -> None:
        try:
            _maybe_reseed(case_idx=case_idx, sample_idx=sample_idx, deterministic=deterministic)
            ep = run_episode_oc(
                agent_c=agent_c,
                agent_o=agent_o,
                agent_m=agent_m,
                critic=critic,
                case_path=case_path,
                cfg=cfg if cfg_override is None else cfg_override,
                deterministic=bool(deterministic),
                collect_transitions=False,
            )
            case_runs_by_idx[case_idx].append(ep)
        except Exception as exc:
            errors.append({"case": str(case_path), "reason": repr(exc)})

    for case_idx, case_path in enumerate(case_paths_list):
        if eval_deterministic and len(det_valonly_cfg_variants) > 1:
            for sample_idx, det_cfg in enumerate(det_valonly_cfg_variants):
                _try_rollout(
                    case_idx=case_idx,
                    case_path=case_path,
                    deterministic=True,
                    sample_idx=sample_idx,
                    cfg_override=det_cfg,
                )
            continue

        if enable_val_only_best_strategy and add_det_baseline:
            _try_rollout(case_idx=case_idx, case_path=case_path, deterministic=True, sample_idx=-1)
        for sample_idx in range(int(eval_sample_times)):
            _try_rollout(
                case_idx=case_idx,
                case_path=case_path,
                deterministic=bool(eval_deterministic),
                sample_idx=sample_idx,
            )

    if (
        enable_val_only_best_strategy
        and eval_sample_reduce == "best"
        and extra_samples_per_case > 0
        and hard_case_topk > 0
    ):
        case_scores: List[Tuple[float, int]] = []
        for case_idx, case_runs in enumerate(case_runs_by_idx):
            if len(case_runs) == 0:
                continue
            best_ep = min(case_runs, key=lambda x: float(x.makespan))
            case_scores.append((float(best_ep.makespan), int(case_idx)))

        case_scores.sort(key=lambda x: float(x[0]), reverse=True)
        max_hard_cases = int(min(hard_case_topk, len(case_scores)))
        if max_hard_cases > 0:
            if adaptive_extra_sampling:
                # Keep total extra budget stable, but allocate samples adaptively to
                # currently hard/uncertain under-explored cases.
                total_extra_budget = int(extra_samples_per_case) * int(max_hard_cases)
                for _ in range(int(total_extra_budget)):
                    current_scores: List[Tuple[float, int]] = []
                    for idx, runs in enumerate(case_runs_by_idx):
                        if len(runs) == 0:
                            continue
                        best_now = float(min(float(x.makespan) for x in runs))
                        current_scores.append((best_now, int(idx)))

                    if len(current_scores) == 0:
                        break

                    current_scores.sort(key=lambda x: float(x[0]), reverse=True)
                    candidate_ids = [int(x[1]) for x in current_scores[: int(max_hard_cases)]]
                    if len(candidate_ids) == 0:
                        break

                    candidate_best = [
                        float(min(float(ep.makespan) for ep in case_runs_by_idx[idx]))
                        for idx in candidate_ids
                    ]
                    best_scale = float(np.median(np.asarray(candidate_best, dtype=np.float64)))
                    best_scale = max(1e-6, abs(best_scale))

                    chosen_idx: Optional[int] = None
                    chosen_score = -float("inf")
                    for idx in candidate_ids:
                        runs = case_runs_by_idx[idx]
                        makes = np.asarray([float(ep.makespan) for ep in runs], dtype=np.float64)
                        if makes.size == 0:
                            continue
                        best_v = float(np.min(makes))
                        mean_v = float(np.mean(makes))
                        std_v = float(np.std(makes))
                        n_v = float(max(1, makes.size))

                        hardness = float(best_v / best_scale)
                        uncertainty = float(std_v / max(1e-6, abs(mean_v)))
                        exploration = float(1.0 / np.sqrt(n_v))
                        score = hardness + float(adaptive_var_bonus) * uncertainty + float(adaptive_ucb_bonus) * exploration

                        if score > chosen_score:
                            chosen_score = float(score)
                            chosen_idx = int(idx)

                    if chosen_idx is None:
                        break

                    _try_rollout(
                        case_idx=int(chosen_idx),
                        case_path=case_paths_list[int(chosen_idx)],
                        deterministic=False,
                        sample_idx=int(len(case_runs_by_idx[int(chosen_idx)])),
                    )
            else:
                chosen = [int(x[1]) for x in case_scores[: int(max_hard_cases)]]
                for case_idx in chosen:
                    case_path = case_paths_list[case_idx]
                    base_count = int(len(case_runs_by_idx[case_idx]))
                    for j in range(int(extra_samples_per_case)):
                        _try_rollout(
                            case_idx=case_idx,
                            case_path=case_path,
                            deterministic=False,
                            sample_idx=int(base_count + j),
                        )

    for case_idx, case_path in enumerate(case_paths_list):
        case_runs = case_runs_by_idx[case_idx]
        if len(case_runs) == 0:
            continue

        if len(case_runs) == 1 or eval_sample_reduce == "mean":
            reward_v = float(np.mean([float(x.episode_reward) for x in case_runs]))
            makespan_v = float(np.mean([float(x.makespan) for x in case_runs]))
            done_v = bool(np.mean([float(bool(x.done)) for x in case_runs]) >= 0.5)
            case_name_v = str(case_runs[0].case_name)
        else:
            best_ep = min(case_runs, key=lambda x: float(x.makespan))
            reward_v = float(best_ep.episode_reward)
            makespan_v = float(best_ep.makespan)
            done_v = bool(best_ep.done)
            case_name_v = str(best_ep.case_name)

        rows.append({
            "case": case_name_v,
            "episode_reward": reward_v,
            "makespan": makespan_v,
            "done": done_v,
            "sample_times": int(len(case_runs)),
            "sample_reduce": str(eval_sample_reduce),
            "eval_deterministic": bool(eval_deterministic),
            "val_only_best_strategy": bool(enable_val_only_best_strategy),
        })

    if len(rows) == 0:
        return {
            "num_cases": int(len(case_paths)),
            "num_success": 0,
            "num_errors": int(len(errors)),
            "mean_reward": 0.0,
            "mean_makespan": float("inf"),
            "rows": rows,
            "errors": errors,
        }

    return {
        "num_cases": int(len(case_paths)),
        "num_success": int(len(rows)),
        "num_errors": int(len(errors)),
        "mean_reward": float(np.mean([x["episode_reward"] for x in rows])),
        "mean_makespan": float(np.mean([x["makespan"] for x in rows])),
        "rows": rows,
        "errors": errors,
    }


def _parse_case_splits(raw: str, *, arg_name: str, default: Tuple[str, ...]) -> List[str]:
    allowed = ("train", "val", "test")
    text = str(raw).strip().lower()
    if len(text) == 0:
        return list(default)

    selected: List[str] = []
    for token in [x.strip().lower() for x in text.split(",") if len(x.strip()) > 0]:
        if token == "all":
            for name in allowed:
                if name not in selected:
                    selected.append(name)
            continue
        if token not in allowed:
            raise ValueError(
                f"{arg_name} contains invalid token '{token}', expected any of {','.join(allowed)} or 'all'"
            )
        if token not in selected:
            selected.append(token)

    if len(selected) == 0:
        raise ValueError(f"{arg_name} cannot be empty")
    return selected


def _collect_cases_by_splits(cfg: OCMAPPOConfig, split_names: List[str]) -> List[str]:
    merged: List[str] = []
    for split_name in split_names:
        if split_name == "train":
            merged.extend([str(x) for x in cfg.train_cases])
        elif split_name == "val":
            merged.extend([str(x) for x in cfg.val_cases])
        elif split_name == "test":
            merged.extend([str(x) for x in cfg.test_cases])

    # Keep order stable while removing duplicates.
    return list(dict.fromkeys(merged))


def _skipped_eval_result() -> Dict[str, object]:
    return {
        "num_cases": 0,
        "num_success": 0,
        "num_errors": 0,
        "mean_reward": 0.0,
        "mean_makespan": float("inf"),
        "rows": [],
        "errors": [],
        "skipped": True,
    }


def _case_names(cases: List[str]) -> List[str]:
    return [Path(x).name for x in cases]


def _print_split_summary(cfg: OCMAPPOConfig) -> None:
    split_source = str(cfg.split_source) if str(cfg.split_source).strip() else "random_split"
    print(
        f"[split] source={split_source}, seed={int(cfg.seed)}, "
        f"train={len(cfg.train_cases)}, val={len(cfg.val_cases)}, test={len(cfg.test_cases)}",
        flush=True,
    )
    print(
        f"[split-select] scan_case_splits={str(getattr(cfg, 'scan_case_splits', 'train,val'))}, "
        f"eval_case_splits={str(getattr(cfg, 'eval_case_splits', 'val,test'))}",
        flush=True,
    )
    print(f"[split-train] {json.dumps(_case_names(cfg.train_cases), ensure_ascii=False)}", flush=True)
    print(f"[split-val] {json.dumps(_case_names(cfg.val_cases), ensure_ascii=False)}", flush=True)
    print(f"[split-test] {json.dumps(_case_names(cfg.test_cases), ensure_ascii=False)}", flush=True)


def _rollout_backend_is_mp(cfg: OCMAPPOConfig) -> bool:
    return str(getattr(cfg, "rollout_backend", "serial")).strip().lower() in {"mp", "multiprocess", "worker"}


def _state_dict_to_cpu(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu()
        else:
            out[k] = v
    return out


def _split_cases_into_shards(case_paths: List[str], num_shards: int) -> List[List[str]]:
    if len(case_paths) == 0:
        return []
    n = int(max(1, min(int(num_shards), len(case_paths))))
    shards: List[List[str]] = [[] for _ in range(n)]
    for i, case in enumerate(case_paths):
        shards[i % n].append(str(case))
    return [x for x in shards if len(x) > 0]


def _run_rollout_shard_worker(
    case_paths: List[str],
    cfg_payload: Dict[str, Any],
    c_cfg_payload: Dict[str, Any],
    o_cfg_payload: Dict[str, Any],
    m_cfg_payload: Dict[str, Any],
    dims_payload: Dict[str, int],
    critic_kwargs: Dict[str, Any],
    model_state_payload: Dict[str, Dict[str, torch.Tensor]],
    reward_progress: float,
    m_entropy_signal: Optional[float] = None,
    worker_seed: Optional[int] = None,
) -> Dict[str, object]:
    # Keep each worker single-threaded to avoid CPU over-subscription when many workers are launched.
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    worker_cfg = OCMAPPOConfig(**cfg_payload)
    worker_device = str(getattr(worker_cfg, "rollout_worker_device", "cpu")).strip() or "cpu"
    worker_cfg.device = worker_device

    if bool(getattr(worker_cfg, "rollout_worker_reseed", True)) and worker_seed is not None:
        seed_everything(int(worker_seed))

    # Keep environment-side reward config aligned with trainer-side runtime config.
    env_configs.reward_balance_coef = float(worker_cfg.reward_balance_coef)
    env_configs.reward_wait_coef = float(worker_cfg.reward_wait_coef)
    env_configs.agent_c_active_job_rules = str(getattr(worker_cfg, "active_job_rules", "")).strip()
    env_configs.agent_c_active_machine_rules = str(getattr(worker_cfg, "active_machine_rules", "")).strip()

    c_cfg_local = AgentCConfig(**c_cfg_payload)
    c_cfg_local.device = worker_device
    o_cfg_local = AgentOConfig(**o_cfg_payload)
    o_cfg_local.device = worker_device
    m_cfg_local = AgentMLearnConfig(**m_cfg_payload)
    m_cfg_local.device = worker_device

    agent_c = Agent_C(
        config=c_cfg_local,
        global_dim=int(dims_payload["c_global_dim"]),
        pair_feat_dim=int(dims_payload["c_pair_feat_dim"]),
    )
    agent_o = Agent_O(
        config=o_cfg_local,
        global_dim=int(dims_payload["o_global_dim"]),
        op_feat_dim=int(dims_payload["o_op_feat_dim"]),
    )
    agent_m = Agent_M_Learn(
        config=m_cfg_local,
        global_dim=int(dims_payload["m_global_dim"]),
        pair_feat_dim=int(dims_payload["m_pair_feat_dim_raw"]),
    )

    critic = CentralCritic(**critic_kwargs).to(agent_c.device)

    agent_c.policy.load_state_dict(model_state_payload["agent_c"], strict=True)
    agent_o.policy.load_state_dict(model_state_payload["agent_o"], strict=True)
    agent_m.policy.load_state_dict(model_state_payload["agent_m"], strict=True)
    critic.load_state_dict(model_state_payload["critic"], strict=True)

    policy_mode = str(getattr(worker_cfg, "rollout_worker_policy_mode", "train")).strip().lower()
    if policy_mode == "eval":
        agent_c.policy.eval()
        agent_o.policy.eval()
        agent_m.policy.eval()
        critic.eval()
    else:
        # Keep rollout mode aligned with serial path by default.
        agent_c.policy.train()
        agent_o.policy.train()
        agent_m.policy.train()
        critic.train()

    shard_eps: List[EpisodeRollout] = []
    shard_errors: List[Dict[str, str]] = []
    rollout_sec = 0.0

    for case in case_paths:
        try:
            t0 = time.perf_counter()
            ep = run_episode_oc(
                agent_c=agent_c,
                agent_o=agent_o,
                agent_m=agent_m,
                critic=critic,
                case_path=str(case),
                cfg=worker_cfg,
                deterministic=False,
                collect_transitions=True,
                reward_progress=float(reward_progress),
                m_entropy_signal=m_entropy_signal,
                timing_meter=None,
                rule_meter=None,
            )
            rollout_sec += float(time.perf_counter() - t0)
            shard_eps.append(ep)
        except Exception as exc:
            shard_errors.append({"case": str(case), "reason": repr(exc)})

    return {
        "episodes": shard_eps,
        "errors": shard_errors,
        "rollout_sec": float(rollout_sec),
        "num_cases": int(len(case_paths)),
    }


def _collect_chunk_rollouts_mp(
    chunk: List[str],
    cfg: OCMAPPOConfig,
    c_cfg: AgentCConfig,
    o_cfg: AgentOConfig,
    m_cfg: AgentMLearnConfig,
    agent_c: Agent_C,
    agent_o: Agent_O,
    agent_m: Agent_M_Learn,
    critic: CentralCritic,
    reward_progress: float,
    m_entropy_signal: Optional[float],
    dims_payload: Dict[str, int],
    critic_kwargs: Dict[str, Any],
    executor: Optional[concurrent.futures.ProcessPoolExecutor] = None,
    rollout_seed_base: Optional[int] = None,
) -> Tuple[List[EpisodeRollout], List[Dict[str, str]], float]:
    if len(chunk) == 0:
        return [], [], 0.0

    auto_workers = int(max(1, min(int(max(1, cfg.num_envs)), int(mp.cpu_count()))))
    worker_n = int(cfg.rollout_workers) if int(cfg.rollout_workers) > 0 else auto_workers
    worker_n = int(max(1, min(worker_n, len(chunk))))
    min_cases_per_worker = int(max(1, getattr(cfg, "rollout_min_cases_per_worker", 2)))
    max_workers_by_cases = int(max(1, len(chunk) // min_cases_per_worker))
    worker_n = int(max(1, min(worker_n, max_workers_by_cases)))
    shards = _split_cases_into_shards(chunk, worker_n)

    cfg_payload = dataclasses.asdict(cfg)
    # Avoid sending large split lists to each worker task; workers only need per-episode runtime config.
    cfg_payload["train_cases"] = []
    cfg_payload["val_cases"] = []
    cfg_payload["test_cases"] = []

    c_cfg_payload = dataclasses.asdict(c_cfg)
    o_cfg_payload = dataclasses.asdict(o_cfg)
    m_cfg_payload = dataclasses.asdict(m_cfg)

    model_state_payload = {
        "agent_c": _state_dict_to_cpu(agent_c.policy.state_dict()),
        "agent_o": _state_dict_to_cpu(agent_o.policy.state_dict()),
        "agent_m": _state_dict_to_cpu(agent_m.policy.state_dict()),
        "critic": _state_dict_to_cpu(critic.state_dict()),
    }

    chunk_eps: List[EpisodeRollout] = []
    chunk_errors: List[Dict[str, str]] = []
    rollout_sec = 0.0

    def _submit_and_collect(ex: concurrent.futures.ProcessPoolExecutor) -> None:
        nonlocal rollout_sec
        future_to_shard_idx: Dict[concurrent.futures.Future, int] = {}
        for shard_idx, shard in enumerate(shards):
            shard_seed = None
            if rollout_seed_base is not None:
                shard_seed = int(rollout_seed_base) + int(shard_idx)
            fut = ex.submit(
                _run_rollout_shard_worker,
                shard,
                cfg_payload,
                c_cfg_payload,
                o_cfg_payload,
                m_cfg_payload,
                dims_payload,
                critic_kwargs,
                model_state_payload,
                float(reward_progress),
                m_entropy_signal,
                shard_seed,
            )
            future_to_shard_idx[fut] = int(shard_idx)

        shard_outputs: Dict[int, Dict[str, object]] = {}
        for fut in concurrent.futures.as_completed(list(future_to_shard_idx.keys())):
            shard_idx = int(future_to_shard_idx[fut])
            shard_outputs[shard_idx] = fut.result()

        for shard_idx in sorted(shard_outputs.keys()):
            out = shard_outputs[shard_idx]
            chunk_eps.extend(list(out.get("episodes", [])))
            chunk_errors.extend(list(out.get("errors", [])))
            rollout_sec += float(out.get("rollout_sec", 0.0))

    if executor is None:
        start_method = str(getattr(cfg, "rollout_mp_start_method", "spawn")).strip().lower() or "spawn"
        try:
            mp_ctx = mp.get_context(start_method)
        except ValueError:
            mp_ctx = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(shards), mp_context=mp_ctx) as ex:
            _submit_and_collect(ex)
    else:
        _submit_and_collect(executor)

    return chunk_eps, chunk_errors, float(rollout_sec)


def train_oc_mappo(cfg: OCMAPPOConfig) -> Dict[str, object]:
    scan_case_splits = _parse_case_splits(
        str(getattr(cfg, "scan_case_splits", "train,val")),
        arg_name="scan_case_splits",
        default=("train", "val"),
    )
    eval_case_splits = _parse_case_splits(
        str(getattr(cfg, "eval_case_splits", "val,test")),
        arg_name="eval_case_splits",
        default=("val", "test"),
    )

    if int(cfg.epochs) > 0:
        if len(cfg.train_cases) == 0:
            raise ValueError("train_cases cannot be empty when epochs > 0")
        if len(cfg.val_cases) == 0:
            raise ValueError("val_cases cannot be empty when epochs > 0")
    else:
        eval_pool_cases = _collect_cases_by_splits(cfg, eval_case_splits)
        if len(eval_pool_cases) == 0:
            raise ValueError("No cases available for requested eval_case_splits")

    train_wall_t0 = time.perf_counter()
    timing_meter = _new_timing_meter() if bool(cfg.profile_timing) else None
    rule_meter = _new_rule_meter() if bool(cfg.profile_rule_stats) else None

    seed_everything(int(cfg.seed))
    _print_split_summary(cfg)

    all_scan_cases = _collect_cases_by_splits(cfg, scan_case_splits)
    if len(all_scan_cases) == 0:
        all_scan_cases = _collect_cases_by_splits(cfg, eval_case_splits)
        if len(all_scan_cases) == 0:
            raise ValueError("No cases available for dimension scan")
        print(
            "[scan] requested scan_case_splits has no cases; fallback to eval_case_splits",
            flush=True,
        )
    t_scan_c0 = time.perf_counter()
    c_global_dim, pair_feat_dim, observed_max_candidates, c_scan = infer_training_dims(
        case_paths=all_scan_cases,
        reset_rule=cfg.reset_rule,
        use_candidate_set_feat=True,
        quiet=cfg.quiet_env,
        extra_step_budget=cfg.extra_step_budget,
        scan_rule=cfg.reset_rule,
        candidate_safety_margin=cfg.c_candidate_safety_margin,
        round_max_candidates_to_power_of_two=bool(cfg.c_round_max_candidates_to_power_of_two),
    )
    _add_timing(timing_meter, "scan_c_sec", time.perf_counter() - t_scan_c0)
    c_auto_detail = {
        "observed": int(observed_max_candidates),
        "fixed_margin": int(max(0, cfg.c_candidate_safety_margin)),
        "ratio_margin": 0,
        "effective_margin": int(max(0, cfg.c_candidate_safety_margin)),
        "min_candidates": int(max(1, cfg.c_min_max_candidates)),
    }
    if bool(cfg.c_auto_max_candidates):
        c_max_candidates, c_auto_detail = _resolve_auto_c_max_candidates(
            observed_max_candidates=int(observed_max_candidates),
            fixed_margin=int(cfg.c_candidate_safety_margin),
            margin_ratio=float(cfg.c_candidate_safety_margin_ratio),
            min_candidates=int(cfg.c_min_max_candidates),
            round_to_power_of_two=bool(cfg.c_round_max_candidates_to_power_of_two),
        )
    else:
        c_max_candidates = max(
            int(cfg.c_max_candidates),
            int(observed_max_candidates),
            int(max(1, cfg.c_min_max_candidates)),
        )

    full_soft_widen_active = bool(cfg.full_soft_widen_enabled) and bool(cfg.agent_c_two_stage_refine_enabled)
    if full_soft_widen_active:
        c_max_candidates = int(c_max_candidates + max(0, int(cfg.full_soft_widen_c_extra)))

    t_scan_o0 = time.perf_counter()
    o_global_dim, op_feat_dim, observed_max_ops = infer_o_dims(
        case_paths=all_scan_cases,
        reset_rule=cfg.reset_rule,
        quiet=cfg.quiet_env,
        use_candidate_set_feat=True,
    )
    _add_timing(timing_meter, "scan_o_sec", time.perf_counter() - t_scan_o0)
    if bool(cfg.o_auto_max_ops):
        o_max_ops = int(observed_max_ops + max(0, int(cfg.o_op_safety_margin)))
    else:
        o_max_ops = max(int(cfg.o_max_ops), int(observed_max_ops))

    if full_soft_widen_active:
        o_max_ops = int(o_max_ops + max(0, int(cfg.full_soft_widen_o_extra)))

    m_global_dim, m_pair_feat_dim_raw = infer_m_dims(
        case_paths=all_scan_cases,
        reset_rule=cfg.reset_rule,
        quiet=cfg.quiet_env,
    )

    c_cfg = AgentCConfig(
        device=cfg.device,
        hidden_dim=cfg.c_hidden_dim,
        attn_heads=cfg.c_attn_heads,
        attn_layers=cfg.c_attn_layers,
        dropout=cfg.c_dropout,
        lr=cfg.c_lr,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_ratio=cfg.clip_ratio_c,
        value_coef=cfg.value_coef_c,
        ent_coef=cfg.c_ent_coef,
        max_grad_norm=cfg.max_grad_norm,
        ppo_epochs=cfg.ppo_epochs,
        minibatch_size=cfg.minibatch_size,
        target_kl=cfg.target_kl_c,
        max_candidates=c_max_candidates,
        use_candidate_set_feat=True,
        use_graph_encoder=bool(cfg.c_use_graph_encoder),
        use_edge_rule_msg=bool(cfg.c_use_edge_rule_msg),
        use_edge_opmch_msg=bool(cfg.c_use_edge_opmch_msg),
        use_adaptive_edge_gates=bool(cfg.c_use_adaptive_edge_gates),
    )

    o_cfg = AgentOConfig(
        device=cfg.device,
        hidden_dim=cfg.o_hidden_dim,
        dropout=cfg.o_dropout,
        lr=cfg.o_lr,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_ratio=cfg.clip_ratio_o,
        value_coef=cfg.value_coef_o,
        ent_coef=cfg.o_ent_coef,
        max_grad_norm=cfg.max_grad_norm,
        ppo_epochs=cfg.ppo_epochs,
        minibatch_size=cfg.minibatch_size,
        target_kl=cfg.target_kl_o,
        max_ops=o_max_ops,
        use_candidate_set_feat=True,
        overflow_strategy="truncate",
        model_type=cfg.o_model_type,
    )

    m_cfg = AgentMLearnConfig(
        device=cfg.device,
        hidden_dim=cfg.m_hidden_dim,
        dropout=cfg.m_dropout,
        lr=cfg.m_lr,
        weight_decay=cfg.m_weight_decay,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_ratio=cfg.clip_ratio_m,
        value_coef=cfg.value_coef_m,
        ent_coef=cfg.m_ent_coef,
        max_grad_norm=cfg.max_grad_norm,
        ppo_epochs=cfg.ppo_epochs,
        minibatch_size=cfg.minibatch_size,
        target_kl=cfg.target_kl_m,
        feature_mode=cfg.m_feature_mode,
        strategy_mode=cfg.m_strategy_mode,
        enable_entropy_backup=bool(cfg.m_enable_entropy_backup),
        backup_entropy_threshold=cfg.m_backup_entropy_threshold,
        backup_max_extra_pairs=cfg.m_backup_max_extra_pairs,
    )

    agent_c = Agent_C(config=c_cfg, global_dim=c_global_dim, pair_feat_dim=pair_feat_dim)
    agent_o = Agent_O(config=o_cfg, global_dim=o_global_dim, op_feat_dim=op_feat_dim)
    agent_m = Agent_M_Learn(config=m_cfg, global_dim=m_global_dim, pair_feat_dim=m_pair_feat_dim_raw)

    if cfg.init_c_checkpoint_path:
        agent_c.load(cfg.init_c_checkpoint_path, strict=False)
    if cfg.init_o_checkpoint_path:
        agent_o.load(cfg.init_o_checkpoint_path, strict=False)
    if cfg.init_m_checkpoint_path:
        _load_agent_m_checkpoint(agent_m, cfg.init_m_checkpoint_path, strict=False)

    dummy_pc = {
        "global_feat": np.zeros((c_global_dim,), dtype=np.float32),
        "pair_feat": np.zeros((c_max_candidates, pair_feat_dim), dtype=np.float32),
        "pair_mask": np.ones((c_max_candidates,), dtype=bool),
    }
    dummy_po = {
        "global_feat": np.zeros((o_global_dim,), dtype=np.float32),
        "op_feat": np.zeros((o_max_ops, op_feat_dim), dtype=np.float32),
        "op_mask": np.ones((o_max_ops,), dtype=bool),
    }
    dummy_pm = {
        "global_feat": np.zeros((m_global_dim,), dtype=np.float32),
        "pair_feat": np.zeros((1, int(agent_m.policy_pair_feat_dim)), dtype=np.float32),
        "pair_mask": np.ones((1,), dtype=bool),
    }
    state_dim = int(build_central_state(dummy_pc, dummy_po, dummy_pm, [0], use_rich_stats=bool(cfg.critic_rich_state)).shape[0])

    critic = CentralCritic(
        state_dim=state_dim,
        hidden_dim=int(cfg.critic_hidden_dim),
        split_tower=bool(cfg.critic_split_tower),
        tower_hidden_dim=int(cfg.critic_tower_hidden_dim),
        c_pair_dim=int(pair_feat_dim),
        o_op_dim=int(op_feat_dim),
        m_pair_dim=int(agent_m.policy_pair_feat_dim),
        use_gnn_branch=bool(cfg.critic_use_gnn_branch),
        use_gate_fusion=bool(cfg.critic_use_gate_fusion),
        gnn_hidden_dim=int(cfg.critic_gnn_hidden_dim),
        gnn_heads=int(cfg.critic_gnn_heads),
    ).to(agent_c.device)
    if str(getattr(cfg, "init_critic_checkpoint_path", "")).strip():
        _cp = str(cfg.init_critic_checkpoint_path).strip()
        payload_cr = _torch_load_compat(_cp, map_location=agent_c.device)
        critic.load_state_dict(payload_cr["state_dict"], strict=True)
    critic_kwargs = {
        "state_dim": int(state_dim),
        "hidden_dim": int(cfg.critic_hidden_dim),
        "split_tower": bool(cfg.critic_split_tower),
        "tower_hidden_dim": int(cfg.critic_tower_hidden_dim),
        "c_pair_dim": int(pair_feat_dim),
        "o_op_dim": int(op_feat_dim),
        "m_pair_dim": int(agent_m.policy_pair_feat_dim),
        "use_gnn_branch": bool(cfg.critic_use_gnn_branch),
        "use_gate_fusion": bool(cfg.critic_use_gate_fusion),
        "gnn_hidden_dim": int(cfg.critic_gnn_hidden_dim),
        "gnn_heads": int(cfg.critic_gnn_heads),
    }
    rollout_dims = {
        "c_global_dim": int(c_global_dim),
        "c_pair_feat_dim": int(pair_feat_dim),
        "o_global_dim": int(o_global_dim),
        "o_op_feat_dim": int(op_feat_dim),
        "m_global_dim": int(m_global_dim),
        "m_pair_feat_dim_raw": int(m_pair_feat_dim_raw),
    }
    critic_optimizer = optim.AdamW(critic.parameters(), lr=float(cfg.critic_lr), weight_decay=1e-4)

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    c_ckpt_path = save_dir / cfg.c_checkpoint_name
    o_ckpt_path = save_dir / cfg.o_checkpoint_name
    m_ckpt_path = save_dir / cfg.m_checkpoint_name
    critic_ckpt_path = save_dir / cfg.critic_checkpoint_name
    c_latest_ckpt_path = save_dir / cfg.c_latest_checkpoint_name
    o_latest_ckpt_path = save_dir / cfg.o_latest_checkpoint_name
    m_latest_ckpt_path = save_dir / cfg.m_latest_checkpoint_name
    critic_latest_ckpt_path = save_dir / cfg.critic_latest_checkpoint_name
    c_final_ckpt_path = save_dir / cfg.c_final_checkpoint_name
    o_final_ckpt_path = save_dir / cfg.o_final_checkpoint_name
    m_final_ckpt_path = save_dir / cfg.m_final_checkpoint_name
    critic_final_ckpt_path = save_dir / cfg.critic_final_checkpoint_name
    progress_json_path = save_dir / cfg.progress_json_name
    history_jsonl_path = save_dir / cfg.history_jsonl_name
    train_state_path = save_dir / cfg.train_state_name

    best_val_makespan = float("inf")
    best_epoch = 0
    history: List[Dict[str, object]] = []
    start_epoch = 1

    if cfg.resume_state_path:
        resume_path = Path(cfg.resume_state_path)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume_state_path not found: {resume_path}")
        resume_meta = _load_latest_train_state(
            train_state_path=resume_path,
            agent_c=agent_c,
            agent_o=agent_o,
            agent_m=agent_m,
            critic=critic,
            critic_optimizer=critic_optimizer,
        )
        best_val_makespan = float(resume_meta["best_val_makespan"])
        best_epoch = int(resume_meta["best_epoch"])
        history = list(resume_meta["history"])
        start_epoch = int(resume_meta["epoch"]) + 1
        print(
            f"[resume] from={resume_path}, start_epoch={start_epoch}, best_val={best_val_makespan:.4f}@{best_epoch}, "
            f"history_rows={len(history)}",
            flush=True,
        )

    if int(start_epoch) <= 1 and bool(getattr(cfg, "save_history_jsonl", True)) and history_jsonl_path.exists():
        history_jsonl_path.unlink()

    print(
        f"[init-oc-mappo] c_global={c_global_dim}, pair_feat={pair_feat_dim}, c_max_candidates={c_max_candidates}, "
        f"c_obs={c_auto_detail['observed']}, c_margin={c_auto_detail['effective_margin']}, "
        f"c_min={c_auto_detail['min_candidates']}, c_round2pow={int(bool(cfg.c_round_max_candidates_to_power_of_two))}, "
        f"o_global={o_global_dim}, op_feat={op_feat_dim}, o_max_ops={o_max_ops}, "
        f"m_global={m_global_dim}, m_pair_feat_raw={m_pair_feat_dim_raw}, m_pair_feat_policy={agent_m.policy_pair_feat_dim}, "
        f"eval_policy={'deterministic' if bool(cfg.eval_deterministic) else 'sampling'}, "
        f"eval_sample_times={int(max(1, cfg.eval_sample_times))}, eval_sample_reduce={str(cfg.eval_sample_reduce)}, "
        f"det_refine_topk={int(max(1, cfg.deterministic_refine_topk))}, "
        f"det_refine_logit_gap={float(cfg.deterministic_refine_logit_gap):.4f}, "
        f"det_refine_min_ect_gain={float(cfg.deterministic_refine_min_ect_gain):.4f}, "
        f"det_valonly_gap_candidates='{str(cfg.deterministic_val_only_gap_candidates)}', "
        f"train_eval_force_det={int(bool(cfg.train_eval_force_deterministic))}, "
        f"val_only_best_strategy={int(bool(cfg.val_only_best_strategy))}, "
        f"val_only_hard_topk={int(max(0, cfg.val_only_hard_case_topk))}, "
        f"val_only_extra={int(max(0, cfg.val_only_extra_samples_per_case))}, "
        f"val_only_adaptive={int(bool(cfg.val_only_adaptive_extra_sampling))}, "
        f"adaptive_var_bonus={float(cfg.val_only_adaptive_var_bonus):.3f}, "
        f"adaptive_ucb_bonus={float(cfg.val_only_adaptive_ucb_bonus):.3f}, "
        f"num_envs={int(max(1, getattr(cfg, 'num_envs', cfg.episodes_per_update)))}, "
        f"rollout_backend={str(cfg.rollout_backend)}, rollout_workers={int(cfg.rollout_workers)}, "
        f"rollout_worker_device={str(cfg.rollout_worker_device)}, rollout_mp_start={str(cfg.rollout_mp_start_method)}, "
        f"rollout_worker_mode={str(cfg.rollout_worker_policy_mode)}, rollout_worker_reseed={int(bool(cfg.rollout_worker_reseed))}, "
        f"p1_fast_skip_c_o_ref={int(bool(getattr(cfg, 'p1_fast_skip_c_o_ref', True)))}, "
        f"p1_skip_c_full_ref={int(bool(getattr(cfg, 'p1_skip_c_full_ref', False)))}, "
        f"o_update_interval={int(max(1, getattr(cfg, 'o_update_interval', 1)))}, "
        f"m_update_interval={int(max(1, getattr(cfg, 'm_update_interval', 1)))}, "
        f"o_redundancy_target={float(getattr(cfg, 'o_redundancy_target_ratio', 0.35)):.3f}, "
        f"o_redundancy_coef={float(getattr(cfg, 'o_redundancy_penalty_coef', 0.05)):.3f}, "
        f"o_coverage_coef={float(getattr(cfg, 'o_coverage_bonus_coef', 0.02)):.3f}, "
        f"o_set_aux_coef={float(getattr(cfg, 'o_set_aux_coef', 0.0)):.3f}, "
        f"o_set_ppo_mix={float(getattr(cfg, 'o_set_ppo_mix', 0.35)):.3f}, "
        f"reward_version={str(getattr(cfg, 'reward_version', 'role_aligned_v1'))}, "
        f"reward_schedule_anneal_epochs={int(getattr(cfg, 'reward_schedule_anneal_epochs', 100))}, "
        f"o_instab_monitor={int(bool(getattr(cfg, 'o_instability_monitor_enabled', True)))}, "
        f"o_instab_kl={float(getattr(cfg, 'o_instability_kl_threshold', 0.04)):.4f}, "
        f"o_instab_guard={float(getattr(cfg, 'o_instability_guard_threshold', 0.10)):.3f}, "
        f"o_instab_ent={float(getattr(cfg, 'o_instability_entropy_threshold', 1.75)):.3f}, "
        f"o_batch_protect={int(bool(getattr(cfg, 'o_batch_protect_enabled', True)))}, "
        f"o_ratio_guard={float(getattr(cfg, 'o_ratio_batch_soft_limit', 0.40)):.2f}/"
        f"{float(getattr(cfg, 'o_ratio_batch_hard_limit', 0.80)):.2f}/"
        f"{float(getattr(cfg, 'o_ratio_batch_soft_scale', 0.5)):.2f}, "
        f"o_teacher_min={float(getattr(cfg, 'o_teacher_coef_min', 0.0)):.3f}, "
        f"m_teacher_min={float(getattr(cfg, 'm_teacher_coef_min', 0.0)):.3f}, "
        f"o_consensus_coef={float(getattr(cfg, 'o_consensus_coef', 0.02)):.3f}->{float(getattr(cfg, 'o_consensus_coef_end', 0.0)):.3f}, "
        f"m_consensus_coef={float(getattr(cfg, 'm_consensus_coef', 0.01)):.3f}->{float(getattr(cfg, 'm_consensus_coef_end', 0.0)):.3f}, "
        f"m_entropy_fb={int(bool(getattr(cfg, 'm_entropy_feedback_enabled', True)))}, "
        f"m_entropy_thr={float(getattr(cfg, 'm_entropy_feedback_threshold', 0.50)):.3f}, "
        f"m_entropy_pow={float(getattr(cfg, 'm_entropy_feedback_power', 1.25)):.2f}, "
        f"m_entropy_min={float(getattr(cfg, 'm_entropy_feedback_shape_min_scale', 0.90)):.2f}/"
        f"{float(getattr(cfg, 'm_entropy_feedback_teacher_min_scale', 0.20)):.2f}/"
        f"{float(getattr(cfg, 'm_entropy_feedback_consensus_min_scale', 0.35)):.2f}, "
        f"ocm_parallel={int(bool(getattr(cfg, 'ocm_parallel_om_enabled', False)))}, "
        f"fusion_mode={str(getattr(cfg, 'ocm_parallel_fusion_mode', 'o_plus_m_backup'))}, "
        f"parallel_min_pairs={int(max(1, getattr(cfg, 'ocm_parallel_min_final_pairs', 3)))}, "
        f"parallel_budget_extra={int(max(0, getattr(cfg, 'ocm_parallel_budget_extra_pairs', 2)))}, "
        f"parallel_max_final={int(max(0, getattr(cfg, 'ocm_parallel_max_final_pairs', 0)))}, "
        f"parallel_m_teacher={str(getattr(cfg, 'ocm_parallel_m_teacher_mode', 'full_ref'))}, "
        f"parallel_rescue_ops={int(max(0, getattr(cfg, 'ocm_parallel_rescue_ops', 2)))}, "
        f"parallel_rescue_pairs_per_op={int(max(1, getattr(cfg, 'ocm_parallel_rescue_pairs_per_op', 2)))}, "
        f"save_dir='{str(save_dir)}', "
        f"train_state='{str(train_state_path)}', "
        f"state_dim={state_dim}, "
        f"critic_gnn={int(bool(cfg.critic_use_gnn_branch))}, gate={int(bool(cfg.critic_use_gate_fusion))}, "
        f"freeze_epochs={int(cfg.critic_freeze_gnn_epochs)}"
        ,
        flush=True,
    )

    if _rollout_backend_is_mp(cfg) and (bool(cfg.profile_timing) or bool(cfg.profile_rule_stats)):
        print(
            "[rollout-mp] profile_timing/profile_rule_stats are collected only in serial rollout; "
            "mp rollout workers return aggregate rollout seconds and errors.",
            flush=True,
        )

    mp_executor_train: Optional[concurrent.futures.ProcessPoolExecutor] = None
    chunk_size_train = int(max(1, getattr(cfg, "num_envs", cfg.episodes_per_update)))
    if _rollout_backend_is_mp(cfg) and len(cfg.train_cases) > 1:
        auto_workers = int(max(1, min(int(max(1, cfg.num_envs)), int(mp.cpu_count()))))
        pool_workers = int(cfg.rollout_workers) if int(cfg.rollout_workers) > 0 else auto_workers
        pool_workers = int(max(1, min(pool_workers, chunk_size_train, len(cfg.train_cases))))
        start_method = str(getattr(cfg, "rollout_mp_start_method", "spawn")).strip().lower() or "spawn"
        try:
            mp_ctx = mp.get_context(start_method)
        except ValueError:
            mp_ctx = mp.get_context("spawn")
        try:
            mp_executor_train = concurrent.futures.ProcessPoolExecutor(max_workers=pool_workers, mp_context=mp_ctx)
            print(
                f"[rollout-mp] persistent_pool=1 workers={int(pool_workers)} "
                f"min_cases_per_worker={int(cfg.rollout_min_cases_per_worker)}",
                flush=True,
            )
        except Exception as mp_pool_exc:
            print(
                f"[rollout-mp-warning] failed to create persistent worker pool with {repr(mp_pool_exc)}; "
                "fallback to serial rollout.",
                flush=True,
            )

    update_step = 0
    c_lr_runtime_scale = 1.0
    o_lr_runtime_scale = 1.0
    m_lr_runtime_scale = 1.0
    o_update_interval_runtime = int(max(1, int(getattr(cfg, "o_update_interval", 1))))
    o_reward_term_runtime_scale = 1.0
    o_aux_runtime_scale = 1.0
    val_no_improve_streak = 0
    last_plateau_restore_epoch = -10**9
    m_entropy_feedback_signal: Optional[float] = None
    if len(history) > 0 and ("update_entropy_m" in history[-1]):
        m_entropy_feedback_signal = float(history[-1]["update_entropy_m"])

    for epoch in range(start_epoch, int(cfg.epochs) + 1):
        if bool(cfg.critic_use_gnn_branch):
            gnn_trainable = not (int(cfg.critic_freeze_gnn_epochs) > 0 and int(epoch) <= int(cfg.critic_freeze_gnn_epochs))
            critic.set_gnn_trainable(gnn_trainable)

        # Keep actor learning rates controllable by runtime decay.
        c_lr_effective = float(cfg.c_lr) * float(c_lr_runtime_scale)
        o_lr_effective = float(cfg.o_lr) * float(o_lr_runtime_scale)
        m_lr_effective = float(cfg.m_lr) * float(m_lr_runtime_scale)
        for pg in agent_c.optimizer.param_groups:
            pg["lr"] = float(c_lr_effective)
        for pg in agent_o.optimizer.param_groups:
            pg["lr"] = float(o_lr_effective)
        for pg in agent_m.optimizer.param_groups:
            pg["lr"] = float(m_lr_effective)

        epoch_rollout_sec = 0.0
        epoch_update_sec = 0.0
        epoch_eval_sec = 0.0
        epoch_ckpt_sec = 0.0

        reward_progress = _resolve_reward_progress(cfg=cfg, epoch=epoch)
        reward_ctrl = _resolve_reward_coeffs(
            cfg=cfg,
            reward_progress=reward_progress,
            c_max_candidates_effective=int(agent_c.config.max_candidates),
            o_max_ops_effective=int(agent_o.config.max_ops),
            m_entropy_signal=m_entropy_feedback_signal,
            o_reward_term_runtime_scale=o_reward_term_runtime_scale,
        )
        train_cases = list(cfg.train_cases)
        random.Random(int(cfg.seed) + epoch).shuffle(train_cases)

        train_eps: List[EpisodeRollout] = []
        train_errors = []
        update_logs = []

        chunk_size = int(max(1, getattr(cfg, "num_envs", cfg.episodes_per_update)))
        mp_executor_epoch = mp_executor_train

        for start in range(0, len(train_cases), chunk_size):
            chunk = train_cases[start : start + chunk_size]
            chunk_eps = []
            used_mp_chunk = False

            if mp_executor_epoch is not None and len(chunk) > 1:
                try:
                    mp_eps, mp_errors, mp_rollout_sec = _collect_chunk_rollouts_mp(
                        chunk=chunk,
                        cfg=cfg,
                        c_cfg=c_cfg,
                        o_cfg=o_cfg,
                        m_cfg=m_cfg,
                        agent_c=agent_c,
                        agent_o=agent_o,
                        agent_m=agent_m,
                        critic=critic,
                        reward_progress=reward_progress,
                        m_entropy_signal=m_entropy_feedback_signal,
                        dims_payload=rollout_dims,
                        critic_kwargs=critic_kwargs,
                        executor=mp_executor_epoch,
                            rollout_seed_base=(int(cfg.seed) * 1000003 + int(epoch) * 10007 + int(start) * 97),
                    )
                    chunk_eps.extend(mp_eps)
                    train_eps.extend(mp_eps)
                    train_errors.extend(mp_errors)
                    epoch_rollout_sec += float(mp_rollout_sec)
                    used_mp_chunk = True
                except Exception as mp_exc:
                    print(
                        f"[rollout-mp-warning] chunk_start={int(start)} failed with {repr(mp_exc)}; "
                        "fallback to serial rollout for this chunk.",
                        flush=True,
                    )

            if not used_mp_chunk:
                for case in chunk:
                    try:
                        t_rollout_case0 = time.perf_counter()
                        ep = run_episode_oc(
                            agent_c=agent_c,
                            agent_o=agent_o,
                            agent_m=agent_m,
                            critic=critic,
                            case_path=case,
                            cfg=cfg,
                            deterministic=False,
                            collect_transitions=True,
                            reward_progress=reward_progress,
                            m_entropy_signal=m_entropy_feedback_signal,
                            timing_meter=timing_meter,
                            rule_meter=rule_meter,
                        )
                        epoch_rollout_sec += float(time.perf_counter() - t_rollout_case0)
                        train_eps.append(ep)
                        chunk_eps.append(ep)
                    except Exception as exc:
                        train_errors.append({"case": str(case), "reason": repr(exc)})

            if len(chunk_eps) > 0:
                t_update0 = time.perf_counter()
                update_step += 1
                o_update_interval_eff = int(o_update_interval_runtime)
                o_update_weight_eff_chunk = 1.0
                if bool(getattr(cfg, "o_smooth_update_enabled", True)) and o_update_interval_eff > 1:
                    o_update_weight_eff_chunk = float(max(
                        float(getattr(cfg, "o_smooth_update_min_weight", 0.20)),
                        1.0 / float(max(1, o_update_interval_eff)),
                    ))
                    o_update_interval_eff = 1
                logs = update_mappo(
                    agent_c=agent_c,
                    agent_o=agent_o,
                    agent_m=agent_m,
                    critic=critic,
                    critic_optimizer=critic_optimizer,
                    episodes=chunk_eps,
                    cfg=cfg,
                    update_step=update_step,
                    o_update_interval_override=int(o_update_interval_eff),
                    o_aux_coef_scale=float(o_aux_runtime_scale),
                    o_update_weight=float(o_update_weight_eff_chunk),
                )
                epoch_update_sec += float(time.perf_counter() - t_update0)
                update_logs.append(logs)

        _add_timing(timing_meter, "rollout_total_sec", epoch_rollout_sec)
        _add_timing(timing_meter, "update_total_sec", epoch_update_sec)

        row = {
            "epoch": int(epoch),
            "train_num_success": int(len(train_eps)),
            "train_num_errors": int(len(train_errors)),
            "train_done_ratio": float(np.mean([float(x.done) for x in train_eps])) if train_eps else 0.0,
            "train_mean_reward": float(np.mean([x.episode_reward for x in train_eps])) if train_eps else 0.0,
            "train_mean_o_reward": float(np.mean([x.episode_o_reward for x in train_eps])) if train_eps else 0.0,
            "train_mean_m_reward": float(np.mean([x.episode_m_reward for x in train_eps])) if train_eps else 0.0,
            "train_mean_makespan": float(np.mean([x.makespan for x in train_eps])) if train_eps else float("inf"),
            "train_mean_o_selected_ops": float(np.mean([x.o_avg_selected_ops for x in train_eps])) if train_eps else 0.0,
            "train_o_fallback_rate": float(np.mean([x.o_fallback_rate for x in train_eps])) if train_eps else 0.0,
            "train_o_invalid_op_ratio": float(np.mean([x.o_invalid_op_ratio for x in train_eps])) if train_eps else 0.0,
            "train_o_shape_term": float(np.mean([x.o_avg_shape_term for x in train_eps])) if train_eps else 0.0,
            "train_o_teacher_term": float(np.mean([x.o_avg_teacher_term for x in train_eps])) if train_eps else 0.0,
            "train_o_consensus_term": float(np.mean([x.o_avg_consensus_term for x in train_eps])) if train_eps else 0.0,
            "train_mean_m_keep_pairs": float(np.mean([x.m_avg_keep_pairs for x in train_eps])) if train_eps else 0.0,
            "train_parallel_m_scope_pairs": float(np.mean([x.parallel_avg_m_scope_pairs for x in train_eps])) if train_eps else 0.0,
            "train_parallel_final_pairs": float(np.mean([x.parallel_avg_final_pairs for x in train_eps])) if train_eps else 0.0,
            "train_parallel_rescue_pairs": float(np.mean([x.parallel_avg_rescue_pairs for x in train_eps])) if train_eps else 0.0,
            "train_m_keep_per_o_ratio": (
                float(np.mean([x.m_avg_keep_pairs for x in train_eps]))
                / max(float(np.mean([x.o_avg_selected_ops for x in train_eps])), 1e-6)
            )
            if train_eps
            else 0.0,
            "train_m_adopt_rate": float(np.mean([x.m_adopt_rate for x in train_eps])) if train_eps else 0.0,
            "train_c_forward_per_step": float(np.mean([x.c_forward_per_step for x in train_eps])) if train_eps else 0.0,
            "update_loss_c": float(np.mean([x.get("loss_c", 0.0) for x in update_logs])) if update_logs else 0.0,
            "update_o_active_chunks": int(sum(1 for x in update_logs if float(x.get("o_metrics_valid", 0.0)) > 0.0)),
            "update_loss_o": _mean_log_value(update_logs, "loss_o", active_key="o_metrics_valid", default=0.0),
            "update_loss_o_aux": _mean_log_value(update_logs, "loss_o_aux", active_key="o_metrics_valid", default=0.0),
            "update_o_set_aux_coef_eff": _mean_log_value(
                update_logs,
                "o_set_aux_coef_eff",
                active_key="o_metrics_valid",
                default=float(getattr(cfg, "o_set_aux_coef", 0.0)) * float(o_aux_runtime_scale),
            ),
            "update_loss_m": float(np.mean([x.get("loss_m", 0.0) for x in update_logs])) if update_logs else 0.0,
            "update_m_ent_coef_eff": float(np.mean([x.get("m_ent_coef_eff", 0.0) for x in update_logs])) if update_logs else float(getattr(cfg, "m_ent_coef", 0.0)),
            "update_value_loss": float(np.mean([x.get("value_loss", 0.0) for x in update_logs])) if update_logs else 0.0,
            "update_value_loss_c": float(np.mean([x.get("value_loss_c", 0.0) for x in update_logs])) if update_logs else 0.0,
            "update_value_loss_o": _mean_log_value(update_logs, "value_loss_o", active_key="o_metrics_valid", default=0.0),
            "update_value_loss_m": float(np.mean([x.get("value_loss_m", 0.0) for x in update_logs])) if update_logs else 0.0,
            "update_entropy_c": float(np.mean([x.get("entropy_c", 0.0) for x in update_logs])) if update_logs else 0.0,
            "update_entropy_o": _mean_log_value(update_logs, "entropy_o", active_key="o_metrics_valid", default=0.0),
            "update_entropy_m": float(np.mean([x.get("entropy_m", 0.0) for x in update_logs])) if update_logs else 0.0,
            "update_kl_c": float(np.mean([x.get("approx_kl_c", 0.0) for x in update_logs])) if update_logs else 0.0,
            "update_kl_o": _mean_log_value(update_logs, "approx_kl_o", active_key="o_metrics_valid", default=0.0),
            "update_kl_m": float(np.mean([x.get("approx_kl_m", 0.0) for x in update_logs])) if update_logs else 0.0,
            "update_kl_guard_c": float(np.mean([x.get("minibatch_kl_guard_c", 0.0) for x in update_logs])) if update_logs else 0.0,
            "update_kl_guard_o": _mean_log_value(update_logs, "minibatch_kl_guard_o", active_key="o_metrics_valid", default=0.0),
            "update_kl_guard_m": float(np.mean([x.get("minibatch_kl_guard_m", 0.0) for x in update_logs])) if update_logs else 0.0,
            "update_o_batch_soft_protect": _mean_log_value(update_logs, "o_batch_soft_protect", active_key="o_metrics_valid", default=0.0),
            "update_o_batch_hard_protect": _mean_log_value(update_logs, "o_batch_hard_protect", active_key="o_metrics_valid", default=0.0),
            "update_o_batch_update_weight": _mean_log_value(update_logs, "o_batch_update_weight", active_key="o_metrics_valid", default=1.0),
            # Backward-compatible aliases for O-side reward schedule.
            "reward_alpha_env": float(reward_ctrl["o_alpha_env"]),
            "reward_beta_shape": float(reward_ctrl["o_beta_shape"]),
            "reward_version": str(reward_ctrl["version"]),
            "o_reward_alpha_env": float(reward_ctrl["o_alpha_env"]),
            "o_reward_beta_shape": float(reward_ctrl["o_beta_shape"]),
            "m_reward_alpha_env": float(reward_ctrl["m_alpha_env"]),
            "m_reward_beta_shape": float(reward_ctrl["m_beta_shape"]),
            "o_teacher_coef": float(reward_ctrl["o_teacher_coef"]),
            "m_teacher_coef": float(reward_ctrl["m_teacher_coef"]),
            "o_consensus_coef": float(reward_ctrl["o_consensus_coef"]),
            "m_consensus_coef": float(reward_ctrl["m_consensus_coef"]),
            "m_candidate_scale": float(reward_ctrl.get("m_candidate_scale", 1.0)),
            "m_entropy_signal": (
                float(reward_ctrl["m_entropy_signal"])
                if reward_ctrl.get("m_entropy_signal", None) is not None
                else -1.0
            ),
            "m_shape_entropy_scale": float(reward_ctrl.get("m_shape_entropy_scale", 1.0)),
            "m_teacher_entropy_scale": float(reward_ctrl.get("m_teacher_entropy_scale", 1.0)),
            "m_consensus_entropy_scale": float(reward_ctrl.get("m_consensus_entropy_scale", 1.0)),
            "o_lr_runtime_scale": float(o_lr_runtime_scale),
            "o_update_interval_runtime": int(o_update_interval_runtime),
            "o_reward_term_runtime_scale": float(o_reward_term_runtime_scale),
            "o_aux_runtime_scale": float(o_aux_runtime_scale),
            "critic_gnn_trainable": int(not (bool(cfg.critic_use_gnn_branch) and int(cfg.critic_freeze_gnn_epochs) > 0 and int(epoch) <= int(cfg.critic_freeze_gnn_epochs))),
        }
        if bool(cfg.profile_timing):
            row["epoch_rollout_sec"] = float(epoch_rollout_sec)
            row["epoch_update_sec"] = float(epoch_update_sec)

        if epoch % int(max(1, cfg.eval_interval)) == 0:
            t_eval0 = time.perf_counter()
            train_eval_force_det = bool(getattr(cfg, "train_eval_force_deterministic", True)) and int(cfg.epochs) > 0
            val_eval = evaluate_oc(
                agent_c,
                agent_o,
                agent_m,
                critic,
                cfg.val_cases,
                cfg,
                eval_deterministic_override=(True if train_eval_force_det else None),
                eval_sample_times_override=(1 if train_eval_force_det else None),
                val_only_best_strategy_override=(False if train_eval_force_det else None),
            )
            epoch_eval_sec += float(time.perf_counter() - t_eval0)
            row["val_mean_reward"] = float(val_eval["mean_reward"])
            row["val_mean_makespan"] = float(val_eval["mean_makespan"])
            row["val_num_errors"] = int(val_eval["num_errors"])

            if float(val_eval["mean_makespan"]) < best_val_makespan:
                best_val_makespan = float(val_eval["mean_makespan"])
                best_epoch = int(epoch)
                val_no_improve_streak = 0
                t_ckpt0 = time.perf_counter()
                agent_c.save(str(c_ckpt_path))
                agent_o.save(str(o_ckpt_path))
                agent_m.save(str(m_ckpt_path))
                torch.save(
                    {
                        "state_dim": int(state_dim),
                        "hidden_dim": int(cfg.critic_hidden_dim),
                        "split_tower": int(bool(cfg.critic_split_tower)),
                        "tower_hidden_dim": int(cfg.critic_tower_hidden_dim),
                        "use_gnn_branch": int(bool(cfg.critic_use_gnn_branch)),
                        "use_gate_fusion": int(bool(cfg.critic_use_gate_fusion)),
                        "gnn_hidden_dim": int(cfg.critic_gnn_hidden_dim),
                        "gnn_heads": int(cfg.critic_gnn_heads),
                        "freeze_gnn_epochs": int(cfg.critic_freeze_gnn_epochs),
                        "state_dict": critic.state_dict(),
                    },
                    str(critic_ckpt_path),
                )
                epoch_ckpt_sec += float(time.perf_counter() - t_ckpt0)
                row["best_updated"] = 1
            else:
                row["best_updated"] = 0
                val_no_improve_streak += 1
                patience = int(max(1, int(getattr(cfg, "val_plateau_patience", 2))))
                if val_no_improve_streak >= patience:
                    decay = float(np.clip(float(getattr(cfg, "val_plateau_decay", 0.7)), 0.1, 1.0))
                    min_scale = float(np.clip(float(getattr(cfg, "val_plateau_min_scale", 0.2)), 0.01, 1.0))
                    if bool(getattr(cfg, "val_plateau_apply_c", True)):
                        c_lr_runtime_scale = _apply_monotonic_decay(
                            scale=float(c_lr_runtime_scale),
                            decay=decay,
                            min_scale=min_scale,
                        )
                    if bool(getattr(cfg, "val_plateau_apply_o", True)):
                        o_lr_runtime_scale = _apply_monotonic_decay(
                            scale=float(o_lr_runtime_scale),
                            decay=decay,
                            min_scale=min_scale,
                        )
                    if bool(getattr(cfg, "val_plateau_apply_m", True)):
                        m_lr_runtime_scale = _apply_monotonic_decay(
                            scale=float(m_lr_runtime_scale),
                            decay=decay,
                            min_scale=min_scale,
                        )

                    restored = False
                    restore_reason = "cooldown_or_small_gap"
                    val_gap_abs = float(val_eval["mean_makespan"]) - float(best_val_makespan)
                    val_gap_rel = float(val_gap_abs / max(abs(float(best_val_makespan)), 1e-6))
                    restore_cooldown = int(max(0, int(getattr(cfg, "val_plateau_restore_cooldown", 2))))
                    restore_rel_gap = float(max(0.0, float(getattr(cfg, "val_plateau_restore_rel_gap", 0.03))))
                    restore_abs_gap = float(max(0.0, float(getattr(cfg, "val_plateau_restore_abs_gap", 0.0))))
                    restore_gap_ok = (
                        float(val_gap_abs) >= float(restore_abs_gap)
                        and float(val_gap_rel) >= float(restore_rel_gap)
                    )
                    restore_cooldown_ok = (int(epoch) - int(last_plateau_restore_epoch)) > int(restore_cooldown)
                    if (
                        bool(getattr(cfg, "val_plateau_restore_best", True))
                        and int(epoch) >= int(max(1, int(getattr(cfg, "val_plateau_restore_min_epoch", 3))))
                        and int(best_epoch) > 0
                        and bool(restore_gap_ok)
                        and bool(restore_cooldown_ok)
                    ):
                        try:
                            if c_ckpt_path.exists():
                                agent_c.load(str(c_ckpt_path), strict=False)
                            if o_ckpt_path.exists():
                                agent_o.load(str(o_ckpt_path), strict=False)
                            if m_ckpt_path.exists():
                                _load_agent_m_checkpoint(agent_m, str(m_ckpt_path), strict=False)
                            if critic_ckpt_path.exists():
                                payload_best = _torch_load_compat(str(critic_ckpt_path), map_location=agent_c.device)
                                critic.load_state_dict(payload_best["state_dict"], strict=True)
                            restored = True
                            restore_reason = "restored_best"
                            last_plateau_restore_epoch = int(epoch)
                        except Exception as restore_exc:
                            restore_reason = "restore_error"
                            print(
                                f"[intervene-val-plateau-restore-warning] epoch={epoch} err={repr(restore_exc)}",
                                flush=True,
                            )
                    elif not bool(getattr(cfg, "val_plateau_restore_best", True)):
                        restore_reason = "restore_disabled"
                    elif int(epoch) < int(max(1, int(getattr(cfg, "val_plateau_restore_min_epoch", 3)))):
                        restore_reason = "before_restore_min_epoch"
                    elif int(best_epoch) <= 0:
                        restore_reason = "no_best_checkpoint"
                    elif not bool(restore_gap_ok):
                        restore_reason = "gap_below_threshold"

                    val_no_improve_streak = 0
                    print(
                        f"[intervene-val-plateau] epoch={epoch} "
                        f"lr_scale(c/o/m)={c_lr_runtime_scale:.4f}/{o_lr_runtime_scale:.4f}/{m_lr_runtime_scale:.4f}, "
                        f"restore_best={int(restored)}, restore_reason={restore_reason}, "
                        f"gap_abs={val_gap_abs:.4f}, gap_rel={val_gap_rel:.4f}",
                        flush=True,
                    )

            print(
                f"[epoch {epoch:03d}] "
                f"train_ms={row['train_mean_makespan']:.4f}, train_r={row['train_mean_reward']:.4f}, "
                f"train_or={row['train_mean_o_reward']:.4f}, train_mr={row['train_mean_m_reward']:.4f}, done={row['train_done_ratio']:.3f}, "
                f"loss_c={row['update_loss_c']:.5f}, loss_o={row['update_loss_o']:.5f}, loss_m={row['update_loss_m']:.5f}, "
                f"vloss_c={row['update_value_loss_c']:.5f}, vloss_o={row['update_value_loss_o']:.5f}, vloss_m={row['update_value_loss_m']:.5f}, "
                f"ent_c={row['update_entropy_c']:.5f}, ent_o={row['update_entropy_o']:.5f}, ent_m={row['update_entropy_m']:.5f}, "
                f"kl_c={row['update_kl_c']:.5f}, kl_o={row['update_kl_o']:.5f}, kl_m={row['update_kl_m']:.5f}, "
                f"rv={row['reward_version']}, "
                f"o_env={row['o_reward_alpha_env']:.3f}, o_shape={row['o_reward_beta_shape']:.3f}, "
                f"m_env={row['m_reward_alpha_env']:.3f}, m_shape={row['m_reward_beta_shape']:.3f}, "
                f"o_t={row['o_teacher_coef']:.3f}, m_t={row['m_teacher_coef']:.3f}, "
                f"o_cs={row['o_consensus_coef']:.3f}, m_cs={row['m_consensus_coef']:.3f}, "
                f"o_rt={row['o_lr_runtime_scale']:.2f}/{int(row['o_update_interval_runtime'])}/{row['o_reward_term_runtime_scale']:.2f}/{row['o_aux_runtime_scale']:.2f}, "
                f"o_upd={int(row.get('update_o_active_chunks', 0))}, "
                f"o_cmp={row['train_o_shape_term']:.3f}/{row['train_o_teacher_term']:.3f}/{row['train_o_consensus_term']:.3f}, "
                f"m_ent={row['update_m_ent_coef_eff']:.4f}, "
                f"m_s={row['m_shape_entropy_scale']:.2f}/{row['m_teacher_entropy_scale']:.2f}/{row['m_consensus_entropy_scale']:.2f}, "
                f"val_ms={row['val_mean_makespan']:.4f}, val_r={row['val_mean_reward']:.4f}, "
                f"val_err={row['val_num_errors']}, best_val={best_val_makespan:.4f}@{best_epoch}, "
                f"p_scope/final/rescue={row['train_parallel_m_scope_pairs']:.2f}/{row['train_parallel_final_pairs']:.2f}/{row['train_parallel_rescue_pairs']:.2f}, "
                f"m_keep/o={row['train_m_keep_per_o_ratio']:.3f}, "
                f"updated={row['best_updated']}, train_ok={row['train_num_success']}, train_err={row['train_num_errors']}"
                ,
                flush=True,
            )
        else:
            print(
                f"[epoch {epoch:03d}] "
                f"train_ms={row['train_mean_makespan']:.4f}, train_r={row['train_mean_reward']:.4f}, "
                f"train_or={row['train_mean_o_reward']:.4f}, train_mr={row['train_mean_m_reward']:.4f}, done={row['train_done_ratio']:.3f}, "
                f"loss_c={row['update_loss_c']:.5f}, loss_o={row['update_loss_o']:.5f}, loss_m={row['update_loss_m']:.5f}, "
                f"vloss_c={row['update_value_loss_c']:.5f}, vloss_o={row['update_value_loss_o']:.5f}, vloss_m={row['update_value_loss_m']:.5f}, "
                f"ent_c={row['update_entropy_c']:.5f}, ent_o={row['update_entropy_o']:.5f}, ent_m={row['update_entropy_m']:.5f}, "
                f"kl_c={row['update_kl_c']:.5f}, kl_o={row['update_kl_o']:.5f}, kl_m={row['update_kl_m']:.5f}, "
                f"rv={row['reward_version']}, "
                f"o_env={row['o_reward_alpha_env']:.3f}, o_shape={row['o_reward_beta_shape']:.3f}, "
                f"m_env={row['m_reward_alpha_env']:.3f}, m_shape={row['m_reward_beta_shape']:.3f}, "
                f"o_t={row['o_teacher_coef']:.3f}, m_t={row['m_teacher_coef']:.3f}, "
                f"o_cs={row['o_consensus_coef']:.3f}, m_cs={row['m_consensus_coef']:.3f}, "
                f"o_rt={row['o_lr_runtime_scale']:.2f}/{int(row['o_update_interval_runtime'])}/{row['o_reward_term_runtime_scale']:.2f}/{row['o_aux_runtime_scale']:.2f}, "
                f"o_upd={int(row.get('update_o_active_chunks', 0))}, "
                f"o_cmp={row['train_o_shape_term']:.3f}/{row['train_o_teacher_term']:.3f}/{row['train_o_consensus_term']:.3f}, "
                f"m_ent={row['update_m_ent_coef_eff']:.4f}, "
                f"m_s={row['m_shape_entropy_scale']:.2f}/{row['m_teacher_entropy_scale']:.2f}/{row['m_consensus_entropy_scale']:.2f}, "
                f"p_scope/final/rescue={row['train_parallel_m_scope_pairs']:.2f}/{row['train_parallel_final_pairs']:.2f}/{row['train_parallel_rescue_pairs']:.2f}, "
                f"m_keep/o={row['train_m_keep_per_o_ratio']:.3f}, "
                f"train_ok={row['train_num_success']}, train_err={row['train_num_errors']}"
                ,
                flush=True,
            )

        ent_m_epoch = float(row.get("update_entropy_m", 0.0))
        keep_ratio_epoch = float(row.get("train_m_keep_per_o_ratio", 0.0))
        if int(row.get("update_o_active_chunks", 0)) > 0 and float(row.get("update_kl_guard_o", 0.0)) > 0.0:
            print(
                f"[guard-o-kl] epoch={epoch} minibatch_guard_hits={row['update_kl_guard_o']:.3f} "
                f"mean_kl_o={row['update_kl_o']:.5f} target={float(getattr(cfg, 'target_kl_o', 0.0)):.5f}",
                flush=True,
            )
        if int(row.get("update_o_active_chunks", 0)) > 0 and (
            float(row.get("update_o_batch_soft_protect", 0.0)) > 0.0
            or float(row.get("update_o_batch_hard_protect", 0.0)) > 0.0
        ):
            print(
                f"[guard-o-batch] epoch={epoch} soft={row['update_o_batch_soft_protect']:.3f} "
                f"hard={row['update_o_batch_hard_protect']:.3f} "
                f"mean_w={row['update_o_batch_update_weight']:.3f}",
                flush=True,
            )
        if bool(getattr(cfg, "o_instability_monitor_enabled", True)):
            o_active_chunks = int(row.get("update_o_active_chunks", 0))
            kl_o_epoch = float(row.get("update_kl_o", 0.0))
            guard_o_epoch = float(row.get("update_kl_guard_o", 0.0))
            ent_o_epoch = float(row.get("update_entropy_o", 0.0))
            unstable_reasons: List[str] = []
            if o_active_chunks > 0 and kl_o_epoch >= float(getattr(cfg, "o_instability_kl_threshold", 0.04)):
                unstable_reasons.append("high_kl")
            if o_active_chunks > 0 and guard_o_epoch >= float(getattr(cfg, "o_instability_guard_threshold", 0.10)):
                unstable_reasons.append("guard_hits")
            if o_active_chunks > 0 and ent_o_epoch <= float(getattr(cfg, "o_instability_entropy_threshold", 1.75)):
                unstable_reasons.append("low_entropy")

            if len(unstable_reasons) > 0:
                old_o_lr_scale = float(o_lr_runtime_scale)
                old_o_update_interval = int(o_update_interval_runtime)
                old_o_reward_scale = float(o_reward_term_runtime_scale)
                old_o_aux_scale = float(o_aux_runtime_scale)

                if bool(getattr(cfg, "o_auto_lr_decay_on_instability", True)):
                    decay_factor = float(np.clip(float(getattr(cfg, "o_auto_lr_decay_factor", 0.85)), 0.1, 1.0))
                    min_scale = float(np.clip(float(getattr(cfg, "o_auto_lr_min_scale", 0.25)), 0.01, 1.0))
                    o_lr_runtime_scale = _apply_monotonic_decay(
                        scale=float(o_lr_runtime_scale),
                        decay=decay_factor,
                        min_scale=min_scale,
                    )
                if bool(getattr(cfg, "o_auto_update_interval_on_instability", True)):
                    max_interval = int(max(1, int(getattr(cfg, "o_auto_update_interval_max", 6))))
                    o_update_interval_runtime = min(max_interval, int(o_update_interval_runtime) + 1)
                if bool(getattr(cfg, "o_auto_reward_scale_on_instability", True)):
                    decay_factor = float(np.clip(float(getattr(cfg, "o_auto_reward_scale_factor", 0.85)), 0.1, 1.0))
                    min_scale = float(np.clip(float(getattr(cfg, "o_auto_reward_min_scale", 0.35)), 0.0, 1.0))
                    o_reward_term_runtime_scale = _apply_monotonic_decay(
                        scale=float(o_reward_term_runtime_scale),
                        decay=decay_factor,
                        min_scale=min_scale,
                    )
                if bool(getattr(cfg, "o_auto_aux_scale_on_instability", True)):
                    decay_factor = float(np.clip(float(getattr(cfg, "o_auto_aux_scale_factor", 0.85)), 0.1, 1.0))
                    min_scale = float(np.clip(float(getattr(cfg, "o_auto_aux_min_scale", 0.25)), 0.0, 1.0))
                    o_aux_runtime_scale = _apply_monotonic_decay(
                        scale=float(o_aux_runtime_scale),
                        decay=decay_factor,
                        min_scale=min_scale,
                    )

                changed = (
                    abs(float(o_lr_runtime_scale) - old_o_lr_scale) > 1e-12
                    or int(o_update_interval_runtime) != int(old_o_update_interval)
                    or abs(float(o_reward_term_runtime_scale) - old_o_reward_scale) > 1e-12
                    or abs(float(o_aux_runtime_scale) - old_o_aux_scale) > 1e-12
                )
                if changed:
                    print(
                        f"[intervene-o-stability] epoch={epoch} reason={'+'.join(unstable_reasons)} "
                        f"kl_o={kl_o_epoch:.5f} guard_o={guard_o_epoch:.3f} ent_o={ent_o_epoch:.4f} "
                        f"lr_scale={old_o_lr_scale:.4f}->{float(o_lr_runtime_scale):.4f} "
                        f"upd_int={int(old_o_update_interval)}->{int(o_update_interval_runtime)} "
                        f"reward_scale={old_o_reward_scale:.4f}->{float(o_reward_term_runtime_scale):.4f} "
                        f"aux_scale={old_o_aux_scale:.4f}->{float(o_aux_runtime_scale):.4f}",
                        flush=True,
                    )
        if ent_m_epoch < float(getattr(cfg, "m_entropy_warn_threshold", 0.45)):
            print(
                f"[warn-m-collapse] epoch={epoch} ent_m={ent_m_epoch:.4f} < "
                f"warn={float(getattr(cfg, 'm_entropy_warn_threshold', 0.45)):.4f}",
                flush=True,
            )
        if keep_ratio_epoch < float(getattr(cfg, "m_keep_per_o_warn_threshold", 1.5)):
            print(
                f"[warn-m-width] epoch={epoch} keep/o={keep_ratio_epoch:.4f} < "
                f"warn={float(getattr(cfg, 'm_keep_per_o_warn_threshold', 1.5)):.4f}",
                flush=True,
            )

        hard_trigger = (
            ent_m_epoch < float(getattr(cfg, "m_entropy_hard_threshold", 0.35))
            or keep_ratio_epoch < float(getattr(cfg, "m_keep_per_o_warn_threshold", 1.5))
        )
        if hard_trigger and bool(getattr(cfg, "m_auto_lr_decay_on_collapse", False)):
            decay_factor = float(np.clip(float(getattr(cfg, "m_auto_lr_decay_factor", 0.8)), 0.1, 1.0))
            min_scale = float(np.clip(float(getattr(cfg, "m_auto_lr_min_scale", 0.25)), 0.01, 1.0))
            old_scale = float(m_lr_runtime_scale)
            m_lr_runtime_scale = _apply_monotonic_decay(
                scale=old_scale,
                decay=decay_factor,
                min_scale=min_scale,
            )
            if m_lr_runtime_scale < old_scale - 1e-12:
                print(
                    f"[intervene-m-lr] epoch={epoch} scale={old_scale:.4f}->{m_lr_runtime_scale:.4f}, "
                    f"next_m_lr={float(cfg.m_lr) * float(m_lr_runtime_scale):.6g}",
                    flush=True,
                )
        m_entropy_feedback_signal = float(ent_m_epoch)
        row["o_lr_runtime_scale"] = float(o_lr_runtime_scale)
        row["o_update_interval_runtime"] = int(o_update_interval_runtime)
        row["o_reward_term_runtime_scale"] = float(o_reward_term_runtime_scale)
        row["o_aux_runtime_scale"] = float(o_aux_runtime_scale)

        _add_timing(timing_meter, "eval_total_sec", epoch_eval_sec)
        _add_timing(timing_meter, "checkpoint_save_sec", epoch_ckpt_sec)
        if bool(cfg.profile_timing):
            row["epoch_eval_sec"] = float(epoch_eval_sec)
            row["epoch_checkpoint_save_sec"] = float(epoch_ckpt_sec)

        history.append(row)

        if bool(getattr(cfg, "save_history_jsonl", True)):
            with history_jsonl_path.open("a", encoding="utf-8") as f_hist:
                f_hist.write(json.dumps(row, ensure_ascii=False) + "\n")

        if bool(getattr(cfg, "save_progress_every_epoch", True)):
            progress_payload = {
                "epoch": int(epoch),
                "best_val_makespan": float(best_val_makespan),
                "best_epoch": int(best_epoch),
                "history_rows": int(len(history)),
                "last_row": row,
                "checkpoint_best": {
                    "c": str(c_ckpt_path),
                    "o": str(o_ckpt_path),
                    "m": str(m_ckpt_path),
                    "critic": str(critic_ckpt_path),
                },
                "checkpoint_latest": {
                    "c": str(c_latest_ckpt_path),
                    "o": str(o_latest_ckpt_path),
                    "m": str(m_latest_ckpt_path),
                    "critic": str(critic_latest_ckpt_path),
                },
                "history_jsonl": str(history_jsonl_path),
                "train_state": str(train_state_path),
            }
            progress_json_path.write_text(
                json.dumps(progress_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        t_latest_ckpt0 = time.perf_counter()
        if bool(getattr(cfg, "save_latest_every_epoch", True)):
            agent_c.save(str(c_latest_ckpt_path))
            agent_o.save(str(o_latest_ckpt_path))
            agent_m.save(str(m_latest_ckpt_path))
            torch.save(
                {
                    "state_dim": int(state_dim),
                    "hidden_dim": int(cfg.critic_hidden_dim),
                    "split_tower": int(bool(cfg.critic_split_tower)),
                    "tower_hidden_dim": int(cfg.critic_tower_hidden_dim),
                    "use_gnn_branch": int(bool(cfg.critic_use_gnn_branch)),
                    "use_gate_fusion": int(bool(cfg.critic_use_gate_fusion)),
                    "gnn_hidden_dim": int(cfg.critic_gnn_hidden_dim),
                    "gnn_heads": int(cfg.critic_gnn_heads),
                    "freeze_gnn_epochs": int(cfg.critic_freeze_gnn_epochs),
                    "state_dict": critic.state_dict(),
                },
                str(critic_latest_ckpt_path),
            )
            _save_latest_train_state(
                train_state_path=train_state_path,
                epoch=int(epoch),
                best_val_makespan=float(best_val_makespan),
                best_epoch=int(best_epoch),
                history=history,
                agent_c=agent_c,
                agent_o=agent_o,
                agent_m=agent_m,
                critic=critic,
                critic_optimizer=critic_optimizer,
            )
            epoch_ckpt_sec += float(time.perf_counter() - t_latest_ckpt0)

    # Save last-epoch model snapshots explicitly for convenient resume/analysis.
    agent_c.save(str(c_final_ckpt_path))
    agent_o.save(str(o_final_ckpt_path))
    agent_m.save(str(m_final_ckpt_path))
    torch.save(
        {
            "state_dim": int(state_dim),
            "hidden_dim": int(cfg.critic_hidden_dim),
            "split_tower": int(bool(cfg.critic_split_tower)),
            "tower_hidden_dim": int(cfg.critic_tower_hidden_dim),
            "use_gnn_branch": int(bool(cfg.critic_use_gnn_branch)),
            "use_gate_fusion": int(bool(cfg.critic_use_gate_fusion)),
            "gnn_hidden_dim": int(cfg.critic_gnn_hidden_dim),
            "gnn_heads": int(cfg.critic_gnn_heads),
            "freeze_gnn_epochs": int(cfg.critic_freeze_gnn_epochs),
            "state_dict": critic.state_dict(),
        },
        str(critic_final_ckpt_path),
    )

    if int(best_epoch) > 0 and c_ckpt_path.exists():
        agent_c.load(str(c_ckpt_path), strict=False)
    if int(best_epoch) > 0 and o_ckpt_path.exists():
        agent_o.load(str(o_ckpt_path), strict=False)
    if int(best_epoch) > 0 and m_ckpt_path.exists():
        _load_agent_m_checkpoint(agent_m, str(m_ckpt_path), strict=False)
    if int(best_epoch) > 0 and critic_ckpt_path.exists():
        payload = _torch_load_compat(str(critic_ckpt_path), map_location=agent_c.device)
        critic.load_state_dict(payload["state_dict"], strict=True)

    t_final_eval0 = time.perf_counter()
    final_eval_force_det = bool(getattr(cfg, "train_eval_force_deterministic", True)) and int(cfg.epochs) > 0
    eval_selected = set(eval_case_splits)
    train_eval_final = (
        evaluate_oc(
            agent_c,
            agent_o,
            agent_m,
            critic,
            cfg.train_cases,
            cfg,
            eval_deterministic_override=(True if final_eval_force_det else None),
            eval_sample_times_override=(1 if final_eval_force_det else None),
            val_only_best_strategy_override=(False if final_eval_force_det else None),
        )
        if "train" in eval_selected
        else _skipped_eval_result()
    )
    val_eval_final = (
        evaluate_oc(
            agent_c,
            agent_o,
            agent_m,
            critic,
            cfg.val_cases,
            cfg,
            eval_deterministic_override=(True if final_eval_force_det else None),
            eval_sample_times_override=(1 if final_eval_force_det else None),
            val_only_best_strategy_override=(False if final_eval_force_det else None),
        )
        if "val" in eval_selected
        else _skipped_eval_result()
    )
    test_eval_final = (
        evaluate_oc(
            agent_c,
            agent_o,
            agent_m,
            critic,
            cfg.test_cases,
            cfg,
            eval_deterministic_override=(True if final_eval_force_det else None),
            eval_sample_times_override=(1 if final_eval_force_det else None),
            val_only_best_strategy_override=(False if final_eval_force_det else None),
        )
        if "test" in eval_selected
        else _skipped_eval_result()
    )
    _add_timing(timing_meter, "final_eval_sec", time.perf_counter() - t_final_eval0)

    summary = {
        "best_val_makespan": float(best_val_makespan),
        "best_epoch": int(best_epoch),
        "checkpoint_c": str(c_ckpt_path),
        "checkpoint_o": str(o_ckpt_path),
        "checkpoint_m": str(m_ckpt_path),
        "checkpoint_critic": str(critic_ckpt_path),
        "checkpoint_c_latest": str(c_latest_ckpt_path),
        "checkpoint_o_latest": str(o_latest_ckpt_path),
        "checkpoint_m_latest": str(m_latest_ckpt_path),
        "checkpoint_critic_latest": str(critic_latest_ckpt_path),
        "checkpoint_c_final": str(c_final_ckpt_path),
        "checkpoint_o_final": str(o_final_ckpt_path),
        "checkpoint_m_final": str(m_final_ckpt_path),
        "checkpoint_critic_final": str(critic_final_ckpt_path),
        "progress_json": str(progress_json_path),
        "history_jsonl": str(history_jsonl_path),
        "train_state_latest": str(train_state_path),
        "resume_state_path": str(cfg.resume_state_path),
        "scan_c": c_scan,
        "train_eval": train_eval_final,
        "val_eval": val_eval_final,
        "test_eval": test_eval_final,
        "history": history,
        "config": dataclasses.asdict(cfg),
    }

    if bool(cfg.profile_timing):
        summary["timing_profile"] = _summarize_timing_profile(
            meter=timing_meter,
            history=history,
            total_wall_sec=float(time.perf_counter() - train_wall_t0),
        )
    if bool(cfg.profile_rule_stats):
        rule_summary = _summarize_rule_meter(rule_meter, topk=int(max(1, cfg.profile_rule_topk)))
        summary["rule_profile"] = rule_summary
        likely_redundant = list(rule_summary.get("likely_redundant_rules", []))
        if len(likely_redundant) > 0:
            print(
                f"[rule-profile] likely_redundant_rules={json.dumps(likely_redundant, ensure_ascii=False)}",
                flush=True,
            )

    summary_path = save_dir / "train_oc_mappo_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    test_eval = summary.get("test_eval", {})
    if int(test_eval.get("num_cases", 0)) > 0:
        test_msg = f"test={float(test_eval.get('mean_makespan', float('inf'))):.4f}"
    else:
        test_msg = "test=NA"
    print(
        f"[done-oc-mappo] best_val={summary['best_val_makespan']:.4f}, "
        f"{test_msg}, summary={summary_path}"
        ,
        flush=True,
    )

    if mp_executor_train is not None:
        mp_executor_train.shutdown(wait=True, cancel_futures=True)

    return summary


def _build_cases(case_dir: str, pattern: str, train_count: int, val_count: int, seed: int):
    all_cases = sorted(Path(case_dir).glob(pattern))
    if len(all_cases) < train_count + val_count + 1:
        raise ValueError("Not enough cases for requested split")
    random.Random(int(seed)).shuffle(all_cases)
    return make_case_split(all_cases=all_cases, train_count=train_count, val_count=val_count)


def parse_args(args=None):
    p = argparse.ArgumentParser(description="Train O+C+M MAPPO with centralized critic")
    p.add_argument("--case-dir", type=str, default="1_Brandimarte")
    p.add_argument("--pattern", type=str, default="BrandimarteMk*.fjs")
    p.add_argument("--split-json", type=str, default="")
    p.add_argument(
        "--scan-case-splits",
        type=str,
        default="train,val",
        help="Comma-separated splits for dimension scan: train,val,test,all",
    )
    p.add_argument(
        "--eval-case-splits",
        type=str,
        default="val,test",
        help="Comma-separated splits for final makespan evaluation: train,val,test,all",
    )
    p.add_argument("--train-count", type=int, default=10)
    p.add_argument("--val-count", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--episodes-per-update", type=int, default=8)
    p.add_argument(
        "--num-envs",
        type=int,
        default=0,
        help="Rollout environments per update chunk (0 means use --episodes-per-update)",
    )
    p.add_argument("--eval-interval", type=int, default=1)
    p.add_argument("--eval-deterministic", type=int, default=1)
    p.add_argument("--eval-sample-times", type=int, default=1)
    p.add_argument("--eval-sample-reduce", type=str, default="best", choices=["best", "mean"])
    p.add_argument(
        "--deterministic-refine-topk",
        type=int,
        default=1,
        help="When eval-deterministic=1, re-rank C top-k logits by deterministic ECT/PT heuristic (1 disables)",
    )
    p.add_argument(
        "--deterministic-refine-logit-gap",
        type=float,
        default=0.05,
        help="Only apply deterministic top-k re-rank when candidate logit gap <= this threshold",
    )
    p.add_argument(
        "--deterministic-refine-min-ect-gain",
        type=float,
        default=0.0,
        help="Only switch from argmax when ECT improves by at least this amount (0 disables)",
    )
    p.add_argument(
        "--train-eval-force-deterministic",
        type=int,
        default=1,
        help="Force deterministic validation during training epochs/final summary when epochs>0",
    )
    p.add_argument(
        "--deterministic-val-only-gap-candidates",
        type=str,
        default="",
        help="In val-only deterministic eval, run extra deterministic refine gap candidates per case (comma-separated, e.g. 'argmax,0.024,0.033')",
    )
    p.add_argument(
        "--val-only-best-strategy",
        type=int,
        default=1,
        help="Enable stronger sampling strategy only for pure val-only evaluation (epochs=0 and empty test)",
    )
    p.add_argument(
        "--val-only-add-deterministic-baseline",
        type=int,
        default=1,
        help="In val-only best strategy, include one deterministic rollout as baseline candidate",
    )
    p.add_argument(
        "--val-only-extra-samples-per-case",
        type=int,
        default=20,
        help="Additional sampling rollouts for each selected hard case in val-only best strategy",
    )
    p.add_argument(
        "--val-only-hard-case-topk",
        type=int,
        default=3,
        help="How many hardest val cases receive extra samples in val-only best strategy",
    )
    p.add_argument(
        "--val-only-seed-jitter",
        type=int,
        default=1,
        help="Use deterministic seed jitter per case/sample to diversify val-only sampling",
    )
    p.add_argument(
        "--val-only-adaptive-extra-sampling",
        type=int,
        default=1,
        help="Use adaptive allocation for val-only extra samples (hardness + uncertainty + exploration)",
    )
    p.add_argument(
        "--val-only-adaptive-var-bonus",
        type=float,
        default=0.35,
        help="Weight of uncertainty term in adaptive val-only extra sampling score",
    )
    p.add_argument(
        "--val-only-adaptive-ucb-bonus",
        type=float,
        default=0.15,
        help="Weight of exploration term in adaptive val-only extra sampling score",
    )
    p.add_argument("--extra-step-budget", type=int, default=40)

    p.add_argument("--save-dir", type=str, default="checkpoints_oc_mappo")
    p.add_argument("--m-checkpoint-name", type=str, default="agent_m_mappo_best.pt")
    p.add_argument("--c-checkpoint-name", type=str, default="agent_c_mappo_best.pt")
    p.add_argument("--o-checkpoint-name", type=str, default="agent_o_mappo_best.pt")
    p.add_argument("--critic-checkpoint-name", type=str, default="critic_mappo_best.pt")
    p.add_argument("--c-latest-checkpoint-name", type=str, default="agent_c_mappo_latest.pt")
    p.add_argument("--o-latest-checkpoint-name", type=str, default="agent_o_mappo_latest.pt")
    p.add_argument("--m-latest-checkpoint-name", type=str, default="agent_m_mappo_latest.pt")
    p.add_argument("--critic-latest-checkpoint-name", type=str, default="critic_mappo_latest.pt")
    p.add_argument("--c-final-checkpoint-name", type=str, default="agent_c_mappo_final.pt")
    p.add_argument("--o-final-checkpoint-name", type=str, default="agent_o_mappo_final.pt")
    p.add_argument("--m-final-checkpoint-name", type=str, default="agent_m_mappo_final.pt")
    p.add_argument("--critic-final-checkpoint-name", type=str, default="critic_mappo_final.pt")
    p.add_argument("--progress-json-name", type=str, default="train_oc_mappo_progress.json")
    p.add_argument("--history-jsonl-name", type=str, default="train_oc_mappo_history.jsonl")
    p.add_argument("--train-state-name", type=str, default="train_oc_mappo_state_latest.pt")
    p.add_argument("--save-latest-every-epoch", type=int, default=1)
    p.add_argument("--save-progress-every-epoch", type=int, default=1)
    p.add_argument("--save-history-jsonl", type=int, default=1)
    p.add_argument("--o-topk", type=int, default=5)
    p.add_argument("--o-model-type", type=str, default="mlp")
    p.add_argument("--o-op-safety-margin", type=int, default=2)

    p.add_argument("--c-max-candidates", type=int, default=16)
    p.add_argument("--c-auto-max-candidates", type=int, default=1)
    p.add_argument("--c-candidate-safety-margin", type=int, default=4)
    p.add_argument("--c-candidate-safety-margin-ratio", type=float, default=0.10)
    p.add_argument("--c-min-max-candidates", type=int, default=16)
    p.add_argument("--c-round-max-candidates-to-power-of-two", type=int, default=0)

    p.add_argument("--c-use-graph-encoder", type=int, default=0)
    p.add_argument("--c-use-edge-rule-msg", type=int, default=1)
    p.add_argument("--c-use-edge-opmch-msg", type=int, default=1)
    p.add_argument("--c-use-adaptive-edge-gates", type=int, default=0)

    p.add_argument("--critic-rich-state", type=int, default=0)
    p.add_argument("--critic-split-tower", type=int, default=0)
    p.add_argument("--critic-tower-hidden-dim", type=int, default=256)
    p.add_argument("--critic-use-gnn-branch", type=int, default=0)
    p.add_argument("--critic-use-gate-fusion", type=int, default=0)
    p.add_argument("--critic-gnn-hidden-dim", type=int, default=128)
    p.add_argument("--val-plateau-patience", type=int, default=2)
    p.add_argument("--val-plateau-decay", type=float, default=0.7)
    p.add_argument("--val-plateau-min-scale", type=float, default=0.2)
    p.add_argument("--val-plateau-apply-c", type=int, default=1)
    p.add_argument("--val-plateau-apply-o", type=int, default=1)
    p.add_argument("--val-plateau-apply-m", type=int, default=1)
    p.add_argument("--val-plateau-restore-best", type=int, default=1)
    p.add_argument("--val-plateau-restore-min-epoch", type=int, default=3)
    p.add_argument("--val-plateau-restore-cooldown", type=int, default=2)
    p.add_argument("--val-plateau-restore-rel-gap", type=float, default=0.03)
    p.add_argument("--val-plateau-restore-abs-gap", type=float, default=0.0)
    p.add_argument("--critic-gnn-heads", type=int, default=4)
    p.add_argument("--critic-freeze-gnn-epochs", type=int, default=0)

    p.add_argument("--init-c-checkpoint-path", type=str, default="")
    p.add_argument("--init-o-checkpoint-path", type=str, default="")
    p.add_argument("--init-m-checkpoint-path", type=str, default="")
    p.add_argument("--init-critic-checkpoint-path", type=str, default="")
    p.add_argument("--resume-state-path", type=str, default="")

    p.add_argument("--c-lr", type=float, default=2e-4)
    p.add_argument("--o-lr", type=float, default=2e-4)
    p.add_argument("--m-lr", type=float, default=1e-4)
    p.add_argument("--m-weight-decay", type=float, default=1e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--clip-ratio", type=float, default=0.2)
    p.add_argument("--clip-ratio-c", type=float, default=0.2)
    p.add_argument("--clip-ratio-o", type=float, default=0.2)
    p.add_argument("--clip-ratio-m", type=float, default=0.1)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--value-coef-c", type=float, default=0.6)
    p.add_argument("--value-coef-o", type=float, default=0.6)
    p.add_argument("--value-coef-m", type=float, default=0.6)
    p.add_argument("--use-huber-value-loss", type=int, default=1)
    p.add_argument("--value-huber-delta", type=float, default=1.0)
    p.add_argument("--value-clip-range", type=float, default=0.2)
    p.add_argument("--target-kl-c", type=float, default=0.02)
    p.add_argument("--target-kl-o", type=float, default=0.01)
    p.add_argument("--target-kl-m", type=float, default=0.01)
    p.add_argument("--kl-penalty-coef-c", type=float, default=0.0)
    p.add_argument("--kl-penalty-coef-o", type=float, default=0.01)
    p.add_argument("--kl-penalty-coef-m", type=float, default=0.015)
    p.add_argument("--minibatch-kl-guard-enabled", type=int, default=1)
    p.add_argument("--minibatch-kl-guard-mult-c", type=float, default=2.0)
    p.add_argument("--minibatch-kl-guard-mult-o", type=float, default=1.5)
    p.add_argument("--minibatch-kl-guard-mult-m", type=float, default=2.0)
    p.add_argument("--o-update-interval", type=int, default=1)
    p.add_argument("--m-update-interval", type=int, default=2)
    p.add_argument("--o-smooth-update-enabled", type=int, default=1)
    p.add_argument("--o-smooth-update-min-weight", type=float, default=0.20)
    p.add_argument("--o-kl-batch-soft-limit-mult", type=float, default=1.5)
    p.add_argument("--o-kl-batch-hard-limit-mult", type=float, default=2.5)
    p.add_argument("--o-kl-batch-soft-scale", type=float, default=0.5)
    p.add_argument("--o-batch-protect-enabled", type=int, default=1)
    p.add_argument("--o-ratio-batch-soft-limit", type=float, default=0.40)
    p.add_argument("--o-ratio-batch-hard-limit", type=float, default=0.80)
    p.add_argument("--o-ratio-batch-soft-scale", type=float, default=0.5)
    p.add_argument("--disable-ppo-early-stop", "--disable-kl-early-stop", action="store_true", default=False)

    p.add_argument(
        "--reward-version",
        type=str,
        default="role_aligned_v1",
        choices=["legacy", "role_aligned_v1", "role_aligned_v2"],
    )
    p.add_argument("--reward-schedule-anneal-epochs", type=int, default=100)

    p.add_argument("--o-reward-alpha-env", type=float, default=1.00)
    p.add_argument("--o-reward-alpha-env-end", type=float, default=1.00)
    p.add_argument("--o-reward-beta-shape", type=float, default=0.10)
    p.add_argument("--o-reward-beta-shape-end", type=float, default=0.04)
    p.add_argument("--o-teacher-coef", type=float, default=0.03)
    p.add_argument("--o-teacher-coef-end", type=float, default=0.005)
    p.add_argument("--o-teacher-coef-min", type=float, default=0.005)
    p.add_argument("--o-consensus-coef", type=float, default=0.02)
    p.add_argument("--o-consensus-coef-end", type=float, default=0.003)
    p.add_argument("--o-consensus-coef-min", type=float, default=0.003)
    p.add_argument("--o-reward-anneal-epochs", type=int, default=180)
    p.add_argument("--o-reward-schedule", type=str, default="linear", choices=["linear", "piecewise"])
    p.add_argument("--o-reward-mid-progress", type=float, default=0.55)
    p.add_argument("--o-reward-mid-ratio", type=float, default=0.35)
    p.add_argument("--o-reward-clip-abs", type=float, default=1.5)
    p.add_argument("--reward-retain-pos", type=float, default=0.20)
    p.add_argument("--reward-retain-neg", type=float, default=-0.45)
    p.add_argument("--reward-quality-coef", type=float, default=0.30)
    p.add_argument("--reward-quality-clip", type=float, default=0.30)
    p.add_argument("--reward-quality-hard-threshold", type=float, default=0.08)
    p.add_argument("--reward-quality-hard-penalty", type=float, default=0.18)
    p.add_argument("--reward-quality-hard-smooth", type=int, default=0)
    p.add_argument("--reward-quality-hard-smooth-width", type=float, default=0.02)
    p.add_argument("--reward-mismatch-penalty", type=float, default=0.08)
    p.add_argument("--reward-mismatch-only-if-not-retained", type=int, default=1)
    p.add_argument("--reward-makespan-terminal-coef", type=float, default=0.01)
    p.add_argument("--reward-balance-coef", type=float, default=0.0)
    p.add_argument("--reward-wait-coef", type=float, default=0.0)
    p.add_argument(
        "--active-job-rules",
        type=str,
        default="FIFO,CRJ,MWKR",
        help="Comma-separated active job rules for candidate generation; empty means all",
    )
    p.add_argument(
        "--active-machine-rules",
        type=str,
        default="SPT,EET,EETLQ,EETD",
        help="Comma-separated active machine rules for candidate generation; empty means all",
    )
    p.add_argument("--agent-c-topk-jobs", type=int, default=3)
    p.add_argument("--agent-c-topk-machines", type=int, default=3)
    p.add_argument("--agent-c-pairs-per-rule", type=int, default=5)
    p.add_argument("--agent-c-extra-explore-pairs", type=int, default=4)
    p.add_argument("--agent-c-two-stage-refine-enabled", type=int, default=1)
    p.add_argument("--agent-c-refine-max-pairs-per-op", type=int, default=2)
    p.add_argument("--agent-c-refine-keep-global-best-ect", type=int, default=1)
    p.add_argument("--agent-c-refine-keep-global-best-pt", type=int, default=1)
    p.add_argument("--agent-c-refine-score-w-eet", type=float, default=1.0)
    p.add_argument("--agent-c-refine-score-w-pt", type=float, default=0.3)
    p.add_argument("--agent-c-refine-score-w-queue", type=float, default=0.2)
    p.add_argument("--agent-c-refine-score-w-support", type=float, default=0.5)
    p.add_argument("--agent-c-refine-explore-bonus", type=float, default=0.1)
    p.add_argument("--agent-c-candidate-v2-enabled", type=int, default=1)
    p.add_argument("--agent-c-critical-pairs-enabled", type=int, default=1)
    p.add_argument("--agent-c-critical-pairs-per-op", type=int, default=2)
    p.add_argument("--agent-c-refine-max-pairs-per-machine", type=int, default=0)
    p.add_argument("--agent-c-refine-global-reserve-size", type=int, default=2)
    p.add_argument("--agent-c-refine-diversity-min-machines", type=int, default=2)
    p.add_argument("--agent-c-refine-keep-explore-min", type=int, default=1)
    p.add_argument("--agent-c-candidate-debug-print", type=int, default=0)

    p.add_argument("--full-soft-widen-enabled", type=int, default=1)
    p.add_argument("--full-soft-widen-c-extra", type=int, default=2)
    p.add_argument("--full-soft-widen-o-extra", type=int, default=1)
    p.add_argument("--full-soft-widen-refine-pairs-per-op-extra", type=int, default=1)
    p.add_argument("--ocm-parallel-om-enabled", type=int, default=0)
    p.add_argument(
        "--ocm-parallel-fusion-mode",
        type=str,
        default="o_plus_m_backup",
        choices=["intersection", "union", "o_plus_m_backup"],
    )
    p.add_argument("--ocm-parallel-min-final-pairs", type=int, default=3)
    p.add_argument("--ocm-parallel-budget-extra-pairs", type=int, default=2)
    p.add_argument("--ocm-parallel-max-final-pairs", type=int, default=0)
    p.add_argument("--ocm-parallel-debug-print", type=int, default=0)
    p.add_argument(
        "--ocm-parallel-m-teacher-mode",
        type=str,
        default="full_ref",
        choices=["full_ref", "disabled"],
    )
    p.add_argument("--ocm-parallel-rescue-ops", type=int, default=2)
    p.add_argument("--ocm-parallel-rescue-pairs-per-op", type=int, default=2)
    p.add_argument("--ocm-parallel-rescue-include-c-full", type=int, default=1)
    p.add_argument("--ocm-parallel-rescue-include-global-best", type=int, default=1)

    p.add_argument("--o-topk-min", type=int, default=2)
    p.add_argument("--o-topk-max", type=int, default=8)
    p.add_argument(
        "--o-scale-topk-floor-divisor",
        type=int,
        default=10,
        help="Scale-aware top-k floor: keep at least ceil(num_ops/divisor) when divisor > 0",
    )
    p.add_argument("--o-topk-entropy-gain", type=float, default=1.0)
    p.add_argument("--o-entropy-fallback-threshold", type=float, default=1.0)
    p.add_argument("--o-entropy-fallback-extra-ops", type=int, default=1)
    p.add_argument("--o-entropy-low-fallback-threshold", type=float, default=1.5)
    p.add_argument("--o-entropy-low-fallback-extra-ops", type=int, default=1)
    p.add_argument("--o-reward-fallback-scale", "--o-reward-fallback-discount", dest="o_reward_fallback_scale", type=float, default=0.85)
    p.add_argument("--o-reference-c-deterministic", type=int, default=1)
    p.add_argument("--o-redundancy-target-ratio", type=float, default=0.35)
    p.add_argument("--o-redundancy-penalty-coef", type=float, default=0.05)
    p.add_argument("--o-coverage-bonus-coef", type=float, default=0.02)
    p.add_argument("--o-set-aux-coef", type=float, default=0.02)
    p.add_argument("--o-set-ppo-mix", type=float, default=0.35)
    p.add_argument("--o-ent-adaptive-target", type=float, default=1.8)
    p.add_argument("--o-ent-adaptive-gain", type=float, default=0.6)
    p.add_argument("--o-ent-adaptive-max-scale", type=float, default=2.5)
    p.add_argument("--o-instability-monitor-enabled", type=int, default=1)
    p.add_argument("--o-instability-kl-threshold", type=float, default=0.04)
    p.add_argument("--o-instability-guard-threshold", type=float, default=0.10)
    p.add_argument("--o-instability-entropy-threshold", type=float, default=1.75)
    p.add_argument("--o-auto-lr-decay-on-instability", type=int, default=1)
    p.add_argument("--o-auto-lr-decay-factor", type=float, default=0.85)
    p.add_argument("--o-auto-lr-min-scale", type=float, default=0.25)
    p.add_argument("--o-auto-update-interval-on-instability", type=int, default=1)
    p.add_argument("--o-auto-update-interval-max", type=int, default=6)
    p.add_argument("--o-auto-aux-scale-on-instability", type=int, default=1)
    p.add_argument("--o-auto-aux-scale-factor", type=float, default=0.85)
    p.add_argument("--o-auto-aux-min-scale", type=float, default=0.25)
    p.add_argument("--o-auto-reward-scale-on-instability", type=int, default=1)
    p.add_argument("--o-auto-reward-scale-factor", type=float, default=0.85)
    p.add_argument("--o-auto-reward-min-scale", type=float, default=0.35)

    p.add_argument("--m-hidden-dim", type=int, default=128)
    p.add_argument("--m-dropout", type=float, default=0.05)
    p.add_argument("--m-feature-mode", type=str, default="machine_heavy")
    p.add_argument("--m-strategy-mode", type=str, default="ppo")
    p.add_argument("--m-enable-entropy-backup", type=int, default=1)
    p.add_argument("--m-backup-entropy-threshold", type=float, default=0.75)
    p.add_argument("--m-backup-max-extra-pairs", type=int, default=1)
    p.add_argument("--m-keep-c-full-top1", type=int, default=1)
    p.add_argument("--m-keep-c-o-top1", type=int, default=1)
    p.add_argument("--m-hard-keep-requires-teacher", type=int, default=0)
    p.add_argument("--m-hard-keep-teacher-min-coef", type=float, default=0.0)
    p.add_argument("--m-hard-keep-entropy-trigger", type=int, default=0)
    p.add_argument("--m-hard-keep-entropy-threshold", type=float, default=0.20)
    p.add_argument("--m-expand-pairs-for-safety", type=int, default=1)
    p.add_argument("--m-safety-min-total-pairs", type=int, default=3)
    p.add_argument("--m-safety-min-machines-per-op", type=int, default=3)
    p.add_argument(
        "--p1-fast-skip-c-o-ref",
        type=int,
        default=1,
        help="P1 speed mode: skip intermediate C(o-filtered) inference via ECT proxy reference",
    )
    p.add_argument(
        "--p1-skip-c-full-ref",
        type=int,
        default=0,
        help="P1 speed mode: skip initial C(full) reference pass and use ECT proxy",
    )
    p.add_argument("--m-ent-coef", type=float, default=0.012)
    p.add_argument("--m-ent-coef-end", type=float, default=0.012)
    p.add_argument("--m-ent-coef-anneal-updates", type=int, default=120)
    p.add_argument("--m-reward-alpha-env", type=float, default=1.00)
    p.add_argument("--m-reward-alpha-env-end", type=float, default=1.00)
    p.add_argument("--m-reward-beta-shape", type=float, default=0.08)
    p.add_argument("--m-reward-beta-shape-end", type=float, default=0.03)
    p.add_argument("--m-teacher-coef", type=float, default=0.05)
    p.add_argument("--m-teacher-coef-end", type=float, default=0.01)
    p.add_argument("--m-teacher-coef-min", type=float, default=0.01)
    p.add_argument("--m-consensus-coef", type=float, default=0.01)
    p.add_argument("--m-consensus-coef-end", type=float, default=0.002)
    p.add_argument("--m-consensus-coef-min", type=float, default=0.002)
    p.add_argument("--m-reward-clip-abs", type=float, default=1.5)
    p.add_argument("--m-reward-fallback-scale", type=float, default=0.90)
    p.add_argument("--m-reward-retain-pos", type=float, default=0.20)
    p.add_argument("--m-reward-retain-neg", type=float, default=-0.30)
    p.add_argument("--m-reward-quality-coef", type=float, default=0.35)
    p.add_argument("--m-reward-quality-hard-threshold", type=float, default=0.08)
    p.add_argument("--m-reward-quality-hard-penalty", type=float, default=0.10)
    p.add_argument("--m-reward-overprune-target-keep-ratio", type=float, default=0.30)
    p.add_argument("--m-reward-overprune-coef", type=float, default=0.10)
    p.add_argument("--m-reward-mismatch-penalty", type=float, default=0.08)
    p.add_argument("--m-reward-terminal-gap-coef", type=float, default=0.01)
    p.add_argument("--m-candidate-aware-scaling-enabled", type=int, default=1)
    p.add_argument("--m-candidate-aware-c-low", type=int, default=22)
    p.add_argument("--m-candidate-aware-c-high", type=int, default=34)
    p.add_argument("--m-candidate-aware-o-low", type=int, default=9)
    p.add_argument("--m-candidate-aware-o-high", type=int, default=12)
    p.add_argument("--m-candidate-aware-min-scale", type=float, default=0.35)
    p.add_argument("--m-candidate-aware-gamma", type=float, default=1.30)
    p.add_argument("--m-candidate-aware-apply-to-beta-shape", type=int, default=1)
    p.add_argument("--m-candidate-aware-apply-to-teacher", type=int, default=1)
    p.add_argument("--m-candidate-aware-apply-to-consensus", type=int, default=1)
    p.add_argument("--m-entropy-feedback-enabled", type=int, default=1)
    p.add_argument("--m-entropy-feedback-threshold", type=float, default=0.50)
    p.add_argument("--m-entropy-feedback-power", type=float, default=1.25)
    p.add_argument("--m-entropy-feedback-shape-min-scale", type=float, default=0.90)
    p.add_argument("--m-entropy-feedback-teacher-min-scale", type=float, default=0.20)
    p.add_argument("--m-entropy-feedback-consensus-min-scale", type=float, default=0.35)
    p.add_argument("--m-entropy-warn-threshold", type=float, default=0.45)
    p.add_argument("--m-entropy-hard-threshold", type=float, default=0.35)
    p.add_argument("--m-keep-per-o-warn-threshold", type=float, default=1.5)
    p.add_argument("--m-auto-lr-decay-on-collapse", type=int, default=0)
    p.add_argument("--m-auto-lr-decay-factor", type=float, default=0.8)
    p.add_argument("--m-auto-lr-min-scale", type=float, default=0.25)

    p.add_argument("--profile-timing", type=int, default=0)
    p.add_argument("--profile-rule-stats", type=int, default=0)
    p.add_argument("--profile-rule-topk", type=int, default=12)

    p.add_argument(
        "--rollout-backend",
        type=str,
        default="serial",
        choices=["serial", "mp"],
        help="serial: single-process rollout; mp: multi-process rollout workers",
    )
    p.add_argument(
        "--rollout-workers",
        type=int,
        default=0,
        help="Number of rollout workers for --rollout-backend mp (0 means auto)",
    )
    p.add_argument(
        "--rollout-min-cases-per-worker",
        type=int,
        default=2,
        help="Minimum cases handled by each rollout worker to reduce over-fragmented multiprocessing overhead",
    )
    p.add_argument(
        "--rollout-worker-device",
        type=str,
        default="cpu",
        help="Device used in rollout workers (recommended: cpu)",
    )
    p.add_argument(
        "--rollout-mp-start-method",
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="Multiprocessing start method for mp rollout workers",
    )
    p.add_argument(
        "--rollout-worker-policy-mode",
        type=str,
        default="train",
        choices=["train", "eval"],
        help="Policy mode used by rollout workers (train keeps parity with serial rollout)",
    )
    p.add_argument(
        "--rollout-worker-reseed",
        type=int,
        default=1,
        help="Whether to reseed each worker task with epoch/chunk/shard-specific seed (1=yes,0=no)",
    )

    p.add_argument("--quiet-env", action="store_true", default=False)
    return p.parse_args(args=args)


def main(args=None):
    cli = parse_args(args=args)
    effective_num_envs = int(cli.num_envs) if int(cli.num_envs) > 0 else int(cli.episodes_per_update)
    scan_case_splits = _parse_case_splits(
        str(cli.scan_case_splits),
        arg_name="--scan-case-splits",
        default=("train", "val"),
    )
    eval_case_splits = _parse_case_splits(
        str(cli.eval_case_splits),
        arg_name="--eval-case-splits",
        default=("val", "test"),
    )
    if cli.split_json:
        split_payload = json.loads(Path(cli.split_json).read_text(encoding="utf-8"))
        train_cases = [str(x) for x in split_payload.get("train_cases", [])]
        val_cases = [str(x) for x in split_payload.get("val_cases", [])]
        test_cases = [str(x) for x in split_payload.get("test_cases", [])]
        if int(cli.epochs) > 0 and len(train_cases) == 0:
            raise ValueError("split-json must contain non-empty train_cases when epochs > 0")
        if int(cli.epochs) > 0 and len(val_cases) == 0:
            raise ValueError("split-json must contain non-empty val_cases when epochs > 0")
        if int(cli.epochs) <= 0:
            case_map = {
                "train": train_cases,
                "val": val_cases,
                "test": test_cases,
            }
            if all(len(case_map[name]) == 0 for name in eval_case_splits):
                raise ValueError("split-json has no cases for requested --eval-case-splits")
    else:
        train_cases, val_cases, test_cases = _build_cases(
            case_dir=cli.case_dir,
            pattern=cli.pattern,
            train_count=cli.train_count,
            val_count=cli.val_count,
            seed=cli.seed,
        )

    cfg = OCMAPPOConfig(
        train_cases=train_cases,
        val_cases=val_cases,
        test_cases=test_cases,
        split_source=str(cli.split_json) if cli.split_json else "",
        scan_case_splits=",".join(scan_case_splits),
        eval_case_splits=",".join(eval_case_splits),
        seed=cli.seed,
        device=cli.device,
        epochs=cli.epochs,
        episodes_per_update=effective_num_envs,
        num_envs=effective_num_envs,
        eval_interval=cli.eval_interval,
        eval_deterministic=bool(int(cli.eval_deterministic)),
        eval_sample_times=int(cli.eval_sample_times),
        eval_sample_reduce=str(cli.eval_sample_reduce),
        deterministic_refine_topk=int(cli.deterministic_refine_topk),
        deterministic_refine_logit_gap=float(cli.deterministic_refine_logit_gap),
        deterministic_refine_min_ect_gain=float(cli.deterministic_refine_min_ect_gain),
        train_eval_force_deterministic=bool(int(cli.train_eval_force_deterministic)),
        deterministic_val_only_gap_candidates=str(cli.deterministic_val_only_gap_candidates),
        val_only_best_strategy=bool(int(cli.val_only_best_strategy)),
        val_only_add_deterministic_baseline=bool(int(cli.val_only_add_deterministic_baseline)),
        val_only_extra_samples_per_case=int(cli.val_only_extra_samples_per_case),
        val_only_hard_case_topk=int(cli.val_only_hard_case_topk),
        val_only_seed_jitter=bool(int(cli.val_only_seed_jitter)),
        val_only_adaptive_extra_sampling=bool(int(cli.val_only_adaptive_extra_sampling)),
        val_only_adaptive_var_bonus=float(cli.val_only_adaptive_var_bonus),
        val_only_adaptive_ucb_bonus=float(cli.val_only_adaptive_ucb_bonus),
        extra_step_budget=cli.extra_step_budget,
        save_dir=cli.save_dir,
        c_checkpoint_name=str(cli.c_checkpoint_name),
        o_checkpoint_name=str(cli.o_checkpoint_name),
        m_checkpoint_name=cli.m_checkpoint_name,
        critic_checkpoint_name=str(cli.critic_checkpoint_name),
        c_latest_checkpoint_name=str(cli.c_latest_checkpoint_name),
        o_latest_checkpoint_name=str(cli.o_latest_checkpoint_name),
        m_latest_checkpoint_name=str(cli.m_latest_checkpoint_name),
        critic_latest_checkpoint_name=str(cli.critic_latest_checkpoint_name),
        c_final_checkpoint_name=str(cli.c_final_checkpoint_name),
        o_final_checkpoint_name=str(cli.o_final_checkpoint_name),
        m_final_checkpoint_name=str(cli.m_final_checkpoint_name),
        critic_final_checkpoint_name=str(cli.critic_final_checkpoint_name),
        progress_json_name=str(cli.progress_json_name),
        history_jsonl_name=str(cli.history_jsonl_name),
        train_state_name=str(cli.train_state_name),
        save_latest_every_epoch=bool(int(cli.save_latest_every_epoch)),
        save_progress_every_epoch=bool(int(cli.save_progress_every_epoch)),
        save_history_jsonl=bool(int(cli.save_history_jsonl)),
        o_topk=cli.o_topk,
        o_model_type=cli.o_model_type,
        o_op_safety_margin=int(cli.o_op_safety_margin),
        c_max_candidates=int(cli.c_max_candidates),
        c_auto_max_candidates=bool(int(cli.c_auto_max_candidates)),
        c_candidate_safety_margin=int(cli.c_candidate_safety_margin),
        c_candidate_safety_margin_ratio=float(cli.c_candidate_safety_margin_ratio),
        c_min_max_candidates=int(cli.c_min_max_candidates),
        c_round_max_candidates_to_power_of_two=bool(int(cli.c_round_max_candidates_to_power_of_two)),
        c_use_graph_encoder=bool(int(cli.c_use_graph_encoder)),
        c_use_edge_rule_msg=bool(int(cli.c_use_edge_rule_msg)),
        c_use_edge_opmch_msg=bool(int(cli.c_use_edge_opmch_msg)),
        c_use_adaptive_edge_gates=bool(int(cli.c_use_adaptive_edge_gates)),
        critic_rich_state=bool(int(cli.critic_rich_state)),
        critic_split_tower=bool(int(cli.critic_split_tower)),
        critic_tower_hidden_dim=int(cli.critic_tower_hidden_dim),
        critic_use_gnn_branch=bool(int(cli.critic_use_gnn_branch)),
        critic_use_gate_fusion=bool(int(cli.critic_use_gate_fusion)),
        critic_gnn_hidden_dim=int(cli.critic_gnn_hidden_dim),
        val_plateau_patience=int(cli.val_plateau_patience),
        val_plateau_decay=float(cli.val_plateau_decay),
        val_plateau_min_scale=float(cli.val_plateau_min_scale),
        val_plateau_apply_c=bool(int(cli.val_plateau_apply_c)),
        val_plateau_apply_o=bool(int(cli.val_plateau_apply_o)),
        val_plateau_apply_m=bool(int(cli.val_plateau_apply_m)),

        o_batch_protect_enabled=bool(int(cli.o_batch_protect_enabled)),
        o_ratio_batch_soft_limit=float(cli.o_ratio_batch_soft_limit),
        o_ratio_batch_hard_limit=float(cli.o_ratio_batch_hard_limit),
        o_ratio_batch_soft_scale=float(cli.o_ratio_batch_soft_scale),
        val_plateau_restore_best=bool(int(cli.val_plateau_restore_best)),
        val_plateau_restore_min_epoch=int(cli.val_plateau_restore_min_epoch),
        val_plateau_restore_cooldown=int(cli.val_plateau_restore_cooldown),
        val_plateau_restore_rel_gap=float(cli.val_plateau_restore_rel_gap),
        val_plateau_restore_abs_gap=float(cli.val_plateau_restore_abs_gap),
        critic_gnn_heads=int(cli.critic_gnn_heads),
        critic_freeze_gnn_epochs=int(cli.critic_freeze_gnn_epochs),
        init_c_checkpoint_path=cli.init_c_checkpoint_path,
        init_o_checkpoint_path=cli.init_o_checkpoint_path,
        init_m_checkpoint_path=cli.init_m_checkpoint_path,
        init_critic_checkpoint_path=str(cli.init_critic_checkpoint_path),
        resume_state_path=str(cli.resume_state_path),
        c_lr=cli.c_lr,
        o_lr=cli.o_lr,
        m_lr=cli.m_lr,
        m_weight_decay=cli.m_weight_decay,
        critic_lr=cli.critic_lr,
        clip_ratio=cli.clip_ratio,
        clip_ratio_c=cli.clip_ratio_c,
        clip_ratio_o=cli.clip_ratio_o,
        clip_ratio_m=cli.clip_ratio_m,
        ppo_epochs=cli.ppo_epochs,
        minibatch_size=cli.minibatch_size,
        value_coef_c=cli.value_coef_c,
        value_coef_o=cli.value_coef_o,
        value_coef_m=cli.value_coef_m,
        use_huber_value_loss=bool(int(cli.use_huber_value_loss)),
        value_huber_delta=cli.value_huber_delta,
        value_clip_range=cli.value_clip_range,
        target_kl_c=cli.target_kl_c,
        target_kl_o=cli.target_kl_o,
        target_kl_m=cli.target_kl_m,
        kl_penalty_coef_c=float(cli.kl_penalty_coef_c),
        kl_penalty_coef_o=float(cli.kl_penalty_coef_o),
        kl_penalty_coef_m=float(cli.kl_penalty_coef_m),
        minibatch_kl_guard_enabled=bool(int(cli.minibatch_kl_guard_enabled)),
        minibatch_kl_guard_mult_c=float(cli.minibatch_kl_guard_mult_c),
        minibatch_kl_guard_mult_o=float(cli.minibatch_kl_guard_mult_o),
        minibatch_kl_guard_mult_m=float(cli.minibatch_kl_guard_mult_m),
        o_update_interval=int(cli.o_update_interval),
        m_update_interval=int(cli.m_update_interval),
        o_smooth_update_enabled=bool(int(cli.o_smooth_update_enabled)),
        o_smooth_update_min_weight=float(cli.o_smooth_update_min_weight),
        o_kl_batch_soft_limit_mult=float(cli.o_kl_batch_soft_limit_mult),
        o_kl_batch_hard_limit_mult=float(cli.o_kl_batch_hard_limit_mult),
        o_kl_batch_soft_scale=float(cli.o_kl_batch_soft_scale),
        ppo_early_stop=not bool(cli.disable_ppo_early_stop),
        reward_version=str(cli.reward_version),
        reward_schedule_anneal_epochs=int(cli.reward_schedule_anneal_epochs),
        o_reward_alpha_env=cli.o_reward_alpha_env,
        o_reward_alpha_env_end=cli.o_reward_alpha_env_end,
        o_reward_beta_shape=cli.o_reward_beta_shape,
        o_reward_beta_shape_end=cli.o_reward_beta_shape_end,
        o_teacher_coef=cli.o_teacher_coef,
        o_teacher_coef_end=cli.o_teacher_coef_end,
        o_teacher_coef_min=cli.o_teacher_coef_min,
        o_consensus_coef=cli.o_consensus_coef,
        o_consensus_coef_end=cli.o_consensus_coef_end,
        o_consensus_coef_min=cli.o_consensus_coef_min,
        o_reward_anneal_epochs=cli.o_reward_anneal_epochs,
        o_reward_schedule=cli.o_reward_schedule,
        o_reward_mid_progress=cli.o_reward_mid_progress,
        o_reward_mid_ratio=cli.o_reward_mid_ratio,
        o_reward_clip_abs=cli.o_reward_clip_abs,
        reward_retain_pos=cli.reward_retain_pos,
        reward_retain_neg=cli.reward_retain_neg,
        reward_quality_coef=cli.reward_quality_coef,
        reward_quality_clip=cli.reward_quality_clip,
        reward_quality_hard_threshold=cli.reward_quality_hard_threshold,
        reward_quality_hard_penalty=cli.reward_quality_hard_penalty,
        reward_quality_hard_smooth=bool(int(cli.reward_quality_hard_smooth)),
        reward_quality_hard_smooth_width=cli.reward_quality_hard_smooth_width,
        reward_mismatch_penalty=cli.reward_mismatch_penalty,
        reward_mismatch_only_if_not_retained=bool(int(cli.reward_mismatch_only_if_not_retained)),
        reward_makespan_terminal_coef=cli.reward_makespan_terminal_coef,
        reward_balance_coef=cli.reward_balance_coef,
        reward_wait_coef=cli.reward_wait_coef,
        active_job_rules=str(cli.active_job_rules),
        active_machine_rules=str(cli.active_machine_rules),
        agent_c_topk_jobs=int(cli.agent_c_topk_jobs),
        agent_c_topk_machines=int(cli.agent_c_topk_machines),
        agent_c_pairs_per_rule=int(cli.agent_c_pairs_per_rule),
        agent_c_extra_explore_pairs=int(cli.agent_c_extra_explore_pairs),
        agent_c_two_stage_refine_enabled=bool(int(cli.agent_c_two_stage_refine_enabled)),
        agent_c_refine_max_pairs_per_op=int(cli.agent_c_refine_max_pairs_per_op),
        agent_c_refine_keep_global_best_ect=int(cli.agent_c_refine_keep_global_best_ect),
        agent_c_refine_keep_global_best_pt=int(cli.agent_c_refine_keep_global_best_pt),
        agent_c_refine_score_w_eet=float(cli.agent_c_refine_score_w_eet),
        agent_c_refine_score_w_pt=float(cli.agent_c_refine_score_w_pt),
        agent_c_refine_score_w_queue=float(cli.agent_c_refine_score_w_queue),
        agent_c_refine_score_w_support=float(cli.agent_c_refine_score_w_support),
        agent_c_refine_explore_bonus=float(cli.agent_c_refine_explore_bonus),
        agent_c_candidate_v2_enabled=bool(int(cli.agent_c_candidate_v2_enabled)),
        agent_c_critical_pairs_enabled=bool(int(cli.agent_c_critical_pairs_enabled)),
        agent_c_critical_pairs_per_op=int(cli.agent_c_critical_pairs_per_op),
        agent_c_refine_max_pairs_per_machine=int(cli.agent_c_refine_max_pairs_per_machine),
        agent_c_refine_global_reserve_size=int(cli.agent_c_refine_global_reserve_size),
        agent_c_refine_diversity_min_machines=int(cli.agent_c_refine_diversity_min_machines),
        agent_c_refine_keep_explore_min=int(cli.agent_c_refine_keep_explore_min),
        agent_c_candidate_debug_print=bool(int(cli.agent_c_candidate_debug_print)),
        full_soft_widen_enabled=bool(int(cli.full_soft_widen_enabled)),
        full_soft_widen_c_extra=int(cli.full_soft_widen_c_extra),
        full_soft_widen_o_extra=int(cli.full_soft_widen_o_extra),
        full_soft_widen_refine_pairs_per_op_extra=int(cli.full_soft_widen_refine_pairs_per_op_extra),
        ocm_parallel_om_enabled=bool(int(cli.ocm_parallel_om_enabled)),
        ocm_parallel_fusion_mode=str(cli.ocm_parallel_fusion_mode),
        ocm_parallel_min_final_pairs=int(cli.ocm_parallel_min_final_pairs),
        ocm_parallel_budget_extra_pairs=int(cli.ocm_parallel_budget_extra_pairs),
        ocm_parallel_max_final_pairs=int(cli.ocm_parallel_max_final_pairs),
        ocm_parallel_debug_print=bool(int(cli.ocm_parallel_debug_print)),
        ocm_parallel_m_teacher_mode=str(cli.ocm_parallel_m_teacher_mode),
        ocm_parallel_rescue_ops=int(cli.ocm_parallel_rescue_ops),
        ocm_parallel_rescue_pairs_per_op=int(cli.ocm_parallel_rescue_pairs_per_op),
        ocm_parallel_rescue_include_c_full=bool(int(cli.ocm_parallel_rescue_include_c_full)),
        ocm_parallel_rescue_include_global_best=bool(int(cli.ocm_parallel_rescue_include_global_best)),
        o_topk_min=cli.o_topk_min,
        o_topk_max=cli.o_topk_max,
        o_scale_topk_floor_divisor=int(cli.o_scale_topk_floor_divisor),
        o_topk_entropy_gain=cli.o_topk_entropy_gain,
        o_entropy_fallback_threshold=cli.o_entropy_fallback_threshold,
        o_entropy_fallback_extra_ops=cli.o_entropy_fallback_extra_ops,
        o_entropy_low_fallback_threshold=cli.o_entropy_low_fallback_threshold,
        o_entropy_low_fallback_extra_ops=cli.o_entropy_low_fallback_extra_ops,
        o_reward_fallback_scale=cli.o_reward_fallback_scale,
        o_reference_c_deterministic=bool(int(cli.o_reference_c_deterministic)),
        o_redundancy_target_ratio=float(cli.o_redundancy_target_ratio),
        o_redundancy_penalty_coef=float(cli.o_redundancy_penalty_coef),
        o_coverage_bonus_coef=float(cli.o_coverage_bonus_coef),
        o_set_aux_coef=float(cli.o_set_aux_coef),
        o_set_ppo_mix=float(cli.o_set_ppo_mix),
        o_ent_adaptive_target=float(cli.o_ent_adaptive_target),
        o_ent_adaptive_gain=float(cli.o_ent_adaptive_gain),
        o_ent_adaptive_max_scale=float(cli.o_ent_adaptive_max_scale),
        o_instability_monitor_enabled=bool(int(cli.o_instability_monitor_enabled)),
        o_instability_kl_threshold=float(cli.o_instability_kl_threshold),
        o_instability_guard_threshold=float(cli.o_instability_guard_threshold),
        o_instability_entropy_threshold=float(cli.o_instability_entropy_threshold),
        o_auto_lr_decay_on_instability=bool(int(cli.o_auto_lr_decay_on_instability)),
        o_auto_lr_decay_factor=float(cli.o_auto_lr_decay_factor),
        o_auto_lr_min_scale=float(cli.o_auto_lr_min_scale),
        o_auto_update_interval_on_instability=bool(int(cli.o_auto_update_interval_on_instability)),
        o_auto_update_interval_max=int(cli.o_auto_update_interval_max),
        o_auto_aux_scale_on_instability=bool(int(cli.o_auto_aux_scale_on_instability)),
        o_auto_aux_scale_factor=float(cli.o_auto_aux_scale_factor),
        o_auto_aux_min_scale=float(cli.o_auto_aux_min_scale),
        o_auto_reward_scale_on_instability=bool(int(cli.o_auto_reward_scale_on_instability)),
        o_auto_reward_scale_factor=float(cli.o_auto_reward_scale_factor),
        o_auto_reward_min_scale=float(cli.o_auto_reward_min_scale),
        m_hidden_dim=cli.m_hidden_dim,
        m_dropout=cli.m_dropout,
        m_feature_mode=cli.m_feature_mode,
        m_strategy_mode=cli.m_strategy_mode,
        m_enable_entropy_backup=bool(int(cli.m_enable_entropy_backup)),
        m_backup_entropy_threshold=cli.m_backup_entropy_threshold,
        m_backup_max_extra_pairs=cli.m_backup_max_extra_pairs,
        m_keep_c_full_top1=bool(int(cli.m_keep_c_full_top1)),
        m_keep_c_o_top1=bool(int(cli.m_keep_c_o_top1)),
        m_hard_keep_requires_teacher=bool(int(cli.m_hard_keep_requires_teacher)),
        m_hard_keep_teacher_min_coef=float(cli.m_hard_keep_teacher_min_coef),
        m_hard_keep_entropy_trigger=bool(int(cli.m_hard_keep_entropy_trigger)),
        m_hard_keep_entropy_threshold=float(cli.m_hard_keep_entropy_threshold),
        m_expand_pairs_for_safety=bool(int(cli.m_expand_pairs_for_safety)),
        m_safety_min_total_pairs=int(cli.m_safety_min_total_pairs),
        m_safety_min_machines_per_op=int(cli.m_safety_min_machines_per_op),
        p1_fast_skip_c_o_ref=bool(int(cli.p1_fast_skip_c_o_ref)),
        p1_skip_c_full_ref=bool(int(cli.p1_skip_c_full_ref)),
        m_ent_coef=cli.m_ent_coef,
        m_ent_coef_end=cli.m_ent_coef_end,
        m_ent_coef_anneal_updates=int(cli.m_ent_coef_anneal_updates),
        m_reward_alpha_env=cli.m_reward_alpha_env,
        m_reward_alpha_env_end=cli.m_reward_alpha_env_end,
        m_reward_beta_shape=cli.m_reward_beta_shape,
        m_reward_beta_shape_end=cli.m_reward_beta_shape_end,
        m_teacher_coef=cli.m_teacher_coef,
        m_teacher_coef_end=cli.m_teacher_coef_end,
        m_teacher_coef_min=cli.m_teacher_coef_min,
        m_consensus_coef=cli.m_consensus_coef,
        m_consensus_coef_end=cli.m_consensus_coef_end,
        m_consensus_coef_min=cli.m_consensus_coef_min,
        m_reward_clip_abs=cli.m_reward_clip_abs,
        m_reward_fallback_scale=cli.m_reward_fallback_scale,
        m_reward_retain_pos=cli.m_reward_retain_pos,
        m_reward_retain_neg=cli.m_reward_retain_neg,
        m_reward_quality_coef=cli.m_reward_quality_coef,
        m_reward_quality_hard_threshold=cli.m_reward_quality_hard_threshold,
        m_reward_quality_hard_penalty=cli.m_reward_quality_hard_penalty,
        m_reward_overprune_target_keep_ratio=cli.m_reward_overprune_target_keep_ratio,
        m_reward_overprune_coef=cli.m_reward_overprune_coef,
        m_reward_mismatch_penalty=cli.m_reward_mismatch_penalty,
        m_reward_terminal_gap_coef=cli.m_reward_terminal_gap_coef,
        m_candidate_aware_scaling_enabled=bool(int(cli.m_candidate_aware_scaling_enabled)),
        m_candidate_aware_c_low=int(cli.m_candidate_aware_c_low),
        m_candidate_aware_c_high=int(cli.m_candidate_aware_c_high),
        m_candidate_aware_o_low=int(cli.m_candidate_aware_o_low),
        m_candidate_aware_o_high=int(cli.m_candidate_aware_o_high),
        m_candidate_aware_min_scale=float(cli.m_candidate_aware_min_scale),
        m_candidate_aware_gamma=float(cli.m_candidate_aware_gamma),
        m_candidate_aware_apply_to_beta_shape=bool(int(cli.m_candidate_aware_apply_to_beta_shape)),
        m_candidate_aware_apply_to_teacher=bool(int(cli.m_candidate_aware_apply_to_teacher)),
        m_candidate_aware_apply_to_consensus=bool(int(cli.m_candidate_aware_apply_to_consensus)),
        m_entropy_feedback_enabled=bool(int(cli.m_entropy_feedback_enabled)),
        m_entropy_feedback_threshold=float(cli.m_entropy_feedback_threshold),
        m_entropy_feedback_power=float(cli.m_entropy_feedback_power),
        m_entropy_feedback_shape_min_scale=float(cli.m_entropy_feedback_shape_min_scale),
        m_entropy_feedback_teacher_min_scale=float(cli.m_entropy_feedback_teacher_min_scale),
        m_entropy_feedback_consensus_min_scale=float(cli.m_entropy_feedback_consensus_min_scale),
        m_entropy_warn_threshold=float(cli.m_entropy_warn_threshold),
        m_entropy_hard_threshold=float(cli.m_entropy_hard_threshold),
        m_keep_per_o_warn_threshold=float(cli.m_keep_per_o_warn_threshold),
        m_auto_lr_decay_on_collapse=bool(int(cli.m_auto_lr_decay_on_collapse)),
        m_auto_lr_decay_factor=float(cli.m_auto_lr_decay_factor),
        m_auto_lr_min_scale=float(cli.m_auto_lr_min_scale),
        profile_timing=bool(int(cli.profile_timing)),
        profile_rule_stats=bool(int(cli.profile_rule_stats)),
        profile_rule_topk=int(cli.profile_rule_topk),
        rollout_backend=str(cli.rollout_backend),
        rollout_workers=int(cli.rollout_workers),
        rollout_min_cases_per_worker=int(cli.rollout_min_cases_per_worker),
        rollout_worker_device=str(cli.rollout_worker_device),
        rollout_mp_start_method=str(cli.rollout_mp_start_method),
        rollout_worker_policy_mode=str(cli.rollout_worker_policy_mode),
        rollout_worker_reseed=bool(int(cli.rollout_worker_reseed)),
        quiet_env=bool(cli.quiet_env),
    )

    # Bridge OCMAPPO CLI args into the environment-side reward config used by FJSP_Env_Agent.
    env_configs.reward_balance_coef = float(cfg.reward_balance_coef)
    env_configs.reward_wait_coef = float(cfg.reward_wait_coef)
    env_configs.agent_c_active_job_rules = str(getattr(cfg, "active_job_rules", "")).strip()
    env_configs.agent_c_active_machine_rules = str(getattr(cfg, "active_machine_rules", "")).strip()
    env_configs.agent_c_topk_jobs = int(getattr(cfg, "agent_c_topk_jobs", 3))
    env_configs.agent_c_topk_machines = int(getattr(cfg, "agent_c_topk_machines", 3))
    env_configs.agent_c_pairs_per_rule = int(getattr(cfg, "agent_c_pairs_per_rule", 5))
    env_configs.agent_c_extra_explore_pairs = int(getattr(cfg, "agent_c_extra_explore_pairs", 4))
    env_configs.agent_c_two_stage_refine_enabled = bool(getattr(cfg, "agent_c_two_stage_refine_enabled", True))
    refine_pairs_per_op_effective = int(getattr(cfg, "agent_c_refine_max_pairs_per_op", 2))
    if bool(getattr(cfg, "full_soft_widen_enabled", True)) and bool(getattr(cfg, "agent_c_two_stage_refine_enabled", True)):
        refine_pairs_per_op_effective += max(0, int(getattr(cfg, "full_soft_widen_refine_pairs_per_op_extra", 1)))
    env_configs.agent_c_refine_max_pairs_per_op = int(refine_pairs_per_op_effective)
    env_configs.agent_c_refine_keep_global_best_ect = int(getattr(cfg, "agent_c_refine_keep_global_best_ect", 1))
    env_configs.agent_c_refine_keep_global_best_pt = int(getattr(cfg, "agent_c_refine_keep_global_best_pt", 1))
    env_configs.agent_c_refine_score_w_eet = float(getattr(cfg, "agent_c_refine_score_w_eet", 1.0))
    env_configs.agent_c_refine_score_w_pt = float(getattr(cfg, "agent_c_refine_score_w_pt", 0.3))
    env_configs.agent_c_refine_score_w_queue = float(getattr(cfg, "agent_c_refine_score_w_queue", 0.2))
    env_configs.agent_c_refine_score_w_support = float(getattr(cfg, "agent_c_refine_score_w_support", 0.5))
    env_configs.agent_c_refine_explore_bonus = float(getattr(cfg, "agent_c_refine_explore_bonus", 0.1))
    env_configs.agent_c_candidate_v2_enabled = bool(getattr(cfg, "agent_c_candidate_v2_enabled", True))
    env_configs.agent_c_critical_pairs_enabled = bool(getattr(cfg, "agent_c_critical_pairs_enabled", True))
    env_configs.agent_c_critical_pairs_per_op = int(getattr(cfg, "agent_c_critical_pairs_per_op", 2))
    env_configs.agent_c_refine_max_pairs_per_machine = int(getattr(cfg, "agent_c_refine_max_pairs_per_machine", 0))
    env_configs.agent_c_refine_global_reserve_size = int(getattr(cfg, "agent_c_refine_global_reserve_size", 2))
    env_configs.agent_c_refine_diversity_min_machines = int(getattr(cfg, "agent_c_refine_diversity_min_machines", 2))
    env_configs.agent_c_refine_keep_explore_min = int(getattr(cfg, "agent_c_refine_keep_explore_min", 1))
    env_configs.agent_c_candidate_debug_print = bool(getattr(cfg, "agent_c_candidate_debug_print", False))
    print(
        f"[reward-c] balance_coef={env_configs.reward_balance_coef:.4f}, "
        f"wait_coef={env_configs.reward_wait_coef:.4f}",
        flush=True,
    )
    print(
        f"[rules-active] job={env_configs.agent_c_active_job_rules or 'ALL'}, "
        f"machine={env_configs.agent_c_active_machine_rules or 'ALL'}",
        flush=True,
    )
    print(
        f"[candidate-width] topk_jobs={int(env_configs.agent_c_topk_jobs)}, "
        f"topk_machines={int(env_configs.agent_c_topk_machines)}, "
        f"pairs_per_rule={int(env_configs.agent_c_pairs_per_rule)}, "
        f"extra_explore_pairs={int(env_configs.agent_c_extra_explore_pairs)}",
        flush=True,
    )

    train_oc_mappo(cfg)


if __name__ == "__main__":
    main()

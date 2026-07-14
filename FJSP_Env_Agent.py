import time
from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Tuple, Any

try:
    import gym
    from gym.utils import EzPickle
except Exception:  # pragma: no cover
    import gymnasium as gym
    from gymnasium.utils import EzPickle

import numpy as np
import torch

from Params import configs
from permissibleLS import permissibleLeftShift
from uniform_instance import override
from updateEndTimeLB import calEndTimeLB, calEndTimeLBm
from decoupled_dispatch_rules import (
    JOB_RULE_NAMES,
    MACHINE_RULE_NAMES,
    compute_machine_queue_len,
    dispatch_decoupled,
    get_job_rule_func,
    get_job_rule_mode,
    get_machine_rule_func,
    prepare_rule_state,
    select_topk_job_indices,
)


class FJSP(gym.Env, EzPickle):
    """
    Stage-1 optimized FJSP environment for Agent_C.

    本版优化重点：
    1. 新增 operation / machine 级缓存，减少重复 np.where 全局扫描
    2. 显式维护 machine_len / op_machine / op_pos_on_machine
    3. 用缓存直接更新 job_time / mch_time，而不是每步全量重算
    4. 邻居查询改为基于缓存，避免 updateAdjMat.py 的全表扫描
    5. 尽量保持对现有 permissibleLS.py / updateEndTimeLB.py 的兼容

    注意：
    - 这是环境侧第一阶段重构版
    - 下一步应继续重构 permissibleLS.py，使其直接使用本环境新增缓存
    """

    RULE_NAMES: Sequence[str] = tuple(
        f"{job_rule}_{machine_rule}"
        for job_rule in JOB_RULE_NAMES
        for machine_rule in MACHINE_RULE_NAMES
    )

    def __init__(self, n_j, n_m, EachJob_num_operation):
        EzPickle.__init__(self)

        self.step_count = 0
        self.number_of_jobs = int(n_j)
        self.number_of_machines = int(n_m)
        self.num_operation = EachJob_num_operation
        self.number_of_tasks = int(EachJob_num_operation.sum(axis=1)[0])
        self.max_operation = int(EachJob_num_operation.max())

        self.last_col = np.cumsum(EachJob_num_operation, -1) - 1
        self.first_col = np.cumsum(EachJob_num_operation, -1) - EachJob_num_operation

        self.getEndTimeLB = calEndTimeLB

        # Agent_C candidate controls
        self.max_pairs_per_rule = int(getattr(configs, "agent_c_pairs_per_rule", 4))
        self.max_machines_per_task = int(getattr(configs, "agent_c_topk_machines", 2))
        self.rule_topk_jobs = int(getattr(configs, "agent_c_topk_jobs", 2))
        self.rule_topk_machines = int(getattr(configs, "agent_c_topk_machines", 2))
        self.extra_explore_pairs = int(getattr(configs, "agent_c_extra_explore_pairs", 2))
        self.active_job_rule_names = self._resolve_active_rules(
            raw_rules=getattr(configs, "agent_c_active_job_rules", ""),
            all_rule_names=JOB_RULE_NAMES,
        )
        self.active_machine_rule_names = self._resolve_active_rules(
            raw_rules=getattr(configs, "agent_c_active_machine_rules", ""),
            all_rule_names=MACHINE_RULE_NAMES,
        )
        self.candidate_generation_mode = str(
            getattr(configs, "agent_c_candidate_generation_mode", "rule")
        ).strip().lower()
        # Fixed-size global features improve cross-dataset transfer by decoupling obs dim from n_j/n_m.
        self.fixed_global_feat = bool(getattr(configs, "agent_c_fixed_global_feat", True))

        # Agent_C candidate generation v2 (multi-recall + budgeted refine)
        self.agent_c_candidate_v2_enabled = bool(getattr(configs, "agent_c_candidate_v2_enabled", True))
        self.two_stage_refine_enabled = bool(getattr(configs, "agent_c_two_stage_refine_enabled", True))

        self.refine_max_pairs_per_op = int(getattr(configs, "agent_c_refine_max_pairs_per_op", 2))
        self.refine_max_pairs_per_machine = int(getattr(configs, "agent_c_refine_max_pairs_per_machine", 0))
        self.refine_global_reserve_size = int(getattr(configs, "agent_c_refine_global_reserve_size", 2))
        self.refine_diversity_min_machines = int(getattr(configs, "agent_c_refine_diversity_min_machines", 2))

        self.refine_keep_global_best_ect = int(getattr(configs, "agent_c_refine_keep_global_best_ect", 1))
        self.refine_keep_global_best_pt = int(getattr(configs, "agent_c_refine_keep_global_best_pt", 1))
        self.refine_keep_explore_min = int(getattr(configs, "agent_c_refine_keep_explore_min", 1))

        self.refine_score_w_eet = float(getattr(configs, "agent_c_refine_score_w_eet", 1.0))
        self.refine_score_w_pt = float(getattr(configs, "agent_c_refine_score_w_pt", 0.3))
        self.refine_score_w_queue = float(getattr(configs, "agent_c_refine_score_w_queue", 0.2))
        self.refine_score_w_support = float(getattr(configs, "agent_c_refine_score_w_support", 0.5))
        self.refine_explore_bonus = float(getattr(configs, "agent_c_refine_explore_bonus", 0.1))

        self.critical_pairs_enabled = bool(getattr(configs, "agent_c_critical_pairs_enabled", True))
        self.critical_pairs_per_op = int(getattr(configs, "agent_c_critical_pairs_per_op", 2))
        self.candidate_debug_print = bool(getattr(configs, "agent_c_candidate_debug_print", False))

        # lightweight debug stats for candidate generation
        self._last_candidate_refine_stats: Dict[str, float] = {}

        # reward
        self.reward_terminal_coef = float(getattr(configs, "reward_terminal_coef", 1.0))
        self.reward_lb_coef = float(getattr(configs, "reward_lb_coef", 1.0))
        self.reward_use_terminal = bool(getattr(configs, "reward_use_terminal", True))
        self.reward_balance_coef = float(getattr(configs, "reward_balance_coef", 0.0))
        self.reward_wait_coef = float(getattr(configs, "reward_wait_coef", 0.0))

        self._state_version = 0

        # caches
        self._obs_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._runtime_state_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._compiled_case_static_cache: Dict[Tuple, Dict[str, np.ndarray]] = {}

        # task maps
        self._task_rc_maps: List[np.ndarray] = []
        self._task_job_maps: List[np.ndarray] = []
        self._task_col_maps: List[np.ndarray] = []

    def _resolve_active_rules(self, raw_rules, all_rule_names: Sequence[str]) -> List[str]:
        if raw_rules is None:
            return list(all_rule_names)

        parsed: List[str] = []
        if isinstance(raw_rules, str):
            txt = raw_rules.strip()
            if txt == "":
                return list(all_rule_names)
            parsed = [x.strip() for x in txt.split(",") if x.strip()]
        elif isinstance(raw_rules, (list, tuple, set)):
            parsed = [str(x).strip() for x in raw_rules if str(x).strip()]
        else:
            return list(all_rule_names)

        valid = []
        for name in parsed:
            if name in all_rule_names and name not in valid:
                valid.append(name)
        return valid if len(valid) > 0 else list(all_rule_names)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    def _invalidate_state_caches(self):
        self._obs_cache.clear()
        self._runtime_state_cache.clear()

    def _bump_state_version(self):
        self._state_version += 1
        self._invalidate_state_caches()

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    def done(self, batch_idx: int = 0):
        return bool(np.all(self.partial_sol_sequeence[batch_idx] >= 0))

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    @override
    def reset(self, data, rule):
        self.rule = rule
        self.batch_sie = int(data.shape[0])

        self._bump_state_version()

        self.job_col = np.zeros((self.batch_sie, self.number_of_jobs), dtype=np.int32)
        self.dispatched_num_opera = np.zeros((self.batch_sie, self.number_of_jobs), dtype=np.int32)

        self.step_count = 0
        self.mchMat = -1 * np.ones((self.batch_sie, self.number_of_jobs, self.max_operation), dtype=np.int64)

        self.dur = data.astype(np.float32)
        self.dur_cp = deepcopy(self.dur)

        self.partial_sol_sequeence = -1 * np.ones((self.batch_sie, self.number_of_tasks), dtype=np.int64)
        self.flags = []
        self.posRewards = np.zeros(self.batch_sie, dtype=np.float32)

        # adjacency
        adj_list = []
        for _ in range(self.batch_sie):
            conj_nei_up_stream = np.eye(self.number_of_tasks, k=-1, dtype=np.single)
            conj_nei_low_stream = np.eye(self.number_of_tasks, k=1, dtype=np.single)
            conj_nei_up_stream[self.first_col] = 0
            conj_nei_low_stream[self.last_col] = 0
            self_as_nei = np.eye(self.number_of_tasks, dtype=np.single)
            adj_list.append(self_as_nei + conj_nei_up_stream)
        self.adj = torch.tensor(np.asarray(adj_list))

        compiled = self._get_compiled_case_static(data)
        self.mask_mch = compiled["mask_mch"].copy()
        self.dur = compiled["dur"].copy()
        self.dur_cp = compiled["dur_cp"].copy()
        self.input_min = compiled["input_min"].copy()
        self.input_mean = compiled["input_mean"].copy()
        self.input_max = compiled["input_max"].copy()
        self.input_2d = compiled["input_2d"].copy()

        self.LBs = np.cumsum(self.input_2d, -2)
        self.LB = np.cumsum(self.input_min, -1)
        self.LBm = np.asarray([self._flatten_lbm(i) for i in range(self.batch_sie)], dtype=np.float32)

        self.initQuality = np.ones(self.batch_sie, dtype=np.float32)
        for i in range(self.batch_sie):
            self.initQuality[i] = self.LBm[i].max() if not bool(getattr(configs, "init_quality_flag", False)) else 0.0
        self.max_endTime = self.initQuality.copy()

        # dynamic times
        self.job_time = np.zeros((self.batch_sie, self.number_of_jobs), dtype=np.float32)
        self.mch_time = np.zeros((self.batch_sie, self.number_of_machines), dtype=np.float32)
        self.machine_workload = np.zeros((self.batch_sie, self.number_of_machines), dtype=np.float32)

        # finished marks
        self.finished_mark = np.zeros((self.batch_sie, self.number_of_tasks), dtype=np.float32)

        fea = np.concatenate(
            (
                self.LBm.reshape(self.batch_sie, -1, 1) / float(configs.et_normalize_coef),
                self.finished_mark.reshape(self.batch_sie, self.number_of_tasks, 1),
            ),
            axis=-1,
        )

        self.omega = self.first_col.astype(np.int64).copy()
        self.mask = np.full((self.batch_sie, self.number_of_jobs), fill_value=0, dtype=bool)

        # machine schedule arrays
        self.mchsStartTimes = -float(configs.high) * np.ones(
            (self.batch_sie, self.number_of_machines, self.number_of_tasks), dtype=np.float32
        )
        self.mchsEndTimes = -float(configs.high) * np.ones(
            (self.batch_sie, self.number_of_machines, self.number_of_tasks), dtype=np.float32
        )
        self.opIDsOnMchs = -self.number_of_jobs * np.ones(
            (self.batch_sie, self.number_of_machines, self.number_of_tasks), dtype=np.int32
        )
        self.up_mchendtime = np.zeros_like(self.mchsEndTimes)
        self.temp1 = np.zeros((self.batch_sie, self.number_of_jobs, self.max_operation), dtype=np.float32)

        # -------- Stage-1 new caches --------
        # task -> (row, col)
        self._task_rc_maps, self._task_job_maps, self._task_col_maps = self._build_task_maps()

        # per-batch op caches
        self.op_to_job_row = np.stack(self._task_job_maps, axis=0).astype(np.int32)     # [B, T]
        self.op_to_job_col = np.stack(self._task_col_maps, axis=0).astype(np.int32)     # [B, T]

        # scheduled op state
        self.op_start_time = -np.ones((self.batch_sie, self.number_of_tasks), dtype=np.float32)
        self.op_end_time = -np.ones((self.batch_sie, self.number_of_tasks), dtype=np.float32)
        self.op_machine = -np.ones((self.batch_sie, self.number_of_tasks), dtype=np.int32)
        self.op_pos_on_machine = -np.ones((self.batch_sie, self.number_of_tasks), dtype=np.int32)

        # per-machine effective length / tail end
        self.machine_len = np.zeros((self.batch_sie, self.number_of_machines), dtype=np.int32)
        self.machine_last_end = np.zeros((self.batch_sie, self.number_of_machines), dtype=np.float32)

        # per-job last completion
        self.job_last_end = np.zeros((self.batch_sie, self.number_of_jobs), dtype=np.float32)

        dur = self.dur_cp.reshape(self.batch_sie, -1, self.max_operation)
        return self.adj, fea, self.omega, self.mask, self.mask_mch, dur, self.mch_time, self.job_time

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    @override
    def step(self, action, mch_a):
        feas, rewards, dones, masks, mch_masks = [], [], [], [], []
        mch_spaces, mchForJobSpaces = [], []
        state_changed = False

        action = np.asarray(action, dtype=np.int64)
        mch_a = np.asarray(mch_a, dtype=np.int64)

        for i in range(self.batch_sie):
            done_before = self.done(i)
            balance_delta_norm = 0.0
            wait_penalty_norm = 0.0
            use_balance_term = abs(float(self.reward_balance_coef)) > 1e-12
            use_wait_term = abs(float(self.reward_wait_coef)) > 1e-12

            if not done_before and (action[i] not in self.partial_sol_sequeence[i]):
                state_changed = True

                op_id = int(action[i])
                mch_id = int(mch_a[i])
                row, col = self._task_to_row_col(op_id, batch_idx=i)

                prev_imbalance = 0.0
                if use_balance_term:
                    prev_machine_load = self.machine_workload[i].astype(np.float32, copy=False)
                    prev_load_mean = float(np.mean(prev_machine_load)) if len(prev_machine_load) > 0 else 0.0
                    prev_load_std = float(np.std(prev_machine_load)) if len(prev_machine_load) > 0 else 0.0
                    prev_imbalance = prev_load_std / max(prev_load_mean, 1e-6) if prev_load_mean > 1e-8 else 0.0
                prev_job_ready_time = float(self.job_time[i, row]) if use_wait_term else 0.0

                self.dispatched_num_opera[i, row] += 1
                if i == 0:
                    self.step_count += 1

                self.finished_mark[i, op_id] = 1.0
                dur_a = float(self.dur_cp[i, row, col, mch_id])

                next_slot = int(np.where(self.partial_sol_sequeence[i] < 0)[0][0])
                self.partial_sol_sequeence[i, next_slot] = op_id
                self.mchMat[i, row, col] = mch_id

                # 兼容现有 permissibleLS.py
                start_time_a, flag = permissibleLeftShift(
                    a=op_id,
                    mch_a=mch_id,
                    durMat=self.dur_cp[i],
                    mchMat=self.mchMat[i],
                    mchsStartTimes=self.mchsStartTimes[i],
                    opIDsOnMchs=self.opIDsOnMchs[i],
                    mchEndTime=self.mchsEndTimes[i],
                    row=row,
                    col=col,
                    first_col=self.first_col[i],
                    last_col=self.last_col[i],
                )
                self.flags.append(flag)

                end_time_a = float(start_time_a) + dur_a
                wait_time = max(float(start_time_a) - prev_job_ready_time, 0.0) if use_wait_term else 0.0

                # update op sequence state
                if op_id not in self.last_col[i]:
                    self.omega[i, row] += 1
                    self.job_col[i, row] += 1
                else:
                    self.mask[i, row] = True

                # temp1 / LB
                self.temp1[i, row, col] = end_time_a
                self.LB[i] = calEndTimeLBm(self.temp1[i], self.input_min[i])
                self.LBm[i] = self._flatten_lbm_from_lb(i)

                # -------- Stage-1 cache sync after insertion --------
                insert_pos = self._locate_inserted_op_pos(batch_idx=i, mch_id=mch_id, op_id=op_id)
                self._update_after_machine_insert(
                    batch_idx=i,
                    op_id=op_id,
                    mch_id=mch_id,
                    row=row,
                    col=col,
                    start_time_a=float(start_time_a),
                    end_time_a=float(end_time_a),
                    insert_pos=insert_pos,
                )

                # cached dynamic times: O(1) update
                self.job_last_end[i, row] = end_time_a
                self.job_time[i, row] = end_time_a

                # machine time = tail end of effective machine sequence
                if self.machine_len[i, mch_id] > 0:
                    tail_pos = self.machine_len[i, mch_id] - 1
                    self.machine_last_end[i, mch_id] = float(self.mchsEndTimes[i, mch_id, tail_pos])
                    self.mch_time[i, mch_id] = self.machine_last_end[i, mch_id]
                else:
                    self.machine_last_end[i, mch_id] = 0.0
                    self.mch_time[i, mch_id] = 0.0

                self.machine_workload[i, mch_id] += dur_a

                if use_balance_term:
                    curr_machine_load = self.machine_workload[i].astype(np.float32, copy=False)
                    curr_load_mean = float(np.mean(curr_machine_load)) if len(curr_machine_load) > 0 else 0.0
                    curr_load_std = float(np.std(curr_machine_load)) if len(curr_machine_load) > 0 else 0.0
                    curr_imbalance = curr_load_std / max(curr_load_mean, 1e-6) if curr_load_mean > 1e-8 else 0.0
                    balance_delta_norm = float(np.clip(curr_imbalance - prev_imbalance, -1.0, 1.0))
                if use_wait_term:
                    wait_penalty_norm = float(np.clip(wait_time / max(float(self.initQuality[i]), 1.0), 0.0, 2.0))

                # cached neighbors
                precd, succd = self._get_action_neighbors_cached(i, op_id)

                self.adj[i, op_id] = 0
                self.adj[i, op_id, op_id] = 1

                if op_id not in self.first_col[i]:
                    self.adj[i, op_id, op_id - 1] = 1

                if precd >= 0:
                    self.adj[i, op_id, precd] = 1
                if succd >= 0:
                    self.adj[i, succd, op_id] = 1

            done_after = self.done(i)

            next_mask, next_m_masks, _, _ = self._compute_next_dispatch_views(
                batch_idx=i,
                rule_name=self.rule,
                done=done_after,
            )
            masks.append(next_mask)
            mch_masks.append(next_m_masks)

            fea = np.concatenate(
                (
                    self.LBm[i].reshape(-1, 1) / float(configs.et_normalize_coef),
                    self.finished_mark[i].reshape(self.number_of_tasks, 1),
                ),
                axis=-1,
            )
            feas.append(fea)

            prev_lb = float(self.max_endTime[i])
            curr_lb = float(self.LBm[i].max())
            norm = max(float(self.initQuality[i]), 1.0)

            reward = -self.reward_lb_coef * (curr_lb - prev_lb) / norm
            reward += -self.reward_balance_coef * balance_delta_norm
            reward += -self.reward_wait_coef * wait_penalty_norm
            if done_after and self.reward_use_terminal:
                final_makespan = float(np.max(self.temp1[i]))
                reward += -self.reward_terminal_coef * final_makespan / norm

            rewards.append(float(reward))
            self.max_endTime[i] = curr_lb
            dones.append(done_after)

        if state_changed:
            self._bump_state_version()

        mch_masks = np.asarray(mch_masks, dtype=object)
        return (
            self.adj,
            np.asarray(feas, dtype=np.float32),
            rewards,
            dones,
            self.omega,
            masks,
            mchForJobSpaces,
            mch_masks,
            self.mch_time,
            self.job_time,
        )

    # ------------------------------------------------------------------
    # Runtime state cache
    # ------------------------------------------------------------------
    def _get_runtime_state(self, batch_idx=0, copy_arrays: bool = False):
        self._validate_batch_idx(batch_idx)
        cache_key = (int(batch_idx), int(self._state_version))
        cached = self._runtime_state_cache.get(cache_key)
        if cached is not None:
            return self._copy_runtime_state(cached) if copy_arrays else cached

        num_operation = self.num_operation[batch_idx]
        omega = self.omega[batch_idx]
        mask = self.mask[batch_idx]
        job_col = self.job_col[batch_idx]
        dispatched_num_opera = self.dispatched_num_opera[batch_idx]
        dur = self.dur_cp[batch_idx]
        mask_mch = self.mask_mch[batch_idx]
        input_min = self.input_min[batch_idx]
        input_max = self.input_max[batch_idx]
        finished_mark = self.finished_mark[batch_idx]
        temp1 = self.temp1[batch_idx]
        mchs_end_times = self.mchsEndTimes[batch_idx]

        available_jobs = np.where(~mask)[0].astype(np.int64)
        remain_num = np.maximum(num_operation - dispatched_num_opera, 0).astype(np.float32)
        remain_work_min = np.asarray(
            [self._compute_remaining_work_min(batch_idx, j) for j in range(self.number_of_jobs)],
            dtype=np.float32,
        )

        # 用 machine_len 直接得到 queue len，避免从整张 mchsEndTimes 再扫
        machine_queue_len = self.machine_len[batch_idx].astype(np.float32)

        state = {
            "batch_idx": batch_idx,
            "number_of_jobs": self.number_of_jobs,
            "number_of_machines": self.number_of_machines,
            "number_of_tasks": self.number_of_tasks,
            "max_operation": self.max_operation,
            "num_operation": num_operation,
            "first_col": self.first_col[batch_idx],
            "last_col": self.last_col[batch_idx],
            "omega": omega,
            "mask": mask,
            "mask_mch": mask_mch,
            "job_col": job_col,
            "dispatched_num_opera": dispatched_num_opera,
            "dur": dur,
            "temp1": temp1,
            "mch_time": self.mch_time[batch_idx],
            "job_time": self.job_time[batch_idx],
            "mchsEndTimes": mchs_end_times,
            "opIDsOnMchs": self.opIDsOnMchs[batch_idx],
            "input_min": input_min,
            "input_max": input_max,
            "LBm": self.LBm[batch_idx],
            "finished_mark": finished_mark,
            "available_jobs": available_jobs,
            "remain_num": remain_num,
            "remain_work_min": remain_work_min,
            "machine_queue_len": machine_queue_len,
            "machine_len": self.machine_len[batch_idx],
            "op_start_time": self.op_start_time[batch_idx],
            "op_end_time": self.op_end_time[batch_idx],
            "op_machine": self.op_machine[batch_idx],
            "op_pos_on_machine": self.op_pos_on_machine[batch_idx],
            # task-map fast path
            "task_to_row": self.op_to_job_row[batch_idx],
            "task_to_col": self.op_to_job_col[batch_idx],
        }
        self._runtime_state_cache[cache_key] = state
        return self._copy_runtime_state(state) if copy_arrays else state

    def export_state(self, batch_idx=0, copy_arrays: bool = False):
        return self._get_runtime_state(batch_idx=batch_idx, copy_arrays=copy_arrays)

    def _copy_runtime_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in state.items():
            out[k] = v.copy() if isinstance(v, np.ndarray) else v
        return out

    # ------------------------------------------------------------------
    # Next dispatch views
    # ------------------------------------------------------------------
    def _compute_next_dispatch_views(self, batch_idx: int, rule_name: str, done: bool):
        state = self._get_runtime_state(batch_idx=batch_idx, copy_arrays=False)

        selected_tasks, next_mask, local_m_masks, _, _ = dispatch_decoupled(
            mch_time=state["mch_time"],
            job_time=state["job_time"],
            mchs_end_times=state["mchsEndTimes"],
            number_of_machines=state["number_of_machines"],
            dur=state["dur"],
            temp=state["temp1"],
            omega=state["omega"],
            mask_last=state["mask"],
            done=done,
            mask_mch=state["mask_mch"],
            num_operation=state["num_operation"],
            dispatched_num_opera=state["dispatched_num_opera"],
            input_min=state["input_min"],
            job_col=state["job_col"],
            input_max=state["input_max"],
            rule=rule_name,
            last_col=state["last_col"],
            first_col=state["first_col"],
            topk_jobs=self.rule_topk_jobs,
            topk_machines=self.rule_topk_machines,
            normalized_mch_time=state["mch_time"],
            normalized_job_time=state["job_time"],
            machine_queue_len=state["machine_queue_len"],
            task_to_row=state["task_to_row"],
            task_to_col=state["task_to_col"],
        )
        return next_mask, local_m_masks, selected_tasks, state

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------
    def get_rule_candidate_pairs(self, batch_idx=0):
        self._validate_batch_idx(batch_idx)
        state = self._get_runtime_state(batch_idx=batch_idx, copy_arrays=False)
        self._last_candidate_refine_stats = {}

        available_jobs = state["available_jobs"]
        if available_jobs.size == 0:
            return [], []

        normalized_mch_time = np.asarray(state["mch_time"], dtype=np.float32)
        normalized_job_time = np.asarray(state["job_time"], dtype=np.float32)

        no_rule_mode = self.candidate_generation_mode in {"no_rule", "norule", "none", "uniform"}
        if no_rule_mode:
            task_pool, task_sources = self._build_no_rule_task_pool(state, available_jobs)
        else:
            rule_state = prepare_rule_state(
                job_time=normalized_job_time,
                num_operation=state["num_operation"],
                dispatched_num_opera=state["dispatched_num_opera"],
                input_min=state["input_min"],
                job_col=state["job_col"],
                input_max=state["input_max"],
                dur=state["dur"],
                mask_mch=state["mask_mch"],
            )
            task_pool, task_sources = self._build_task_pool(state, rule_state, available_jobs)
        if len(task_pool) == 0:
            fallback_pair = self._fallback_candidate_pair_from_state(state)
            if fallback_pair is None:
                return [], []
            return [fallback_pair], [["FALLBACK"]]

        if no_rule_mode:
            rule_pairs, rule_sources = self._build_no_rule_recall_pairs(
                batch_idx=batch_idx,
                state=state,
                task_pool=task_pool,
            )
        else:
            machine_state = {
                "mch_time": normalized_mch_time,
                "job_time": normalized_job_time,
                "machine_queue_len": state["machine_queue_len"],
                "dur": np.asarray(state["dur"], dtype=np.float32),
                "mask_mch": np.asarray(state["mask_mch"]).astype(bool),
                "last_col": np.asarray(state["last_col"], dtype=np.int64),
                "first_col": np.asarray(state["first_col"], dtype=np.int64),
                "task_to_row": np.asarray(state["task_to_row"], dtype=np.int64),
                "task_to_col": np.asarray(state["task_to_col"], dtype=np.int64),
                "number_of_machines": int(state["number_of_machines"]),
            }
            rule_pairs, rule_sources = self._build_rule_recall_pairs(
                batch_idx=batch_idx,
                state=state,
                task_pool=task_pool,
                task_sources=task_sources,
                machine_state=machine_state,
            )
        critical_pairs, critical_sources = self._build_critical_pairs_from_state(state=state, task_pool=task_pool)
        explore_pairs = self._build_exploration_pairs_from_state(state)
        explore_sources = [["EXPLORE"] for _ in explore_pairs]

        candidate_pairs, pair_sources = self._merge_candidate_pairs_with_sources(
            pairs_with_sources=[
                (rule_pairs, rule_sources),
                (critical_pairs, critical_sources),
                (explore_pairs, explore_sources),
            ]
        )

        original_candidate_count = len(candidate_pairs)
        if self.two_stage_refine_enabled and len(candidate_pairs) > 0:
            if self.agent_c_candidate_v2_enabled:
                candidate_pairs, pair_sources = self._budgeted_refine_candidates(
                    state=state,
                    candidate_pairs=candidate_pairs,
                    pair_sources=pair_sources,
                )
            else:
                candidate_pairs, pair_sources = self._safe_two_stage_refine_candidates(
                    state=state,
                    candidate_pairs=candidate_pairs,
                    pair_sources=pair_sources,
                )

        if len(candidate_pairs) == 0:
            fallback_pair = self._fallback_candidate_pair_from_state(state)
            if fallback_pair is not None:
                candidate_pairs = [fallback_pair]
                pair_sources = [["FALLBACK"]]

        if not self._last_candidate_refine_stats:
            self._last_candidate_refine_stats = {
                "original_candidate_count": float(original_candidate_count),
                "refined_candidate_count": float(len(candidate_pairs)),
                "num_ops_covered_after_refine": float(len({int(op) for op, _ in candidate_pairs})),
                "num_global_kept_pairs": 0.0,
                "num_explore_pairs_kept": float(sum(1 for src in pair_sources if "EXPLORE" in src)),
                "source_rule_pairs": float(sum(1 for src in pair_sources if any(("_" in s) for s in src))),
                "source_critical_pairs": float(sum(1 for src in pair_sources if any(("CRITICAL" in s) for s in src))),
                "source_explore_pairs": float(sum(1 for src in pair_sources if "EXPLORE" in src)),
            }

        if self.candidate_debug_print:
            st = self._last_candidate_refine_stats
            print(
                "[candidate-v2] "
                f"orig={int(st.get('original_candidate_count', 0))}, "
                f"refined={int(st.get('refined_candidate_count', 0))}, "
                f"ops={int(st.get('num_ops_covered_after_refine', 0))}, "
                f"rule={int(st.get('source_rule_pairs', 0))}, "
                f"critical={int(st.get('source_critical_pairs', 0))}, "
                f"explore={int(st.get('source_explore_pairs', 0))}",
                flush=True,
            )

        return candidate_pairs, pair_sources

    def _build_task_pool(self, state, rule_state, available_jobs):
        task_pool_set = set()
        task_sources: Dict[int, List[str]] = {}

        for job_rule_name in self.active_job_rule_names:
            score_func = get_job_rule_func(job_rule_name)
            scores = score_func(rule_state)
            selected_jobs = select_topk_job_indices(
                scores=scores,
                available_jobs=available_jobs,
                mode=get_job_rule_mode(job_rule_name),
                topk_jobs=self.rule_topk_jobs,
            )
            selected_tasks = np.asarray(state["omega"][selected_jobs], dtype=np.int64).reshape(-1)
            for op_id in selected_tasks.tolist():
                op_id = int(op_id)
                if op_id not in task_pool_set:
                    task_pool_set.add(op_id)
                    task_sources[op_id] = [job_rule_name]
                else:
                    task_sources[op_id].append(job_rule_name)

        task_pool = list(task_pool_set)
        task_pool.sort(key=lambda op: self._estimate_best_completion_for_task_from_state(op, state))
        return task_pool, task_sources

    def _build_no_rule_task_pool(self, state, available_jobs):
        task_pool: List[int] = []
        task_sources: Dict[int, List[str]] = {}
        for job_idx in available_jobs.tolist():
            op_id = int(state["omega"][int(job_idx)])
            task_pool.append(op_id)
            task_sources[op_id] = ["NO_RULE_OP"]
        task_pool.sort()
        return task_pool, task_sources

    def _estimate_best_completion_for_task_from_state(self, op_id: int, state) -> float:
        row, col = self._task_to_row_col(op_id, batch_idx=state["batch_idx"])
        feasible = self._get_feasible_machine_indices_from_state(state, row=row, col=col)
        if feasible.size == 0:
            return float("inf")
        job_ready = float(state["job_time"][row])

        best = float("inf")
        for mch_id in feasible.tolist():
            proc = float(state["dur"][row, col, mch_id])
            est = max(job_ready, float(state["mch_time"][mch_id])) + proc
            if est < best:
                best = est
        return best

    def _collect_pairs_for_single_rule(self, op_id: int, mch_list: List[int], state) -> List[Tuple[int, int]]:
        pairs: List[Tuple[int, int, float, float]] = []
        row, col = self._task_to_row_col(op_id, batch_idx=state["batch_idx"])
        job_ready_time = float(state["job_time"][row])

        for mch_id in mch_list[: self.max_machines_per_task]:
            proc = float(state["dur"][row, col, mch_id])
            mch_available = float(state["mch_time"][mch_id])
            est_start = max(job_ready_time, mch_available)
            est_end = est_start + proc
            pairs.append((int(op_id), int(mch_id), est_end, proc))

        pairs.sort(key=lambda x: (x[2], x[3], x[0], x[1]))
        return [(op_id_, mch_id_) for op_id_, mch_id_, _, _ in pairs[: self.max_pairs_per_rule]]

    def _sanitize_ranked_machines(self, ranked, feasible_base, row, col, state):
        if ranked is None:
            ranked = []
        feasible_set = set(int(m) for m in feasible_base.tolist())
        out = []
        for m in ranked:
            m = int(m)
            if m in feasible_set and state["dur"][row, col, m] > 0:
                out.append(m)
        if len(out) == 0:
            return []
        seen = set()
        unique_out = []
        for m in out:
            if m not in seen:
                seen.add(m)
                unique_out.append(m)
        return unique_out

    def _get_feasible_machine_indices_from_state(self, state, row: int, col: int):
        return np.where((state["dur"][row, col] > 0) & (~state["mask_mch"][row, col]))[0].astype(np.int64)

    def _build_exploration_pairs_from_state(self, state):
        if self.extra_explore_pairs <= 0:
            return []

        candidate_scores = []
        for job_idx in state["available_jobs"].tolist():
            op_id = int(state["omega"][job_idx])
            row, col = self._task_to_row_col(op_id, batch_idx=state["batch_idx"])
            feasible = self._get_feasible_machine_indices_from_state(state, row=row, col=col)
            if feasible.size == 0:
                continue

            job_ready = float(state["job_time"][row])
            for mch_id in feasible.tolist():
                proc = float(state["dur"][row, col, mch_id])
                ect = max(job_ready, float(state["mch_time"][mch_id])) + proc
                candidate_scores.append((op_id, int(mch_id), proc, ect))

        if len(candidate_scores) == 0:
            return []

        result = []
        best_spt = min(candidate_scores, key=lambda x: (x[2], x[3], x[0], x[1]))
        result.append((int(best_spt[0]), int(best_spt[1])))

        best_ect = min(candidate_scores, key=lambda x: (x[3], x[2], x[0], x[1]))
        result.append((int(best_ect[0]), int(best_ect[1])))

        dedup = []
        seen = set()
        for p in result:
            if p not in seen:
                seen.add(p)
                dedup.append(p)
        return dedup[: self.extra_explore_pairs]

    def _build_rule_recall_pairs(self, batch_idx, state, task_pool, task_sources, machine_state):
        candidate_pairs: List[Tuple[int, int]] = []
        pair_rule_names: List[str] = []

        for op_id in task_pool:
            row, col = self._task_to_row_col(int(op_id), batch_idx=batch_idx)
            feasible_base = self._get_feasible_machine_indices_from_state(state, row=row, col=col)
            if feasible_base.size == 0:
                continue

            machine_candidates_by_rule: Dict[str, List[int]] = {}
            for machine_rule_name in self.active_machine_rule_names:
                machine_rank_func = get_machine_rule_func(machine_rule_name)
                ranked = machine_rank_func(machine_state, int(op_id))
                ranked = self._sanitize_ranked_machines(
                    ranked=ranked,
                    feasible_base=feasible_base,
                    row=row,
                    col=col,
                    state=state,
                )
                if len(ranked) == 0:
                    ranked = feasible_base.tolist()
                machine_candidates_by_rule[machine_rule_name] = ranked[: max(self.rule_topk_machines, 1)]

            job_rule_names = task_sources.get(int(op_id), ["UNKNOWN_JOB_RULE"])
            for job_rule_name in job_rule_names:
                for machine_rule_name, mch_list in machine_candidates_by_rule.items():
                    pairs = self._collect_pairs_for_single_rule(op_id=int(op_id), mch_list=mch_list, state=state)
                    source_name = f"{job_rule_name}_{machine_rule_name}"
                    for pair in pairs:
                        candidate_pairs.append(pair)
                        pair_rule_names.append(source_name)

        pairs, sources = self._deduplicate_candidate_pairs(candidate_pairs, pair_rule_names)
        return pairs, sources

    def _build_no_rule_recall_pairs(self, batch_idx, state, task_pool):
        candidate_pairs: List[Tuple[int, int]] = []
        pair_rule_names: List[str] = []
        keep_machines = max(int(self.rule_topk_machines), 1)
        keep_pairs = max(int(self.max_pairs_per_rule), 1)

        for op_id in task_pool:
            row, col = self._task_to_row_col(int(op_id), batch_idx=batch_idx)
            feasible = self._get_feasible_machine_indices_from_state(state, row=row, col=col)
            if feasible.size == 0:
                continue
            kept = sorted(int(m) for m in feasible.tolist())[:keep_machines]
            for mch_id in kept[:keep_pairs]:
                candidate_pairs.append((int(op_id), int(mch_id)))
                pair_rule_names.append("NO_RULE")

        pairs, sources = self._deduplicate_candidate_pairs(candidate_pairs, pair_rule_names)
        return pairs, sources

    def _build_critical_pairs_from_state(self, state, task_pool):
        if not self.critical_pairs_enabled:
            return [], []

        pairs: List[Tuple[int, int]] = []
        sources: List[List[str]] = []
        keep_per_op = max(int(self.critical_pairs_per_op), 1)

        for op_id in task_pool:
            row, col = self._task_to_row_col(int(op_id), batch_idx=state["batch_idx"])
            feasible = self._get_feasible_machine_indices_from_state(state, row=row, col=col)
            if feasible.size == 0:
                continue

            scored = []
            for mch_id in feasible.tolist():
                pt = float(state["dur"][row, col, mch_id])
                ect = max(float(state["job_time"][row]), float(state["mch_time"][mch_id])) + pt
                q = float(state["machine_queue_len"][mch_id])
                scored.append((int(mch_id), ect, pt, q))

            scored.sort(key=lambda x: (x[1], x[2], x[3], x[0]))
            if len(scored) > 0:
                best_m = int(scored[0][0])
                pairs.append((int(op_id), best_m))
                sources.append(["CRITICAL_ECT"])
            if len(scored) > 1 and keep_per_op >= 2:
                alt_m = int(scored[1][0])
                pairs.append((int(op_id), alt_m))
                sources.append(["CRITICAL_ALT"])

        return pairs, sources

    def _merge_candidate_pairs_with_sources(self, pairs_with_sources):
        merged_pairs: List[Tuple[int, int]] = []
        merged_sources: List[List[str]] = []
        index_map: Dict[int, int] = {}

        for pairs, sources in pairs_with_sources:
            pairs = pairs or []
            sources = sources or []
            for i, pair in enumerate(pairs):
                op_id = int(pair[0])
                mch_id = int(pair[1])
                src_names = list(sources[i]) if i < len(sources) else []
                code = op_id * self.number_of_machines + mch_id
                if code not in index_map:
                    index_map[code] = len(merged_pairs)
                    merged_pairs.append((op_id, mch_id))
                    merged_sources.append(src_names)
                else:
                    idx = index_map[code]
                    merged_sources[idx].extend(src_names)

        normalized = []
        for src in merged_sources:
            seen = set()
            uniq = []
            for s in src:
                name = str(s)
                if name not in seen:
                    seen.add(name)
                    uniq.append(name)
            normalized.append(uniq)

        return merged_pairs, normalized

    def _compute_pair_light_score(self, pair_info, norm_ctx):
        score = (
            self.refine_score_w_eet * float(norm_ctx["ect"][pair_info["idx"]])
            + self.refine_score_w_pt * float(norm_ctx["pt"][pair_info["idx"]])
            + self.refine_score_w_queue * float(norm_ctx["queue"][pair_info["idx"]])
            - self.refine_score_w_support * float(norm_ctx["support"][pair_info["idx"]])
        )
        if pair_info.get("has_explore", False):
            score -= float(self.refine_explore_bonus)
        return float(score)

    def _budgeted_refine_candidates(self, state, candidate_pairs, pair_sources):
        if len(candidate_pairs) == 0:
            return candidate_pairs, pair_sources

        pair_infos = []
        eps = 1e-8
        for idx, pair in enumerate(candidate_pairs):
            op_id = int(pair[0])
            mch_id = int(pair[1])
            src = list(pair_sources[idx]) if idx < len(pair_sources) else []
            row, col = self._task_to_row_col(op_id, batch_idx=state["batch_idx"])
            pt = float(state["dur"][row, col, mch_id])
            q = float(state["machine_queue_len"][mch_id])
            ect = max(float(state["job_time"][row]), float(state["mch_time"][mch_id])) + pt
            support = float(len(set(src)))
            pair_infos.append(
                {
                    "idx": idx,
                    "pair": (op_id, mch_id),
                    "sources": src,
                    "op_id": op_id,
                    "mch_id": mch_id,
                    "ect": ect,
                    "pt": pt,
                    "queue": q,
                    "support": support,
                    "has_explore": "EXPLORE" in src,
                }
            )

        def _norm(values):
            arr = np.asarray(values, dtype=np.float32)
            if arr.size == 0:
                return arr
            mn = float(np.min(arr))
            mx = float(np.max(arr))
            if mx - mn <= eps:
                return np.zeros_like(arr, dtype=np.float32)
            return ((arr - mn) / (mx - mn)).astype(np.float32)

        norm_ctx = {
            "ect": _norm([x["ect"] for x in pair_infos]),
            "pt": _norm([x["pt"] for x in pair_infos]),
            "queue": _norm([x["queue"] for x in pair_infos]),
            "support": _norm([x["support"] for x in pair_infos]),
        }
        for info in pair_infos:
            info["score"] = self._compute_pair_light_score(info, norm_ctx)

        by_op: Dict[int, List[Dict[str, Any]]] = {}
        for info in pair_infos:
            by_op.setdefault(info["op_id"], []).append(info)

        selected_idx = set()
        ordered = []

        max_per_op = max(int(self.refine_max_pairs_per_op), 1)
        for op_id in sorted(by_op.keys()):
            group = by_op[op_id]
            by_ect = sorted(group, key=lambda x: (x["ect"], x["pt"], x["score"], x["mch_id"], x["idx"]))
            by_score = sorted(group, key=lambda x: (x["score"], x["ect"], x["pt"], x["mch_id"], x["idx"]))

            keep = []
            if len(by_ect) > 0:
                keep.append(by_ect[0])
            if len(by_ect) > 1:
                keep.append(by_ect[1])

            for item in keep:
                if item["idx"] not in selected_idx:
                    selected_idx.add(item["idx"])
                    ordered.append(item["idx"])

            for item in by_score:
                op_count = sum(1 for i in ordered if pair_infos[i]["op_id"] == op_id)
                if op_count >= max_per_op:
                    break
                if item["idx"] not in selected_idx:
                    selected_idx.add(item["idx"])
                    ordered.append(item["idx"])

        # machine quota
        max_per_machine = int(self.refine_max_pairs_per_machine)
        if max_per_machine > 0:
            filtered = []
            mch_cnt: Dict[int, int] = {}
            for i in ordered:
                m = int(pair_infos[i]["mch_id"])
                c = mch_cnt.get(m, 0)
                if c < max_per_machine:
                    filtered.append(i)
                    mch_cnt[m] = c + 1
            ordered = filtered
            selected_idx = set(ordered)

        global_kept = 0
        if int(self.refine_keep_global_best_ect) > 0:
            best_ect = min(pair_infos, key=lambda x: (x["ect"], x["pt"], x["score"], x["idx"]))
            if best_ect["idx"] not in selected_idx:
                ordered.append(best_ect["idx"])
                selected_idx.add(best_ect["idx"])
                global_kept += 1
        if int(self.refine_keep_global_best_pt) > 0:
            best_pt = min(pair_infos, key=lambda x: (x["pt"], x["ect"], x["score"], x["idx"]))
            if best_pt["idx"] not in selected_idx:
                ordered.append(best_pt["idx"])
                selected_idx.add(best_pt["idx"])
                global_kept += 1

        # global reserve and diversity
        reserve = max(int(self.refine_global_reserve_size), 0)
        if reserve > 0:
            ranked_global = sorted(pair_infos, key=lambda x: (x["score"], x["ect"], x["pt"], x["idx"]))
            added = 0
            for item in ranked_global:
                if item["idx"] in selected_idx:
                    continue
                ordered.append(item["idx"])
                selected_idx.add(item["idx"])
                added += 1
                if added >= reserve:
                    break

        min_div_m = max(int(self.refine_diversity_min_machines), 1)
        current_m = {int(pair_infos[i]["mch_id"]) for i in ordered}
        if len(current_m) < min_div_m:
            ranked = sorted(pair_infos, key=lambda x: (x["score"], x["ect"], x["pt"], x["idx"]))
            for item in ranked:
                if item["idx"] in selected_idx:
                    continue
                if int(item["mch_id"]) not in current_m:
                    ordered.append(item["idx"])
                    selected_idx.add(item["idx"])
                    current_m.add(int(item["mch_id"]))
                if len(current_m) >= min_div_m:
                    break

        # keep a few explore pairs
        keep_explore_min = max(int(self.refine_keep_explore_min), 0)
        if keep_explore_min > 0:
            kept_explore = sum(1 for i in ordered if pair_infos[i]["has_explore"])
            if kept_explore < keep_explore_min:
                ranked_explore = [x for x in sorted(pair_infos, key=lambda y: (y["score"], y["ect"], y["pt"], y["idx"])) if x["has_explore"]]
                for item in ranked_explore:
                    if item["idx"] in selected_idx:
                        continue
                    ordered.append(item["idx"])
                    selected_idx.add(item["idx"])
                    kept_explore += 1
                    if kept_explore >= keep_explore_min:
                        break

        refined_pairs = [pair_infos[i]["pair"] for i in ordered]
        refined_sources = [pair_infos[i]["sources"] for i in ordered]
        dedup_pairs, dedup_sources = self._merge_candidate_pairs_with_sources([(refined_pairs, refined_sources)])

        self._last_candidate_refine_stats = {
            "original_candidate_count": float(len(candidate_pairs)),
            "refined_candidate_count": float(len(dedup_pairs)),
            "num_ops_covered_after_refine": float(len({int(op) for op, _ in dedup_pairs})),
            "num_global_kept_pairs": float(global_kept),
            "num_explore_pairs_kept": float(sum(1 for src in dedup_sources if "EXPLORE" in src)),
            "source_rule_pairs": float(sum(1 for src in dedup_sources if any(("_" in s) for s in src))),
            "source_critical_pairs": float(sum(1 for src in dedup_sources if any(("CRITICAL" in s) for s in src))),
            "source_explore_pairs": float(sum(1 for src in dedup_sources if "EXPLORE" in src)),
        }
        return dedup_pairs, dedup_sources

    def _safe_two_stage_refine_candidates(self, state, candidate_pairs, pair_sources):
        # compatibility fallback: old entry now routes to budgeted refine
        return self._budgeted_refine_candidates(state, candidate_pairs, pair_sources)

    # ------------------------------------------------------------------
    # Feature builders
    # ------------------------------------------------------------------
    def build_pair_features(self, batch_idx, candidate_pairs, pair_sources):
        self._validate_batch_idx(batch_idx)
        candidate_pairs = candidate_pairs or []
        pair_sources = pair_sources or []

        state = self._get_runtime_state(batch_idx=batch_idx, copy_arrays=False)
        rule_names = list(self.RULE_NAMES) + ["EXPLORE", "FALLBACK"]

        pair_machine_counts = np.zeros(self.number_of_machines, dtype=np.float32)
        pair_job_counts = np.zeros(self.number_of_jobs, dtype=np.float32)
        est_completion_list = []
        op_cache: Dict[int, Dict[str, float]] = {}

        for op_id, mch_id in candidate_pairs:
            op_id = int(op_id)
            mch_id = int(mch_id)
            row, col = self._task_to_row_col(op_id, batch_idx=batch_idx)

            pair_machine_counts[mch_id] += 1.0
            pair_job_counts[row] += 1.0

            proc = float(state["dur"][row, col, mch_id])
            ect = max(float(state["job_time"][row]), float(state["mch_time"][mch_id])) + proc
            est_completion_list.append(ect)

            if op_id not in op_cache:
                feasible_same_op = self._get_feasible_machine_indices_from_state(state, row=row, col=col)
                if feasible_same_op.size == 0:
                    feasible_same_op = np.asarray([mch_id], dtype=np.int64)

                same_op_proc = state["dur"][row, col, feasible_same_op]
                min_same_op_proc = float(np.min(same_op_proc))
                mean_same_op_proc = float(np.mean(same_op_proc))
                max_same_op_proc = float(np.max(same_op_proc))

                num_op_job = int(state["num_operation"][row])
                dispatched = int(state["dispatched_num_opera"][row])
                remaining_num_operations = max(num_op_job - dispatched, 0)
                remaining_work_min = float(state["remain_work_min"][row])
                finished_ratio = float(dispatched) / float(max(num_op_job, 1))

                op_cache[op_id] = {
                    "row": row,
                    "col": col,
                    "num_feasible_machines": float(len(feasible_same_op)),
                    "min_same_op_proc": min_same_op_proc,
                    "mean_same_op_proc": mean_same_op_proc,
                    "max_same_op_proc": max_same_op_proc,
                    "proc_gap": float(max_same_op_proc - min_same_op_proc),
                    "remaining_num_operations": float(remaining_num_operations),
                    "remaining_work_min": float(remaining_work_min),
                    "finished_ratio": float(finished_ratio),
                }

        min_candidate_completion = min(est_completion_list) if est_completion_list else 1.0
        mean_machine_load = float(np.mean(state["mch_time"])) if len(state["mch_time"]) > 0 else 0.0
        max_machine_queue = float(np.max(state["machine_queue_len"])) if len(state["machine_queue_len"]) > 0 else 1.0

        feat_rows = []
        for idx, (op_id, mch_id) in enumerate(candidate_pairs):
            op_id = int(op_id)
            mch_id = int(mch_id)

            info = op_cache[op_id]
            row = int(info["row"])
            col = int(info["col"])

            processing_time = float(state["dur"][row, col, mch_id])
            machine_available_time = float(state["mch_time"][mch_id])
            job_ready_time = float(state["job_time"][row])
            estimated_start_time = max(job_ready_time, machine_available_time)
            estimated_completion_time = estimated_start_time + processing_time

            machine_load = machine_available_time
            machine_queue_len = float(state["machine_queue_len"][mch_id])

            min_same_op_proc = float(info["min_same_op_proc"])
            mean_same_op_proc = float(info["mean_same_op_proc"])
            max_same_op_proc = float(info["max_same_op_proc"])
            proc_gap = float(info["proc_gap"])
            num_feasible_machines = float(info["num_feasible_machines"])

            op_id_norm = float(op_id) / float(max(self.number_of_tasks - 1, 1))
            machine_id_norm = float(mch_id) / float(max(self.number_of_machines - 1, 1))

            pair_sources_names = pair_sources[idx] if idx < len(pair_sources) else []
            source_multihot = self._build_rule_source_multihot(pair_sources_names, rule_names)

            is_bottleneck_machine = float(mch_id == int(np.argmax(state["mch_time"]))) if self.number_of_machines > 0 else 0.0

            feat_vec = [
                processing_time,
                machine_available_time,
                job_ready_time,
                estimated_start_time,
                estimated_completion_time,
                float(info["remaining_num_operations"]),
                float(info["remaining_work_min"]),
                float(info["finished_ratio"]),
                machine_load,
                op_id_norm,
                machine_id_norm,
                processing_time / max(min_same_op_proc, 1e-6),
                processing_time / max(mean_same_op_proc, 1e-6),
                estimated_completion_time / max(min_candidate_completion, 1e-6),
                machine_load / max(mean_machine_load, 1e-6) if mean_machine_load > 1e-8 else 0.0,
                float(pair_machine_counts[mch_id]),
                float(pair_job_counts[row]),
                num_feasible_machines,
                is_bottleneck_machine,
                machine_queue_len,
                machine_queue_len / max(max_machine_queue, 1e-6),
                proc_gap,
                (processing_time - min_same_op_proc) / max(proc_gap, 1e-6) if proc_gap > 1e-8 else 0.0,
                max_same_op_proc / max(min_same_op_proc, 1e-6),
            ] + source_multihot
            feat_rows.append(feat_vec)

        pair_feat_dim = 24 + len(rule_names)
        if len(feat_rows) == 0:
            return np.zeros((0, pair_feat_dim), dtype=np.float32)
        return np.asarray(feat_rows, dtype=np.float32)

    def get_global_features(self, batch_idx=0, precomputed_candidate_pairs=None, precomputed_pair_sources=None):
        del precomputed_pair_sources
        self._validate_batch_idx(batch_idx)
        state = self._get_runtime_state(batch_idx=batch_idx, copy_arrays=False)

        remain_num = state["remain_num"].astype(np.float32)
        remain_work_min = state["remain_work_min"].astype(np.float32)

        mch_time_norm = self._safe_normalize(state["mch_time"].astype(np.float32))
        job_time_norm = self._safe_normalize(state["job_time"].astype(np.float32))
        remain_num_norm = self._safe_normalize(remain_num)
        remain_work_norm = self._safe_normalize(remain_work_min)
        queue_len_norm = self._safe_normalize(state["machine_queue_len"].astype(np.float32))

        total_remaining_work = float(remain_work_min.sum())
        total_remaining_work_norm = total_remaining_work / float(max(state["input_min"].sum(), 1.0))
        num_available_jobs = int(np.sum(~state["mask"]))
        num_available_jobs_norm = float(num_available_jobs) / float(max(self.number_of_jobs, 1))
        finished_task_ratio = float(np.sum(state["finished_mark"] > 0)) / float(max(self.number_of_tasks, 1))

        candidate_pairs = precomputed_candidate_pairs
        if candidate_pairs is None:
            candidate_pairs, _ = self.get_rule_candidate_pairs(batch_idx=batch_idx)

        unique_jobs = len({self._task_to_row_col(op_id, batch_idx=batch_idx)[0] for op_id, _ in candidate_pairs}) if candidate_pairs else 0
        unique_machines = len({mch_id for _, mch_id in candidate_pairs}) if candidate_pairs else 0
        candidate_count = len(candidate_pairs)

        machine_load_stats = np.array([
            np.mean(state["mch_time"]) if len(state["mch_time"]) > 0 else 0.0,
            np.std(state["mch_time"]) if len(state["mch_time"]) > 0 else 0.0,
            np.max(state["mch_time"]) if len(state["mch_time"]) > 0 else 0.0,
        ], dtype=np.float32)

        job_ready_stats = np.array([
            np.mean(state["job_time"]) if len(state["job_time"]) > 0 else 0.0,
            np.std(state["job_time"]) if len(state["job_time"]) > 0 else 0.0,
            np.max(state["job_time"]) if len(state["job_time"]) > 0 else 0.0,
        ], dtype=np.float32)

        queue_stats = np.array([
            np.mean(state["machine_queue_len"]) if len(state["machine_queue_len"]) > 0 else 0.0,
            np.std(state["machine_queue_len"]) if len(state["machine_queue_len"]) > 0 else 0.0,
            np.max(state["machine_queue_len"]) if len(state["machine_queue_len"]) > 0 else 0.0,
        ], dtype=np.float32)

        if bool(self.fixed_global_feat):
            mch_time_stats = self._summarize_fixed_global_vector(state["mch_time"])
            job_time_stats = self._summarize_fixed_global_vector(state["job_time"])
            remain_num_stats = self._summarize_fixed_global_vector(remain_num)
            remain_work_stats = self._summarize_fixed_global_vector(remain_work_min)
            queue_len_stats = self._summarize_fixed_global_vector(state["machine_queue_len"])

            candidate_per_available_job_raw = float(candidate_count) / float(max(num_available_jobs, 1))
            candidate_per_machine_raw = float(candidate_count) / float(max(self.number_of_machines, 1))
            candidate_density = float(candidate_count) / float(max(num_available_jobs * self.number_of_machines, 1))

            global_feat = np.concatenate([
                mch_time_stats,
                job_time_stats,
                remain_num_stats,
                remain_work_stats,
                queue_len_stats,
                self._safe_normalize(machine_load_stats),
                self._safe_normalize(job_ready_stats),
                self._safe_normalize(queue_stats),
                np.array([
                    total_remaining_work_norm,
                    num_available_jobs_norm,
                    finished_task_ratio,
                    float(candidate_count) / float(max(len(self.RULE_NAMES) * self.max_pairs_per_rule + self.extra_explore_pairs, 1)),
                    float(unique_jobs) / float(max(self.number_of_jobs, 1)),
                    float(unique_machines) / float(max(self.number_of_machines, 1)),
                    candidate_density,
                    float(np.log1p(candidate_per_available_job_raw)),
                    float(np.log1p(candidate_per_machine_raw)),
                    float(np.log1p(candidate_count)),
                ], dtype=np.float32),
            ]).astype(np.float32)
            return global_feat

        global_feat = np.concatenate([
            mch_time_norm,
            job_time_norm,
            remain_num_norm,
            remain_work_norm,
            queue_len_norm,
            np.array([
                total_remaining_work_norm,
                num_available_jobs_norm,
                finished_task_ratio,
                float(candidate_count) / float(max(len(self.RULE_NAMES) * self.max_pairs_per_rule + self.extra_explore_pairs, 1)),
                float(unique_jobs) / float(max(self.number_of_jobs, 1)),
                float(unique_machines) / float(max(self.number_of_machines, 1)),
            ], dtype=np.float32),
            self._safe_normalize(machine_load_stats),
            self._safe_normalize(job_ready_stats),
            self._safe_normalize(queue_stats),
        ]).astype(np.float32)

        return global_feat

    def _summarize_fixed_global_vector(self, x):
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return np.zeros((18,), dtype=np.float32)

        scale = float(np.max(np.abs(arr)))
        scale = max(scale, 1e-6)
        norm = arr / scale

        q10, q25, q50, q75, q90 = np.percentile(norm, [10.0, 25.0, 50.0, 75.0, 90.0]).astype(np.float32)
        min_v = float(np.min(norm))
        max_v = float(np.max(norm))
        mu = float(np.mean(norm))
        std = float(np.std(norm))
        centered = norm - mu
        denom = max(std, 1e-6)

        skew = float(np.mean((centered / denom) ** 3))
        kurt = float(np.mean((centered / denom) ** 4) - 3.0)
        mean_abs = float(np.mean(np.abs(norm)))
        rms = float(np.sqrt(np.mean(norm ** 2)))
        zero_ratio = float(np.mean(np.abs(norm) <= 1e-6))

        if max_v - min_v > 1e-8:
            hist, _ = np.histogram(norm, bins=8, range=(min_v, max_v))
            p = hist.astype(np.float32) / max(float(np.sum(hist)), 1.0)
            entropy = float(-np.sum(p * np.log(p + 1e-8)) / np.log(8.0))
        else:
            entropy = 0.0

        return np.asarray(
            [
                mu,
                std,
                min_v,
                float(q10),
                float(q25),
                float(q50),
                float(q75),
                float(q90),
                max_v,
                max_v - min_v,
                mean_abs,
                rms,
                zero_ratio,
                skew,
                kurt,
                entropy,
                float(np.log1p(scale)),
                float(np.log1p(arr.size)),
            ],
            dtype=np.float32,
        )

    def build_op_node_features(self, batch_idx=0):
        self._validate_batch_idx(batch_idx)
        state = self._get_runtime_state(batch_idx=batch_idx, copy_arrays=False)

        max_proc = float(np.max(state["dur"])) if state["dur"].size > 0 else 1.0
        max_job_time = float(np.max(state["job_time"])) if state["job_time"].size > 0 else 1.0
        max_remain_work = float(np.max(state["remain_work_min"])) if state["remain_work_min"].size > 0 else 1.0
        max_remain_num = float(np.max(state["remain_num"])) if state["remain_num"].size > 0 else 1.0
        max_lb = float(np.max(self.LBm[batch_idx])) if self.LBm[batch_idx].size > 0 else 1.0

        feats = np.zeros((self.number_of_tasks, 10), dtype=np.float32)
        for op_id in range(self.number_of_tasks):
            row, col = self._task_to_row_col(op_id, batch_idx=batch_idx)
            feasible = self._get_feasible_machine_indices_from_state(state, row=row, col=col)
            if feasible.size > 0:
                proc_vals = state["dur"][row, col, feasible]
                proc_min = float(np.min(proc_vals))
                proc_mean = float(np.mean(proc_vals))
            else:
                proc_min = 0.0
                proc_mean = 0.0

            current_op = int(state["omega"][row]) if not bool(state["mask"][row]) else -1
            is_candidate_head = 1.0 if int(op_id) == int(current_op) else 0.0

            num_op_job = int(state["num_operation"][row])
            dispatched = int(state["dispatched_num_opera"][row])
            finished_ratio = float(dispatched) / float(max(num_op_job, 1))

            feats[op_id] = np.asarray(
                [
                    proc_min / max(max_proc, 1e-6),
                    proc_mean / max(max_proc, 1e-6),
                    float(state["job_time"][row]) / max(max_job_time, 1e-6),
                    float(state["remain_work_min"][row]) / max(max_remain_work, 1e-6),
                    float(state["remain_num"][row]) / max(max_remain_num, 1e-6),
                    finished_ratio,
                    float(feasible.size) / float(max(self.number_of_machines, 1)),
                    float(self.finished_mark[batch_idx, op_id]),
                    is_candidate_head,
                    float(self.LBm[batch_idx, op_id]) / max(max_lb, 1e-6),
                ],
                dtype=np.float32,
            )
        return feats

    def build_machine_node_features(self, batch_idx=0):
        self._validate_batch_idx(batch_idx)
        state = self._get_runtime_state(batch_idx=batch_idx, copy_arrays=False)

        max_mch_time = float(np.max(state["mch_time"])) if state["mch_time"].size > 0 else 1.0
        max_queue = float(np.max(state["machine_queue_len"])) if state["machine_queue_len"].size > 0 else 1.0
        max_workload = float(np.max(self.machine_workload[batch_idx])) if self.machine_workload[batch_idx].size > 0 else 1.0
        min_job_ready = float(np.min(state["job_time"])) if state["job_time"].size > 0 else 0.0

        feats = np.zeros((self.number_of_machines, 6), dtype=np.float32)
        for mch_id in range(self.number_of_machines):
            mch_time = float(state["mch_time"][mch_id])
            queue_len = float(state["machine_queue_len"][mch_id])
            workload = float(self.machine_workload[batch_idx, mch_id])
            is_bottleneck = 1.0 if mch_time >= float(np.max(state["mch_time"])) else 0.0
            is_idle = 1.0 if mch_time <= min_job_ready else 0.0

            feats[mch_id] = np.asarray(
                [
                    mch_time / max(max_mch_time, 1e-6),
                    queue_len / max(max_queue, 1e-6),
                    workload / max(max_workload, 1e-6),
                    is_bottleneck,
                    is_idle,
                    float(mch_id) / float(max(self.number_of_machines - 1, 1)),
                ],
                dtype=np.float32,
            )
        return feats

    # ------------------------------------------------------------------
    # Agent_C observation
    # ------------------------------------------------------------------
    def get_agent_c_obs(self, batch_idx=0):
        self._validate_batch_idx(batch_idx)
        cache_key = (int(batch_idx), int(self._state_version))
        cached = self._obs_cache.get(cache_key)
        if cached is not None:
            return cached

        candidate_pairs, pair_sources = self.get_rule_candidate_pairs(batch_idx=batch_idx)

        if len(candidate_pairs) == 0:
            fallback_pair = self._fallback_candidate_pair(batch_idx=batch_idx)
            if fallback_pair is not None:
                candidate_pairs = [fallback_pair]
                pair_sources = [["FALLBACK"]]

        pair_feat = self.build_pair_features(
            batch_idx=batch_idx,
            candidate_pairs=candidate_pairs,
            pair_sources=pair_sources,
        )
        pair_mask = np.ones(len(candidate_pairs), dtype=bool)
        pair_op_idx = np.asarray([int(op_id) for op_id, _ in candidate_pairs], dtype=np.int64)
        pair_mch_idx = np.asarray([int(mch_id) for _, mch_id in candidate_pairs], dtype=np.int64)

        candidate_set_feat = np.array([
            float(len(candidate_pairs)),
            float(len({op for op, _ in candidate_pairs})),
            float(len({m for _, m in candidate_pairs})),
        ], dtype=np.float32)

        rule_names = list(self.RULE_NAMES) + ["EXPLORE", "FALLBACK"]
        rule_feat_start = 24
        if pair_feat.ndim == 2 and pair_feat.shape[1] >= rule_feat_start + len(rule_names):
            edge_rule_to_pair_feat = np.asarray(pair_feat[:, rule_feat_start : rule_feat_start + len(rule_names)], dtype=np.float32)
        else:
            edge_rule_to_pair_feat = np.zeros((len(candidate_pairs), len(rule_names)), dtype=np.float32)

        if pair_feat.ndim == 2 and pair_feat.shape[1] >= 24:
            edge_opmch_to_pair_feat = np.asarray(
                np.stack(
                    [
                        pair_feat[:, 0],
                        pair_feat[:, 4],
                        pair_feat[:, 13],
                        pair_feat[:, 15],
                    ],
                    axis=1,
                ),
                dtype=np.float32,
            )
        else:
            edge_opmch_to_pair_feat = np.zeros((len(candidate_pairs), 4), dtype=np.float32)

        obs = {
            "global_feat": self.get_global_features(
                batch_idx=batch_idx,
                precomputed_candidate_pairs=candidate_pairs,
                precomputed_pair_sources=pair_sources,
            ),
            "pair_feat": pair_feat,
            "pair_node_feat": np.asarray(pair_feat, dtype=np.float32),
            "candidate_pairs": candidate_pairs,
            "pair_sources": pair_sources,
            "pair_mask": pair_mask,
            "candidate_set_feat": candidate_set_feat,
            "op_node_feat": self.build_op_node_features(batch_idx=batch_idx),
            "op_adj": np.asarray(self.adj[batch_idx], dtype=np.float32),
            "machine_node_feat": self.build_machine_node_features(batch_idx=batch_idx),
            "pair_op_idx": pair_op_idx,
            "pair_mch_idx": pair_mch_idx,
            "edge_op_to_pair": np.asarray(pair_op_idx, dtype=np.int64),
            "edge_mch_to_pair": np.asarray(pair_mch_idx, dtype=np.int64),
            "edge_rule_to_pair_feat": edge_rule_to_pair_feat,
            "edge_opmch_to_pair_feat": edge_opmch_to_pair_feat,
        }
        obs["legal_pairs_set"] = {(int(a), int(b)) for a, b in candidate_pairs}
        self._obs_cache[cache_key] = obs
        return obs

    # ------------------------------------------------------------------
    # Pair action
    # ------------------------------------------------------------------
    def step_with_pair(self, op_id, mch_id, batch_idx=0):
        self._validate_batch_idx(batch_idx)
        if self.batch_sie != 1 or batch_idx != 0:
            raise ValueError("step_with_pair 当前为 batch_size=1 安全版本，请使用 batch_idx=0 且 batch_sie=1")

        self.validate_pair_action(op_id=op_id, mch_id=mch_id, batch_idx=batch_idx)
        action = np.array([int(op_id)], dtype=np.int64)
        mch_a = np.array([int(mch_id)], dtype=np.int64)
        return self.step(action=action, mch_a=mch_a)

    def validate_pair_action(self, op_id, mch_id, batch_idx=0, current_obs=None):
        self._validate_batch_idx(batch_idx)
        state = self._get_runtime_state(batch_idx=batch_idx, copy_arrays=False)
        row, col = self._task_to_row_col(int(op_id), batch_idx=batch_idx)

        if bool(state["mask"][row]):
            raise ValueError(f"job {row} 当前不可调度")
        if int(state["omega"][row]) != int(op_id):
            raise ValueError(f"op_id={op_id} 不是当前 job {row} 的可调度工序，当前应为 {int(state['omega'][row])}")
        if state["dur"][row, col, int(mch_id)] <= 0:
            raise ValueError(f"machine {mch_id} 对 op {op_id} 不可加工")
        if bool(state["mask_mch"][row, col, int(mch_id)]):
            raise ValueError(f"machine {mch_id} 在当前状态下被基础工艺约束屏蔽")

        obs = current_obs if current_obs is not None else self.get_agent_c_obs(batch_idx=batch_idx)
        legal_pairs = obs.get("legal_pairs_set")
        if legal_pairs is None:
            legal_pairs = {(int(a), int(b)) for a, b in obs["candidate_pairs"]}
        if len(legal_pairs) > 0 and (int(op_id), int(mch_id)) not in legal_pairs:
            raise ValueError(f"({op_id}, {mch_id}) 不在当前 Agent_C 候选集中")

    # ------------------------------------------------------------------
    # Stage-1 cache update helpers
    # ------------------------------------------------------------------
    def _locate_inserted_op_pos(self, batch_idx: int, mch_id: int, op_id: int) -> int:
        """
        兼容旧 permissibleLS.py：
        它已经原地修改了 opIDsOnMchs/mchsStartTimes/mchsEndTimes，
        这里仅在目标机器单行上查找新插入 op 的位置，而不是全表扫描。
        """
        row_ops = self.opIDsOnMchs[batch_idx, mch_id]
        pos_arr = np.where(row_ops == int(op_id))[0]
        if len(pos_arr) == 0:
            raise RuntimeError(f"未能在机器 {mch_id} 的排程序列中找到 op {op_id}")
        return int(pos_arr[0])

    def _update_after_machine_insert(
        self,
        batch_idx: int,
        op_id: int,
        mch_id: int,
        row: int,
        col: int,
        start_time_a: float,
        end_time_a: float,
        insert_pos: int,
    ) -> None:
        # 有效长度：只更新目标机器这一行
        self.machine_len[batch_idx, mch_id] = self._compute_machine_len_for_machine(batch_idx, mch_id)
        L = int(self.machine_len[batch_idx, mch_id])

        # 更新新插入 op 的缓存
        self.op_start_time[batch_idx, op_id] = float(start_time_a)
        self.op_end_time[batch_idx, op_id] = float(end_time_a)
        self.op_machine[batch_idx, op_id] = int(mch_id)
        self.op_pos_on_machine[batch_idx, op_id] = int(insert_pos)

        # 更新该机器有效段上的 position cache
        effective_ops = self.opIDsOnMchs[batch_idx, mch_id, :L]
        for pos, scheduled_op in enumerate(effective_ops.tolist()):
            if scheduled_op >= 0:
                self.op_machine[batch_idx, scheduled_op] = int(mch_id)
                self.op_pos_on_machine[batch_idx, scheduled_op] = int(pos)
                self.op_start_time[batch_idx, scheduled_op] = float(self.mchsStartTimes[batch_idx, mch_id, pos])
                self.op_end_time[batch_idx, scheduled_op] = float(self.mchsEndTimes[batch_idx, mch_id, pos])

        # 机器尾结束时间
        if L > 0:
            self.machine_last_end[batch_idx, mch_id] = float(self.mchsEndTimes[batch_idx, mch_id, L - 1])
        else:
            self.machine_last_end[batch_idx, mch_id] = 0.0

    def _compute_machine_len_for_machine(self, batch_idx: int, mch_id: int) -> int:
        """
        兼容旧 sentinel 结构的过渡函数：
        只扫描目标机器单行，而不是整张表。
        """
        row_ops = self.opIDsOnMchs[batch_idx, mch_id]
        valid = np.where(row_ops >= 0)[0]
        return int(valid[-1] + 1) if len(valid) > 0 else 0

    def _get_action_neighbors_cached(self, batch_idx: int, action: int) -> Tuple[int, int]:
        """
        使用 op_machine / op_pos_on_machine cache 获取邻居，
        避免 updateAdjMat.py 原来的全表 np.where(opIDsOnMchs == action)。
        """
        mch_id = int(self.op_machine[batch_idx, action])
        pos = int(self.op_pos_on_machine[batch_idx, action])

        if mch_id < 0 or pos < 0:
            return -1, -1

        L = int(self.machine_len[batch_idx, mch_id])
        if L <= 0:
            return -1, -1

        precd = -1
        succd = -1

        if pos - 1 >= 0:
            prev_op = int(self.opIDsOnMchs[batch_idx, mch_id, pos - 1])
            if prev_op >= 0:
                precd = prev_op

        if pos + 1 < L:
            next_op = int(self.opIDsOnMchs[batch_idx, mch_id, pos + 1])
            if next_op >= 0:
                succd = next_op

        return precd, succd

    # ------------------------------------------------------------------
    # Task maps
    # ------------------------------------------------------------------
    def _build_task_maps(self):
        rc_maps: List[np.ndarray] = []
        job_maps: List[np.ndarray] = []
        col_maps: List[np.ndarray] = []

        for b in range(self.batch_sie):
            task_map = np.zeros((self.number_of_tasks, 2), dtype=np.int32)
            task_job_map = np.zeros((self.number_of_tasks,), dtype=np.int32)
            task_col_map = np.zeros((self.number_of_tasks,), dtype=np.int32)

            for row in range(self.number_of_jobs):
                start = int(self.first_col[b][row])
                end = int(self.last_col[b][row])
                if end < start:
                    continue
                cols = np.arange(0, end - start + 1, dtype=np.int32)
                task_map[start:end + 1, 0] = int(row)
                task_map[start:end + 1, 1] = cols
                task_job_map[start:end + 1] = int(row)
                task_col_map[start:end + 1] = cols

            rc_maps.append(task_map)
            job_maps.append(task_job_map)
            col_maps.append(task_col_map)

        return rc_maps, job_maps, col_maps

    # ------------------------------------------------------------------
    # Static compile
    # ------------------------------------------------------------------
    def _get_compiled_case_static(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        data_arr = np.asarray(data, dtype=np.float32)
        signature = (
            tuple(data_arr.shape),
            str(data_arr.dtype),
            float(np.sum(data_arr)),
            float(np.mean(data_arr)),
            float(np.max(data_arr)) if data_arr.size > 0 else 0.0,
        )
        cached = self._compiled_case_static_cache.get(signature)
        if cached is not None:
            return cached

        dur_cp = data_arr.copy()
        dur = data_arr.copy()
        mask_mch = dur_cp <= 0
        dur = np.where(mask_mch, 100.0, dur).astype(np.float32)

        input_min = np.zeros((self.batch_sie, self.number_of_jobs, self.max_operation), dtype=np.float32)
        input_mean = np.zeros_like(input_min)
        input_max = np.zeros_like(input_min)

        for t in range(self.batch_sie):
            for i in range(self.number_of_jobs):
                for j in range(self.max_operation):
                    durmch = dur_cp[t, i, j][dur_cp[t, i, j] > 0]
                    if len(durmch) == 0:
                        input_min[t, i, j] = 1.0
                        input_mean[t, i, j] = 1.0
                        input_max[t, i, j] = 1.0
                    else:
                        input_min[t, i, j] = float(np.min(durmch))
                        input_mean[t, i, j] = float(np.mean(durmch))
                        input_max[t, i, j] = float(np.max(durmch))

        input_2d = np.concatenate(
            [
                input_min.reshape((self.batch_sie, self.number_of_jobs, self.max_operation, 1)),
                input_mean.reshape((self.batch_sie, self.number_of_jobs, self.max_operation, 1)),
            ],
            axis=-1,
        ).astype(np.float32)

        compiled = {
            "dur": dur,
            "dur_cp": dur_cp,
            "mask_mch": mask_mch.astype(bool),
            "input_min": input_min,
            "input_mean": input_mean,
            "input_max": input_max,
            "input_2d": input_2d,
        }
        self._compiled_case_static_cache[signature] = compiled
        return compiled

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------
    def _validate_batch_idx(self, batch_idx):
        if not hasattr(self, "batch_sie"):
            raise RuntimeError("环境尚未 reset，无法读取 batch 状态")
        if batch_idx < 0 or batch_idx >= self.batch_sie:
            raise IndexError("batch_idx 超出范围")

    def _flatten_lbm(self, batch_idx):
        lbm = []
        for j in range(self.number_of_jobs):
            for k in range(self.num_operation[batch_idx][j]):
                lbm.append(self.LB[batch_idx, j, k])
        return np.asarray(lbm, dtype=np.float32)

    def _flatten_lbm_from_lb(self, batch_idx):
        lbm = []
        for j in range(self.number_of_jobs):
            num_op_j = int(self.num_operation[batch_idx][j])
            for k in range(num_op_j):
                lbm.append(self.LB[batch_idx, j, k])
        return np.asarray(lbm, dtype=np.float32)

    def _compute_remaining_work_min(self, batch_idx, job_idx):
        start_col = int(self.job_col[batch_idx, job_idx])
        num_op = int(self.num_operation[batch_idx][job_idx])
        if start_col >= num_op:
            return 0.0
        return float(np.sum(self.input_min[batch_idx, job_idx, start_col:num_op]))

    def _task_to_row_col(self, task_id, batch_idx=0):
        return int(self.op_to_job_row[batch_idx, int(task_id)]), int(self.op_to_job_col[batch_idx, int(task_id)])

    def _build_rule_source_multihot(self, source_names, rule_names):
        source_set = set(source_names)
        return [1.0 if rule in source_set else 0.0 for rule in rule_names]

    def _safe_normalize(self, x, denom=None):
        x = np.asarray(x, dtype=np.float32)
        if denom is None:
            denom = float(np.max(np.abs(x))) if x.size > 0 else 0.0
        denom = float(denom)
        if denom <= 1e-8:
            return np.zeros_like(x, dtype=np.float32)
        return (x / denom).astype(np.float32)

    def _deduplicate_candidate_pairs(self, candidate_pairs, pair_rule_names):
        pair_to_idx = {}
        dedup_pairs = []
        pair_sources = []

        for pair, rule_name in zip(candidate_pairs, pair_rule_names):
            op_id = int(pair[0])
            mch_id = int(pair[1])
            pair_code = op_id * self.number_of_machines + mch_id

            if pair_code not in pair_to_idx:
                pair_to_idx[pair_code] = len(dedup_pairs)
                dedup_pairs.append((op_id, mch_id))
                pair_sources.append([rule_name])
            else:
                pair_sources[pair_to_idx[pair_code]].append(rule_name)

        return dedup_pairs, pair_sources

    def _fallback_candidate_pair(self, batch_idx=0) -> Optional[Tuple[int, int]]:
        state = self._get_runtime_state(batch_idx=batch_idx, copy_arrays=False)
        return self._fallback_candidate_pair_from_state(state)

    def _fallback_candidate_pair_from_state(self, state) -> Optional[Tuple[int, int]]:
        available_jobs = state["available_jobs"]
        for job_idx in available_jobs.tolist():
            op_id = int(state["omega"][job_idx])
            row, col = self._task_to_row_col(op_id, batch_idx=state["batch_idx"])
            feasible_mchs = self._get_feasible_machine_indices_from_state(state, row=row, col=col)
            if len(feasible_mchs) > 0:
                best = min(
                    feasible_mchs.tolist(),
                    key=lambda m: (
                        max(float(state["job_time"][row]), float(state["mch_time"][m])) + float(state["dur"][row, col, m]),
                        float(state["dur"][row, col, m]),
                        int(m),
                    ),
                )
                return op_id, int(best)
        return None
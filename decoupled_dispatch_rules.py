from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple, Optional, NewType

import numpy as np

from Params import configs


ArrayLike = NewType("ArrayLike", np.ndarray)
JobScoreFunc = Callable[[Dict[str, ArrayLike]], ArrayLike]
MachineRankFunc = Callable[[Dict[str, ArrayLike], int], List[int]]

# ---------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------
# Backward-compatible old rules + a few stronger optional rules.
JOB_RULE_NAMES = (
    "FIFO",
    "MOPNR",
    "LWKR",
    "MWKR",
    "LOR",
    "MOR",
    "SPTJ",
    "LPTJ",
    "LFMJ",   # least flexible machines for current op
    "CRJ",    # critical-ratio-like score
)

MACHINE_RULE_NAMES = (
    "SPT",
    "EET",
    "LQ",
    "LLD",
    "EETLQ",  # earliest end + queue
    "EETD",   # earliest end + relative delay against best machine
)

DEFAULT_TOPK_JOBS = getattr(configs, "agent_c_topk_jobs", 2)
DEFAULT_TOPK_MACHINES = getattr(configs, "agent_c_topk_machines", 2)


# ---------------------------------------------------------------------
# Basic normalization helpers
# ---------------------------------------------------------------------
def update_mch_time(
    mch_time: ArrayLike,
    mchs_end_times: ArrayLike,
    number_of_machines: int,
) -> ArrayLike:
    """Normalize machine availability from machine end-time table."""
    _ = mch_time
    mchs_end_times = np.asarray(mchs_end_times, dtype=np.float32)

    if mchs_end_times.ndim != 2 or mchs_end_times.shape[0] == 0:
        return np.zeros((int(number_of_machines),), dtype=np.float32)

    valid = mchs_end_times >= 0
    last_end = np.max(np.where(valid, mchs_end_times, -np.inf), axis=1)
    updated = np.where(valid.any(axis=1), last_end, 0.0).astype(np.float32)

    if updated.shape[0] != int(number_of_machines):
        padded = np.zeros((int(number_of_machines),), dtype=np.float32)
        n = min(int(number_of_machines), int(updated.shape[0]))
        padded[:n] = updated[:n]
        return padded
    return updated


def update_job_time(job_time: ArrayLike, temp: ArrayLike) -> ArrayLike:
    """Normalize job readiness from operation completion table."""
    _ = job_time
    temp = np.asarray(temp, dtype=np.float32)

    if temp.ndim != 2 or temp.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)

    valid = temp > 0
    last_end = np.max(np.where(valid, temp, -np.inf), axis=1)
    return np.where(valid.any(axis=1), last_end, 0.0).astype(np.float32)


# ---------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------
def get_task_row_col(
    task: int,
    last_col: ArrayLike,
    first_col: ArrayLike,
    task_to_row: Optional[ArrayLike] = None,
    task_to_col: Optional[ArrayLike] = None,
) -> Tuple[int, int]:
    """
    Fast path if task_to_row / task_to_col are provided.
    Falls back to old np.where behavior for compatibility.
    """
    task = int(task)
    if task_to_row is not None and task_to_col is not None:
        return int(task_to_row[task]), int(task_to_col[task])

    row = int(np.where(task <= last_col)[0][0])
    col = int(task - first_col[row])
    return row, col


def get_feasible_machines(
    task: int,
    dur: ArrayLike,
    last_col: ArrayLike,
    first_col: ArrayLike,
    mask_mch: Optional[ArrayLike] = None,
    task_to_row: Optional[ArrayLike] = None,
    task_to_col: Optional[ArrayLike] = None,
) -> ArrayLike:
    row, col = get_task_row_col(
        task=task,
        last_col=last_col,
        first_col=first_col,
        task_to_row=task_to_row,
        task_to_col=task_to_col,
    )
    feasible = dur[row, col] > 0
    if mask_mch is not None:
        feasible = np.logical_and(feasible, ~np.asarray(mask_mch[row, col]).astype(bool))
    return np.where(feasible)[0].astype(np.int64)


# ---------------------------------------------------------------------
# Rule parsing
# ---------------------------------------------------------------------
def parse_compound_rule(rule: str) -> Tuple[str, str]:
    try:
        job_rule_name, machine_rule_name = rule.split("_", 1)
    except ValueError as exc:
        raise ValueError(f"Invalid rule format: {rule}") from exc

    if job_rule_name not in JOB_RULE_NAMES:
        raise ValueError(f"Unsupported job rule: {job_rule_name}")
    if machine_rule_name not in MACHINE_RULE_NAMES:
        raise ValueError(f"Unsupported machine rule: {machine_rule_name}")
    return job_rule_name, machine_rule_name


# ---------------------------------------------------------------------
# Job-level statistics
# ---------------------------------------------------------------------
def _compute_remaining_num_ops(num_operation: ArrayLike, dispatched_num_opera: ArrayLike) -> ArrayLike:
    return np.maximum(np.asarray(num_operation) - np.asarray(dispatched_num_opera), 0).astype(np.float32)


def _compute_remaining_work(input_min: ArrayLike, num_operation: ArrayLike, job_col: ArrayLike) -> ArrayLike:
    """
    Remaining minimum workload from current operation onward.
    Robust to padded operations.
    """
    input_min = np.asarray(input_min, dtype=np.float32)
    num_operation = np.asarray(num_operation, dtype=np.int32)
    job_col = np.asarray(job_col, dtype=np.int32)

    work = np.zeros(shape=(input_min.shape[0],), dtype=np.float32)
    for j in range(input_min.shape[0]):
        total_ops = int(num_operation[j])
        cur_col = int(job_col[j])
        if total_ops <= 0 or cur_col >= total_ops:
            work[j] = 0.0
            continue
        work[j] = float(np.sum(input_min[j, cur_col:total_ops]))
    return work


def _compute_current_operation_min_proc(
    input_min: ArrayLike,
    job_col: ArrayLike,
    num_operation: ArrayLike,
) -> ArrayLike:
    input_min = np.asarray(input_min, dtype=np.float32)
    job_col = np.asarray(job_col, dtype=np.int32)
    num_operation = np.asarray(num_operation, dtype=np.int32)

    current = np.zeros((input_min.shape[0],), dtype=np.float32)
    for j in range(input_min.shape[0]):
        total_ops = int(num_operation[j])
        cur_col = int(job_col[j])
        if total_ops <= 0 or cur_col >= total_ops:
            current[j] = 0.0
            continue
        current[j] = float(input_min[j, cur_col])
    return current


def _compute_current_operation_flexibility(
    input_min: ArrayLike,
    job_col: ArrayLike,
    num_operation: ArrayLike,
    dur: Optional[ArrayLike] = None,
    mask_mch: Optional[ArrayLike] = None,
) -> ArrayLike:
    """
    Number of feasible machines for each job's current operation.
    If dur is unavailable, falls back to zeros.
    """
    num_operation = np.asarray(num_operation, dtype=np.int32)
    job_col = np.asarray(job_col, dtype=np.int32)

    if dur is None:
        return np.zeros((len(num_operation),), dtype=np.float32)

    dur = np.asarray(dur, dtype=np.float32)
    out = np.zeros((dur.shape[0],), dtype=np.float32)

    for j in range(dur.shape[0]):
        total_ops = int(num_operation[j])
        cur_col = int(job_col[j])
        if total_ops <= 0 or cur_col >= total_ops:
            out[j] = 0.0
            continue

        feasible = dur[j, cur_col] > 0
        if mask_mch is not None:
            feasible = np.logical_and(feasible, ~np.asarray(mask_mch[j, cur_col]).astype(bool))
        out[j] = float(np.sum(feasible))
    return out


def _prepare_rule_state(
    job_time: ArrayLike,
    num_operation: ArrayLike,
    dispatched_num_opera: ArrayLike,
    input_min: ArrayLike,
    job_col: ArrayLike,
    input_max: ArrayLike,
    dur: Optional[ArrayLike] = None,
    mask_mch: Optional[ArrayLike] = None,
) -> Dict[str, ArrayLike]:
    del input_max  # kept for API compatibility
    remain_num = _compute_remaining_num_ops(num_operation, dispatched_num_opera)
    remain_work = _compute_remaining_work(input_min, num_operation, job_col)
    current_proc = _compute_current_operation_min_proc(input_min, job_col, num_operation)
    current_flex = _compute_current_operation_flexibility(
        input_min=input_min,
        job_col=job_col,
        num_operation=num_operation,
        dur=dur,
        mask_mch=mask_mch,
    )

    # CR-like: larger means more urgent if a lot of work remains but job is not yet advanced.
    cr_like = remain_work / np.maximum(np.asarray(job_time, dtype=np.float32) + 1.0, 1e-6)

    return {
        "job_time": np.asarray(job_time, dtype=np.float32),
        "remain_num": remain_num,
        "remain_work": remain_work,
        "current_proc": current_proc,
        "current_flex": current_flex,
        "cr_like": cr_like.astype(np.float32),
    }


# ---------------------------------------------------------------------
# Job rules
# ---------------------------------------------------------------------
def score_jobs_fifo(state: Dict[str, ArrayLike]) -> ArrayLike:
    return state["job_time"]


def score_jobs_mopnr(state: Dict[str, ArrayLike]) -> ArrayLike:
    return state["remain_num"]


def score_jobs_lwkr(state: Dict[str, ArrayLike]) -> ArrayLike:
    return state["remain_work"]


def score_jobs_mwkr(state: Dict[str, ArrayLike]) -> ArrayLike:
    return state["remain_work"]


def score_jobs_lor(state: Dict[str, ArrayLike]) -> ArrayLike:
    return state["remain_num"]


def score_jobs_mor(state: Dict[str, ArrayLike]) -> ArrayLike:
    return state["remain_num"]


def score_jobs_sptj(state: Dict[str, ArrayLike]) -> ArrayLike:
    return state["current_proc"]


def score_jobs_lptj(state: Dict[str, ArrayLike]) -> ArrayLike:
    return state["current_proc"]


def score_jobs_lfmj(state: Dict[str, ArrayLike]) -> ArrayLike:
    # Least flexible machines first
    return state["current_flex"]


def score_jobs_crj(state: Dict[str, ArrayLike]) -> ArrayLike:
    return state["cr_like"]


def get_job_rule_func(job_rule_name: str) -> JobScoreFunc:
    return {
        "FIFO": score_jobs_fifo,
        "MOPNR": score_jobs_mopnr,
        "LWKR": score_jobs_lwkr,
        "MWKR": score_jobs_mwkr,
        "LOR": score_jobs_lor,
        "MOR": score_jobs_mor,
        "SPTJ": score_jobs_sptj,
        "LPTJ": score_jobs_lptj,
        "LFMJ": score_jobs_lfmj,
        "CRJ": score_jobs_crj,
    }[job_rule_name]


def _job_rule_mode(job_rule_name: str) -> str:
    return {
        "FIFO": "min",
        "MOPNR": "max",
        "LWKR": "min",
        "MWKR": "max",
        "LOR": "min",
        "MOR": "max",
        "SPTJ": "min",
        "LPTJ": "max",
        "LFMJ": "min",
        "CRJ": "max",
    }[job_rule_name]


def get_job_rule_mode(job_rule_name: str) -> str:
    return _job_rule_mode(job_rule_name)


def prepare_rule_state(
    job_time: ArrayLike,
    num_operation: ArrayLike,
    dispatched_num_opera: ArrayLike,
    input_min: ArrayLike,
    job_col: ArrayLike,
    input_max: ArrayLike,
    dur: Optional[ArrayLike] = None,
    mask_mch: Optional[ArrayLike] = None,
) -> Dict[str, ArrayLike]:
    return _prepare_rule_state(
        job_time=job_time,
        num_operation=num_operation,
        dispatched_num_opera=dispatched_num_opera,
        input_min=input_min,
        job_col=job_col,
        input_max=input_max,
        dur=dur,
        mask_mch=mask_mch,
    )


def select_topk_job_indices(
    scores: ArrayLike,
    available_jobs: ArrayLike,
    mode: str,
    topk_jobs: int,
) -> ArrayLike:
    if available_jobs.size == 0:
        return np.array([], dtype=np.int64)

    available_jobs = np.asarray(available_jobs, dtype=np.int64)
    sub_scores = np.asarray(scores, dtype=np.float32)[available_jobs]

    # Stable lexicographic ordering: by score first, then by job index.
    if mode == "min":
        order = np.lexsort((available_jobs, sub_scores))
    elif mode == "max":
        order = np.lexsort((available_jobs, -sub_scores))
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    k = min(max(int(topk_jobs), 1), available_jobs.size)
    return available_jobs[order[:k]]


# ---------------------------------------------------------------------
# Machine-level helpers
# ---------------------------------------------------------------------
def _state_task_row_col(state: Dict[str, ArrayLike], task: int) -> Tuple[int, int]:
    return get_task_row_col(
        task=task,
        last_col=state["last_col"],
        first_col=state["first_col"],
        task_to_row=state.get("task_to_row", None),
        task_to_col=state.get("task_to_col", None),
    )


def _get_valid_machine_indices(state: Dict[str, ArrayLike], row: int, col: int) -> ArrayLike:
    feasible = state["dur"][row, col] > 0
    if "mask_mch" in state and state["mask_mch"] is not None:
        feasible = np.logical_and(feasible, ~np.asarray(state["mask_mch"][row, col]).astype(bool))
    return np.where(feasible)[0].astype(np.int64)


def _rank_machines_spt(state: Dict[str, ArrayLike], task: int) -> List[int]:
    row, col = _state_task_row_col(state, task)
    feasible = _get_valid_machine_indices(state, row, col)
    if feasible.size == 0:
        return []

    proc = state["dur"][row, col, feasible]
    order = np.lexsort((feasible, proc))
    return [int(feasible[idx]) for idx in order]


def _rank_machines_eet(state: Dict[str, ArrayLike], task: int) -> List[int]:
    row, col = _state_task_row_col(state, task)
    feasible = _get_valid_machine_indices(state, row, col)
    if feasible.size == 0:
        return []

    ready = float(state["job_time"][row])
    mch_time = state["mch_time"][feasible]
    proc = state["dur"][row, col, feasible]
    est_completion = np.maximum(ready, mch_time) + proc
    order = np.lexsort((feasible, proc, est_completion))
    return [int(feasible[idx]) for idx in order]


def _rank_machines_lq(state: Dict[str, ArrayLike], task: int) -> List[int]:
    row, col = _state_task_row_col(state, task)
    feasible = _get_valid_machine_indices(state, row, col)
    if feasible.size == 0:
        return []

    ready = float(state["job_time"][row])
    proc = state["dur"][row, col, feasible]
    mch_time = state["mch_time"][feasible]
    est_completion = np.maximum(ready, mch_time) + proc
    queue_len = state["machine_queue_len"][feasible]
    order = np.lexsort((feasible, proc, est_completion, queue_len))
    return [int(feasible[idx]) for idx in order]


def _rank_machines_lld(state: Dict[str, ArrayLike], task: int) -> List[int]:
    row, col = _state_task_row_col(state, task)
    feasible = _get_valid_machine_indices(state, row, col)
    if feasible.size == 0:
        return []

    ready = float(state["job_time"][row])
    proc = state["dur"][row, col, feasible]
    mch_time = state["mch_time"][feasible]
    est_start = np.maximum(ready, mch_time)
    queue_len = state["machine_queue_len"][feasible]
    load_score = mch_time + 0.25 * queue_len
    order = np.lexsort((feasible, proc, est_start, load_score))
    return [int(feasible[idx]) for idx in order]


def _rank_machines_eetlq(state: Dict[str, ArrayLike], task: int) -> List[int]:
    """
    earliest estimated completion + queue length tie-break
    """
    row, col = _state_task_row_col(state, task)
    feasible = _get_valid_machine_indices(state, row, col)
    if feasible.size == 0:
        return []

    ready = float(state["job_time"][row])
    proc = state["dur"][row, col, feasible]
    mch_time = state["mch_time"][feasible]
    queue_len = state["machine_queue_len"][feasible]
    est_completion = np.maximum(ready, mch_time) + proc

    order = np.lexsort((feasible, proc, queue_len, est_completion))
    return [int(feasible[idx]) for idx in order]


def _rank_machines_eetd(state: Dict[str, ArrayLike], task: int) -> List[int]:
    """
    earliest estimated completion + delay against best proc tie-break.
    """
    row, col = _state_task_row_col(state, task)
    feasible = _get_valid_machine_indices(state, row, col)
    if feasible.size == 0:
        return []

    ready = float(state["job_time"][row])
    proc = state["dur"][row, col, feasible]
    mch_time = state["mch_time"][feasible]
    est_completion = np.maximum(ready, mch_time) + proc
    min_proc = float(np.min(proc))
    proc_gap = proc - min_proc

    order = np.lexsort((feasible, proc_gap, proc, est_completion))
    return [int(feasible[idx]) for idx in order]


def get_machine_rule_func(machine_rule_name: str) -> MachineRankFunc:
    return {
        "SPT": _rank_machines_spt,
        "EET": _rank_machines_eet,
        "LQ": _rank_machines_lq,
        "LLD": _rank_machines_lld,
        "EETLQ": _rank_machines_eetlq,
        "EETD": _rank_machines_eetd,
    }[machine_rule_name]


# ---------------------------------------------------------------------
# Machine queue length
# ---------------------------------------------------------------------
def _compute_machine_queue_len(mchs_end_times: ArrayLike, number_of_machines: int) -> ArrayLike:
    mchs_end_times = np.asarray(mchs_end_times, dtype=np.float32)
    if mchs_end_times.ndim != 2 or mchs_end_times.shape[0] == 0:
        return np.zeros((int(number_of_machines),), dtype=np.float32)

    valid = mchs_end_times >= 0
    queue_len = valid.sum(axis=1).astype(np.float32)

    if queue_len.shape[0] != int(number_of_machines):
        padded = np.zeros((int(number_of_machines),), dtype=np.float32)
        n = min(int(number_of_machines), int(queue_len.shape[0]))
        padded[:n] = queue_len[:n]
        return padded
    return queue_len


def compute_machine_queue_len(mchs_end_times: ArrayLike, number_of_machines: int) -> ArrayLike:
    return _compute_machine_queue_len(mchs_end_times, number_of_machines)


# ---------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------
def build_dispatch_mask(omega: ArrayLike, selected_tasks: ArrayLike, mask_last: ArrayLike) -> ArrayLike:
    """
    Job-level dispatch mask:
    False -> selected/available
    True  -> masked/finished
    """
    omega = np.asarray(omega)
    mask_last = np.asarray(mask_last).astype(bool)
    selected_tasks = np.asarray(selected_tasks).reshape(-1)

    if selected_tasks.size == 0:
        return mask_last.copy()

    selected_jobs = np.isin(omega, selected_tasks)
    mask = ~selected_jobs
    return np.logical_or(mask, mask_last)


def build_selected_task_machine_masks(
    selected_tasks: ArrayLike,
    machine_rank_func: MachineRankFunc,
    state: Dict[str, ArrayLike],
    topk_machines: int,
) -> ArrayLike:
    """
    Build local machine masks aligned with selected_tasks.
    Shape: [num_selected_tasks, number_of_machines]
    False means selectable, True means masked.
    """
    selected_tasks = np.asarray(selected_tasks).reshape(-1)
    num_selected_tasks = len(selected_tasks)
    num_machines = int(state["number_of_machines"])

    if num_selected_tasks == 0:
        return np.ones((0, num_machines), dtype=bool)

    local_masks = np.ones((num_selected_tasks, num_machines), dtype=bool)

    for local_idx, task in enumerate(selected_tasks):
        task = int(task)
        row, col = _state_task_row_col(state, task)

        ranked = machine_rank_func(state, task)
        if len(ranked) == 0:
            fallback = _get_valid_machine_indices(state, row, col)
            ranked = [int(m) for m in fallback.tolist()]

        ranked = ranked[: max(int(topk_machines), 1)]

        row_mask = np.ones((num_machines,), dtype=bool)
        if len(ranked) > 0:
            row_mask[np.asarray(ranked, dtype=np.int64)] = False
        local_masks[local_idx] = row_mask

    return local_masks


# ---------------------------------------------------------------------
# Prepared-state dispatch API
# ---------------------------------------------------------------------
def dispatch_decoupled_prepared(
    rule_state: Dict[str, ArrayLike],
    machine_state: Dict[str, ArrayLike],
    omega: ArrayLike,
    mask_last: ArrayLike,
    done: bool,
    rule: str,
    topk_jobs: int = DEFAULT_TOPK_JOBS,
    topk_machines: int = DEFAULT_TOPK_MACHINES,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, str, str]:
    """
    Lighter dispatch API when caller already prepared rule_state and machine_state.
    """
    job_rule_name, machine_rule_name = parse_compound_rule(rule)
    job_rule_func = get_job_rule_func(job_rule_name)
    machine_rule_func = get_machine_rule_func(machine_rule_name)

    available_jobs = np.where(~np.asarray(mask_last).astype(bool))[0]
    number_of_machines = int(machine_state["number_of_machines"])

    if done or available_jobs.size == 0:
        empty_tasks = np.array([], dtype=np.int64)
        empty_mask = np.asarray(mask_last).astype(bool)
        empty_m_masks = np.ones((0, number_of_machines), dtype=bool)
        return empty_tasks, empty_mask, empty_m_masks, job_rule_name, machine_rule_name

    scores = job_rule_func(rule_state)
    selected_job_indices = select_topk_job_indices(
        scores=scores,
        available_jobs=available_jobs,
        mode=_job_rule_mode(job_rule_name),
        topk_jobs=topk_jobs,
    )
    selected_tasks = np.asarray(omega[selected_job_indices], dtype=np.int64)
    mask = build_dispatch_mask(omega, selected_tasks, mask_last)

    m_masks = build_selected_task_machine_masks(
        selected_tasks=selected_tasks,
        machine_rank_func=machine_rule_func,
        state=machine_state,
        topk_machines=topk_machines,
    )
    return selected_tasks, mask, m_masks, job_rule_name, machine_rule_name


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def dispatch_decoupled(
    mch_time: ArrayLike,
    job_time: ArrayLike,
    mchs_end_times: ArrayLike,
    number_of_machines: int,
    dur: ArrayLike,
    temp: ArrayLike,
    omega: ArrayLike,
    mask_last: ArrayLike,
    done: bool,
    mask_mch: ArrayLike,
    num_operation: ArrayLike,
    dispatched_num_opera: ArrayLike,
    input_min: ArrayLike,
    job_col: ArrayLike,
    input_max: ArrayLike,
    rule: str,
    last_col: ArrayLike,
    first_col: ArrayLike,
    topk_jobs: int = DEFAULT_TOPK_JOBS,
    topk_machines: int = DEFAULT_TOPK_MACHINES,
    normalized_mch_time: ArrayLike | None = None,
    normalized_job_time: ArrayLike | None = None,
    machine_queue_len: ArrayLike | None = None,
    task_to_row: ArrayLike | None = None,
    task_to_col: ArrayLike | None = None,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, str, str]:
    """
    Decoupled rule execution.

    Returns
    -------
    selected_tasks : np.ndarray
        1-D array of selected operation ids.
    mask : np.ndarray
        Job-level dispatch mask aligned with omega.
    m_masks : np.ndarray
        2-D machine mask aligned with selected_tasks.
        Shape [len(selected_tasks), number_of_machines].
    job_rule_name : str
    machine_rule_name : str
    """
    if normalized_mch_time is None:
        normalized_mch_time = update_mch_time(mch_time, mchs_end_times, number_of_machines)
    else:
        normalized_mch_time = np.asarray(normalized_mch_time, dtype=np.float32)

    if normalized_job_time is None:
        normalized_job_time = update_job_time(job_time, temp)
    else:
        normalized_job_time = np.asarray(normalized_job_time, dtype=np.float32)

    if machine_queue_len is None:
        machine_queue_len = _compute_machine_queue_len(mchs_end_times, number_of_machines)
    else:
        machine_queue_len = np.asarray(machine_queue_len, dtype=np.float32)

    rule_state = _prepare_rule_state(
        job_time=normalized_job_time,
        num_operation=num_operation,
        dispatched_num_opera=dispatched_num_opera,
        input_min=input_min,
        job_col=job_col,
        input_max=input_max,
        dur=dur,
        mask_mch=mask_mch,
    )

    machine_state = {
        "mch_time": np.asarray(normalized_mch_time, dtype=np.float32),
        "job_time": np.asarray(normalized_job_time, dtype=np.float32),
        "machine_queue_len": machine_queue_len,
        "dur": np.asarray(dur, dtype=np.float32),
        "mask_mch": np.asarray(mask_mch).astype(bool),
        "last_col": np.asarray(last_col, dtype=np.int64),
        "first_col": np.asarray(first_col, dtype=np.int64),
        "task_to_row": None if task_to_row is None else np.asarray(task_to_row, dtype=np.int64),
        "task_to_col": None if task_to_col is None else np.asarray(task_to_col, dtype=np.int64),
        "number_of_machines": int(number_of_machines),
    }

    return dispatch_decoupled_prepared(
        rule_state=rule_state,
        machine_state=machine_state,
        omega=omega,
        mask_last=mask_last,
        done=done,
        rule=rule,
        topk_jobs=topk_jobs,
        topk_machines=topk_machines,
    )


def DRs_decoupled(
    mch_time: ArrayLike,
    job_time: ArrayLike,
    mchsEndTimes: ArrayLike,
    number_of_machines: int,
    dur: ArrayLike,
    temp: ArrayLike,
    omega: ArrayLike,
    mask_last: ArrayLike,
    done: bool,
    mask_mch: ArrayLike,
    num_operation: ArrayLike,
    dispatched_num_opera: ArrayLike,
    input_min: ArrayLike,
    job_col: ArrayLike,
    input_max: ArrayLike,
    rule: str,
    last_col: ArrayLike,
    first_col: ArrayLike,
    topk_jobs: int = DEFAULT_TOPK_JOBS,
    topk_machines: int = DEFAULT_TOPK_MACHINES,
    normalized_mch_time: ArrayLike | None = None,
    normalized_job_time: ArrayLike | None = None,
    machine_queue_len: ArrayLike | None = None,
    task_to_row: ArrayLike | None = None,
    task_to_col: ArrayLike | None = None,
) -> Tuple[ArrayLike, ArrayLike]:
    _, mask, m_masks, _, _ = dispatch_decoupled(
        mch_time=mch_time,
        job_time=job_time,
        mchs_end_times=mchsEndTimes,
        number_of_machines=number_of_machines,
        dur=dur,
        temp=temp,
        omega=omega,
        mask_last=mask_last,
        done=done,
        mask_mch=mask_mch,
        num_operation=num_operation,
        dispatched_num_opera=dispatched_num_opera,
        input_min=input_min,
        job_col=job_col,
        input_max=input_max,
        rule=rule,
        last_col=last_col,
        first_col=first_col,
        topk_jobs=topk_jobs,
        topk_machines=topk_machines,
        normalized_mch_time=normalized_mch_time,
        normalized_job_time=normalized_job_time,
        machine_queue_len=machine_queue_len,
        task_to_row=task_to_row,
        task_to_col=task_to_col,
    )
    return mask, m_masks


__all__ = [
    "JOB_RULE_NAMES",
    "MACHINE_RULE_NAMES",
    "DRs_decoupled",
    "dispatch_decoupled",
    "dispatch_decoupled_prepared",
    "parse_compound_rule",
    "get_job_rule_func",
    "get_job_rule_mode",
    "get_machine_rule_func",
    "prepare_rule_state",
    "select_topk_job_indices",
    "compute_machine_queue_len",
    "score_jobs_fifo",
    "score_jobs_mopnr",
    "score_jobs_lwkr",
    "score_jobs_mwkr",
    "score_jobs_lor",
    "score_jobs_mor",
    "score_jobs_sptj",
    "score_jobs_lptj",
    "score_jobs_lfmj",
    "score_jobs_crj",
    "update_mch_time",
    "update_job_time",
    "get_task_row_col",
    "get_feasible_machines",
]
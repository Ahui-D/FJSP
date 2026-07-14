from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from Params import configs


def permissibleLeftShift(
    a,
    mch_a,
    durMat,
    mchMat,
    mchsStartTimes,
    opIDsOnMchs,
    mchEndTime,
    row,
    col,
    first_col,
    last_col,
):
    """
    Optimized compatible version.

    输入 / 输出接口保持与旧版一致：
        return startTime_a, flag

    其中:
    - flag=False: 尾插
    - flag=True : 插入到机器中间空隙（left shift）

    主要优化:
    1. 不再使用 np.insert
    2. 不再依赖 startTimes == -configs.high 找尾部
    3. 所有扫描都限制在目标机器的有效段上
    4. 统一使用 mchEndTime 而不是从 start + dur 反推 machine predecessor end
    """

    a = int(a)
    mch_a = int(mch_a)
    row = int(row)
    col = int(col)

    dur_a = float(durMat[row, col, mch_a])

    startTimesForMchOfa = mchsStartTimes[mch_a]
    endTimesForMchOfa = mchEndTime[mch_a]
    opsIDsForMchOfa = opIDsOnMchs[mch_a]

    machine_len = _get_machine_effective_len(opsIDsForMchOfa)

    jobRdyTime_a, mchRdyTime_a = calJobAndMchRdyTimeOfa(
        a=a,
        mch_a=mch_a,
        mchMat=mchMat,
        durMat=durMat,
        mchsStartTimes=mchsStartTimes,
        opIDsOnMchs=opIDsOnMchs,
        mchEndTime=mchEndTime,
        row=row,
        col=col,
        first_col=first_col,
        last_col=last_col,
        machine_len=machine_len,
    )

    if machine_len == 0:
        startTime_a = putInTheEnd(
            a=a,
            jobRdyTime_a=jobRdyTime_a,
            mchRdyTime_a=mchRdyTime_a,
            startTimesForMchOfa=startTimesForMchOfa,
            opsIDsForMchOfa=opsIDsForMchOfa,
            endTimesForMchOfa=endTimesForMchOfa,
            dur_a=dur_a,
            machine_len=machine_len,
        )
        return startTime_a, False

    idxLegalPos, legalPos, endTimesForPossiblePos = calLegalPos(
        dur_a=dur_a,
        mch_a=mch_a,
        jobRdyTime_a=jobRdyTime_a,
        durMat=durMat,
        startTimesForMchOfa=startTimesForMchOfa,
        endTimesForMchOfa=endTimesForMchOfa,
        opsIDsForMchOfa=opsIDsForMchOfa,
        first_col=first_col,
        last_col=last_col,
        machine_len=machine_len,
    )

    if len(legalPos) == 0:
        startTime_a = putInTheEnd(
            a=a,
            jobRdyTime_a=jobRdyTime_a,
            mchRdyTime_a=mchRdyTime_a,
            startTimesForMchOfa=startTimesForMchOfa,
            opsIDsForMchOfa=opsIDsForMchOfa,
            endTimesForMchOfa=endTimesForMchOfa,
            dur_a=dur_a,
            machine_len=machine_len,
        )
        return startTime_a, False

    startTime_a = putInBetween(
        a=a,
        idxLegalPos=idxLegalPos,
        legalPos=legalPos,
        endTimesForPossiblePos=endTimesForPossiblePos,
        startTimesForMchOfa=startTimesForMchOfa,
        opsIDsForMchOfa=opsIDsForMchOfa,
        endTimesForMchOfa=endTimesForMchOfa,
        dur_a=dur_a,
        machine_len=machine_len,
    )
    return startTime_a, True


def putInTheEnd(
    a,
    jobRdyTime_a,
    mchRdyTime_a,
    startTimesForMchOfa,
    opsIDsForMchOfa,
    endTimesForMchOfa,
    dur_a,
    machine_len,
):
    """
    尾插:
    直接写入有效段末尾 machine_len，不再扫描 sentinel。
    """
    index = int(machine_len)
    startTime_a = max(float(jobRdyTime_a), float(mchRdyTime_a))

    startTimesForMchOfa[index] = startTime_a
    endTimesForMchOfa[index] = startTime_a + float(dur_a)
    opsIDsForMchOfa[index] = int(a)

    return float(startTime_a)


def calLegalPos(
    dur_a,
    mch_a,
    jobRdyTime_a,
    durMat,
    startTimesForMchOfa,
    endTimesForMchOfa,
    opsIDsForMchOfa,
    first_col,
    last_col,
    machine_len,
):
    """
    在目标机器当前有效调度段 [0:machine_len] 上寻找最早可插入空隙。

    返回:
    - idxLegalPos: 对 possible positions 的局部索引
    - legalPos: 真正机器序列中的插入位置
    - endTimesForPossiblePos: 每个候选位置对应的最早可开始时间
    """
    if machine_len <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    valid_starts = startTimesForMchOfa[:machine_len]
    valid_ends = endTimesForMchOfa[:machine_len]
    valid_ops = opsIDsForMchOfa[:machine_len]

    # 只考虑那些“当前任务的 job ready time 早于该位置已有任务 start”的位置
    possiblePos = np.where(float(jobRdyTime_a) < valid_starts)[0]
    if len(possiblePos) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    # 对于插入到 pos 前:
    # gap_start = pos==0 ? 0 : valid_ends[pos-1]
    # gap_end   = valid_starts[pos]
    # earliest_start = max(job_ready, gap_start)
    # 若 earliest_start + dur_a <= gap_end 则可插
    endTimesForPossiblePos = np.empty(len(possiblePos), dtype=np.float32)

    for local_idx, pos in enumerate(possiblePos.tolist()):
        if pos == 0:
            gap_start = 0.0
        else:
            gap_start = float(valid_ends[pos - 1])

        earliest_start = max(float(jobRdyTime_a), gap_start)
        endTimesForPossiblePos[local_idx] = earliest_start

    possibleGaps = valid_starts[possiblePos] - endTimesForPossiblePos
    idxLegalPos = np.where(float(dur_a) <= possibleGaps)[0]
    legalPos = possiblePos[idxLegalPos]

    return idxLegalPos.astype(np.int64), legalPos.astype(np.int64), endTimesForPossiblePos


def putInBetween(
    a,
    idxLegalPos,
    legalPos,
    endTimesForPossiblePos,
    startTimesForMchOfa,
    opsIDsForMchOfa,
    endTimesForMchOfa,
    dur_a,
    machine_len,
):
    """
    中间插入:
    使用原地后移替代 np.insert，避免新数组分配与整段拷贝。
    """
    earliest_idx = int(idxLegalPos[0])
    insert_pos = int(legalPos[0])
    startTime_a = float(endTimesForPossiblePos[earliest_idx])

    L = int(machine_len)

    # 原地后移 [insert_pos:L) -> [insert_pos+1:L+1)
    if insert_pos < L:
        startTimesForMchOfa[insert_pos + 1 : L + 1] = startTimesForMchOfa[insert_pos:L]
        endTimesForMchOfa[insert_pos + 1 : L + 1] = endTimesForMchOfa[insert_pos:L]
        opsIDsForMchOfa[insert_pos + 1 : L + 1] = opsIDsForMchOfa[insert_pos:L]

    # 写入新任务
    startTimesForMchOfa[insert_pos] = startTime_a
    endTimesForMchOfa[insert_pos] = startTime_a + float(dur_a)
    opsIDsForMchOfa[insert_pos] = int(a)

    return float(startTime_a)


def calJobAndMchRdyTimeOfa(
    a,
    mch_a,
    mchMat,
    durMat,
    mchsStartTimes,
    opIDsOnMchs,
    mchEndTime,
    row,
    col,
    first_col,
    last_col,
    machine_len: Optional[int] = None,
):
    """
    计算:
    - jobRdyTime_a: 该工序的 job 前驱完工时间
    - mchRdyTime_a: 目标机器尾部最后一个任务的完工时间

    相比旧版优化:
    1. 机器 ready time 直接读取 mchEndTime 尾项，不再 start+dur 反推
    2. 对 machine predecessor 不再重复扫描 np.where(op>=0)
    3. job predecessor 仍需在其所属机器队列中查位置，但只查单机单行
    """
    a = int(a)
    mch_a = int(mch_a)
    row = int(row)
    col = int(col)

    # -------- job ready time --------
    if col != 0:
        jobPredecessor = a - 1
        mchJobPredecessor = int(mchMat[row, col - 1])

        if mchJobPredecessor < 0:
            raise RuntimeError(
                f"job predecessor of op {a} (row={row}, col={col}) has not been assigned machine."
            )

        pred_ops = opIDsOnMchs[mchJobPredecessor]
        pred_pos_arr = np.where(pred_ops == int(jobPredecessor))[0]
        if len(pred_pos_arr) == 0:
            raise RuntimeError(
                f"cannot locate predecessor op {jobPredecessor} on machine {mchJobPredecessor}."
            )

        pred_pos = int(pred_pos_arr[0])
        jobRdyTime_a = float(mchEndTime[mchJobPredecessor, pred_pos])
    else:
        jobRdyTime_a = 0.0

    # -------- machine ready time --------
    if machine_len is None:
        machine_len = _get_machine_effective_len(opIDsOnMchs[mch_a])

    if machine_len > 0:
        mchRdyTime_a = float(mchEndTime[mch_a, machine_len - 1])
    else:
        mchRdyTime_a = 0.0

    return float(jobRdyTime_a), float(mchRdyTime_a)


def _get_machine_effective_len(opsIDsForMchOfa) -> int:
    """
    获取单台机器当前有效调度长度。

    旧代码通过:
        np.where(startTimes == -configs.high)[0][0]
    查空位，这会多扫一遍且依赖 sentinel。
    这里改为基于 opIDsOnMchs 的有效任务段。
    """
    valid = np.where(np.asarray(opsIDsForMchOfa) >= 0)[0]
    return int(valid[-1] + 1) if len(valid) > 0 else 0


if __name__ == "__main__":
    print("Optimized permissibleLS.py loaded successfully.")
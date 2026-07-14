from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def getActionNbghs(
    action: int,
    opIDsOnMchs: np.ndarray,
    mch_id: Optional[int] = None,
    pos: Optional[int] = None,
    machine_len: Optional[np.ndarray] = None,
) -> Tuple[int, int]:
    """
    Optimized neighbor query for a scheduled operation.

    Parameters
    ----------
    action : int
        当前插入/已调度的 operation id
    opIDsOnMchs : np.ndarray
        机器上的工序排程序列，形状通常为 [M, T]
    mch_id : Optional[int]
        如果环境已经知道该 op 所在机器，则直接传入，避免全表扫描
    pos : Optional[int]
        如果环境已经知道该 op 在机器序列中的位置，则直接传入
    machine_len : Optional[np.ndarray]
        每台机器当前有效长度，形状通常为 [M]
        若提供，则不会把尾部负值空位误当成真实后继

    Returns
    -------
    precd : int
        同机器前驱工序；若不存在则返回 -1
    succd : int
        同机器后继工序；若不存在则返回 -1

    Notes
    -----
    优先级:
    1) 若 mch_id 和 pos 都提供，则 O(1) 查询
    2) 否则退化为在目标矩阵中定位 action（兼容旧代码）
    """

    action = int(action)

    if opIDsOnMchs.ndim != 2:
        raise ValueError(
            f"opIDsOnMchs 应为二维数组 [num_machines, max_tasks]，当前 shape={opIDsOnMchs.shape}"
        )

    # ------------------------------------------------------------
    # Fast path: 环境已提供机器号和位置，直接 O(1) 查邻居
    # ------------------------------------------------------------
    if mch_id is not None and pos is not None:
        mch_id = int(mch_id)
        pos = int(pos)
        return _get_neighbors_by_position(
            opIDsOnMchs=opIDsOnMchs,
            mch_id=mch_id,
            pos=pos,
            machine_len=machine_len,
        )

    # ------------------------------------------------------------
    # Compatible fallback: 老代码路径，仍允许全表搜索
    # ------------------------------------------------------------
    coordAction = np.where(opIDsOnMchs == action)
    if len(coordAction[0]) == 0:
        raise ValueError(f"在 opIDsOnMchs 中找不到 action={action}")

    mch_id = int(coordAction[0][0])
    pos = int(coordAction[1][0])

    return _get_neighbors_by_position(
        opIDsOnMchs=opIDsOnMchs,
        mch_id=mch_id,
        pos=pos,
        machine_len=machine_len,
    )


def _get_neighbors_by_position(
    opIDsOnMchs: np.ndarray,
    mch_id: int,
    pos: int,
    machine_len: Optional[np.ndarray] = None,
) -> Tuple[int, int]:
    """
    根据 (machine_id, pos) 直接获取同机器前驱/后继。
    """
    num_machines, max_slots = opIDsOnMchs.shape

    if mch_id < 0 or mch_id >= num_machines:
        raise IndexError(f"mch_id 越界: {mch_id}, num_machines={num_machines}")
    if pos < 0 or pos >= max_slots:
        raise IndexError(f"pos 越界: {pos}, max_slots={max_slots}")

    # 有效长度优先
    if machine_len is not None:
        effective_len = int(machine_len[mch_id])
    else:
        effective_len = _infer_machine_len(opIDsOnMchs[mch_id])

    if effective_len <= 0:
        return -1, -1

    # 若给定位置已经落在有效段外，也视为非法
    if pos >= effective_len:
        return -1, -1

    precd = -1
    succd = -1

    if pos - 1 >= 0:
        prev_op = int(opIDsOnMchs[mch_id, pos - 1])
        if prev_op >= 0:
            precd = prev_op

    if pos + 1 < effective_len:
        next_op = int(opIDsOnMchs[mch_id, pos + 1])
        if next_op >= 0:
            succd = next_op

    return precd, succd


def _infer_machine_len(machine_ops_row: np.ndarray) -> int:
    """
    在没有 machine_len 缓存时，根据单机排程序列推断有效长度。
    默认认为:
    - >= 0 的值表示已调度 op
    - < 0 的值表示空位 / sentinel
    """
    valid = np.where(np.asarray(machine_ops_row) >= 0)[0]
    return int(valid[-1] + 1) if len(valid) > 0 else 0


if __name__ == "__main__":
    opIDsOnMchs = np.array([
        [7, 29, 33, 16, -6, -6],   # machine 0
        [6, 18, 28, 34, 2, -6],    # machine 1
        [26, 31, 14, 21, 11, 1],   # machine 2
        [30, 19, 27, 13, 10, -6],  # machine 3
        [25, 20, 9, 15, -6, -6],   # machine 4
        [24, 12, 8, 32, 0, -6],    # machine 5
    ], dtype=np.int32)

    machine_len = np.array([4, 5, 6, 5, 4, 5], dtype=np.int32)

    action = 29

    # 兼容旧接口
    precd, succd = getActionNbghs(action, opIDsOnMchs)
    print("fallback:", precd, succd)

    # 新接口：直接给机器号和位置
    precd, succd = getActionNbghs(action, opIDsOnMchs, mch_id=0, pos=1, machine_len=machine_len)
    print("fastpath:", precd, succd)
import numpy as np


def lastNonZero(arr, axis=1, invalid_val=-1):
    """
    找每行最后一个非零元素位置
    优化版本：避免 flip
    """
    mask = arr != 0

    any_valid = mask.any(axis=axis)

    idx = mask.shape[axis] - np.argmax(mask[:, ::-1], axis=axis) - 1

    idx = np.where(any_valid, idx, invalid_val)

    x = np.arange(arr.shape[0], dtype=np.int64)
    x = x[idx >= 0]
    y = idx[idx >= 0]

    return x, y


def calEndTimeLBm(temp1, min_mat):
    """
    计算 LBm
    """

    temp1 = np.asarray(temp1)
    min_mat = np.asarray(min_mat)

    mask = temp1 != 0

    x, y = lastNonZero(temp1, axis=1, invalid_val=-1)

    # 已调度任务设为0
    min_mat = min_mat.copy()
    min_mat[mask] = 0

    # action completion time
    min_mat[x, y] = temp1[x, y]

    # prefix sum
    temp20 = np.cumsum(min_mat, axis=1)

    temp20[mask] = 0

    ret = temp1 + temp20

    return ret


def calEndTimeLB(temp1, min_mat, mean_mat):
    """
    计算 LB 和 LBm
    """

    temp1 = np.asarray(temp1)

    min_mat = np.asarray(min_mat).copy()
    mean_mat = np.asarray(mean_mat).copy()

    mask = temp1 != 0

    x, y = lastNonZero(temp1, axis=1, invalid_val=-1)

    # 已调度清零
    min_mat[mask] = 0
    mean_mat[mask] = 0

    # action completion time
    min_mat[x, y] = temp1[x, y]
    mean_mat[x, y] = temp1[x, y]

    # prefix sums
    temp20 = np.cumsum(min_mat, axis=1)
    temp21 = np.cumsum(mean_mat, axis=1)

    temp20[mask] = 0
    temp21[mask] = 0

    # 预分配结果
    J, O = temp1.shape

    ret = np.empty((J, O, 2), dtype=temp1.dtype)

    ret[:, :, 0] = temp1 + temp20
    ret[:, :, 1] = temp1 + temp21

    return ret


if __name__ == "__main__":

    dur = np.random.randint(1, 10, (3, 3))

    temp1 = np.zeros((3, 3))

    temp1[0, 0] = 1
    temp1[1, 0] = 3
    temp1[1, 1] = 5

    ret = calEndTimeLBm(temp1, dur)

    print(ret)
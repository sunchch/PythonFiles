import numpy as np
from AllCellForward import run_forward


"""
首先定义需要计算的梯度变量符号
ds_next -- 表示当前cell的损失对输出s_t的导数
dtanh -- 表示当前cell的损失对激活函数tanh的导数
dx_t -- 表示当前cell的损失对输入x_t的导数
dU -- 当前cell的损失对U的导数
dW -- 当前cell的损失对W的导数
dba -- 当前cell的损失对ba的导数
ds_prev -- 当前cell的损失对s_prev的导数
"""


def run_cell_backward(ds_next, cache):
    """
    对单个cell进行反向传播
    :param ds_next: 当前隐层输出结果相对于损失的导数
    :param cache: 当前cell的缓存  cache = (s_next, s_prev, x_t, parameters)
    :return: gradients 该cell的6个梯度值
    """
    # 获取cache里面的参数
    (s_next, s_prev, x_t, parameters) = cache

    U = parameters["U"]
    # V = parameters["V"]
    W = parameters["W"]
    # ba = parameters["ba"]
    # by = parameters["by"]

    # 根据公式进行反向传播计算
    dtanh = (1 - s_next ** 2) * ds_next
    dU = np.dot(dtanh, x_t.T)
    dW = np.dot(dtanh, s_prev.T)
    dba = np.sum(dtanh, axis=1, keepdims=True)  # keepdims=True 保持维度不变
    dx_t = np.dot(U.T, dtanh)
    ds_prev = np.dot(W.T, dtanh)

    gradients = {"dtanh": dtanh, "dU": dU, "dW": dW, "dba": dba, "dx_t": dx_t, "ds_prev": ds_prev}

    return gradients


def run_backward(ds, caches):
    """
    对给定的一个序列进行RNN的反向传播
    :param ds: 假设T=4, 则ds形状为(n, 1, 4)
    :param caches:
    :return:
    """
    # 获取第一个cell的数据，参数，输入输出值
    (s_1, s_0, x_1, parameters) = caches[0]

    # 获取m和n的值
    n, _, T = ds.shape  # ds.shape = (n, 1, T)
    m, _ = x_1.shape  # x_1.shape = (m, 1)

    # 初始化梯度值
    dx = np.zeros((m, 1, T))
    dU = np.zeros((n, m))
    dW = np.zeros((n, n))
    dba = np.zeros((n, 1))
    ds0 = np.zeros((n, 1))
    ds_prevt = np.zeros((n, 1))

    # 循环从后往前进行传播
    for t in reversed(range(T)):
        # 取出最后一个时刻的梯度值 + 0
        gradients = run_cell_backward(ds[:, :, t] + ds_prevt, caches[t])

        # 获取梯度值准备更新参数
        dx_t, ds_prevt, dUt, dWt, dbat = gradients["dx_t"], gradients["ds_prev"], gradients["dU"], gradients["dW"], gradients["dba"]

        # 进行每次t时间上的梯度相加，作为最终更新的梯度
        dx[:, :, t] = dx_t
        dU += dUt
        dW += dWt
        dba += dbat

    # 最后计算ds0的输出梯度值
    ds0 = ds_prevt

    # 存储需要更新的梯度到字典中
    gradients = {"dx": dx, "ds0": ds0, "dU": dU, "dW": dW, "dba": dba}

    return gradients


if __name__ == "__main__":
    np.random.seed(1)

    # 定义4个cell，每个形状为(3, 1)
    x = np.random.randn(3, 1, 4)
    s0 = np.random.randn(5, 1)
    W = np.random.randn(5, 5)
    U = np.random.randn(5, 3)
    V = np.random.randn(3, 5)
    ba = np.random.randn(5, 1)
    by = np.random.randn(3, 1)
    parameters = {"W": W, "U": U, "V": V, "ba": ba, "by": by}

    s, y, caches = run_forward(x, s0, parameters)

    # 随机给每个cell的隐层输出的导数结果（真实情况下需要计算损失的导数）
    ds = np.random.randn(5, 1, 4)

    gradients = run_backward(ds, caches)

    print(gradients)
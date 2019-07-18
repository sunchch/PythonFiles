import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def run_cell_forward(x_t, s_prev, parameters):
    """
    单个RNN-cell的前向传播过程
    :param x_t: 单元的输入
    :param s_prev: 上一个单元的输出
    :param parameters: 单元中的参数
    :return: s_next, out_pred, cache
    """
    # 获取参数
    U = parameters["U"]
    W = parameters["W"]
    V = parameters["V"]
    ba = parameters["ba"]
    by = parameters["by"]

    # 计算激活函数
    s_next = np.tanh(np.dot(U, x_t) + np.dot(W, s_prev) + ba)

    # 计算当前cell的输出预测结果
    out_pred = softmax(np.dot(V, s_next) + by)

    # 以便后面的反向传播计算
    cache = (s_next, s_prev, x_t, parameters)

    return s_next, out_pred, cache


"""
测试前向传播过程，假设创建下面形状的数据进行测试，m=3是词的个数，n=5为自定义的数字。

UX + WS + ba = S
[n, m] * [m, 1] + [n, n] * [n, 1] + [n, 1] = [n, 1]
[5, 3] * [3, 1] + [5, 5] * [5, 1] + [5, 1] = [5, 1]
U: (5, 3)
X: (3, 1)
W: (5, 5)
S: (5, 1)
ba: (5, 1)

VS + by = out
[m, n] * [n, 1] + [m, 1] = [m, 1]
[3, 5] * [5, 1] + [3, 1] = [3, 1]
V: (3, 5)
by: (3, 1)
"""
if __name__ == "__main__":
    np.random.seed(1)

    x_t = np.random.randn(3, 1)
    s_prev = np.random.randn(5, 1)
    U = np.random.randn(5, 3)
    W = np.random.randn(5, 5)
    V = np.random.randn(3, 5)
    ba = np.random.randn(5, 1)
    by = np.random.randn(3, 1)
    parameters = {"U": U, "W": W, "V": V, "ba": ba, "by": by}

    s_next, out_pred, cache = run_cell_forward(x_t, s_prev, parameters)
    print("s_next = ", s_next)
    print("s_next.shape = ", s_next.shape)
    print("out_pred = ", out_pred)
    print("out_pred.shape = ", out_pred.shape)

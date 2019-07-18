import numpy as np
from SingleCellForward import run_cell_forward


def run_forward(X, s0, parameters):
    """
    实现所有cell的RNN前向传播
    :param X: T时刻的X总输入，形状 (m, 1, T), T表示序列长度
    :param s0: 隐层第一次输入
    :param parameters: 参数字典
    :return: S, Y, caches
    """
    caches = []

    # 根据X输入的形状确定cell的个数 (3, 1, T)
    # m=3为词的个数， n=5为自定义的数字
    m, _, T = X.shape

    # 根据输出
    m, n = parameters["V"].shape

    # 初始化所有cell的S, 用于保存所有cell的隐层结果
    S = np.zeros((n, 1, T))
    # 初始化所有cell的输出Y，保存所有输出结果
    Y = np.zeros((m, 1, T))

    # 初始化第一个输入
    s_next = s0

    # 根据cell的个数进行循环，并保存每组的cache
    for t in range(T):
        # 跟新每个隐层的输出计算结果，这里的s_next需要每次循环都要变
        # s_next, out_pred, cache = run_cell_forward(X[:, :, t], s0, parameters)
        s_next, out_pred, cache = run_cell_forward(X[:, :, t], s_next, parameters)
        # 保存隐层的输出值
        S[:, :, t] = s_next
        # 保存cell的预测值out_pred
        Y[:, :, t] = out_pred
        # 保存每个cell的缓存
        caches.append(cache)

    return S, Y, caches


if __name__ == "__main__":
    np.random.seed(1)

    # 定义4个cell，每个的形状为(3, 1)
    x = np.random.randn(3, 1, 4)  # (m, 1, T)
    s0 = np.random.randn(5, 1)
    W = np.random.randn(5, 5)
    U = np.random.randn(5, 3)
    V = np.random.randn(3, 5)
    ba = np.random.randn(5, 1)
    by = np.random.randn(3, 1)
    parameters = {"W": W, "U": U, "V": V, "ba": ba, "by": by}

    s, y, caches = run_forward(x, s0, parameters)
    print("s = ", s)
    print("s.T = ", s.T)
    print("s.shape = ", s.shape)
    print("y = ", y)
    print("y.T = ", y.T)
    print("y.shape = ", y.shape)
import numpy as np


def initialize_adam(parameters):
    """
    Initialize v and s as two python dictionaries with:
        - keys: "dW1", "db1", ..., "dWL", "dbL"
        - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    Arguments:
        parameters -- python dictionary containing your parameters.
            parameters["W" + str(l)] = W1
            parameters["b" + str(l)] = b1
    Returns:
        v -- python dictionary that will contain the exponentially weighted average of the gradient.
            v["dW" + str(l)] = ...
            v["db" + str(l)] = ...
        s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
            s["dW" + str(l)] = ...
            s["db" + str(l)] = ...
    """
    L = len(parameters) // 2  # number of layers in the neural networks
    v = {}
    s = {}

    # Initialize v, s. Input: "parameters". Output: "v, s"
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
        v["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)
        s["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
        s["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)

    return v, s


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update parameters using Adam

    Arguments:
        parameters -- python dictionary containing your parameters.
            parameters["W" + str(l)] = W1
            parameters["b" + str(l)] = b1
        grads -- python dictionary containing your gradients for each parameters:
            grads["dW" + str(l)] = dW1
            grads["db" + str(l)] = db1
        v -- Adam variable, moving average of the first gradient, python dictionary
        s -- Adam variable, moving average of the squared gradient(平方梯度的移动平均), python dictionary
        beta1 -- Exponential decay hyperparameter for the first moment estimates (第一个矩的指数衰减超参数估计)
        beta2 -- Exponential decay hyperparameter for the second moment estimates
        epsilon -- hyperparameter preventing division by zeros in Adam updates (在Adam更新中防止被零除法的超参数)
    Returns:
        parameters -- python dictionary containing your update parameters
        v -- Adam variable, moving average of the first gradient, python dictionary
        s -- Adam variable, moving average of the squared gradient, python dictionary
    """
    L = len(parameters) // 2
    v_corrected = {}  # Initializing first moment estimate, python dictionary
    s_corrected = {}  # Initializing second moment estimate, python dictionary

    # Perform Adam update on all parameters
    for l in range(L):
        # 动量下降结果
        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v"
        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads["db" + str(l + 1)]
        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected"
        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - beta1 ** t)
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - beta1 ** t)

        # RMSProp梯度下降结果
        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * (grads["dW" + str(l + 1)] ** 2)
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * (grads["db" + str(l + 1)] ** 2)
        # Compute bias-corrected second raw moment estimate. Input: "s, beta2, t". Output: "s_corrected"
        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - beta2 ** t)
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - beta2 ** t)

        # Adam算法的参数更新
        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters"
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v_corrected["dW" + str(l + 1)] / (s_corrected["dW" + str(l + 1)] ** 0.5 + epsilon)
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v_corrected["db" + str(l + 1)] / (s_corrected["db" + str(l + 1)] ** 0.5 + epsilon)

    return parameters, v, s
import numpy as np


def initialize_velocity(parameters):
    """
    Initializes the velocity as a python dictionary with:
        - keys: "dW1", "db1", ..., "dWL", "dbL"
        - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    Arguments:
        parameters -- python dictionary containing your parameters.
            parameters["W" + str(l)] = W1
            parameters["b" + str(l)] = b1
    Returns:
        v -- python dictionary containing the current velocity.
            v["dW" + str(l)] = velocity of dW1
            v["db" + str(l)] = velocity of db1
    """
    L = len(parameters) // 2  # number of layers in the neural networks
    v = {}

    # Initialize velocity
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
        v["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)

    return v


def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    Upgrade parameters using Momentum

    Arguments:
        parameters -- python dictionary containing your parameters:
            parameters['W' + str(l)] = W1
            parameters['b' + str(l)] = b1
        grads -- python dictionary containing your gradients for each parameters:
            grads['dW' + str(l)] = dW1
            grads['db' + str(l)] = db1
        v -- python dictionary containing the current velocity:
            v['dW' + str(l)] = ...
            v['db' + str(l)] = ...
        beta -- the momentum hyperparameter, scalar
        learning_rate -- the learning rate, scalar

    Returns:
        parameters -- python dictionary containing your update parameters
        v -- python dictionary containing your updated velocities
    """
    L = len(parameters) // 2  # number of layers in the neural networks

    # Momentum update for each parameter
    for l in range(L):
        # compute velocities
        v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads["db" + str(l + 1)]
        # update parameters
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v["db" + str(l + 1)]

    return parameters, v
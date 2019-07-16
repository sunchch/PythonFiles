import numpy as np
import math
import matplotlib.pyplot as plt
from opt_utils import initialize_parameters, forward_propagation, backward_propagation
from optimize.momentum import initialize_velocity, update_parameters_with_momentum
from optimize.adam import initialize_adam, update_parameters_with_adam
from opt_utils import compute_cost, load_dataset, predict


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Create a list of minibatches from (X, Y)

    Arguments:
        X -- input data, of shape (input size, number of examples)
        Y -- true "label" vector (1 of blue dot / 0 for red dot), of shape (1, number of examples)
        mini_batch_size -- size of mini-batches, integer
    Returns:
        mini_bauches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    np.random.seed(seed)  # To make your "random" minibatches the same as ours
    m = X.shape[1]  # number of training examples
    mini_batches = []

    # Step 1: shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m / mini_batch_size)  # number of mini batches of size mini_batches_size in your partitionning(分区)
    for k in range(0, int(num_complete_minibatches)):
        mini_batch_X = shuffled_X[:, mini_batch_size * k: (k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * k: (k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini_batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, mini_batch_size * (k + 1): m]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * (k + 1): m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def model(X, Y, layers_dims, optimizer, learning_rate=0.0007, mini_batch_size=64, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=10000, print_cost=True):
    """
    3-layer neural network model which can be run in different optimizer models.

    Arguments:
        X -- input data, of shape (2, number of examples)
        Y -- true "label" vector (1 of blue dot / 0 for red dot), of shape (1, number of examples)
        layers_dims -- python list, containing rate, scalar
        mini_batch_size -- the size of a mini batch
        beta -- Momentum hyperparameter
        beta1 -- Exponential decay hyperparameter for the past gradients estimates
        beta2 -- Exponential decay hyperparameter for the past squared gradients estimates
        epsilon -- hyperparameter preventing division by zero in Adam updates
        num_epochs -- number of epochs
        print_cost -- True to print the cost every 1000 epochs

    Returns；
        parameter -- python dictionary containing your updated parameters
    """
    L = len(layers_dims)
    costs = []
    t = 0  # initializing the counter required for Adam update
    seed = 10  # For grading purposes, so that your "random" minibatches are the same as ours

    # Initialize parameters
    parameters = initialize_parameters(layers_dims)
    # print("parameters: ", parameters)

    # Initialize the optimizer
    if optimizer == "gd":
        pass  # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    # Optimization loop
    for i in range(num_epochs):
        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)

        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # forward propagation
            a3, caches = forward_propagation(minibatch_X, parameters)

            # compute cost
            cost = compute_cost(a3, minibatch_Y)

            # backward propagation
            grads = backward_propagation(minibatch_X, minibatch_Y, caches)

            # update parameters
            # if optimizer == "gd":
            #     parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            # elif optimizer == "momentum":
            #     parameters = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            # elif optimizer == "adam":
            #     t = t + 1  # Adam counter
            #     update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)

            if optimizer == "momentum":
                parameters = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1  # Adam counter
                update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)

        # print the cost every 1000 epoch
        if print_cost and i % 1000 == 0:
            print("Cost after epoch %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters


if __name__ == "__main__":
    train_X, train_Y = load_dataset()
    layers_dims = [train_X.shape[0], 5, 2, 1]
    parameters = model(train_X, train_Y, layers_dims, optimizer="momentum")
    predictions = predict(train_X, train_Y, parameters)

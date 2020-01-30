import numpy as np
import RBF as r
import matplotlib.pyplot as plt

datapoints = 63
feature = 1
nodes = 63
sigma = 0.5
noise = False


def find_closest(x, W):
    """
    :param x: current datapoint
    :param W: W Matrix
    :return: Column index for the closest weight
    """
    closest = 0
    for i, w in enumerate(W):
        if np.linalg.norm(x - w) < np.linalg.norm(x - W[int(closest)]):
            closest = i
    return closest


def delta_w(x, w, eta=0.2):
    return eta * (x - w)


def train_x(x, W):
    w_index = find_closest(x, W)
    # print(W[w_index])
    W[w_index] += delta_w(x, W[w_index])
    # print(W[w_index])
    return W


def init_weights():
    return np.random.rand(datapoints)


def gaussian_rbf(x1, x2, sigma):
    return np.exp(-(x1 - x2) ** 2 / 2 * (sigma ** 2))


def predict_x(weights, x_values):
    predictions = np.zeros(nodes)  # will be features x hidden_nodes
    for i in range(nodes):
        predictions[i] = gaussian_rbf(x_values, weights[i], sigma)
        print(predictions[i])
    return np.sum(predictions)


def train(W, X):
    for x in X:
        W = train_x(x, W)
    return W


def generate_phi_matrix(mu_node_list, x_values):
    phi = np.zeros((nodes, datapoints))  # will be features x hidden_nodes
    for i in range(nodes):
        phi[i] = gaussian_rbf(x_values, mu_node_list[i], sigma)
    return phi.T


def least_squares(phi, function_target):
    return np.linalg.solve(phi.T @ phi, phi.T @ function_target)  # W


def network(weights, training, testing, targets):
    phi = generate_phi_matrix(weights, training)
    W = least_squares(phi, targets)
    phi_test_matrix = generate_phi_matrix(weights, testing)
    total_error = np.square(np.subtract(phi_test_matrix @ W, targets)).mean()
    plt.plot(testing, phi_test_matrix @ W, label="Approximate")
    plt.plot(testing, targets, label="True")
    plt.legend()
    plt.show()


def competitive_learning(training):
    iterations = 100
    W = init_weights()
    for i in range(0, iterations):
        W = train(W, training)
    return W


training, testing, sin_target, square_target = r.generate_data(noise)
weights = competitive_learning(training)
network(weights, training, testing, square_target)

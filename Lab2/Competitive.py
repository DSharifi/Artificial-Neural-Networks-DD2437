import numpy as np
import RBF as r
import matplotlib.pyplot as plt

datapoints = 63
feature = 1
nodes = 30
sigma = 1
noise = True


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


def delta_w(x, w, eta=0.5):
    return eta * (x - w)


def train_x(x, W):
    w_index = find_closest(x, W)
    # print(W[w_index])
    W[w_index] += delta_w(x, W[w_index])
    # print(W[w_index])
    return W


def init_weights():
    return np.random.rand(nodes)


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


def plot_error(nodes, errors):
    plt.plot(nodes, errors, color="blue", label="Error")

    plt.title(
        "Absolute residual error depending on size of the hidden layer\nfor least_square learning, width=" + str(sigma))
    plt.xlabel("Amount of Hidden Nodes")
    plt.ylabel("Absolute residual error")
    plt.legend(loc="upper right")
    plt.grid()
    plt.show()


def network(weights, training, testing, targets, plot=False):
    phi = generate_phi_matrix(weights, training)
    W = least_squares(phi, targets)
    phi_test_matrix = generate_phi_matrix(weights, testing)
    total_error = np.abs(np.subtract(phi_test_matrix @ W, targets)).mean()
    print(total_error)
    if plot:
        plt.plot(testing, phi_test_matrix @ W, label="Approximate")
        plt.plot(testing, targets, label="True")
        plt.scatter(weights, np.zeros(nodes), label="weights")
        plt.legend()
        plt.title("RBF Network with Competitive Learning \n  hidden nodes: " + str(nodes) + ", sigma: " + str(sigma))
        plt.grid()
        plt.show()
    return total_error


def competitive_learning(training):
    iterations = 20
    W = init_weights()
    for i in range(0, iterations):
        W = train(W, training)
    return W


def error_testing():
    global nodes
    training, testing, sin_target, square_target = r.generate_data(noise)
    errors = []
    amt_nodes = []
    for i in range(2, nodes):
        print(i)
        nodes = i
        weights = competitive_learning(training)
        amt_nodes.append(i)
        errors.append(network(weights, training, testing, sin_target))
    plot_error(amt_nodes, errors)


training, testing, sin_target, square_target = r.generate_data(noise)
weights = competitive_learning(training)
network(weights, training, testing, sin_target, True)
#error_testing()



import numpy as np
import RBF as r
import matplotlib.pyplot as plt

datapoints = 63
feature = 1
nodes = 90
sigma = 1
amt_learn = 1
eta = 0.2
noise = True
winners = np.zeros(nodes)


def find_X_closest(x, W, amt=amt_learn):
    """
    :param x: current datapoint
    :param W: W Matrix
    :return: Column index for the closest weight
    """
    # closest_list = np.ones(amt) * np.inf
    # closest_list = sorted(closest_list, key=lambda row: row[0])
    distances = np.zeros(W.shape[0])
    for i, w in enumerate(W):
        distances[i] = np.linalg.norm(x - w)
    weights = []
    for i in range(amt):
        if i >= nodes:
            break
        index = np.argmin(distances)
        winners[index] += 1
        weights.append(index)
        distances[index] = np.inf
    return weights


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
    return np.array([closest])


def delta_w(x, w, eta):
    # print(x-w)
    return eta * (x - w)


def train_x(x, W):
    w_index = find_X_closest(x, W)
    for i, id in enumerate(w_index):
        # print("id: " + str(id))
        # print(W[id])
        W[id] += delta_w(x, W[id], eta / (i + 1))
        # print(W[id])
    # print("-----------------")
    return W


def init_weights():
    return np.random.rand(nodes, feature)


def gaussian_rbf(x1, x2, sigma):
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / 2 * (sigma ** 2))


def predict_x(weights, x_values):
    predictions = np.zeros(nodes)  # will be features x hidden_nodes
    for i in range(nodes):
        predictions[i] = gaussian_rbf(x_values, weights[i], sigma)
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


def network(weights, training, testing, targets_training, targets_test, plot=False):
    phi = generate_phi_matrix(weights, training)
    W = least_squares(phi, targets_training)
    phi_test_matrix = generate_phi_matrix(weights, testing)
    total_error = np.abs(np.subtract(phi_test_matrix @ W, targets_test)).mean()
    if plot:
        plt.plot(testing, phi_test_matrix @ W, label="Approximate")
        plt.plot(testing, targets_test, label="True")
        # plt.scatter(weights.T[0], weights.T[1], label="weights")
        plt.legend()
        plt.title("RBF Network with Competitive Learning \n  hidden nodes: " + str(nodes) + ", sigma: " + str(
            sigma) + ", learning rate: " + str(eta) + ", Amount of winners: " + str(amt_learn))
        plt.grid()
        plt.show()
    return total_error


def competitive_learning(training):
    iterations = 100
    W = init_weights()
    for i in range(0, iterations):
        W = train(W, training)
    return W


def error_testing(training, testing, target, targety):
    global nodes
    errors = []
    amt_nodes = []
    for i in range(5, nodes):
        nodes = i
        weights = competitive_learning(training)
        amt_nodes.append(i)
        errors.append(network(weights, training, testing, target, targety))
    plot_error(amt_nodes, errors)


def read_file_data(filename):
    file = "data_lab2/" + filename

    file = open(file, "r")
    list = []
    templist = []
    for i, line in enumerate(file.read().split()):
        if i % 4 == 0 and i != 0:
            list.append(templist)
            templist = []
        templist.append(line)
    list.append(templist)
    training = np.asfarray(np.array(list), float)
    training_X, training_Y = np.hsplit(training, 2)
    return training_X, training_Y


if __name__ == "__main__":
    # training_x, training_y = read_file_data("ballist.dat")
    # test_x, test_y = read_file_data("balltest.dat")
    training_x, testing_x, sin2x_target, square2x_target, sintarget_y = r.generate_data(False)
    error_testing(training_x, testing_x, sin2x_target, sintarget_y)
    # weights = competitive_learning(training_x)
    # network(weights, training_x, testing_x, sin2x_target, sintarget_y, True)

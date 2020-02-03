import numpy as np
import RBF as r
import matplotlib.pyplot as plt

datapoints = 100
feature = 2
nodes = 100
sigma = 1
amt_learn = 10
eta = 0.5
noise = True
ball = True
winners = np.zeros(nodes)
if noise:
    data_noise = 0.1
else:
    data_noise = 0

if ball:
    datapoints = 100
    feature = 2
else:
    datapoints = 63
    feature = 1

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
        if i == 0:
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
    phi = np.zeros((datapoints, nodes))  # will be features x hidden_nodes

    for i in range(nodes):
        for j in range(datapoints):
            phi[j][i] = gaussian_rbf(x_values[j], mu_node_list[i], sigma)
    return phi


def least_squares(phi, function_target):
    return np.linalg.solve(phi.T @ phi, phi.T @ function_target)  # W


def plot_error(nodes, errors):
    plt.plot(nodes, errors, color="blue", label="Error")

    plt.title(
        "Absolute residual error depending on size of the hidden layer\n, width=" + str(
            sigma) + ", Data noise (SD): " + str(data_noise))
    plt.xlabel("Amount of Hidden Nodes")
    plt.ylabel("Absolute residual error")
    plt.legend(loc="upper right")
    plt.grid()
    plt.show()


def network(weights, training, testing, targets_training, targets_test, plot=False):
    phi = generate_phi_matrix(weights, training)
    W = least_squares(phi, targets_training)
    phi_test_matrix = generate_phi_matrix(weights, testing)
    sum = 0
    total_error = np.linalg.norm(phi_test_matrix@W - targets_test, axis=1).mean()
    print(total_error)

    if plot:
        if ball:
            plt.scatter(testing, phi_test_matrix @ W, label="Approximate")
            plt.scatter(testing, targets_test, label="True")
        else:
            plt.plot(testing,targets_test, color="green", label="True")
            plt.plot(testing,phi_test_matrix@W,color="red", label="Approximated")

        # plt.scatter(weights.T[0], weights.T[1], label="weights")
        plt.legend()
        plt.title("RBF Network with Competitive Learning hidden nodes: " + str(nodes) + "\n sigma: " + str(
            sigma) + ", learning rate: " + str(eta) + ", Amount of winners: " + str(
            amt_learn) + "\n Gaussian Noise with SD: " + str(0.1))
        plt.grid()
        plt.show()
    return total_error


def competitive_learning(training):
    iterations = 10
    W = init_weights()
    for i in range(0, iterations):
        W = train(W, training)
    return W


def error_testing(training, testing, target, target_test):
    global nodes
    errors = []
    amt_nodes = []
    iters = 3
    for i in range(5, nodes, 1):
        print(i)
        sum = 0
        for j in range(iters):
            nodes = i
            weights = competitive_learning(training)
            sum += (network(weights, training, testing, target, target_test))
        amt_nodes.append(i)
        errors.append(sum / iters)
    plot_error(amt_nodes, errors)


def read_file_data(filename, noise=noise):
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
    if noise:
        training_X += np.random.normal(0, data_noise, [datapoints, feature])
    return training_X, training_Y


def weight_plot(training_x):
    weights = competitive_learning(training_x)
    plt.scatter(weights.T[0], weights.T[1], label="weights")
    plt.legend()
    plt.xlabel("Angle")
    plt.ylabel("Velocity")
    plt.title("Weight positioning with " + str(nodes) + " hidden nodes\n, sigma: " + str(
        sigma) + ", learning rate: " + str(eta) + ", Amount of winners: " + str(
        amt_learn) + ", Gaussian Noise with SD: " + str(data_noise))
    plt.grid()
    plt.show()


if __name__ == "__main__":
    if ball:
        training_x, training_y = read_file_data("ballist.dat")
        test_x, test_y = read_file_data("balltest.dat")
    else:
        training_x, test_x, training_y, square2x_target, test_y, square2x_test_target = r.generate_data(noise)
    error_testing(training_x, test_x, training_y, test_y)
    weights = competitive_learning(training_x)

    print((network(weights, training_x, test_x, training_y, test_y, False)))

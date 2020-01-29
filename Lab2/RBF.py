import numpy as np
import matplotlib.pyplot as plt

# ---------Network settings----------#
# Feature is based on the range from interval, [0, 2pi], with step size 0.1
features = 63
# Hidden nodes...
hidden_nodes = 63
# --------------------------------#


# ---------Node_i settings----------#
mu_expected_value = 1
mu_standard_deviation = 0.8

sigma = 1


# --------------------------------#

def gaussian_rbf(x, mean, sigma):
    return np.exp((-(x - mean) ** 2) / (2 * (sigma ** 2)))


def init_hidden_nodes():
    # mu_node = np.zeros(hidden_nodes)
    mu_node = np.random.uniform(0, 2 * np.pi, hidden_nodes)
    # mu_node = np.random.uniform(0,hidden_nodes, hidden_nodes)
    # mu_node = np.random.normal(mu_expected_value, mu_standard_deviation, hidden_nodes)
    # mu_node = np.ones(hidden_nodes)
    sigma_node = sigma * np.ones(hidden_nodes)
    return mu_node, sigma_node


def generate_weights():
    W = np.random.normal(0, 0.1, hidden_nodes)
    return W


def generate_data(settings=None):
    training_points = np.arange(0, 2 * np.pi, 0.1)
    testing_points = np.arange(0.05, 2 * np.pi, 0.1)
    features = (training_points.size)
    sin2x_target = np.sin(2 * training_points)

    square2x_target = np.ones(sin2x_target.size)
    square2x_target = np.where(sin2x_target < 0, square2x_target, -1)
    return training_points, testing_points, sin2x_target, square2x_target


def generate_phi_matrix(mu_node_list, sigma_node_list, x_values):
    phi = np.zeros((hidden_nodes, features))  # will be features x hidden_nodes
    for i in range(hidden_nodes):
        phi[i] = gaussian_rbf(x_values, mu_node_list[i], sigma_node_list[i])
    return phi.T


def least_squares(phi, function_target):
    return np.linalg.solve(phi.T @ phi, phi.T @ function_target)  # W


def delta_rule(error, phi_matrix, k, eta=0.01):
    return eta * error * phi_matrix[k]


def task3_1(use_square=False):
    global hidden_nodes
    iterations = features
    i = 2
    errors = []
    nodes = []
    while i < iterations:
        i += 1
        print(i)
        hidden_nodes = i
        training_points, testing_points, sin2x_target, square2x_target = generate_data()
        mu_node_list, sigma_node_list = init_hidden_nodes()
        phi_matrix = generate_phi_matrix(mu_node_list, sigma_node_list, training_points)
        if not use_square:
            W = least_squares(phi_matrix, sin2x_target)
            phi_test_matrix = generate_phi_matrix(mu_node_list, sigma_node_list, testing_points)
            #total_error = np.square(np.subtract(phi_test_matrix @ W, sin2x_target))
            total_error = np.abs(phi_test_matrix @ W - sin2x_target)
            print(total_error)
            errors.append(total_error.mean())
            nodes.append(hidden_nodes)
        else:
            W = least_squares(phi_matrix, square2x_target)
            phi_test_matrix = generate_phi_matrix(mu_node_list, sigma_node_list, testing_points)
            #total_error = np.square(np.subtract(phi_test_matrix @ W, square2x_target))
            total_error = np.abs(phi_test_matrix@W - square2x_target)
            errors.append(total_error.mean())
            nodes.append(hidden_nodes)

    plt.plot(nodes, errors, color="blue", label="error")
    plt.title("Absolute Residual Error depending on Hidden Nodes")
    plt.xlabel("Amount of Hidden Nodes")
    plt.ylabel("Absolute Residual Error")
    plt.grid()
    plt.show()


def plot_function():
    training_points, testing_points, sin2x_target, square2x_target = generate_data()

    mu_node_list, sigma_node_list = init_hidden_nodes()

    phi_matrix = generate_phi_matrix(mu_node_list, sigma_node_list, training_points)

    W = least_squares(phi_matrix, sin2x_target)

    phi_test_matrix = generate_phi_matrix(mu_node_list, sigma_node_list, testing_points)
    total_error = np.square(np.subtract(phi_test_matrix @ W, sin2x_target))
    plt.plot(np.arange(0.05, 2 * np.pi, 0.1), (phi_test_matrix @ W), color="blue", label="Trained")
    plt.plot(np.arange(0.05, 2 * np.pi, 0.1), sin2x_target, color="green", label="True")
    # plt.plot(nodes, errors, label='Error Line')
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Function approximation of sin(2x) using \nhidden nodes=" + str(hidden_nodes) + ", sigma=" + str(
        sigma) + ", mean error = " + str(total_error.mean()))
    plt.legend(loc="lower left")
    plt.grid()
    plt.show()


def task3_2():
    return 1


def task3_3():
    return 1


# ------------Function calls--------------#
# task3_1(True)
#plot_function()
task3_1(False)
# ----------------------------------------#

import numpy as np
import matplotlib.pyplot as plt

#---------Network settings----------#
# Feature is based on the range from interval, [0, 2pi], with step size 0.1
total_points = 63
# Hidden nodes...
hidden_nodes = 25
#--------------------------------#


#---------Node_i settings----------#
sigma = 1
#--------------------------------#

def gaussian_rbf(x, mean, sigma):
    return np.exp((-(x - mean)**2) / (2*(sigma**2)))

def init_hidden_nodes():
    mu_node = np.random.uniform(0, 2*np.pi, hidden_nodes)
    sigma_node = sigma * np.ones(hidden_nodes)
    return mu_node, sigma_node

def generate_weights():
    W = np.random.uniform(0, 2*np.pi, hidden_nodes)
    return W

def generate_data(add_noise = False):
    training_points = np.arange(0, 2*np.pi, 0.1)
    testing_points = np.arange(0.05, 2*np.pi, 0.1)    
    noise = np.random.normal(0, np.sqrt(0.1), (total_points,))
    sin2x_target = np.sin(2*training_points) + (noise if add_noise else 0)
    sin2x_test_target = np.sin(2*testing_points) + (noise if add_noise else 0)
    
    square2x_target = np.ones(sin2x_target.size)
    square2x_target = np.where(sin2x_target < 0, square2x_target, -1) + (noise if add_noise else 0)
    square2x_test_target = np.ones(sin2x_test_target.size)
    square2x_test_target = np.where(sin2x_test_target < 0, square2x_test_target, -1) + (noise if add_noise else 0)
    return training_points, testing_points, sin2x_target, square2x_target, sin2x_test_target, square2x_test_target

def generate_phi_matrix(mu_node_list, sigma_node_list, x_values):
    phi = np.zeros((hidden_nodes, total_points)) #will be features x hidden_nodes
    for i in range(hidden_nodes):
        phi[i] = gaussian_rbf(x_values, mu_node_list[i], sigma_node_list[i])
    return phi.T

def least_squares(phi, function_target):
    return np.linalg.solve(phi.T @ phi, phi.T @ function_target) #W

def delta_rule(error, phi_matrix, k, eta = 0.01):
    return eta*error*phi_matrix[k]
    
def task3_1(use_square = False):
    global hidden_nodes
    iterations = 65
    i = 63
    errors = []
    nodes = []
    W = list()
    phi_test_matrix = list()
    result_matrix = list()
    
    while i < iterations:
        #i += 1
        hidden_nodes = i
        i += 10
        training_points, testing_points, sin2x_target, square2x_target = generate_data()
        mu_node_list, sigma_node_list = init_hidden_nodes()
        phi_matrix = generate_phi_matrix(mu_node_list, sigma_node_list, training_points)
        if not use_square:
            W = least_squares(phi_matrix, sin2x_target)
            phi_test_matrix = generate_phi_matrix(mu_node_list, sigma_node_list, testing_points)
            result_matrix = phi_test_matrix @ W
            total_error = np.abs(np.subtract(result_matrix, sin2x_target))
            print(total_error.mean())
            print(hidden_nodes)
            if total_error.mean()>0.1:
                print(hidden_nodes)
                print(total_error.mean())
                print("-----------")
            errors.append(total_error.mean())
            nodes.append(hidden_nodes)
        else:
            W = least_squares(phi_matrix, square2x_target)
            phi_test_matrix = generate_phi_matrix(mu_node_list, sigma_node_list, testing_points)
            result_matrix = phi_test_matrix @ W
            for k in range(phi_test_matrix.shape[0]):
                if result_matrix[k] >= 0:
                    result_matrix[k] = 1
                else:
                    result_matrix[k] = -1
            total_error = np.abs(np.subtract(result_matrix, square2x_target))
            if total_error.mean()>0.1:
                print(hidden_nodes)
                print(total_error.mean())
                print("-----------")
            errors.append(total_error.mean())
            nodes.append(hidden_nodes)

        if i >= iterations:
            plot_error(nodes, errors, 0.01)
            plot_function(square2x_target if use_square else sin2x_target, result_matrix)

def plot_error(nodes, errors, eta, delta_plot=False):
    plt.plot(nodes, errors, color="blue", label="Error")
    if delta_plot:
        plt.title("Absolute residual error depending on size of the hidden layer\nfor delta learning, width=" + str(sigma) + ", eta=" + str(eta))
    else:
        plt.title("Absolute residual error depending on size of the hidden layer\nfor least_square learning, width=" + str(sigma)+ ", eta=" + str(eta))
    plt.xlabel("Amount of Hidden Nodes")
    plt.ylabel("Absolute residual error")
    plt.legend(loc="upper right")
    plt.grid()
    plt.show()

def plot_function(true_function, approx_function):
    plt.plot(np.arange(0.0, 2*np.pi, 0.1), approx_function, color="blue", label="Trained")
    plt.plot(np.arange(0.0, 2*np.pi, 0.1), true_function, color="green", label="True")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Function approximation of sin(2x) using \nhidden nodes=" + str(hidden_nodes) + ", sigma=" + str(sigma))
    plt.legend(loc="upper right")
    plt.grid()
    plt.show()


def shuffle(a, b):
    p = np.random.permutation(len(a))
    #print(p)
    return a[p], b[p]


def delta_learning(sin2x_target, square2x_target, sin2x_test_target, square2x_test_target, n_epochs, eta, use_square=False):
    training_points, testing_points, _, _, _, _ = generate_data()
    chosen_target = sin2x_target if not use_square else square2x_target
    mu_node_list, sigma_node_list = init_hidden_nodes()
    phi_matrix = generate_phi_matrix(mu_node_list, sigma_node_list, training_points)
    phi_test_matrix = generate_phi_matrix(mu_node_list, sigma_node_list, testing_points)
    W = generate_weights()
    k = 0
    iters = 0
    #TODO: SHUFFLE
    while iters < n_epochs:
        #phi_matrix, chosen_target = shuffle(phi_matrix, chosen_target)
        error = chosen_target[k] - (phi_matrix[k] @ W)
        k = (k+1) % total_points
        delta_w = delta_rule(error, phi_matrix, k, eta)
        W += delta_w
        iters += 1
    result_matrix = phi_test_matrix @ W
    mean_error = np.abs(np.subtract(result_matrix, square2x_test_target if use_square else sin2x_test_target)).mean()
    return mean_error, W, phi_test_matrix

def least_square_learning(sin2x_target, square2x_target, sin2x_test_target, square2x_test_target, use_square=False):
    training_points, testing_points, _, _, _, _ = generate_data()
    mu_node_list, sigma_node_list = init_hidden_nodes()
    phi_matrix = generate_phi_matrix(mu_node_list, sigma_node_list, training_points)
    W = list()
    phi_test_matrix = list()
    if not use_square:
        W = least_squares(phi_matrix, sin2x_target)
        phi_test_matrix = generate_phi_matrix(mu_node_list, sigma_node_list, testing_points)
        result_matrix = phi_test_matrix @ W
        mean_error = np.abs(np.subtract(result_matrix, sin2x_test_target)).mean()
    else:
        W = least_squares(phi_matrix, square2x_target)
        phi_test_matrix = generate_phi_matrix(mu_node_list, sigma_node_list, testing_points)
        result_matrix = phi_test_matrix @ W
        for k in range(phi_test_matrix.shape[0]):
            if result_matrix[k] >= 0:
                result_matrix[k] = 1
            else:
                result_matrix[k] = -1
        mean_error = np.abs(np.subtract(result_matrix, square2x_test_target)).mean()
    return mean_error, W, result_matrix

def task3_2(use_square=False, use_delta=False, use_noise=False, eta=0.01, n_epochs=5000):
    global sigma, hidden_nodes
    mean_error_list = list()
    W = list()
    result_matrix = list()
    phi_test_matrix = list()
    _, _, sin2x_target, square2x_target, sin2x_test_target, square2x_test_target = generate_data(True if use_noise else False)
    for i in range(1,total_points):
        hidden_nodes = i
        if use_delta:
            abs_mean_error, W, phi_test_matrix = delta_learning(sin2x_target, square2x_target, sin2x_test_target, square2x_test_target, n_epochs, eta, use_square)
            mean_error_list.append(abs_mean_error)
        else:
            abs_mean_error, W, result_matrix = least_square_learning(sin2x_target, square2x_target, sin2x_test_target, square2x_test_target, use_square)
            mean_error_list.append(abs_mean_error)
    plot_error(np.arange(1, total_points), mean_error_list, eta, True if use_delta else False)
    plot_function(square2x_target if use_square else sin2x_target, phi_test_matrix @ W if use_delta else result_matrix)
    return abs_mean_error
#------------Task calls--------------#
#task3_1(True)
#task3_1(use_square=False)
task3_2(use_square=False, use_delta=True, use_noise=True, eta=0.01, n_epochs=5000)
#----------------------------------------#

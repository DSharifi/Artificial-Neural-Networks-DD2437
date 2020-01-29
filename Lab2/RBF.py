import numpy as np
import matplotlib as plt

#---------Network settings----------#
# Feature is based on the range from interval, [0, 2pi], with step size 0.1
features = 63
# Hidden nodes...
hidden_nodes = 10
#--------------------------------#


#---------Node_i settings----------#
mu_expected_value = 1
mu_standard_deviation = 0.5
sigma = 1.5
#--------------------------------#

def gaussian_rbf(x, mean, sigma):
    return np.exp((-(x-mean)**2)/ (2*(sigma**2)))

def init_hidden_nodes():
    mu_node = np.random.normal(mu_expected_value, mu_standard_deviation, hidden_nodes)
    sigma_node = sigma * np.ones(hidden_nodes)
    return mu_node, sigma_node

def generate_weights():
    W = np.random.normal(0, 0.1, hidden_nodes)
    return W

def generate_data(settings=None):
    training_points = np.arange(0.0, 2*np.pi, 0.1)
    testing_points = np.arange(0.05, 2*np.pi, 0.1)
    
    sin2x_target = np.sin(2*training_points)
    
    square2x_target = np.ones(sin2x_target.size)
    square2x_target = np.where(sin2x_target < 0, square2x_target, -1)

    return training_points, testing_points, sin2x_target, square2x_target

def generate_phi_matrix(mu_node_list, sigma_node_list, x_values):
    phi = np.zeros((features, hidden_nodes))
    
    for i in len(features):
        for j in len(hidden_nodes):
            
    print(phi.shape)
    return
    
training_points, testing_points, sin2x_target, square2x_target = generate_data()
W = generate_weights()
#generate_phi_matrix()

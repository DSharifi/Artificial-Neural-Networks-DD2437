import numpy as np

datapoints = 63
nodes = 50
def distance(x, w):
    return np.sqrt(np.sum((x - w) ** 2))


def find_closest(x, W):
    """
    :param x: current datapoint
    :param W: W Matrix
    :return: Column index for the closest weight
    """
    closest = 0
    for i, w in enumerate(W):
        if distance(x, w[i]) < distance(x, w[closest]):
            closest = i
    return closest


def delta_w(x, w, eta=0.01):
    return eta * (x - w)


def train_x(x, W):
    w_index = find_closest(x, W)
    W[w_index] += delta_w(x, W[w_index])
    return W


def init_weights():
    return "lol"


def train(X):
    W = init_weights()
    for x in X:
        W = train_x(x, W)

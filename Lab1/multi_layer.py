import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def mackey_glass(t):
    beta = 0.2
    gamma = 0.1
    n = 10
    tao = 25
    x = np.zeros([t])
    x[0] = 1.5
    return
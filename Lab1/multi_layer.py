import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

import numpy as np
import tensorflow

lower = 301
higher = 1501
predict = 5
points = 1200
validation_points = 200
test_points = 200
training_points = 800
features = 5


def generate_io_data(x):
    t = np.arange(lower, higher, 1)
    inputs = np.array([x[t - 20], x[t - 15], x[t - 10], x[t - 5], x[t]])
    output = x[t + predict]
    return inputs, output


def split_data(inputs, output):
    training = np.zeros([features, training_points])
    validation = np.zeros([features, validation_points])
    test = np.zeros([features, test_points])

    training_T = np.zeros(training_points)
    validation_T = np.zeros(validation_points)
    test_T = np.zeros(test_points)

    # Input splits
    for i in range(features):
        training[i] = inputs[i, :training_points]
        validation[i] = inputs[i, training_points:validation_points + training_points]
        test[i] = inputs[i, validation_points + training_points:]

    # Output Split
    training_T = output[:training_points]
    validation_T = output[training_points:training_points + validation_points]
    test_T = output[training_points + validation_points:]

    return training, validation, test, training_T, validation_T, test_T


def mackey_glass(t_iter):
    beta = 0.2
    gamma = 0.1
    n = 10
    tao = 25
    x = np.zeros(t_iter + 1)
    x[0] = 1.5
    for t in range(t_iter):
        res = t - tao
        if res < 0:
            res = 0
        elif res == 0:
            res = x[0]
        else:
            res = x[res]
        x[t + 1] = x[t] + (beta * res) / (1 + res ** n) - gamma * x[t]
    return x


def network(X_training, Y_training, X_test, Y_test):
    # Compiling
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])

    # Fit
    model.fit(X_training, Y_training, epochs=100, batch_size=   )


x = mackey_glass(higher + predict)
inputs, output = generate_io_data(x)

training, validation, test, training_T, validation_T, test_T = split_data(inputs, output)

# print(inputs)
# print(output)
# print(inputs)
# tf2.random_normal_initializer()
# trains_ds = tf.load(inputs, split='train')
# print(trains_ds)

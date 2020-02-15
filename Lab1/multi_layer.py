import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization

from matplotlib import pyplot
import time

lower = 301
higher = 1501
predict = 5
points = 1200
test_points = 200
training_points = 1000
features = 5
hidden_nodes2 = 300
hidden_nodes1 = 50

output_nodes = 1
mom = 0.99
eta = 0.01
epochs = 1000
bs = 32
iterations = 3
sd = 0.09


def generate_io_data(x):
    t = np.arange(lower, higher, 1)
    inputs = np.array([x[t - 20], x[t - 15], x[t - 10], x[t - 5], x[t]])
    output = x[t + predict] + np.random.normal(0, sd, points)
    return inputs, output


def split_data(inputs, output):
    training = np.zeros([features, training_points])
    test = np.zeros([features, test_points])

    training_T = np.zeros(training_points)
    test_T = np.zeros(test_points)

    # Input splits
    for i in range(features):
        training[i] = inputs[i, :training_points]
        test[i] = inputs[i, training_points:]

    # Output Split
    training_T = output[:training_points]
    test_T = output[training_points:]

    return training, test, training_T, test_T


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


def network(X_training, Y_training, X_test, plot=True, regularization=True, hidden_nodes=hidden_nodes2):
    # Compiling
    model = Sequential()
    model.add(Dense(3, input_shape=(features,)))
    if regularization:
        model.add(Dropout(0.2))
    model.add(Dense(hidden_nodes1))
    if regularization:
        model.add(Dropout(0.2))
    model.add(Dense(hidden_nodes))
    model.add(Dense(output_nodes))
    sgd = SGD(learning_rate=eta, momentum=mom)
    model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', patience=10)

    # verbose = 2 for prints
    start_time = time.time()
    history = model.fit(X_training, Y_training, epochs=epochs, batch_size=bs, verbose=0,
                        validation_split=0.2, callbacks=[es])
    elapsed_time = time.time() - start_time

    yhat = model.predict(X_test)
    weights = []
    for m in model.layers[0].get_weights()[0]:
            weights.append(m)
    #for m in model.layers[1].get_weights()[0]:
    #         weights.append(m)

    if plot:
        pyplot.title('Learning Curves')
        pyplot.xlabel('Epochs')
        pyplot.ylabel('MSE')
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='val')
        pyplot.legend()
        pyplot.show()
    return yhat, history.history['val_loss'][len(history.history['val_loss']) - 1], weights, elapsed_time


def MSE(T, y):
    return np.square(np.subtract(T, y)).mean()


def plot_histo(weights):
    x = np.linspace(1, features, features)
    y = weights
    pyplot.hist(y, bins=5)
    pyplot.ylabel('Weight')
    pyplot.xlabel('Feature')
    pyplot.show()


def plot_time_series(training_T, test_T, y_predicted=None):
    # x1 = np.linspace(lower, higher - test_points, training_points)
    x1 = np.arange(lower, higher - test_points, 1)
    x2 = np.arange(higher - test_points, higher, 1)

    # x2 = np.linspace(higher - test_points, higher, test_points)
    print(x1)
    print(x2)
    pyplot.plot(x1, training_T, color='green')
    if y_predicted is not None:
        pyplot.plot(x2, y_predicted, color='red', label='Error')
    pyplot.plot(x2, test_T, color='green', label='Training')
    pyplot.xlabel('Time')
    pyplot.ylabel('Output')
    pyplot.legend(loc='lower-right')
    pyplot.title("Time Series for test-set")
    pyplot.show()


def averageMSE(training, training_T, test):
    i = 0
    errors = []
    while i < iterations:
        i += 1
        print(i)
        y_predicted, error, weight = network(training.T, training_T, test.T, False, True)
        # errors.append(MSE(test_T, y_predicted))
        errors.append(error)
    print(np.mean(np.array(errors)))


def takeTime(X_training, Y_training, X_test):
    i = 0
    iterations = 10
    times = []
    nodes = []
    while i < iterations:
        i += 1
        hidden_nodes = i * 100
        nodes.append(hidden_nodes)
        x,y,z, time = network(X_training.T, Y_training, X_test.T, False, True, hidden_nodes)
        print(time)
        times.append(time)
    pyplot.plot(nodes, times)
    pyplot.title("Time elapsed depending on hidden nodes")
    pyplot.xlabel("Amount of hidden nodes in 2nd layer")
    pyplot.ylabel("Time Elapsed")
    pyplot.show()


x = mackey_glass(higher + predict)
inputs, output = generate_io_data(x)
training, test, training_T, test_T = split_data(inputs, output)
yhat, history, w, time = network(training.T,training_T,test.T, False,True)
print(w)
plot_histo(w)

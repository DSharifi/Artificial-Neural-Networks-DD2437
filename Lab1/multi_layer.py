import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot

lower = 301
higher = 1501
predict = 5
points = 1200
test_points = 200
training_points = 1000
features = 5
hidden_nodes = 10
output_nodes = 1
mom = 0.8
eta = 0.001
epochs = 1000
bs = 32


def generate_io_data(x):
    t = np.arange(lower, higher, 1)
    inputs = np.array([x[t - 20], x[t - 15], x[t - 10], x[t - 5], x[t]])
    output = x[t + predict]
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


def network(X_training, Y_training, X_test, Y_test):
    # Compiling
    model = Sequential()
    model.add(Dense(2, input_shape=(features,)))
    model.add(Dense(hidden_nodes))
    model.add(Dense(output_nodes))
    sgd = SGD(learning_rate=eta, momentum=mom)
    # opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])
    # Fit
    print(X_training.shape)
    es = EarlyStopping(monitor='val_loss', patience=5)

    history = model.fit(X_training, Y_training, epochs=epochs, batch_size=bs, verbose=2,
                        validation_split=0.2, callbacks=[es])

    yhat = model.predict(X_test)
    pyplot.title('Learning Curves')
    pyplot.xlabel('Epochs')
    pyplot.ylabel('MSE')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='val')
    pyplot.legend()
    pyplot.show()

    print(yhat)


x = mackey_glass(higher + predict)
inputs, output = generate_io_data(x)

training, test, training_T, test_T = split_data(inputs, output)

network(training.T, training_T, test.T, test_T)
# print(inputs)
# print(output)
# print(inputs)
# tf2.random_normal_initializer()
# trains_ds = tf.load(inputs, split='train')
# print(trains_ds)

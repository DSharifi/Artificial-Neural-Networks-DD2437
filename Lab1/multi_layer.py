import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#import tensorflow as tf2

lower = 301
higher = 1501
predict = 5
points = 1200
validation_points = 200
test_points = 200
training_points = 800
features = 5

def generate_io_data(x):
    t = np.arange(lower, higher,1)
    inputs = np.array([x[t-20], x[t-15], x[t-10], x[t-5], x[t]])
    output = x[t+predict]
    return inputs, output

def split_data(inputs, output):
    training = np.zeros([features, training_points])
    validation = np.zeros([features, validation_points])
    test = np.zeros([features, test_points])
    
    training_T = np.zeros(training_points)
    validation_T = np.zeros(validation_points)
    test_T = np.zeros(test_points)

    #Input splits
    for i in range(features):
        training[i] = inputs[i, :training_points]
        validation[i] = inputs[i, training_points:validation_points+training_points]
        test[i] = inputs[i, validation_points+training_points:]
    
    #Output Split
    training_T = output[:training_points]
    validation_T = output[training_points:training_points+validation_points]
    test_T = output[training_points+validation_points:]

    return training, validation, test, training_T, validation_T, test_T

def mackey_glass(t_iter):
    beta = 0.2
    gamma = 0.1
    n = 10
    tao = 25
    x = np.zeros(t_iter+1)
    x[0] = 1.5
    for t in range(t_iter):
        res = t - tao
        if res < 0:
            res = 0
        elif res == 0:
            res = x[0]
        else:
            res = x[res]
        x[t+1] = x[t] + (beta * res) / (1 + res**n) - gamma * x[t]
    return x


def tensorflowa_oss(x_train, y_train, x_val, y_val):
    return


#W_temp = tf.Variable(tf.zeros([784,10], tf.float32))
#print(W_temp)

def youtube(x_data, y_data):
    
    x_data = np.array([[0,0], [0,1], [1,0], [1,1]])
    y_data = np.array([[0], [1], [1], [0]])

    n_input = 5
    n_hidden = 2
    n_output = 1
    learning_rate = 0.1
    epochs = 10000

    X = tf.placeholder('float')
    Y = tf.placeholder('float')
      
    W1 = tf.Variable(tf.random_uniform([n_input, n_hidden], -1.0, 1.0))
    W2 = tf.Variable(tf.random_uniform([n_hidden, n_output], -1.0, 1.0))

    b1 = tf.Variable(tf.zeros([n_hidden]), name = "Bias1")
    b2 = tf.Variable(tf.zeros([n_hidden]), name = "Bias2")


    L2 = tf.sigmoid(tf.matmul(X,W1) + b1)
    hy = tf.sigmoid(tf.matmul(L2, W2) + b2)

    cost = tf.reduce_mean(-Y*tf.log(hy) - (1-Y)*tf.log(1-hy))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init = tf.initialize_all_variables()

    with tf.Session() as session:
        session.run(init)

        for step in range(epochs):
            session.run(optimizer, feed_dict={X: x_data, Y: y_data})

            if step % 1000 == 0:
                print(session.run(cost, feed_dict={X: x_data, Y: y_data}))

        answer = tf.equal(tf.floor(hy + 0.5), Y)
        accuracy = tf.reduce_mean(tf.cast(answer, "float"))

        print(session.run([hy], feed_dict={X: x_data, Y: y_data}))
        print("Accuracy: ", accuracy.eval({X: x_data, Y: y_data}))


x = mackey_glass(higher+predict)
inputs, output = generate_io_data(x)
training, validation, test, training_T, validation_T, test_T = split_data(inputs, output)

#print(inputs)
#print(output)
youtube(training.T,training_T.T)
#print(inputs)
#tf2.random_normal_initializer()
#trains_ds = tf.load(inputs, split='train')
print(inputs)
#print(trains_ds)
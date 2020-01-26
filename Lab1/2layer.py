from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

# NETWORK SETTINGS
n_epochs = 20000
hidden_nodes = 10
features = 2  # input_nodes, for 3.2.1-3.2.2
eta = 0.001
output_nodes = 1

# DATA generation for 3.2.1
mA = [1.0, 0.5]
mB = [0.2, 0.3]
sigmaA = 0.3
sigmaB = 0.3
n = 200
n_classes = 2



def generate_weights(X):
    # Weight, W
    W = np.random.normal(1, 0.5, (hidden_nodes, X.shape[0]))
    print(hidden_nodes)
    # Weight, V
    V = np.random.normal(1, 0.5, (output_nodes, hidden_nodes + 1))
    return W, V


def generate_io_matrix(useBias=True):
    # np.random.seed(1000)

    X = np.zeros([features + 1, n * 2])

    # A
    X[0, :n] = np.concatenate((np.random.normal(2, 1, int(n / 2)) * sigmaA - mA[0],
                                         np.random.normal(2, 1, int(n / 2)) * sigmaA + mA[0]))
    
    np.random.shuffle(X[0, :n])

    X[1, :n] = np.random.normal(0, 1, n) * sigmaA + mA[1]
    X[2, :n] = 1 if useBias else 0

    # B
    X[0, n:] = np.random.normal(0, 1, n) * sigmaB + mB[0]
    X[1, n:] = np.random.normal(0, 1, n) * sigmaB + mB[1]
    X[2, n:] = 1 if useBias else 0

    # T
    T = np.zeros(n * 2)
    T[:n] = 1
    T[n:] = -1
    T = phi(T)

    #idx = np.random.permutation(n)
    

    return X, T


def generate_io_encoder_matrix():
    rand = np.random.randint(features, size=n)
    X = np.ones([n, features]) * -1
    for i, x in enumerate(X):
        x[rand[i]] = 1
    print(X)
    return X.T

def generate_bell_data():
    x = np.arange(-5, 5, 0.5)
    y = np.arange(-5, 5, 0.5)
    xx, yy = np.meshgrid(x, y, sparse=False)
    xx, yy = xx.flatten(), yy.flatten()
    X = np.vstack([xx, yy, np.ones(len(xx))])
    T = np.atleast_2d(gauss_bell_function(xx, yy))
    return X, T, x, y

def gauss_bell_function(x, y):
    return np.exp(-(x**2 + y**2)/10) - 0.5

def MSE(T, y):
    return np.square(np.subtract(T, y)).mean()


def compute_cost(T, y):
    return (1 / 2) * np.sum((T - y) ** 2)


def phi(x):
    return (2 / (1 + np.exp(-x))) - 1


def phi_prim(x):
    return ((1 + phi(x)) * (1 - phi(x))) / 2


def forward_pass(X, W, V):
    # X = np.concatenate((X, np.ones((1, X.shape[1]))))     ## Eventuellt senare
    # h_in = W @ np.concatenate((X, np.ones((1, X.shape[1])))) ###viktigt
    #print(X.shape)
    #print(W.shape)
    h_in = W @ X
    # h_out = np.array([phi(h_in)], [1]*X.shape[1])         ### Kommer ge error
    h_out = np.concatenate((phi(h_in), np.ones((1, X.shape[1]))))
    oin = V @ h_out
    o_out = phi(oin)
    return o_out, h_out


def backward_pass(out, hout, T, V):
    delta_o = (out - T) * phi_prim(out)
    delta_h = (np.transpose(V) @ delta_o) * phi_prim(hout)
    delta_h = delta_h[0:hidden_nodes, :]
    return delta_h, delta_o


def delta_W(delta_h, X, use_momentum=False, dw=0, alpha=0.9):
    if not use_momentum:
        return -eta * (delta_h @ X.T)
    else:
        return eta * ((dw * alpha) - (delta_h @ X.T) * (1 - alpha))


def delta_V(delta_o, H, use_momentum=False, dv=0, alpha=0.9):
    if not use_momentum:
        return -eta * (delta_o @ H.T)
    else:
        return eta * ((dv * alpha) - (delta_o @ H.T) * (1 - alpha))


def weight_update(delta_h, delta_o, X, W, H, V, use_momentum=False, dw=0, dv=0):
    if not use_momentum:
        dw = delta_W(delta_h, X)
        dv = delta_V(delta_o, H)
        W += dw
        V += dv
        return W, V, dw, dv
    else:
        dw = delta_W(delta_h, X, True, dw)
        dv = delta_V(delta_o, H, True, dv)
        W += dw
        V += dv
        return W, V, dw, dv


def calculate_accuracy(o_out, testT):
    counter = 0
    for i in range(len(testT)):
        if testT[i] > 0 and o_out[0, i] > 0:
            counter += 1
        elif testT[i] < 0 and o_out[0, i] < 0:
            counter += 1

    if counter == 0:
        return 1.0
    else:
        return 1.0 - ((len(testT) - counter) / len(testT))


def two_layer_perceptron(X, T, W, V, n_epoch, testX, testT, use_momentum=False, use_bell=False, x=None, y=None):
    mse = np.zeros(n_epoch)
    dw = 0
    dv = 0
    test_accuracy = np.zeros(n_epoch)
    train_accuracy = np.zeros(n_epoch)
    
    for i in range(n_epoch):
        o_out, h_out = forward_pass(X, W, V)
        delta_h, delta_o = backward_pass(o_out, h_out, T, V)
        if not use_momentum:
            W, V, dw, dv = weight_update(delta_h, delta_o, X, W, h_out, V)
        else:
            W, V, dw, dv = weight_update(delta_h, delta_o, X, W, h_out, V, True, dw, dv)
        #mse[i] = MSE(testT, o_out)
        #if use_bell:
            #if (i%100) == 0:
                #plot_bell_3d(o_out, x, y)
        #print(mse[i])
        o_out_test, h_out_test = forward_pass(testX, W, V)
        #print(testT.shape, o_out_test.shape)
        #print(trainT.shape, o_out_test.shape)

        test_accuracy[i] = MSE(testT, o_out_test)       #calculate_accuracy(o_out_test, testT)
        train_accuracy[i] = MSE(T, o_out)          #calculate_accuracy(o_out, T)


    #o_out_test, h_out_test = forward_pass(testX, W, V)
    #last_error = MSE(testT, o_out_test)
    #last_error = calculate_accuracy(o_out_test, testT)
    return W, V, dw, dv, o_out, h_out, train_accuracy, test_accuracy


def plot(a1, a2, b1, b2):
    plt.scatter(a1, a2, color="red")
    plt.scatter(b1, b2, color="blue")
    plt.show()

def plot_data_points(X):
    plt.ylim(top=100, bottom=-100)
    plt.xlim(right=100, left=-100)
    classAx = X[0, :n]
    classAy = X[1, :n]
    classBx = X[0, n:]
    classBy = X[1, n:]
    plt.scatter(classAx, classAy, color="red")
    plt.scatter(classBx, classBy, color="blue")
    plt.xlabel("X1-Feature")
    plt.ylabel("X2-Feature")
    plt.title("Classification of linearly non-separable data, eta=" + str(eta))
    plt.grid()
    #plt.show()

def splitData(InputData, TargetData, A_Ratio=0.0, B_Ratio=0.0):

    testA = InputData[:, :int(n * A_Ratio)]
    trainingA = InputData[:, int(n * A_Ratio):n]
    testB = InputData[:, n: n + int((B_Ratio * n))]
    trainingB = InputData[:, n + int((B_Ratio * n)): n * 2]

    testAT = TargetData[0:int(A_Ratio * n):1]
    trainingAT = TargetData[int(A_Ratio * n):n:1]
    testBT = TargetData[n:n + int(B_Ratio * n):1]
    trainingBT = TargetData[n + int(B_Ratio * n):2 * n:1]

    if B_Ratio == 0:
        testX = testA
        testT = testAT
    elif A_Ratio == 0:
        testX = testB
        testT = testBT
    else:
        testX = np.concatenate((testA, testB), axis=1)
        testT = np.concatenate((testAT, testBT))

    trainingX = np.concatenate((trainingA, trainingB), axis=1)
    trainingT = np.concatenate((trainingAT, trainingBT))

    #plot(testA[0], testA[1], testB[0], testB[1])
    return testX, testT, trainingX, trainingT


def plot_mse(mse_array):
    plt.ylim(top=0.5, bottom=0)
    plt.xlim(right=n_epochs, left=0)
    x = np.array(range(n_epochs))
    y = mse_array
    plt.xlabel("Number of Epochs")
    plt.ylabel("Error rate")
    plt.title("Mean Square Error for every epoch with eta = " + str(eta))  ##ändra till bättre, tack
    plt.plot(x, y, color='green', label="Mean Square Error")
    plt.legend(loc="lower left")
    plt.show()


def plotLine(W, bias=True):
    plt.ylim(top=50, bottom=-10)
    y = []
    x = np.linspace(-100, 100, 200)
    #Tror line ska vara det här: y = (W[2] - W[0]*x)/W[1]
    if bias:
        b = W[2]
        k = -(b / W[1]) / (b / W[0])
        m = -b / W[1]
        y = k * x + m
        # y = (W[0]*x+W[2])/W[1]
    else:
        k = (W[0]) / W[1]
        y = k * x
    plt.plot(x, y, color='green', label="Decision Boundary")

def plot_bell(output, x, y):
    xx, yy = np.meshgrid(x, y)
    zz = np.reshape(output, xx.shape)
    plt.axis([-5, 5, -5, 5])
    plt.contourf(xx, yy, zz)
    plt.show()

def plot_bell_3d(output, x, y, use_ratio=False, ratio=0.0):
    ax = plt.axes(projection="3d")
    print(output.shape, x.shape, y.shape)
    xx, yy = np.meshgrid(x, y)
    zz = np.reshape(output, xx.shape)
    ax.plot_surface(xx, yy, zz)
    #plt.show()
    plt.pause(0.0001)
    #plt.waitforbuttonpress()

def plot_accuracy(train_acc, test_acc):
    plt.ylim(top=1, bottom=0)
    plt.xlim(right=n_epochs, left=0)
    epochs = np.arange(0, n_epochs)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Mean Square Error")
    plt.title("Mean Square Error of the function approximation \nwith " + str(hidden_nodes) + " hidden nodes and eta =" + str(eta))  ##ändra till bättre, tack
    plt.plot(epochs, train_acc, color='green', label="MSE change for train")
    plt.plot(epochs, test_acc, color='red', label="MSE change for test")

    plt.legend(loc="upper right")
    plt.show()

def plot_for_nodes(train, test, nodes_used):
    plt.ylim(top=1, bottom=0)
    plt.xlim(right=24, left=0)
    nodes = np.arange(0, nodes_used)
    print(nodes.shape)
    print(train.shape)
    print(test.shape)
    plt.xlabel("Size of hidden layer")
    plt.ylabel("Mean Square Error")
    plt.title("Mean Square Error for the function approximation")# \nwith " + str(hidden_nodes) + " hidden nodes and eta =" + str(eta))  ##ändra till bättre, tack
    plt.plot(nodes, train, color='green', label="MSE change for train")
    plt.plot(nodes, test, color='red', label="MSE change for test")

    plt.legend(loc="upper right")
    plt.show()


def split_bell_data(X, T, test_ratio=0.2):
    points = int(X.shape[1])
    test_points = int(X.shape[1]*test_ratio)
    testX = np.zeros([X.shape[0], test_points])
    testT = np.zeros([T.shape[0], test_points])
    trainX = np.zeros([X.shape[0], int(points-test_points)])
    trainT = np.zeros([T.shape[0], int(points-test_points)])

    for i in range(test_points):
        testX[:, i] = X[:, i]
        testT[0, i] = T[0, i]
    
    for i in range(points-test_points):
        trainX[:, i] = X[:, i+test_points]
        trainT[0, i] = T[0, i+test_points]

    return trainX, trainT, testX, testT

""" TASK 3.2.1 """
#iterations = 25
#for i in range(1, iterations+1):
    #hidden_nodes = i
#X, T = generate_io_matrix()
#W, V = generate_weights(X)
#plot_data_points(X)
#plt.show()
#testX, testT, trainingX, trainingT = splitData(X, T, 0.2, 0.2)
#print(testX.shape)
#print(testX.shape)
#W, V, dw, dv, o_out, h_out, train_accuracy, test_accuracy = two_layer_perceptron(trainingX, trainingT, W, V, n_epochs, testT, testX)
#print(error)
#print("Iteration " + str(i) + " done.")

#print(test_accuracy)
#plot_accuracy(train_accuracy, test_accuracy)

""" TASK 3.2.2 """
#X = generate_io_encoder_matrix()
#T = phi(X)

""" TASK 3.2.3 """
#print(gauss_bell_function(1,1))
"""
iterations = 2
train_acc = np.zeros(iterations+1)
test_acc = np.zeros(iterations+1)
for i in range(1, iterations+1):
    hidden_nodes = i
    X, T, x, y = generate_bell_data()
    W, V = generate_weights(X)
    trainX, trainT, testX, testT = split_bell_data(X, T, 0.2)

    
    W, V, dw, dv, o_out, h_out, train_accuracy, test_accuracy = two_layer_perceptron(trainX, trainT, W, V, n_epochs, testX, testT, False, True, x, y)
    train_acc[i-1] = train_accuracy
    test_acc[i-1] = test_accuracy
"""
X, T, x, y = generate_bell_data()
W, V = generate_weights(X)
trainX, trainT, testX, testT = split_bell_data(X, T, 0.2)
W, V, dw, dv, o_out, h_out, train_accuracy, test_accuracy = two_layer_perceptron(trainX, trainT, W, V, n_epochs, testX, testT, False, True, x, y)
#print(o_out)
plot_accuracy(train_accuracy, test_accuracy)
#plot_for_nodes(train_acc, test_acc, iterations+1)

""" Plotting """
#plot_data_points(X)
#for w in W:
    #plotLine(w)
    #break
#plt.legend(loc="lower left")
#plt.grid()
#plt.show()
#plot_mse(mse_array)
#print("-------O--------")
#print(o_out)
#print("-------T--------")
#print(T)
#print("------MSE----------")
#print(mse_array)
#plot_bell(o_out, x, y)
#plot_bell_3d(o_out, x, y)
#plt.show()


from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

# NETWORK SETTINGS
n_epochs = 100
hidden_nodes = 5
features = 8  # input_nodes, for 3.2.1-3.2.2
eta = 0.1
output_nodes = 1

# DATA generation for 3.2.1
mA = [1.0, 0.5]
mB = [0.2, 0.3]
sigmaA = 0.1
sigmaB = 0.1
data_points = 100
n_classes = 2


# WEIGHT


def generate_weights(X):
    # Weight, W
    W = np.random.normal(0, 0.001, (hidden_nodes, X.shape[0]))

    # Weight, V
    V = np.random.normal(0, 0.001, (output_nodes, hidden_nodes + 1))
    return W, V


def generate_io_matrix(useBias=True):
    # np.random.seed(1000)

    X = np.zeros([features + 1, data_points * 2])

    # A
    X[0, :data_points] = np.concatenate((np.random.normal(2, 1, int(data_points / 2)) * sigmaA - mA[0],
                                         np.random.normal(2, 1, int(data_points / 2)) * sigmaA + mA[0]))
    X[1, :data_points] = np.random.normal(0, 1, data_points) * sigmaA + mA[1]
    X[2, :data_points] = 1 if useBias else 0

    # B
    X[0, data_points:] = np.random.normal(0, 1, data_points) * sigmaB + mB[0]
    X[1, data_points:] = np.random.normal(0, 1, data_points) * sigmaB + mB[1]
    X[2, data_points:] = 1 if useBias else 0

    # T
    T = np.zeros(data_points * 2)
    T[:data_points] = 1
    T[data_points:] = -1
    T = phi(T)
    return X, T


def generate_io_encoder_matrix():
    rand = np.random.randint(features, size=data_points)
    X = np.ones([data_points, features]) * -1
    for i, x in enumerate(X):
        x[rand[i]] = 1
    print(X)
    return X.T

def generate_bell_data():
    x = np.reshape(np.arange(-5, 5, 0.5), (20, 1))
    y = np.reshape(np.arange(-5, 5, 0.5), (20, 1))
    z = np.exp(-x * x * 0.1) * np.exp(-y * y * 0.1).T - 0.5
    ndata = x.shape[0]*y.shape[0]
    T = np.reshape(z, (1, ndata))
    xx, yy = np.meshgrid(x, y)
    X = np.vstack((np.reshape(xx, (1, ndata)), np.reshape(yy, (1, ndata))))
    return x, y, X, T


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
    h_in = W @ X
    # h_out = np.array([phi(h_in)], [1]*X.shape[1])         ### Kommer ge error
    h_out = np.concatenate((phi(h_in), np.ones((1, X.shape[1]))))
    oin = V @ h_out
    return phi(oin), h_out


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


def calculate_accuracy(o_out):
    counter = 0
    for i in range(0, data_points, 1):
        if o_out[0, i] <= 0:
            counter += 1
    for i in range(data_points, data_points * 2, 1):
        if o_out[0, i] > 0:
            counter += 1
    if counter == 0:
        return 1.0
    else:
        return (data_points * 2 - counter) / (data_points * 2)


def two_layer_perceptron(X, T, W, V, n_epoch, use_momentum=False):
    mse = np.zeros(n_epoch)
    dw = 0
    dv = 0
    for i in range(n_epoch):
        o_out, h_out = forward_pass(X, W, V)
        delta_h, delta_o = backward_pass(o_out, h_out, T, V)
        if not use_momentum:
            W, V, dw, dv = weight_update(delta_h, delta_o, X, W, h_out, V)
        else:
            W, V, dw, dv = weight_update(delta_h, delta_o, X, W, h_out, V, True, dw, dv)
        mse[i] = MSE(T, o_out)
        # print(mse[i])

    return W, V, dw, dv, o_out, h_out, mse


def plot_data_points(X):
    plt.ylim(top=100, bottom=-100)
    plt.xlim(right=100, left=-100)
    classAx = X[0, :data_points]
    classAy = X[1, :data_points]
    classBx = X[0, data_points:]
    classBy = X[1, data_points:]
    plt.scatter(classAx, classAy, color="red")
    plt.scatter(classBx, classBy, color="blue")
    plt.grid()
    #plt.show()


def plot_mse(mse_array):
    plt.ylim(top=0.5, bottom=0)
    plt.xlim(right=n_epochs, left=0)
    x = np.array(range(n_epochs))
    y = mse_array
    plt.xlabel("Number of Epochs")
    plt.ylabel("Error rate")
    plt.title("Mean Square Error for every epoch with eta = " + str(eta))  ##ändra till bättre, tack
    plt.plot(x, y, color='green', label="Mean Square Error")
    plt.legend(loc="upper right")
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

def plot_bell_3d(output, x, y):
    ax = plt.axes(projection="3d")
    xx, yy = np.meshgrid(x, y)
    zz = np.reshape(output, xx.shape)
    ax.plot_surface(xx, yy, zz)
    plt.show()


""" TASK 3.2.1 """
#X, T = generate_io_matrix()
#W, V = generate_weights(X)

""" TASK 3.2.2 """
#X = generate_io_encoder_matrix()
#T = phi(X)

""" TASK 3.2.3 """
#print(gauss_bell_function(1,1))
x, y, X, T = generate_bell_data()
W, V = generate_weights(X)

""" Training """
W, V, dw, dv, o_out, h_out, mse_array = two_layer_perceptron(X, T, W, V, n_epochs)
#print(o_out)

""" Plotting """
#plot_data_points(X)
#for w in W:
#    plotLine(w)
#plt.grid()
#plt.show()
#plt.grid()
#plt.show()
#plot_mse(mse_array)
print("-------O--------")
print(o_out)
print("-------T--------")
print(T)
print("------MSE----------")
print(mse_array)
plot_bell(o_out, x, y)
plot_bell_3d(o_out, x, y)

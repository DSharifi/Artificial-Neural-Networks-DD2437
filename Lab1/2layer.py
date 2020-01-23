import numpy as np
import matplotlib.pyplot as plt

#NETWORK SETTINGS
n_epochs = 10000
hidden_nodes = 1
features = 2 #input_nodes
eta= 0.01
output_nodes = 1

#DATA STRUCTURE
mA = [1.0, 0.3]
mB = [0.0, -0.1]
sigmaA = 0.2
sigmaB = 0.3
data_points = 100
n_classes = 2

#WEIGHT


def generate_weights():
    #Weight, W
    W = np.random.normal(0, 0.001, (hidden_nodes, X.shape[0]))

    #Weight, V
    V = np.random.normal(0, 0.001, (output_nodes, hidden_nodes+1))
    return W, V

def generate_io_matrix(useBias = True):
    #np.random.seed(1000)
 
    X = np.zeros([features+1, data_points*2])
    
    #A
    X[0, :data_points] = np.random.normal(0, 1, data_points) * sigmaA + mA[0]
    X[1, :data_points] = np.random.normal(0, 1, data_points) * sigmaA + mA[1]
    X[2, :data_points] = 1 if useBias else 0
    
    #B
    X[0, data_points:] = np.random.normal(0, 1, data_points) * sigmaB + mB[0]
    X[1, data_points:] = np.random.normal(0, 1, data_points) * sigmaB + mB[1]
    X[2, data_points:] = 1 if useBias else 0

    #T
    T = np.zeros(data_points*2)
    T[:data_points] = 1
    T[data_points:] = -1
    T = phi(T)
    
    return X, T


def MSE(T, y):
    return np.square(np.subtract(T,y)).mean() 

def compute_cost(T, y):
    return (1/2) * np.sum((T-y)**2)

def phi(x):
     return (2 / (1 + np.exp(-x))) - 1

def phi_prim(x):
    return ((1+phi(x)) * (1-phi(x))) / 2
   
def forward_pass(X, W, V):
    #X = np.concatenate((X, np.ones((1, X.shape[1]))))     ## Eventuellt senare
    #h_in = W @ np.concatenate((X, np.ones((1, X.shape[1])))) ###viktigt
    h_in = W @ X
    #h_out = np.array([phi(h_in)], [1]*X.shape[1])         ### Kommer ge error
    h_out= np.concatenate((phi(h_in), np.ones((1, X.shape[1]))))
    oin = V @ h_out
    return phi(oin), h_out

def backward_pass(out, hout, T, V):
    delta_o = (out - T) * phi_prim(out)
    delta_h = (np.transpose(V) @ delta_o) * phi_prim(hout)
    delta_h = delta_h[0:hidden_nodes,:]
    return delta_h, delta_o

def delta_W(delta_h, X, use_momentum = False, dw=0, alpha=0.9):
    if not use_momentum:
        return -eta*(delta_h@X.T)
    else:
        return eta*((dw*alpha)-(delta_h @ X.T) * (1-alpha))


def delta_V(delta_o, H, use_momentum=False, dv=0,alpha=0.9):
    if not use_momentum:
        return -eta*(delta_o@H.T)
    else:
        return eta*((dv*alpha)-(delta_o @ H.T) * (1-alpha))

def weight_update(delta_h, delta_o, X, W, H, V, use_momentum = False, dw = 0, dv = 0):
    if not use_momentum:
        dw = delta_W(delta_h, X)
        #print("-----dw----")
        dv = delta_V(delta_o, H)
        #print(np.shape(dw))
        #print("-----W----")
        #print(np.shape(W))
        W += dw
        V += dv
        return W, V, dw, dv
    else:
        dw = delta_W(delta_h, X, True, dw)
        dv = delta_V(delta_o, H, True, dv)
        W += dw
        V += dv
        return W, V, dw, dv

def two_layer_perceptron(X, T, W, V, n_epoch):
    mse = np.zeros(n_epoch)
    for i in range(n_epoch):
        o_out, h_out = forward_pass(X, W, V)
        delta_h, delta_o = backward_pass(o_out, h_out, T, V)
        W, V, dw, dv =  weight_update(delta_h,delta_o,X,W,h_out, V)
        mse[i] = MSE(T, o_out)
        print(mse[i])

    
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
    plt.show()

def plot_mse(mse_array):
    plt.ylim(top=0.5, bottom=0)
    plt.xlim(right=n_epochs, left=0)
    x = np.array(range(n_epochs))
    y = mse_array
    plt.xlabel("Number of Epochs")
    plt.ylabel("Error rate")
    plt.title("Mean Square Error for every epoch with eta = " + str(eta)) ##ändra till bättre, tack
    plt.plot(x, y, color='green', label="Mean Square Error")
    plt.legend(loc="upper right")
    plt.show()


""" Code here """
X, T = generate_io_matrix()
W, V = generate_weights()
W, V, dw, dv, o_out, h_out, mse_array = two_layer_perceptron(X,T,W,V,n_epochs)

#print(mse_array)
#print("-------------- O ---------------- ")
#print(o_out)
#print("H")
#print(h_out)
#print("TARGET")
#print(T)
#plot_data_points(X)
plot_mse(mse_array)


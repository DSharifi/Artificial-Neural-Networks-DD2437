import numpy as np
import matplotlib.pyplot as plt

n = 100


#Generate Input/Output and Weight Matricies
def generateIOWMatrix(useBias = True):
    mA = [1.0, 0.5]
    mB = [-1.0, 0.0]
    sigmaA = 0.5
    sigmaB = 0.5

    X = np.zeros([3, n*2])
    
    #A
    X[0, :n] = np.random.normal(0, sigmaA, n) * sigmaA + mA[0]
    X[1, :n] = np.random.normal(0, sigmaA, n) * sigmaA + mA[1]
    X[2, :n] = 1 if useBias else 0
    
    #B
    X[0, n:] = np.random.normal(0, sigmaB, n) * sigmaB + mB[0]
    X[1, n:] = np.random.normal(0, sigmaB, n) * sigmaB + mB[1]
    X[2, n:] = 1 if useBias else 0

    #T
    T = np.zeros(n*2)
    T[:n] = 1
    T[n:] = -1

    #W 
    W = np.random.normal(0, 0.01, X.shape[0])
    
    return X, T, W

def delta_rule(X, T, W, eta=0.00001):
    return -eta * ((W@X - T) @ np.transpose(X))

def delta_learning(X, T, W, n_epoch):
    for i in range(n_epoch):
        delta_W = delta_rule(X, T, W)
        W = W - delta_W 
        print(W)
    return W

def perceptron_rule():
    return 
def perceptron_learning(X,T,W,n_epoch):
    y_prim = X @ W
    if y_prim[2] > W[2]:
        return 1
    else:
        return 0
    return None

def plot(X, T):
    classAx = X[0, :n]
    classAy = X[1, :n]
    classBx = X[0, n:]
    classBy = X[1, n:]
    
    plt.ylim(top=100, bottom=-100)
    plt.scatter(classAx, classAy, color="red")
    plt.scatter(classBx, classBy, color="blue")

def drawLine(W):
    x = np.linspace(-100, 100, 200)
    k = W[0]
    m = W[2]
    y = k*x + m
    plt.plot(x, y, color='green', label="Decision Boundary")
    plt.legend(loc="upper right")
    plt.xlabel("X-feature")
    plt.ylabel("Y-feature")
    plt.title("Delta Rule for two linearly seperable datasets")
    plt.grid()

#Function calls
n_epochs = 1000
X, T, W = generateIOWMatrix()
plot(X, T)
W = delta_learning(X, T, W, n_epochs)
drawLine(W)
plt.show()

#perceptron_learning(X,T,W,n_epochs)
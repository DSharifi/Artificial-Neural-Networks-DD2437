import numpy as np
import matplotlib.pyplot as plt

n = 100

#Generate Input/Output and Weight Matricies
def generateIOWMatrix(useBias = True):
    #np.random.seed(1000)
    mA = [1.0, 0.3]
    mB = [0.0, -0.1]
    sigmaA = 0.2
    sigmaB = 0.3

    X = np.zeros([3, n*2])
    
    #A
    X[0, :n] = np.random.normal(0, sigmaA, n) * sigmaA + mA[0]
    X[1, :n] = np.random.normal(0, sigmaA, n) * sigmaA + mA[1]
    X[2, :n] = 1 if useBias else 0
    
    #B
    X[0, n:] = np.random.normal(0, sigmaB, n) * sigmaB + mB[0]
    X[1, n:] = np.random.normal(0, sigmaB, n) * sigmaB + mB[1]
    X[2, n:] = 1 if useBias else 0

    """
    X[0, :n] = np.array([-3, -2, -3, -2])
    X[1, :n] = np.array([3, 3, 2, 2])
    X[2, :n] = 1 if useBias else 0

    X[0, n:] = np.array([-1, 0, -1, 0])
    X[1, n:] = np.array([0,0,1,1])
    X[2, n:] = 1 if useBias else 0
    """
    #T_Per
    T = np.zeros(n*2)
    T[:n] = 1
    T[n:] = 0
    #T_Del
    #T_del = np.zeros(n*2)
    #T_del[n:] = np.random.normal(0, sigmaB, n) 
    #T_del[n:] = np.random.normal(0, sigmaB, n)
    #W 
    W = np.random.normal(0, 0.01, X.shape[0])
    
    return X, T, W

def delta_rule(X, T, W, eta=0.0000001):
    return eta * ((T - W@X) @ np.transpose(X))

def delta_learning(X, T, W, n_epoch):
    for i in range(n_epoch):
        delta_W = delta_rule(X, T, W)
        W = W - delta_W
        print(W)
    return W

def perceptron_rule(X, e, eta):
    return eta * (e @ np.transpose(X))

def perceptron_learning(X,T,W,n_epoch, use_batch = True, eta=0.001):
    output_T = np.array([-1]*n*2)
    e = np.zeros(n*2)
    if not use_batch:
        delta_W = np.array([-1, -1, -1])
        for j in range(n_epoch):
            for i in range(n*2):
                sum = 0
                for k in range(len(W)):#####
                    sum += W[k] * X[k, i]
                #sum -= W[len(W)-1] * X[len(W)-1, i]
                output_value = 0
                if sum > 0:
                    output_value = 1
                else:
                   output_value = 0  #-1
                
                e[i]  = T[i] - output_value

            for k in range(len(W)):
                for i in range(n*2):
                    delta_W[k] += e[i] * X[k, i]
            delta_W = delta_W*eta
            W = W + delta_W
            print(delta_W)
            drawLine(W)

        return W
    else:
        done = False
        for j in range(n_epoch):
            y_vector = W @ X
            for i in range(n*2):
                if y_vector[i] > 0:
                    output_T[i] = 1
                else:
                    output_T[i] = 0   # -1
            
            e = T-output_T
            #print(e)
            delta_W = perceptron_rule(X, e, eta)
            #if delta_W[0] == 0 and delta_W[1] == 0 and delta_W[2] == 0 and not done:
                #done = True
                #W[2] = W[2]*-1
            #print(delta_W)
            print("-----delta_W-----")
            print(delta_W)
            print("---W----")
            print(W)
            W = W + delta_W       # W = [1,2,3]     delta-W = []
            #if not done:
                #drawLine(W)
            #print(W)
    return W


def plot(X, T):
    classAx = X[0, :n]
    classAy = X[1, :n]
    classBx = X[0, n:]
    classBy = X[1, n:]
    
    
    plt.scatter(classAx, classAy, color="red")
    plt.scatter(classBx, classBy, color="blue")
    plt.xlabel("X-feature")
    plt.ylabel("Y-feature")
    plt.title("Delta Rule for two linearly seperable datasets")
    plt.grid()


def drawLine(W):
    plt.ylim(top=10, bottom=-10)
    plt.xlim(right=10, left=-10)

    x = np.linspace(-100, 100, 200)
    #x = np.array([2, -2])
    #b = W[2]
    #k = -(b / W[1]) / (b / W[0])
    #m = -b / W[1]
    #y = k * x + m
    #W[0] och W[1]
    y = (W[2] - W[0]*x)/W[1]
    #y = (W[0]/W[1])*x+W[2]
    #target = np.array([W[0], W[1]])

    #k = W[0]
    #m = W[2]
    #y = k*x + m
    ln, = plt.plot(x, y, color='green', label="Decision Boundary")
    plt.pause(0.01)
    plt.plot()
    plt.legend(loc="upper right")
    #ln.remove()

#Function calls
n_epochs = 2000
X, T, W = generateIOWMatrix()
#print(T)
plot(X, T)
#plt.show()
#W = delta_learning(X, T, W, n_epochs)
W = perceptron_learning(X, T, W, n_epochs)
drawLine(W)
plt.show()

#perceptron_learning(X,T,W,n_epochs)
import numpy as np
import matplotlib.pyplot as plt

n = 100
mA = [1.0, 0.5]
mB = [-1.0, 0.0]
sigmaA = 0.5
sigmaB = 0.5

def generateInputMatrix(useBias = True):
    np.random.seed(100)
    classA1 = np.array(np.random.randint(1, n, n) * np.array([sigmaA + mA[0]] * n))
    classA2 = np.array(np.random.randint(1, n, n) * np.array([sigmaA + mA[1]] * n))
    classB1 = np.array(np.random.randint(1, n, n) * np.array([sigmaB + mB[0]] * n))
    classB2 = np.array(np.random.randint(1, n, n) * np.array([sigmaB + mB[1]] * n))

    #classA1 = np.arange(50, (n + 50), 1)
    #classA2 = np.arange(50, (n + 50), 1)
    #classB1 = np.arange(-(n + 1), -1, 1)
    #classB2 = np.arange(-(n + 1), -1, 1)

    classX = np.array([np.concatenate((classA1, classB1)),np.array([1] * (n * 2))])
    classW = np.random.normal(0, 0.01, classX.shape[0])

    classAT = np.array([1] * n)#Random ClassA1,ClassA2 -> Eps 1 eller -1
    classBT = np.array([-1] * n)#Random
    classT = np.concatenate((classA2, classB2))

    plot(classA1, classA2, classB1, classB2)

    return classX, classT, classW

def plot(a1, a2, b1, b2):
    plt.ylim(top=100, bottom=-100)
    plt.xlim(right=100, left=-100)

    plt.scatter(a1, a2, color="red")
    plt.scatter(b1, b2, color="blue")

def delta(W, X, T, eta=0.0001):
    """
    @param  W - Weight matrix
    @param  X - Input matrix
    @param  T - Target matrix
    @param  eta - Learning rate
    @return - Delta Matrix
    """
    xt = X.T
    return eta * (T - W @ X) @ np.transpose(X)  # <-- F2

def converge_check(old, change, percentage=0.2):
    for i, d in np.ndenumerate(old):
        if (np.abs(change[i] / old[i])) > percentage:
            return False
    return True

def sum_square(X,W,T):
    return np.sum((T - W @ X))

def delta_learning(X, T, W):
    i = 0
    converged = False
    while not converged and i < 100:
        i += 1
        old_W = W
        changeW = delta(old_W, X, T)
        W = old_W - changeW
        print(sum_square(X,W,T))
    if not converged:
        print("NO CONVERGENCE!")
    print(i)
    return W

def plotLine(W):
    plt.ylim(top=100, bottom=-100)
    plt.xlim(right=100, left=-100)

    x = np.linspace(-100, 100, 200)
    #b = W[2]
    y = (W[0]*x + W[1])

    #k = -(b / W[1]) / (b / W[0])
    #m = -b / W[1]
    #y = k * x + m
    plt.plot(x, y, color='green', label="Decision Boundary")
    plt.legend(loc="upper right")
    plt.xlabel("X-feature")
    plt.ylabel("Y-feature")
    plt.title("Delta Rule for two linearly seperable datasets")
    plt.grid()
    plt.show()

classX, classT, classW = generateInputMatrix()

W = delta_learning(classX, classT, classW)

plotLine(W)
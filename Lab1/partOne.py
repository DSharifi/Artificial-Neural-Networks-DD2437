import numpy as np
import matplotlib.pyplot as plt

n = 100
mA = [1.0, 0, 5]
mB = [-1.0, 0.0]
sigmaA = 0.5
sigmaB = 0.5


def generateInputMatrix():
    classA1 = np.array(np.random.randint(0, n, n) * np.array([sigmaA + mA[0]] * n))
    classA2 = np.array(np.random.randint(0, n, n) * np.array([sigmaA + mA[1]] * n))

    classB1 = np.array(np.random.randint(0, n, n) * np.array([sigmaB + mB[0]] * n))
    classB2 = np.array(np.random.randint(0, n, n) * np.array([sigmaB + mB[1]] * n))

    classX = np.array([np.concatenate((classA1, classB1)), np.concatenate((classA2, classB2)), np.array([1] * (n * 2))])
    classW = np.random.normal(0, 0.01, classX.shape[0])
    print(classW)
    classAT = np.array([1] * n)
    classBT = np.array([-1] * n)
    classT = np.concatenate((classAT, classBT))

    plot(classA1, classA2, classB1, classB2)

    return classX, classT, classW


def plot(a1, a2,b1,b2):
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


def delta_learning(X, T, W):
    i = 0
    converged = False
    while not converged and i < 100:
        i += 1
        old_W = W
        changeW = delta(old_W, X, T)
        W = old_W - changeW
        # if np.isnan[W[0]] or np.isnan(W[1]):
        # return old_W
        print(W)
        print("---------------")
        if converge_check(old_W, changeW):
            converged = True
    if not converged:
        print("NO CONVERGENCE!")
    print(i)
    return W


classX, classT, classW = generateInputMatrix()

W = delta_learning(classX, classT, classW)

def plotLine(W):
    plt.ylim(top=55, bottom=-5)
    x = np.linspace(-100, 100, 200)
    y = (W[0] * x + W[1])/W[2]
    plt.plot(x, y, color='green', label="Seperable line from weights")
    plt.legend(loc="upper right")
    plt.xlabel("X-feature")
    plt.ylabel("Y-feature")
    plt.title("Delta Rule for two linearly seperable datasets")
    plt.grid()
    plt.show()


plotLine(W)

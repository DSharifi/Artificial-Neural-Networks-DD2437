import numpy as np
import matplotlib.pyplot as plt

n = 100
mA = [1.0, 0.5]
mB = [-1.0, 0.0]
sigmaA = 0.5
sigmaB = 0.5
eta = 0.0001

def generateIOWMatrix(useBias=True):
    return


def generateInputMatrix(useBias=True):
    np.random.seed(100)
    classA1 = np.array(np.random.normal(2, sigmaA, n) * sigmaA + mA[0])
    classA2 = np.array(np.random.normal(3, sigmaA, n) * sigmaA + mA[1])
    classB1 = np.array(np.random.normal(7, sigmaB, n) * sigmaB + mB[0])
    classB2 = np.array(np.random.normal(8, sigmaB, n) * sigmaB + mB[1])

    # classA1 = np.arange(1, (n + 1), 1)
    # classA2 = np.arange(1, (n + 1), 1)
    # classB1 = np.arange(-(n + 1), -1, 1)
    # classB2 = np.arange(-(n + 1), -1, 1)

    # classA1 = np.array([0])
    # classA2 = np.array([1])
    # classB1= np.array([0])
    # classB2 = np.array([2])

    classX = np.array([np.concatenate((classA1, classB1)), np.concatenate((classA2, classB2)), np.array([1] * (n * 2))])
    classW = np.random.normal(0, 0.01, classX.shape[0])
    classAT = np.array([1] * n)  # Random ClassA1,ClassA2 -> Eps 1 eller -1
    classBT = np.array([0] * n)  # Random
    classT = np.concatenate(([1] * n, [-1] * n))

    plot(classA1, classA2, classB1, classB2)
    return classX, classT, classW


def plot(a1, a2, b1, b2):
    plt.scatter(a1, a2, color="red")
    plt.scatter(b1, b2, color="blue")


def delta(W, X, T):
    """
    @param  W - Weight matrix
    @param  X - Input matrix
    @param  T - Target matrix
    @param  eta - Learning rate
    @return - Delta Matrix
    """
    return eta * (T - W @ X) @ np.transpose(X)


def converge_check(old, new, percentage=0.00001):
    if (np.abs(old - new) / old) > percentage:
        return False
    return True


def sum_square(X, W, T):
    return np.sum((T - W @ X) ** 2)/2


def sequential_delta_learning(X, T, W):
    i = 0
    error_values = []
    iterations = []
    while True:
        i += 1
        old_W = W
        old_error = sum_square(X, old_W, T)
        print(old_error)
        for j, x in enumerate(X.T):
            W += eta * ((T[j] - W @ x) * x)
        error = sum_square(X, W, T)
        error_values.append(error)
        iterations.append(i*n*2)
        if converge_check(old_error, error):
            print(W)
            return W, error_values, iterations


def delta_learning(X, T, W):
    i = 0
    converged = False
    old_error = sum_square(X, W, T)
    error_values = []
    iterations = []
    while not converged:
        i += 1
        old_W = W
        changeW = delta(old_W, X, T)
        W = old_W + changeW
        old_error = sum_square(X, old_W, T)
        print(old_error)
        error = sum_square(X, W, T)
        error_values.append(error)
        iterations.append(i)
        if converge_check(old_error, error):
            print(W)
            return W, error_values, iterations


def plotIters(errors, iters):
    plt.ylim(top=100, bottom=-10)
    plt.plot(iters, errors, color='green', label="Error Rate")
    plt.legend(loc="upper right")
    plt.xlabel("Iterations")
    plt.ylabel("Sum of Squares Error")
    plt.title("Error rate converges after several iterations")
    plt.grid()
    plt.show()


def plotLine(W):
    plt.ylim(top=50, bottom=-10)
    x = np.linspace(-10, 10, 200)
    b = W[2]
    k = -(b / W[1]) / (b / W[0])
    m = -b / W[1]
    y = k * x + m
    plt.plot(x, y, color='green', label="Decision Boundary")
    plt.legend(loc="upper right")
    plt.xlabel("X-feature")
    plt.ylabel("Y-feature")
    plt.title("Delta Rule for two linearly seperable datasets, eta = " + str(eta))
    plt.grid()
    plt.show()


classX, classT, classW = generateInputMatrix()

# W, errors, iters = sequential_delta_learning(classX, classT, classW)
W, errors, iters = delta_learning(classX, classT, classW)

plotLine(W)
plotIters(errors, iters)

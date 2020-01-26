import numpy as np
import matplotlib.pyplot as plt

n = 100
mA = [1.0, 0.3]
mB = [0.0, -0.1]
sigmaA = 0.2
sigmaB = 0.3
eta = 0.001
ratioA = 0.0
ratioB = 0.0


def generateInputMatrix(useBias=True):
    np.random.seed(100)
    classA1 = np.array(np.concatenate((np.random.normal(0, sigmaA, int(n / 2)) * sigmaA - mA[0],
                                           np.random.normal(0, sigmaA, int(n / 2)) * sigmaA + mA[0])))

    np.random.shuffle(classA1)
    classA2 = np.array(np.random.normal(0, sigmaA, n) * sigmaA + mA[1])
    classB1 = np.array(np.random.normal(0, sigmaB, n) * sigmaB + mB[0])
    classB2 = np.array(np.random.normal(0, sigmaB, n) * sigmaB + mB[1])

    if useBias:
        classX = np.array(
            [np.concatenate((classA1, classB1)), np.concatenate((classA2, classB2)), np.array([1] * (n * 2))])
    else:
        classX = np.array([np.concatenate((classA1, classB1)), np.concatenate((classA2, classB2))])

    classW = np.random.normal(0, 5, classX.shape[0])
    classT = np.concatenate(([1] * n, [-1] * n))
    print(classX.shape)
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


def converge_check(old, new, percentage=0.0000001):
    if (np.abs(old - new) / old) > percentage:
        return False
    return True


def sum_square(X, W, T):
    return np.sum((T - W @ X) ** 2) / 2


def sequential_delta_learning(X, T, W):
    i = 0
    error_values = []
    iterations = []
    while True:
        i += 1
        old_W = W
        old_error = sum_square(X, old_W, T)
        for j, x in enumerate(X.T):
            W += eta * ((T[j] - W @ x) * x)
        error = sum_square(X, W, T)
        error_values.append(error)
        iterations.append(i * n * 2)
        if converge_check(old_error, error):
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
        error = sum_square(X, W, T)
        error_values.append(error)
        print(error)
        iterations.append(i)
        if converge_check(old_error, error):
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


def calculate_y(W, x):
    b = W[2]
    k = -(b / W[1]) / (b / W[0])
    m = -b / W[1]
    y = k * x + m
    return y


def plotLine(W, bias=True):
    plt.ylim(top=50, bottom=-10)
    y = []
    x = np.linspace(-100, 100, 200)
    if bias:
        b = W[2]
        k = -(b / W[1]) / (b / W[0])
        m = -b / W[1]
        y = k * x + m
    else:
        k = (W[0]) / W[1]
        y = k * x

    plt.plot(x, y, color='green', label="Decision Boundary")
    plt.legend(loc="upper right")
    plt.xlabel("X-feature")
    plt.ylabel("Y-feature")
    plt.title("Delta Rule for two linearly seperable datasets, eta = " + str(eta))
    plt.grid()
    plt.show()


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

    plot(testA[0], testA[1], testB[0], testB[1])

    return trainingX, trainingT, testX, testT
    # print(testA.shape)


classX, classT, classW = generateInputMatrix()

trainingX, trainingT, testX, testT = splitData(classX, classT, ratioA, ratioB)
W, errors, iters = delta_learning(trainingX, trainingT, classW)


def calculate_accuracy(W, testX, testT):
    predictT = np.zeros(testT.size)
    for i, x in enumerate(testX.T):
        if calculate_y(W, x[0]) >= x[1]:
            predictT[i] = -1
        else:
            predictT[i] = 1

    counter = 0
    sizeA = 0
    counterA = 0
    counterB = 0
    sizeB = 0
    size = 0
    for i, t in enumerate(testT):
        size += 1
        if t == 1:
            sizeA += 1
        else:
            sizeB += 1
        if predictT[i] == t:
            if t == 1:
                counterA += 1
            else:
                counterB += 1
            counter += 1
    if sizeA == 0:
        accuracyA = None
    else:
        accuracyA = counterA / sizeA
    if sizeB == 0:
        accuracyB = None
    else:
        accuracyB = counterB / sizeB
    #accuracy = counter / size
    print(accuracyA)
    print(accuracyB)
    #print(accuracy)


calculate_accuracy(W, testX, testT)
# W, errors, iters = sequential_delta_learning(classX, classT, classW)
plt.show()
#plotLine(W)
# plotIters(errors, iters)

import numpy as np
import matplotlib.pyplot as plt

n = 100
mA = [1.0, 0.5]
mB = [-1.0, 0.0]
sigmaA = 0.5
sigmaB = 0.5

def generateInputMatrix():
    classA1 = np.array(np.random.randint(0, n, n) * np.array([sigmaA + mA[0]]*n))
    classAT = np.array(np.random.randint(0, n, n) * np.array([sigmaA + mA[1]]*n))
    classA = np.array([classA1, classAT])
    
    classB1 = np.array(np.random.randint(0, n, n) * np.array([sigmaB + mB[0]]*n))
    classBT = np.array(np.random.randint(0, n, n) * np.array([sigmaB + mB[1]]*n))
    classB = np.array([classB1, classBT])
    
    classT = np.concatenate((classAT,classBT))
    classX = np.array([np.concatenate((classA1, classB1)),   np.array([1]*(n*2))])
    classW = np.array([np.random.normal(0, 1), 1])

    plot(classA, classB)
    
    return classX, classT, classW

def plot(dataset1, dataset2):
    plt.scatter(dataset1[0], dataset1[1], color="red")
    plt.scatter(dataset2[0], dataset2[1], color="blue")

def delta(W, X, T, eta=0.000001):
    """
    @param  W - Weight matrix
    @param  X - Input matrix
    @param  T - Target matrix
    @param  eta - Learning rate
    @return - Delta Matrix
    """

    return -eta * (W@X - T)@np.transpose(X)

classX, classT, classW = generateInputMatrix()

def perceptron_learning(X, T, W):
    i = 0
    converged = False
    while not converged and i <= 100:
        i+=1
        print(W)
        print("---------------")
        old_W = W
        W = delta(old_W, X, T,0.000001)
        if np.array_equal(old_W, W):
            converged = True
    return W


W = perceptron_learning(classX, classT, classW)
#print(W)
#W = np.array([1,1,1])
#T = np.array([1]*100)

#print(delta(W, classA, T))

#print(float(np.min(classA[0])))
#print(float(np.min(classA[1])))

#print(np.max(classB[0]))
#print(np.max(classB[1]))

x = classX[0]
y = W[0]*x + W[1]
plt.plot(x, y, '-r', label='y=2x+1')
plt.grid()
plt.show()

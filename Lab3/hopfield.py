import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Hopfield_Network(object):
    def __init__(self, units=8, patterns=3, epochs=10, use_bias=False):
        self.units = units
        self.patterns = patterns
        self.epochs = epochs
        self.bias = use_bias
    
    def recall(self, test_patterns):
        for _ in range(self.epochs):
            test_patterns = np.dot(test_patterns, self.W)
            test_patterns[test_patterns >= 0] = 1
            test_patterns[test_patterns < 0] = -1
        return test_patterns

    def store_patterns(self, inputs):
        self.X = inputs
        self.W = np.zeros([self.units, self.units])
        for pattern in inputs:
            self.W += np.outer(np.transpose(pattern),pattern)
        np.fill_diagonal(self.W, 0)
        self.W /= self.units
            

def data_file_to_matrix(filename, slice_at):
    file = open(filename, "r")
    images_string = file.read()
    images = list(map(int, images_string.split(",")))
    images = np.asarray(images)
    if slice_at > 0: 
        images = np.reshape(images, (-1, slice_at))
    return images 
    

def generate_inputs_task31():
    # 8 neurons, 3 patterns for task 3.1

    #Inputs used for storing (True inputs)
    x1 = np.array([-1, -1, 1, -1, 1, -1, -1, 1]).reshape(1, -1)
    x2 = np.array([-1, -1, -1, -1, -1, 1, -1, -1]).reshape(1, -1)
    x3 = np.array([-1, 1, 1, -1, -1, 1, -1, 1]).reshape(1, -1)
    inputs = np.concatenate((x1, x2, x3))
    
    #Inputs used to test the network (Distorted inputs)
    x1d = np.array([1, -1, -1, 1, -1, 1, -1, 1]).reshape(1, -1)
    x2d = np.array([1, 1, -1, -1, -1, 1, -1, -1]).reshape(1, -1)
    x3d = np.array([1, 1, 1, -1, 1, 1, -1, 1]).reshape(1, -1)
    distorted_inputs = np.concatenate((x1d, x2d, x3d))

    return inputs, distorted_inputs
    
def generate_inputs_task32(filename=None, slice_at=0):
    data_matrix = data_file_to_matrix(filename, slice_at)
    inputs = data_matrix[:9]
    distorted_inputs = data_matrix[9:]

    return inputs, distorted_inputs #TODO: testade .T

def data_to_image(data_matrix):
    vector_of_images = np.zeros((data_matrix.shape[0], 32, 32))
    for i in range(data_matrix.shape[0]):
        vector_of_images[i] = np.reshape(data_matrix[i], (32 ,-1))
    return vector_of_images


figure_number = 0
def plot_image(image):
    global figure_number
    plt.matshow(image)
    plt.savefig("fig" + str(figure_number))
    figure_number += 1
    plt.show()

def task3_1():
    hopfield = Hopfield_Network(units=8, patterns=3, epochs=3)
    inputs, distorted_inputs = generate_inputs_task31()
    hopfield.store_patterns(inputs)
    recall = hopfield.recall(distorted_inputs)
    return recall

def task3_2():      
    #inputs, distorted_inputs = generate_inputs_task31()
    inputs, distorted_inputs = generate_inputs_task32(filename="data_lab3/pict.dat", slice_at=1024)
    
    #extracting images p1, p2 and p3 into a matrix where each row corresponds to an image
    train_data = np.concatenate((inputs[0].reshape(1, -1), inputs[1].reshape(1, -1), inputs[2].reshape(1, -1)))

    test_data = np.concatenate((distorted_inputs[0].reshape(1, -1), distorted_inputs[1].reshape(1, -1)))
    
    test_images = data_to_image(test_data)
    
    hopfield = Hopfield_Network(units=train_data.shape[1], patterns=train_data.shape[0], epochs=100)

    hopfield.store_patterns(train_data)
    
    recall = hopfield.recall(test_data)
    
    recall_images = data_to_image(recall)
    real_images = data_to_image(train_data)
    
    plot_image(real_images[0])
    plot_image(test_images[0])
    plot_image(recall_images[0])
    


#Task calls

#task3_1()
task3_2()
#task3_3()
#task3_4()



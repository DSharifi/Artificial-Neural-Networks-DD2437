import numpy as np
import matplotlib.pyplot as plt


class Hopfield_Network(object):
    def __init__(self, units=8, use_bias=False, ndr_weights=False):
        self.units = units
        self.bias = use_bias
        self.energies = list()
        self.ndr = ndr_weights

    def recall_sequential_random(self, test_patterns):
        #SEQUENTIAL
        patterns = test_patterns.copy()
        for p in range(patterns.shape[0]):
            p1 = np.random.randint(0, patterns.shape[0]) if self.ndr else p

            for _ in range(self.units):
                i = np.random.randint(0, self.units)
                sum_un = 0
                for j in range(self.units):
                    sum_un += (self.W[i][j] * patterns[p1][j])
                
                if sum_un < 0:
                    sum_un = -1
                else:
                    sum_un = 1

                patterns[p1][i] = sum_un
        #test_image = data_to_image(patterns)
        #plot_image(test_image[1])
        self.energies.append(self.energy(patterns))
        return patterns

    def recall_batch(self, test_patterns):
        patterns = test_patterns.copy()
        patterns = np.dot(test_patterns, self.W)
        patterns[patterns >= 0] = 1
        patterns[patterns < 0] = -1
        self.energies.append(self.energy(patterns))
        return patterns

    def store_patterns(self, inputs):
        self.X = inputs

        if self.ndr:
            self.W = np.random.normal(0, 0.1, (self.units, self.units))
        else:
            self.W = np.zeros([self.units, self.units])
        for pattern in inputs:
            self.W += np.outer(pattern, np.transpose(pattern))
        np.fill_diagonal(self.W, 0)
        self.W /= self.units
        #print("Stored patterns, weight matrix: \n", self.W)

    def energy(self, patterns):         
        return np.array([-0.5*np.dot(np.dot(p.T,self.W),p) for p in patterns])

    
    
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

def plot_image(image):
    plt.matshow(image)
    plt.show()

def plot_energy(energy_list, epochs, attractor_energy):
    iterations = np.arange(epochs)
    energy_list = np.transpose(energy_list)
    list_sym = {
        0: 'o',
        1: '^',
        2: 's'
    }
    colors = {
        0: "magenta",
        1: "pink",
        2: "blue",
        3: "red"

    }

    for i in range(energy_list.shape[0]):
        plt.plot(iterations, energy_list[i], label="(Recall) Image " + str(i+10), marker=list_sym[i])
    
    for i in range(attractor_energy.shape[0]):
        plt.axhline(y=attractor_energy[i], label="(True) Image " + str(i+1), linestyle="-", color=colors[i])
    
    plt.title("Energy for true (fixed) and recall functions every iteration")
    plt.xlabel("Iterations")
    plt.ylabel("Energy")
    plt.legend(loc="upper right")
    plt.show()

def apply_x_error_bits(test_data, error_bits=0):
    pattern = test_data.copy()
    perm_array = np.random.permutation(1024)
    idx_array = perm_array[:error_bits]
    for idx in idx_array:
        pattern[0][idx] *= -1
    
    return pattern



def task3_1():
    hopfield = Hopfield_Network(units=8)
    inputs, distorted_inputs = generate_inputs_task31()
    hopfield.store_patterns(inputs)
    recall = hopfield.recall_batch(distorted_inputs)
    return recall

def task3_23(use_ndr=False, mode="batch"):      
    #inputs, distorted_inputs = generate_inputs_task31()
    inputs, distorted_inputs = generate_inputs_task32(filename="data_lab3/pict.dat", slice_at=1024)
    
    #extracting images p1, p2 and p3 into a matrix where each row corresponds to an image
    train_data = np.concatenate((inputs[0].reshape(1, -1), inputs[1].reshape(1, -1), inputs[2].reshape(1, -1)))

    test_data = np.concatenate((distorted_inputs[0].reshape(1, -1), distorted_inputs[1].reshape(1, -1)))

    hopfield = Hopfield_Network(units=train_data.shape[1], ndr_weights=use_ndr)
    
    hopfield.store_patterns(train_data)
    
    attractor_energy = hopfield.energy(train_data)

    epochs = int(np.log2(hopfield.units)) - 9
    
    recall = test_data.copy()
    for epoch in range(epochs):
        if mode == "batch":
            recall = hopfield.recall_batch(recall)
        elif mode == "seq":
            recall = hopfield.recall_sequential_random(recall)
        print("Epoch " + str(epoch) + " done")
    energy_list = np.asarray(hopfield.energies, list)
    print(recall.shape)
    plot_energy(energy_list, epochs, attractor_energy)
    recall_images = data_to_image(recall)

    plot_image(recall_images[1])
    
def plot_convergence(error_list, error_bits_list, image_number):
    for i in range(error_list.shape[0]):
        epochs = 0
        for err in error_list[i]:
            if err == -1:
                break
            epochs += 1
        plt.plot(np.arange(epochs), error_list[i][:epochs], label=str(error_bits_list[i]) + " error bits")
    
    plt.legend(loc="upper right")
    plt.title("Error rate during each epoch depending on the amount of error bits (Image " + str(image_number) + ")")
    plt.xlabel("Epochs")
    plt.ylabel("Error (bits)")
    plt.show()

    
def task3_4(image_number=0, max_iter=25):
    inputs, distorted_inputs = generate_inputs_task32(filename="data_lab3/pict.dat", slice_at=1024)
    
    train_data = np.concatenate((inputs[0].reshape(1, -1), inputs[1].reshape(1, -1), inputs[2].reshape(1, -1)))

    test_data = inputs[image_number].reshape(1, -1)
    hopfield = Hopfield_Network(units=train_data.shape[1], ndr_weights=False)
    hopfield.store_patterns(train_data)
    """ Applying more and more noise, checking number of epochs until conversion """
    error_bits = np.arange(0, 1024, 50)
    
    max_epochs = max_iter
    error_list = -1 * np.ones((len(error_bits), max_epochs+1))
    images = data_to_image(train_data)
    
    for i in range(len(error_bits)):
        epochs = max_epochs
        noise = error_bits[i]
        distorted_data = apply_x_error_bits(test_data, error_bits=noise)
        error = np.inf
        result = distorted_data.copy()
        error_list[i][0] = error_bits[i]
        while error > 0.0 and epochs > 0:
            result = hopfield.recall_sequential_random(result)
            epochs -= 1
            error = np.sum(np.abs(result-train_data[image_number]))
            error = error/2
            print("Error: " + str(error) + ", (" + str(error_bits[i]) + " error bits) \t epoch_number=" + str(max_epochs-epochs))
            error_list[i][max_epochs-epochs] = error
        
    plot_convergence(error_list, error_bits, image_number)
    
    #plotting


""" Task calls """
#task3_1()
#task3_23(use_ndr=False, mode="seq") #TODO: Symmetric matrix
#task3_4(image_number=2, max_iter=25)



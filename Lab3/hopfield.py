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
        #self.energies.append(self.energy(patterns))
        return patterns

    def store_patterns(self, inputs):
        self.X = inputs

        if self.ndr:
            self.W = np.random.normal(0, 1, (self.units, self.units))
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

def plot_energy(energy_list, attractor_energy, epochs, task3_5=False, stored_patterns=0):
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
        3: "red",
        4: "#33D7DC",
        5: "#59DC33",
        6: "#CADC33",
        7: "#8C25AD",
        8: "#58126E",
        9: "#923754"
    }

    for i in range(energy_list.shape[0]):
        plt.plot(iterations, energy_list[i], label="(Recall) Image " + str(i+10), marker=list_sym[i])
    
    for i in range(attractor_energy.shape[0]):
        plt.axhline(y=attractor_energy[i], label="(True) Image " + str(i+1), linestyle="-", color=colors[i])
    
    if task3_5:
        plt.title("Energy for true (fixed) and recall functions every iteration" + ", stored patterns: " + str(stored_patterns))
    else:
        plt.title("Energy for true (fixed) and recall functions every iteration")
    plt.xlabel("Iterations")
    plt.ylabel("Energy")
    plt.legend(loc="upper right")
    if not task3_5:
        plt.show()

def apply_x_error_bits(units, test_data, error_bits=0):
    pattern = test_data.copy()
    for i in range(pattern.shape[0]):
        perm_array = np.random.permutation(units)
        idx_array = perm_array[:error_bits]
        for idx in idx_array:
            pattern[i][idx] *= -1
    
    return pattern



def task3_1():
    hopfield = Hopfield_Network(units=8)
    inputs, distorted_inputs = generate_inputs_task31()
    hopfield.store_patterns(inputs)
    recall = hopfield.recall_batch(distorted_inputs)
    return recall

def task3_23(use_ndr=False, mode="batch", epochs=10):      
    #inputs, distorted_inputs = generate_inputs_task31()
    inputs, distorted_inputs = generate_inputs_task32(filename="data_lab3/pict.dat", slice_at=1024)
    
    #extracting images p1, p2 and p3 into a matrix where each row corresponds to an image
    train_data = np.concatenate((inputs[0].reshape(1, -1), inputs[1].reshape(1, -1), inputs[2].reshape(1, -1)))

    test_data = np.concatenate((distorted_inputs[0].reshape(1, -1), distorted_inputs[1].reshape(1, -1)))

    hopfield = Hopfield_Network(units=train_data.shape[1], ndr_weights=use_ndr)
    
    hopfield.store_patterns(train_data)
    
    attractor_energy = hopfield.energy(train_data)
    
    recall = test_data.copy()
    for epoch in range(epochs):
        if mode == "batch":
            recall = hopfield.recall_batch(recall)
        elif mode == "seq":
            recall = hopfield.recall_sequential_random(recall)
        print("Epoch " + str(epoch) + " done")
    energy_list = np.asarray(hopfield.energies, list)
    print(recall.shape)
    plot_energy(energy_list, attractor_energy, epochs)
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
        distorted_data = apply_x_error_bits(hopfield.units, test_data, error_bits=noise)
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


def generate_inputs_task35(size, amount):
    data = np.ones((amount, size))
    data = data*np.random.randint(2, size=(amount,size))
    data[data==0] = -1
    return data


def task3_5_1():
    inputs, distorted_inputs = generate_inputs_task32(filename="data_lab3/pict.dat", slice_at=1024)

    train_data = np.concatenate((inputs[0].reshape(1, -1), inputs[1].reshape(1, -1), inputs[2].reshape(1, -1)))

    distorted_data = np.concatenate((distorted_inputs[0].reshape(1, -1), distorted_inputs[1].reshape(1, -1)))

    total_images = 9
    
    max_epochs = 25
    
    for i in range(3, total_images):
        train_data = np.concatenate((train_data, inputs[i].reshape(1, -1)), axis=0)
        hopfield = Hopfield_Network(units=train_data.shape[1], ndr_weights=False)
        hopfield.store_patterns(train_data)
        attractor_energy = hopfield.energy(train_data)
        epochs = 0
        recall = distorted_data.copy()
        while epochs < max_epochs:
            recall = hopfield.recall_sequential_random(recall)
            #print("Stored images: " + str(i+1), ", epoch=" + str(epochs), ", error="+str(error))
            epochs += 1
            print("Epoch: " + str(epochs))

        energy_list = np.asarray(hopfield.energies, list)
        plot_energy(energy_list, attractor_energy, epochs, True, (i+1))
        plt.show()

def task3_5_2(patterns=300, units=100, mode="batch", use_noise=False, distortion=0.0, bias=False):
    data = generate_inputs_task35(patterns, units)
    if bias:
        data = 0.5+np.random.normal(0, 1, (300,100))
        data[data >= 0] = 1
        data[data < 0] = -1
    patterns_amt = []
    stable_amt = []
    for i in range(1, 50):
        new_data = data[:i]
        hopfield = Hopfield_Network(units=data.shape[1], ndr_weights=False)
        hopfield.store_patterns(new_data)
        if use_noise:
            noisy_data = apply_x_error_bits(new_data.shape[1], new_data, error_bits=int(data.shape[1]*distortion))
        patterns_amt.append(i)
        if mode == "seq":
            recall_result = hopfield.recall_sequential_random(noisy_data if use_noise else new_data)
        else:
            recall_result = hopfield.recall_batch(noisy_data if use_noise else new_data)
        #total_error = (np.sum(np.abs(recall_result-new_data))) / 2
        print(recall_result.shape)
        counter = 0
        for j in range(new_data.shape[0]):
            if np.array_equal(recall_result[j] - new_data[j], np.zeros(new_data.shape[1])):
                counter+=1
        stable_amt.append(counter)
    
    """ Plotting """
    plt.plot(patterns_amt, stable_amt, label="Capacity", color="green")
    plt.title("Capacity of the hopfield network with " + str(patterns) + " stored patterns and " + str(units) + " units\n (using " + mode + " update) " + \
        "Distorition: " + str(distortion) + ", Bias: " + str(bias))
    plt.xlabel("Amount of Patterns Stored")
    plt.ylabel("Amount of Stable Patterns")
    plt.legend(loc="upper right")
    plt.show()

def task3_6():
    return
    
""" Task calls """
#task3_1()
#task3_23(use_ndr=False, mode="seq", epochs=15) #TODO: Symmetric matrix
#task3_4(image_number=2, max_iter=25)
#task3_5_1()
#task3_5_2(patterns=300, units=100, mode="seq", use_noise=False, distortion=0.0, bias=True)
#task3_6()
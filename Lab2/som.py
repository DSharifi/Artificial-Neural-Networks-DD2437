import numpy as np
import matplotlib.pyplot as plt

#---------------------------------------------------#
#----------Read-in-file-animal-to-matrix------------#
#---Argument: 1.File name, 2.number of attributes---#
#----------- Return: a matrix of size: -------------#
#------ [file size/numb_of_atr, numb_of_atr] -------#
#---------------------------------------------------#
def read_in_file_to_matrix(file_name, n_of_slices):
    file = open(file_name, "r")
    animal = file.read()
    animal = list(map(int,animal.split(",")))
    animal = np.asarray(animal)
    animal = np.reshape(animal,(-1, n_of_slices))
    return animal


def read_in_file_to_matrix_float(file_name, n_of_slices):
    file = open(file_name, "r")
    animal = file.read()
    animal = list(map(float,animal.split(",")))
    animal = np.asarray(animal)
    animal = np.reshape(animal,(-1, n_of_slices))
    return animal
#---------------------------------------------------#
#----------- Read-in-file-animal-to-list -----------#
#-------------- Argument: 1.File name --------------#
#--- Return: A vector with each row of the file  ---#
#---------------------------------------------------#
def read_in_file_to_vector(file_name):
    file = open(file_name, "r")
    animal = file.read()
    animal = list(animal.splitlines())
    animal = np.asarray(animal)
    return animal

#---------------------------------------------------#
#------------ Genereate_weight_matrix --------------#
#---- Argument: 1.numb_of_nodes, 2.numb_of_feat ----#
#--- Return: A matrix, random weights from [0,1] ---#
#-------- Size: [numb_nodes, numb_features] --------#
#---------------------------------------------------#
def genereate_weight_matrix(number_of_hidden_nodes,number_of_features):
    weight_matrix = np.random.rand(number_of_hidden_nodes,number_of_features)
    return weight_matrix

    
#------- 1.Calculate the similarity between --------# 
#-------- the input pattern and the weights --------# 
#---------- arriving at each output node. ----------#
#---------------------------------------------------#
#-- 2. Find the most similar node; often referred --#
#---------------- to as the winner. ----------------#
#---------------------------------------------------#
def measuring_similarity(animal,weight_matrix):
    ## Size of the distance & index vector is equal to amount of nodes
    distance_vector = np.zeros(weight_matrix.shape[0]).reshape(-1,1)
    index_vector = np.arange(weight_matrix.shape[0]).reshape(-1,1)
    neighbourhood_vector = np.zeros(weight_matrix.shape[0]).reshape(-1,1)

    # Loop thought all node.
    for i in range(weight_matrix.shape[0]):
        temp = animal-weight_matrix[i]
        distance_vector[i] =  temp.T @ temp

    # Sort the nodes from lowest distance to largest.
    distance_matrix = np.concatenate((distance_vector,index_vector),axis=1)
    distance_matrix = sorted(distance_matrix, key=lambda row: row[0])
    distance_matrix = np.array(distance_matrix)
    distance_matrix = np.concatenate((distance_matrix,neighbourhood_vector),axis=1)

    # Return the matrix which contains the placement of the nodes.
    return distance_matrix.T


#---------------------------------------------------#
#---- 3. Select a set of output nodes which are ----#
#---- located close to the winner in the output ----#
#----- grid. This is called the neighbourhood. -----#
#---------------------------------------------------#
def neighbourhood(distance_matrix,weight_matrix,step_size,neighbourhood_set,task,epoch, factor1=1, factor2=1,):
    # ------------NOTE---------------
    # Lite osäker på den här logiken!
    # dvs, ifall man får jämför såhär.
    #--------              ---------
    # altr, vi räkna om dist från
    # varje node till den vinnande.
    #--------------------------------
    
    # The winning node & neighbourhood factor
    winner_distance = distance_matrix[0][0]
    winner_index = distance_matrix[1][0]

    distance_matrix[2][0] = step_size

    # Loop thought the other nodes
    # Set a learning rate if they belong 
    # to the neighbourhood.
    print(distance_matrix.shape)
    #while 1 :
    #    1
    if task == "animal":
        for i in range(1,neighbourhood_set):
        
            if distance_matrix[0][i] - winner_distance <= winner_distance/factor1:
                distance_matrix[2][i] = step_size/2 ###### DENNA TWEAKAR VI######
            elif distance_matrix[0][i] - winner_distance <= winner_distance/factor2:
                distance_matrix[2][i] = step_size/4 ###### DENNA TWEAKAR VI######
    
    elif task == "tour":
        

        distance_matrix = distance_matrix.T
        distance_matrix = sorted(distance_matrix, key=lambda row: row[1])
        distance_matrix = np.array(distance_matrix).T
        #print(distance_matrix)

        #while 1 :
        #    1
        print(epoch)
        #if epoch > 10:
        print((winner_index-1)%10)
        distance_matrix[2][int(winner_index-1)%10] = step_size
        distance_matrix[2][int(winner_index+1)%10] = step_size
        #elif epoch > 2:
        #    distance_matrix[2][(winner_index-1)%10] = step_size
        #print(epoch)             
        distance_matrix = distance_matrix.T
        distance_matrix = sorted(distance_matrix, key=lambda row: row[0])
        distance_matrix = np.array(distance_matrix).T
        
    return distance_matrix

    
#---------------------------------------------------#
#---- 4. Update the weights of all nodes in the ----#
#------ neighbourhood such that their weights ------#
#----- are moved closer to the input pattern. ------#
#---------------------------------------------------#
def update_weights(animal, weight_matrix, distance_matrix):
    
    distance_matrix = distance_matrix.T
    distance_matrix = sorted(distance_matrix, key=lambda row: row[1])
    distance_matrix = np.array(distance_matrix).T
    
    # Loop thought all node & update weights.
    for i in range(weight_matrix.shape[0]):
        weight_matrix[i] = weight_matrix[i] + distance_matrix[2][i]*(animal-weight_matrix[i])
    
    return weight_matrix


#---------------------------------------------------#
#---------------- Traning the network --------------#
#------- Argument: Weights, [100,1]features, -------#
#--------------- epochs, l_rate --------------------#
#------------ Return: A trained model --------------#
#---------------------------------------------------#
def train_network(animal_matrix,weight_matrix,neighbourhood_set, number_of_epochs = 20, learning_rate = 0.2):

    factor1 = 4
    factor2 = 2
    task = "tour"
    # Loop the amount epochs 
    for epoch in range(number_of_epochs):
        # Loop thought the amount of aminals
        for animal in range(animal_matrix.shape[0]):

            distance_matrix = measuring_similarity(animal_matrix[animal],weight_matrix)
        
            distance_matrix = neighbourhood(distance_matrix,weight_matrix, learning_rate,neighbourhood_set,task,epoch,factor1,factor2)
            #factor1 = factor1-(2*(epoke+5))
            #factor2 = factor2+epoke
            
            #factor1 = factor1 + (epoke+1)
            #factor2 = factor2 + (epoke+1)
            #neighbourhood_set = neighbourhood_set - 2
            neighbourhood_set = int(neighbourhood_set * (1 - (epoch / number_of_epochs)))
            
            weight_matrix = update_weights(animal_matrix[animal],weight_matrix,distance_matrix)

    return weight_matrix


#---------------------------------------------------#
#---------------- testing the network --------------#
#------- Argument: Weights, [10,1]features, --------#
#--------- Return: A vector pos of elements --------#
#---------------------------------------------------#
def test_network(animal_matrix,weight_matrix):

    # The vector where we save all the nodes each data picked.
    pos_vector = np.zeros(animal_matrix.shape[0])

    # Loop thought the amount of aminals
    for animal in range(animal_matrix.shape[0]):

        distance_matrix = measuring_similarity(animal_matrix[animal],weight_matrix)
        pos_vector[animal] = distance_matrix[1][0]

    return pos_vector


  
#---------------------------------------------------#
#----------------- PRINT COOL STUFF ----------------#
#---------------------------------------------------#

def plot_points(x, y):
    plt.scatter(x, y, color="red")
    plt.grid()
    plt.show()
    
#---------------------------------------------------#
#---- 4.1 TOpological Ordering of Animal Species ---#
#---------------------------------------------------#
def animal_species():
    animal_matrix = read_in_file_to_matrix("data_lab2/animals.dat", 84)

    animal_names = read_in_file_to_vector("data_lab2/animalnames.txt").reshape(-1,1)

    weight_matrix = genereate_weight_matrix(100,84)

    neighbourhood_set = int(weight_matrix.shape[0]/2) # 50% of the set as neighbours
    weight_matrix = train_network(animal_matrix,weight_matrix,neighbourhood_set)

    pos_vector = test_network(animal_matrix,weight_matrix).reshape(-1,1)

    result = np.concatenate((pos_vector, animal_names),axis=1)
    sorted_result = sorted(result, key=lambda row: row[0])
    sorted_result = np.array(sorted_result)
    print(sorted_result)

#---------------------------------------------------#
#----------------- 4.2 Cyclic Tour -----------------#
#---------------------------------------------------#
def cyclic_tour():
    
    cities_matrix = read_in_file_to_matrix_float("data_lab2/cities.dat",2)
    
    weight_matrix = genereate_weight_matrix(10,2)

    neighbourhood_set = int(weight_matrix.shape[0]/5) # 20% of the set as neighbours

    weight_matrix = train_network(cities_matrix,weight_matrix,neighbourhood_set)

    pos_vector = test_network(cities_matrix,weight_matrix).reshape(-1,1)
    
    cities_index = np.arange(0,10).reshape(-1,1)

    result = np.concatenate((pos_vector, cities_matrix,cities_index),axis=1)
    sorted_result = sorted(result, key=lambda row: row[0])
    sorted_result = np.array(sorted_result)


    print(sorted_result)
    plt.title("Cyclic tour estimation using SOM")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.scatter(cities_matrix[:, :1], cities_matrix[:, 1:2], color = "red")
    plt.plot(sorted_result[:,1],sorted_result[:, 2], label="Path")
    plt.legend(loc="upper right")
    plt.show()
    
   
    #print(cities_matrix)
    return 1
#---------------------------------------------------#
#-------------------Function calls------------------#
#---------------------------------------------------#

#animal_species()

cyclic_tour()
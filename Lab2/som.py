import numpy as np
import matplotlib.pyplot as plt
#---------------------------------------------------#
#----------Read-in-file-animal-to-matrix------------#
#---Argument: 1.File name, 2.number of attributes---#
#----------- Return: a matrix of size: -------------#
#------ [file size/numb_of_atr, numb_of_atr] -------#
#---------------------------------------------------#
def read_in_file_to_matrix(file_name, n_of_slices ):
    file = open(file_name, "r")
    animal = file.read()
    animal = list(map(int,animal.split(",")))
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
   
    # Loop thought all node.
    for i in range(weight_matrix.shape[0]):
        temp = animal-weight_matrix[i]
        distance_vector[i] =  temp.T @ temp

    # Sort the nodes from lowest distance to largest.
    distance_matrix = np.concatenate((distance_vector,index_vector),axis=1)
    distance_matrix = sorted(distance_matrix, key=lambda row: row[0])
    distance_matrix = np.array(distance_matrix)

    # Return the matrix which contains the placement of the nodes.
    return distance_matrix.T


#---------------------------------------------------#
#---- 3. Select a set of output nodes which are ----#
#---- located close to the winner in the output ----#
#----- grid. This is called the neighbourhood. -----#
#---------------------------------------------------#
def neighbourhood(distance_matrix, amount_of_affected_neighbours):
    
    #TODO Implement this fuction
    
    return 1
#---------------------------------------------------#
#---- 4. Update the weights of all nodes in the ----#
#------ neighbourhood such that their weights ------#
#----- are moved closer to the input pattern. ------#
#---------------------------------------------------#
def update_weights():
    
    #TODO Implement this fuction
    
    return 1


#---------------------------------------------------#
#---------------- Traning the network --------------#
#------- Argument: Weights, [10,1]features, --------#
#--------------- epochs, l_rate --------------------#
#------------ Return: A trained model --------------#
#---------------------------------------------------#
def train_network(animal_matrix,weight_matrix, number_of_epochs = 20, learning_rate = 0.2):
    
    #print(animal_matrix.shape)

    # Loop the amount epochs 
    #for epoke in range(number_of_epochs):
        # Loop thought the amount of aminals
        #for animal in range(animal_matrix.shape[0]):


    distance_matrix = measuring_similarity(animal_matrix[0],weight_matrix)
    print(distance_matrix.shape)

    #TODO Keep coding here.

    return 1


    
#----------------------------------------#
#------------Function calls--------------#
#----------------------------------------#
animal_matrix = read_in_file_to_matrix("data_lab2/animals.dat", 84)
#print(animal_matrix.shape)

animal_names = read_in_file_to_vector("data_lab2/animalnames.txt")
#print(animal_names.shape)

weight_matrix = genereate_weight_matrix(100,84)
#print(weight_matrix.shape)


train_network(animal_matrix,weight_matrix)


from util import *
from rbm import RestrictedBoltzmannMachine 
from dbn import DeepBeliefNet
from dbm1 import DeepBeliefNet1

if __name__ == "__main__":

    image_size = [28,28]
    train_imgs,train_lbls,test_imgs,test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)
    
    ''' restricted boltzmann machine '''
    
    print ("\nStarting a Restricted Boltzmann Machine..")

    
    hidden_node_list = np.array([150, 200, 250, 300, 350, 400, 450, 500])
    recon_loss_records = np.zeros((len(hidden_node_list), 10))

    for idx, ndim in enumerate(hidden_node_list):
        rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                        ndim_hidden=ndim,
                                        is_bottom=True,
                                        image_size=image_size,
                                        is_top=False,
                                        n_labels=10,
                                        batch_size=20
        )
        #rbm.cd1(visible_trainset=train_imgs, n_iterations=10)

        recon_loss_records[idx] = rbm.cd1(visible_trainset=train_imgs, n_iterations=10)
    
    #plotting reconstruction losses
    plt.title("Reconstruction losses during each epoch \ndepending on amount of hidden nodes (batch size=20)")
    plt.xlabel("Epochs")
    plt.ylabel("Reconstruction loss")
    epochs = np.arange(10)
    for i in range(len(hidden_node_list)):
        plt.plot(epochs, recon_loss_records[i], label=str(hidden_node_list[i]) + " hidden nodes")
    
    plt.legend(loc="upper right")
    plt.show()
    
    
    

    ''' deep- belief net '''

    """
    print ("\nStarting a Deep Belief Net..")
    
    dbn = DeepBeliefNet(sizes={"vis":image_size[0]*image_size[1], "hid":500, "pen":500, "top":2000, "lbl":10},
                        image_size=image_size,
                        n_labels=10,
                        batch_size=20
    )
    
    ''' greedy layer-wise training '''

    recon_loss_records = dbn.train_greedylayerwise(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=10)

    #plotting reconstruction losses
    plt.title("Reconstruction losses for each layer in the DBN (Batch size=20)")
    plt.xlabel("Epochs")
    plt.ylabel("Reconstruction loss")
    epochs = np.arange(10)
    labels_recon= {
        0: "vis--hid",
        1: "hid--pen",
        2: "pen+lbl--top"
    }
    for i in range(0, 3):
        plt.plot(epochs, recon_loss_records[i], label=labels_recon[i])
    
    plt.legend(loc="upper right")
    plt.show()

    #dbn.recognize(train_imgs, train_lbls)
    
    #dbn.recognize(test_imgs, test_lbls)
    
    
    for digit in range(10):
        digit_1hot = np.zeros(shape=(1,10))
        digit_1hot[0,digit] = 1
        dbn.generate(digit_1hot, name="rbms")
  

    ''' fine-tune wake-sleep training '''
    
    dbn.train_wakesleep_finetune(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=20)

    dbn.recognize(train_imgs, train_lbls)
    
    dbn.recognize(test_imgs, test_lbls)
    
    for digit in range(10):
        digit_1hot = np.zeros(shape=(1,10))
        digit_1hot[0,digit] = 1
        dbn.generate(digit_1hot, name="dbn")
    

    dbn = DeepBeliefNet1(sizes={"vis":image_size[0]*image_size[1], "hid":0, "pen":500, "top":2000, "lbl":10},
                        image_size=image_size,
                        n_labels=10,
                        batch_size=20
    )
    recon_loss_records = dbn.train_greedylayerwise(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=10)
  
  
    dbn.train_wakesleep_finetune(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=20)

    #dbn.recognize(train_imgs, train_lbls)
    
    #dbn.recognize(test_imgs, test_lbls)
    
    for digit in range(10):
        digit_1hot = np.zeros(shape=(1,10))
        digit_1hot[0,digit] = 1
        dbn.generate(digit_1hot, name="dbn")
    
    dbn.recognize(train_imgs, train_lbls)
    
    dbn.recognize(test_imgs, test_lbls)

    for digit in range(10):
        digit_1hot = np.zeros(shape=(1,10))
        digit_1hot[0,digit] = 1
        dbn.generate(digit_1hot, name="rbms")
    """
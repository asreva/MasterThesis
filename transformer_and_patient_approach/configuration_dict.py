"""
Aim: Implement the configuration skeleton that defines the whole training procedure for transformer with patient data
Author: Ivan-Daniel Sievering for the LTS4 Lab (EPFL)
"""

# --- Dictionnaries --- #
train_configuration_default = {
    # General
    "seed": 42, # seed to use to enforce the training-testing set separation
    "nb_cv": None, # if not using cross validation -> None, else specify the number of cross validation to use
    "save_best_net": False, # if true will save at each new F1_valid best score and at the end
    "load_network": None, # path to the network
    
    # Network structure
    "network_class": None, # pytorch class to construct the network
    "dropout": None, # dropout for the main network
    "weights_init": None, # "Xavier Std", "Xavier Uniform", "Xavier Normal", "Kaiming Uniform" and "Kaiming Normal"
    
    # Transformer specific parameters
    "patch_size_l" : None, # list of the size of the patch extract for each kind of box (! for the network, size has to be same for all)
    "nb_patch_l": None, # list of the number of patches to extract for each kind of box (values can be different)
    
    # Patient network specific parameters (! also if patient network used inside of another network !)
    "dropout_patient_net": None, # dropout for the patient network
    "nb_neur_per_hidden_layer_patient": None, # list with the number of neurons for each hidden layer
    "batch_norm_patient":None, # True or False
    "weights_init_patient": None, # "Std", "Xavier Uniform", "Xavier Normal", "Kaiming Uniform" and "Kaiming Normal"
    
    # Dataset information
    "balance_method": None, # "no", "oversample" and "undersample"
    "train_test_ratio": 0.2, # ratio to use in testing vs in training (also in validation vs in training)
    "test": False, # False: remove test data and then separate train and valid. True: separate test and train (impossible to use CV)
    "normalise": None, # True or False (on images)
    "gaussian_blur": None, # probability 
    "random_rotation": None, # probability
    "random_crop": None, # probability
    "random_color_modifs": None, # for CNN, probability
    "patch_randomness": None, # for transformers, probability to NOT take a sample on centerline
    "normalise_patient": None, # for patient network, normalise or not the data, True or False
    
    # Define training
    "n_epochs": None, 
    "batch_size": None, 
    "change_strategy_epoch_l": None, # list indicating at which epoch change the optimizer, loss and lr, start with -1 for the first one
    
    # Define the optimiser
    "optimizer_l": None, # list of optimizer to use, "SGD", "Adam", "PESG", "PDSCA"
    "weight_decay": None,
    "lr_l": None, # list of the lr
    "SGD_momentum": None, # HP of the SGD
    "PESG_gamma": None, # HP of the PSEG
    "PESG_margin": None, # HP of the PSEG
    "PESG_imratio": None, # HP of the PSEG
    "Compo_gamma": None, # HP of the PDSCA
    "Compo_margin": None, # HP of the PDSCA
    "Compo_imratio": None, # HP of the PDSCA
    "Compo_beta1": None, # HP of the PDSCA
    "Compo_beta2": None, # HP of the PDSCA
    
    # Define the scheduler
    "scheduler_patience": None, # nb of epochs without improvement before reducing lr
    "scheduler_factor": None, # how much to reduce the lr when plateau (lr*=scheduler_factor)
    
    # Define the loss
    "criterion_l": None, # list of optimizer to use, "BCE", "AUC", "Focal"
    "siamese_pred_loss_ratio": None, # ratio btw artery MI prediction and global MI prediction losses
    "arteries_pred_loss_ratio": None, # ratio btw siamese loss and global MI predicition losses
    "patient_data_loss_ratio": None, # ratio btw patient data prediction loss and global MI predicition losses
    "focal_alpha": None, # HP of the focal loss
    "focal_gamma": None, # HP of the focal loss
    "focal_reduction": None, # HP of the focal loss
         
}
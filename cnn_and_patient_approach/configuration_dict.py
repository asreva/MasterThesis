"""
Aim: Implement the configuration skeleton that defines the whole training procedure
Author: Ivan-Daniel Sievering for the LTS4 Lab (EPFL)
"""

# --- Dictionnaries --- #
train_configuration_default = {
    # General
    "seed": 42, # seed to use, always 42 to keep always the same testing set
    
    # Define training strategy
    "nb_cv": None, # if not using cross validation -> None, else specify the number of cross validation to use
    "n_epochs": None, # nb of epochs to run
    "batch_size": None, # batch size to use when training
    "optimizer_type": None, # a list with all the optimizer that are used, choose btw "SGD" and "Adam" and "PESG"
    "learning_rate": None, # learning rate of the optimier, a list of the start lr of each optimizer
    "change_opti_and_crit_epochs": None, # list indicating at which epoch change the optimizer/lr, start with -1 for the first one
    "SGD_momentum": None, # HP of the SGD optimisation
    "PESG_gamma": None, # HP of the PSEG optimisation
    "PESG_margin": None, # HP of the PSEG optimisation algorithm, margin between classes
    "PESG_imratio": None, # HP of the PSEG optimisation algorithm, percentage of the samples belonging to the minority class
    "focal_alpha": None, # HP of the focal loss
    "focal_gamma": None, # HP of the focal loss
    "focal_reduction": None, # HP of the focal loss
    "weight_decay": None, # weight decay of the optimizer
    "criterion_type": None, # list of criterions to use, choose btw "BCE" "AUC" and "Focal"
    "scheduler_patience": None, # if using SGD or PESG, the nb of epochs without improvement before reducing lr
    "scheduler_factor": None, # if using SGD or PESG, how much to reduce the lr when plateau (lr*=scheduler_factor)
    "siamese_prediction_loss_ratio": None, # ratio btw artery MI prediction and global MI prediction losses
    "arteries_prediction_loss_ratio": None, # ratio btw siamese loss and global MI predicition losses
    "pred_from_patient_data_loss_ratio": None, # ratio btw patient data prediction loss and global MI predicition losses
    
     # Saving and loading network
    "save_best_net": False, # if true will save at each new F1_valid best score and at the end
    "load_network": None, # load an existing network, give the path to the network from the run.py file
    
    # Network structure
    "network_class": None, # pytorch class to construct the network
    "dropout": None, # dropout to apply at the end of the network (btw 0 and 1)
    "dropout_patient_net": None, # dropout to apply at the end of the network (btw 0 and 1)
    "nb_neur_per_hidden_layer_patient": None, # array with the number of neurons for each hidden layer, length of the array will be the nb of layers
    "batch_norm_patient":None, # actaivate or not batch normalisation
    "init": None, # weights initialisation method, choose btw "Std", "Xavier Uniform", "Xavier Normal", "Kaiming Uniform" and "Kaiming Normal". Std means no special intialisation, Xavier init is also known as Glorot and Kaiming as He
    "init_patient": None, # weights initialisation method, choose btw "Std", "Xavier Uniform", "Xavier Normal", "Kaiming Uniform" and "Kaiming Normal". Std means no special intialisation, Xavier init is also known as Glorot and Kaiming as He
    
    # Dataset information
    "balance_method": None, 
    "dataset_ratio": 1, # percentage of the dataset to use (in order to make fast tests)
    "train_test_ratio": 0.2, # ratio to use in testing vs in training (also in validation vs in training)
    "gaussian_blur": None, # the probability to apply gaussian blur (btw 0 and 1)
    "normalise": None, # the probability to apply gaussian blur (btw 0 and 1)
    "normalise_patient": None, # the probability to apply gaussian blur (btw 0 and 1)
    "random_rotation": None, # the probability to apply random rotation (btw 0 and 1)
    "random_crop": None, # the probability to apply random crop (btw 0 and 1)
    "random_color_modifs": None, # the probability to apply random color modifs (brightness, saturation, ...) (btw 0 and 1)
    
    # Old, used for sweeps because we cannot give arrays as input
    # "nb_neur_1":None,
    # "nb_neur_2":None,
    # "nb_neur_per_layer":None,
    # "nb_layer":None,
    # "learning_rate_1": None, # learning rate of the optimier
    # "learning_rate_2": None, # learning rate of the optimier
    
}
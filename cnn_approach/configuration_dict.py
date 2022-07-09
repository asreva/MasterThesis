"""
Aim: Implement the configuration skeleton that defines the whole training procedure of MI prediction with CNN
Author: Ivan-Daniel Sievering for the LTS4 Lab (EPFL)
"""

# --- Dictionnaries --- #
train_configuration_default = {
    "train_test_ratio": 0.2, # ratio to use in testing vs in training (also in validation vs in training)
    "seed": 42, # seed to use, always 42 to keep always the same testing set
    "nb_cv": None, # if not using cross validation -> None, else specify the number of cross validation to use
    "save_best_net": False, # if true will save the network at each new F1_valid best score and at the end
    
    "balance_method": None, # balance method to use, choose btw "undersample", "oversample" and "no"
    
    "gaussian_blur": None, # the probability to apply gaussian blur (btw 0 and 1)
    "normalise": None, # the probability to apply gaussian blur (btw 0 and 1)
    "random_rotation": None, # the probability to apply random rotation (btw 0 and 1)
    "random_crop": None, # the probability to apply random crop (btw 0 and 1)
    "random_color_modifs": None, # the probability to apply random color modifs (brightness, saturation, ...) (btw 0 and 1)
    
    "network_class": None, # pytorch class to construct the network
    "dropout": None, # dropout to apply at the end of the network (btw 0 and 1)
    "init": None, # weights initialisation method, choose btw "Std", "Xavier Uniform", "Xavier Normal", "Kaiming Uniform" and "Kaiming Normal". Std means no special intialisation, Xavier init is also known as Glorot and Kaiming as He
    
    "n_epochs": None, # nb of epochs to run
    "batch_size": None, # batch size to use when training
    "change_opti_and_crit_epochs": None, # list indicating at which epoch change the optimizer/lr, start with -1 for the first one
    "optimizer_type": None, # a list with all the optimizer that are used, choose btw "SGD" and "Adam" and "PESG"
    "learning_rate": None, # learning rate of the optimier, a list of the start lr of each optimizer
    
    "PESG_gamma": None, # hyper parameter of the PSEG optimisation algorithm, see paper for details
    "PESG_margin": None, # hyper parameter of the PSEG optimisation algorithm, margin between classes
    "PESG_imratio": None, # hyper parameter of the PSEG optimisation algorithm, percentage of the samples belonging to the minority class
    "SGD_momentum": None, # HP of SGD
    "focal_alpha": None, # HP of Focal loss
    "focal_gamma": None, # HP of Focal loss
    "focal_reduction": None, # HP of Focal loss
    "weight_decay": None, # weight decay of the optimizer
    "criterion_type": None, # list of criterions to use, choose btw "BCE" "AUC" and "Focal"
    "siamese_prediction_loss_ratio": None, # ratio btw artery MI prediction and global MI prediction losses
    "arteries_prediction_loss_ratio": None, # ratio btw siamese loss and global MI predicition losses
    "scheduler_patience": None, # if using SGD, the nb of epochs without improvement before reducing lr
    "scheduler_factor": None, # if using SGD, how much to reduce the lr when plateau (lr*=scheduler_factor)
    
    # Parameters used for W&B sweeps because we cannot give arrays as input
    # "nb_neur_1":None,
    # "nb_neur_2":None,
    # "nb_neur_per_layer":None,
    # "nb_layer":None,
    # "learning_rate_1": None,
    # "learning_rate_2": None,
}
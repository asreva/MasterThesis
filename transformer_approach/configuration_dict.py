"""
Aim: Implement the configuration skeleton that defines the whole training procedure
Author: Ivan-Daniel Sievering for the LTS4 Lab (EPFL)
"""

# --- Dictionnaries --- #
train_configuration_default = {
    "save_best_net": False,
    "load_network": None,
    
    "train_test_ratio": 0.2, # ratio to use in testing vs in training (also in validation vs in training)
    "seed": 42, # seed to use, always 42 to keep always the same testing set
    "nb_cv": None, # if not using cross validation -> None, else specify the number of cross validation to use
   
    "patch_randomness": None, # probability to choose y position of the patch indenpendtly of the centerline
    "patch_size_l" : None, # list of the size of the patch extract for each kind of box (! for the network, size has to be same for all)
    "nb_patch_l": None, # number of patches to extract for each kind of box (values can be different)
    
    "balance_method": None, # balance method to use, choose btw "undersample", "oversample" and "no"
    "gaussian_blur": None, # the probability to apply gaussian blur (btw 0 and 1)
    "normalise": None, # the probability to apply gaussian blur (btw 0 and 1)
    "random_rotation": None, # the probability to apply random rotation (btw 0 and 1)
    "random_crop": None, # the probability to apply random crop (btw 0 and 1)
    
    "network_class": None, # pytorch class to construct the network
    "dropout": None, # dropout to apply at the end of the network (btw 0 and 1)
    "init": None, # weights initialisation method, choose btw "Std", "Xavier Uniform", "Xavier Normal", "Kaiming Uniform" and "Kaiming Normal". Std means no special intialisation, Xavier init is also known as Glorot and Kaiming as He
    
    "n_epochs": None, # nb of epochs to run
    "batch_size": None, # batch size to use when training
    "optimizer_type": None, # optimizer to use to change the weights, choose btw "SGD" and "Adam"
    "learning_rate": None, # learning rate of the optimier
    "learning_rate_1": None, # learning rate of the optimier
    "learning_rate_2": None, # learning rate of the optimier
    
    "SGD_momentum": None,
    "PESG_gamma": None, # hyper parameter of the PSEG optimisation algorithm, see paper for details
    "PESG_margin": None, # hyper parameter of the PSEG optimisation algorithm, margin between classes
    "PESG_imratio": None, # hyper parameter of the PSEG optimisation algorithm, percentage of the samples belonging to the minority class
    
    "focal_alpha": None,
    "focal_gamma": None,
    "focal_reduction": None,
    
    "weight_decay": None, # weight decay of the optimizer
    "criterion_type": None, # criterion to compare the predicted MI and the true MI labels, choose btw "BCE"
    "siamese_prediction_loss_ratio": None, # ratio btw artery MI prediction and global MI prediction losses
    "arteries_prediction_loss_ratio": None, # ratio btw siamese loss and global MI predicition losses
    "scheduler_patience": None, # if using SGD, the nb of epochs without improvement before reducing lr
    "scheduler_factor": None, # if using SGD, how much to reduce the lr when plateau (lr*=scheduler_factor)
    
    
    "Compo_gamma": None,
    "Compo_margin": None,
    "Compo_imratio": None,
    "Compo_beta1": None,
    "Compo_beta2": None,
}
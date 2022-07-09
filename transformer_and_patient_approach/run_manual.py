"""
Aim: run the training of a network manually chose
Author: Ivan-Daniel Sievering for the LTS4 Lab (EPFL)
"""

# --- Settings --- #
# Libraries
import torch
import wandb

from train_valid import train_valid

from transfo_and_pat_softmax_net import PatientLevelDNN_PatientDataSoftmax

# --- Define dict --- #
train_config_manual = {
        # General
        "seed": 42, # seed to use to enforce the training-testing set separation
        "nb_cv": None, # if not using cross validation -> None, else specify the number of cross validation to use
        "save_best_net": True, # if true will save at each new F1_valid best score and at the end
        "load_network": None, # path to the network

        # Network structure
        "network_class": PatientLevelDNN_PatientDataSoftmax, # pytorch class to construct the network
        "dropout": 0.014775322741393282 , # dropout for the main network
        "weights_init": "Xavier Uniform", # "Xavier Uniform", "Xavier Normal", "Kaiming Uniform" and "Kaiming Normal"

        # Transformer specific parameters
        "patch_size_l" : [64, 64, 64, 64], # list of the size of the patch extract for each kind of box (! for the network, size has to be same for all)
        "nb_patch_l": [32, 64, 128, 128], # list of the number of patches to extract for each kind of box (values can be different)

        # Patient network specific parameters (! also if patient network used inside of another network !)
        "dropout_patient_net": 0.0700010465677528 , # dropout for the patient network
        "nb_neur_per_hidden_layer_patient": [50, 10], # list with the number of neurons for each hidden layer
        "batch_norm_patient": False, # True or False
        "weights_init_patient": "Kaiming Uniform", # "Std", "Xavier Uniform", "Xavier Normal", "Kaiming Uniform" and "Kaiming Normal"

        # Dataset information
        "dataset_type": "transformer_patient", # "CNN", "transformer", "CNN_patient" and "transformer_patient"
        "balance_method": "undersample", # "no", "oversample" and "undersample"
        "train_test_ratio": 0.2, # ratio to use in testing vs in training (also in validation vs in training)
        "test": False, # False: remove test data and then separate train and valid. True: separate test and train (impossible to use CV)
        "normalise": False, # True or False (on images)
        "gaussian_blur": None, # probability 
        "random_rotation": 0.1, # probability
        "random_crop": 0.1, # probability
        # "random_color_modifs": 0.2, # for CNN, probability
        "patch_randomness": 0.1, # for transformers, probability to NOT take a sample on centerline
        "normalise_patient": True, # for patient network, normalise or not the data, True or False

        # Define training
        "n_epochs": 2, 
        "batch_size": 4, 
        "change_strategy_epoch_l": [-1], # list indicating at which epoch change the optimizer, loss and lr, start with -1 for the first one

        # Define the optimiser
        "optimizer_l": ["SGD"], # list of optimizer to use, "SGD", "Adam", "PESG", "PDSCA"
        "weight_decay": 0.00634788540665404,
        "lr_l": [0.0004493489445028419], # list of the lr
        "SGD_momentum": 0.6566222749612993 , # HP of the SGD
        # "PESG_gamma": None, # HP of the PSEG
        # "PESG_margin": None, # HP of the PSEG
        # "PESG_imratio": None, # HP of the PSEG
        # "Compo_gamma": None, # HP of the PDSCA
        # "Compo_margin": None, # HP of the PDSCA
        # "Compo_imratio": None, # HP of the PDSCA
        # "Compo_beta1": None, # HP of the PDSCA
        # "Compo_beta2": None, # HP of the PDSCA

        # Define the scheduler
        "scheduler_patience": 5, # nb of epochs without improvement before reducing lr
        "scheduler_factor": 0.1, # how much to reduce the lr when plateau (lr*=scheduler_factor)

        # Define the loss
        "criterion_l": ["BCE"], # list of optimizer to use, "BCE", "AUC", "Focal"
        "siamese_pred_loss_ratio": 7.399590987258929e-05, # ratio btw artery MI prediction and global MI prediction losses
        "arteries_pred_loss_ratio": 0.2071422613609661, # ratio btw siamese loss and global MI predicition losses
        "patient_data_loss_ratio": 0.09457471894183037, # ratio btw patient data prediction loss and global MI predicition losses
        # "focal_alpha": None, # HP of the focal loss
        # "focal_gamma": None, # HP of the focal loss
        # "focal_reduction": None, # HP of the focal loss

    }

if __name__ == '__main__':

    # Constants and global variables
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- Load the configuration dictionnary --- #
    train_configuration = train_config_manual
    
    # --- Launch the training and its record --- #
    wandb.init(config=train_configuration, project="dl_mi_pred_transformers")
    train_valid(train_configuration, device, cv_split=None)

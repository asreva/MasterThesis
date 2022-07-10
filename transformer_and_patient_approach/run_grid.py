"""
Aim: run a grid search with the W&B API
Author: Ivan-Daniel Sievering for the LTS4 Lab (EPFL)
"""

# --- Settings --- #
# Libraries
import sys
import argparse
import torch
import wandb
import numpy as np

from train_valid import train_valid
from transfo_and_pat_softmax_net import PatientLevelDNN_PatientDataSoftmax


if __name__ == '__main__':

    # Constants and global variables
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- Train definition --- #
    # load base value (default, will be changed by the grid search)
    train_config =  {
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
        "balance_method": "oversample", # "no", "oversample" and "undersample"
        "train_test_ratio": 0.2, # ratio to use in testing vs in training (also in validation vs in training)
        "test": True, # False: remove test data and then separate train and valid. True: separate test and train (impossible to use CV)
        "normalise": False, # True or False (on images)
        "gaussian_blur": None, # probability 
        "random_rotation": 0.1, # probability
        "random_crop": 0.1, # probability
        # "random_color_modifs": 0.2, # for CNN, probability
        "patch_randomness": 0.1, # for transformers, probability to NOT take a sample on centerline
        "normalise_patient": True, # for patient network, normalise or not the data, True or False

        # Define training
        "n_epochs": 50, 
        "batch_size": 4, 
        "change_strategy_epoch_l": [-1], # list indicating at which epoch change the optimizer, loss and lr, start with -1 for the first one

        # Define the optimiser
        "optimizer_l": ["SGD"], # list of optimizer to use, "SGD", "Adam", "PESG", "PDSCA"
        "weight_decay": 0.00634788540665404,
        "lr_l": [0.0004493489445028419], # list of the lr
        "SGD_momentum": 0.6566222749612993 , # HP of the SGD

        # Define the scheduler
        "scheduler_patience": 5, # nb of epochs without improvement before reducing lr
        "scheduler_factor": 0.1, # how much to reduce the lr when plateau (lr*=scheduler_factor)

        # Define the loss
        "criterion_l": ["BCE"], # list of optimizer to use, "BCE", "AUC", "Focal"
        "siamese_pred_loss_ratio": 7.399590987258929e-05, # ratio btw artery MI prediction and global MI prediction losses
        "arteries_pred_loss_ratio": 0.2071422613609661, # ratio btw siamese loss and global MI predicition losses
        "patient_data_loss_ratio": 0.09457471894183037, # ratio btw patient data prediction loss and global MI predicition losses

    }
    
    # Add void values used by the grid search because we cannot use arrays in the API
    train_config["learning_rate_1"] = None
    train_config["learning_rate_2"] = None

    # --- Set the w&b and launche this iteration of the train --- #

    # Get the parameters of the grid from w&b
    train_config_keys = train_config.keys()
    parser = argparse.ArgumentParser()
    for key in train_config_keys:
        parser.add_argument('--'+key)
    args = parser.parse_args()

    # Apply them to the training dictionnary
    args_dict = vars(args)
    for arg_name, arg_value in args_dict.items():
        if arg_value is not None:
            train_config[arg_name] = arg_value

    # Indicate values that will be changed by the API and change their type
    train_config["lr_l"] = [float(train_config["learning_rate_1"])]
    train_config["weight_decay"] = float(train_config["weight_decay"])
    train_config["dropout"] = float(train_config["dropout"])
    train_config["dropout_patient_net"] = float(train_config["dropout_patient_net"])
    train_config["SGD_momentum"] = float(train_config["SGD_momentum"])
    train_config["siamese_pred_loss_ratio"] = float(train_config["siamese_pred_loss_ratio"])
    train_config["arteries_pred_loss_ratio"] = float(train_config["arteries_pred_loss_ratio"])
    train_config["patient_data_loss_ratio"] = float(train_config["patient_data_loss_ratio"])

    # Run the train and log it
    nb_cv = train_config["nb_cv"]
    all_perf = []
    wandb.init(project="test", config=train_config)
    for i_cv in range(0,nb_cv):
        print("\n\nCross validation {}/{}\n\n".format(i_cv, nb_cv-1)) 
        perf = train_valid(train_config, device, cv_split=[i_cv, nb_cv], grid=True)
        all_perf.append(perf)


    # Sadly we have to wait all the model to have run to compute the mean at each epoch
    # https://towardsdatascience.com/how-i-learned-to-stop-worrying-and-track-my-machine-learning-experiments-d9f2dfe8e4b3
    # https://github.com/wandb/examples/blob/master/examples/wandb-sweeps/sweeps-cross-validation/train-cross-validation.py
    perf_record_skeleton = all_perf[0][0]
    mean_record = [perf_record_skeleton.copy() for i in range(0,train_config["n_epochs"])]
    best_f1, best_f1_valid = 0, 0
    
    for metric in perf_record_skeleton.keys():
        best_acc = []
        for epoch in range(0,train_config["n_epochs"]):
            accumulator=[]

            for i_cv in range(0,nb_cv):
                perf = all_perf[i_cv][epoch][metric]
                if torch.is_tensor(perf):
                    perf = perf.cpu().numpy()
                accumulator.append(perf)

            mean_record[epoch][metric] = np.mean(accumulator)
            mean_record[epoch][metric+"_std"] = np.std(accumulator)
            best_acc.append(np.mean(accumulator))
        
        if metric == "f1":
            best_f1 = np.max(best_acc)
        if metric == "f1_valid":
            best_f1_valid = np.max(best_acc)

    for epoch in range(0, train_config["n_epochs"]):
        wandb.log(mean_record[epoch])
        
    wandb.log({"best_f1":best_f1, "best_f1_valid":best_f1_valid})
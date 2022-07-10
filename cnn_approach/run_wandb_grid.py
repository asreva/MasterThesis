"""
Aim: run a grid search with the W&B API
Author: Ivan-Daniel Sievering for the LTS4 Lab (EPFL)
"""

# --- Settings --- #
# Libraries
import sys
import argparse
from train_valid import train_valid
from configuration_dict import train_configuration_default
from network import MiPredArteryLevel, MiPredArteryLevel_Or
import torch
import wandb
import numpy as np

if __name__ == '__main__':

    # Constants and global variables
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- Initial skeleton definition --- #
    # Define the non grid search parameters
    train_config = train_configuration_default
    
    train_config["nb_cv"] = 5

    train_config["balance_method"] = "oversample"
    train_config["gaussian_blur"] = None
    train_config["normalise"] = False
    train_config["random_rotation"] = 0.2
    train_config["random_crop"] = 0.2
    train_config["random_color_modifs"] = 0.2

    train_config["network_class"] = MiPredArteryLevel_Or
    train_config["init"] = "Xavier Uniform"

    train_config["n_epochs"] = 20
    train_config["batch_size"] = 4
    train_config["change_opti_and_crit_epochs"] = [-1]
    train_config["optimizer_type"] = ["SGD"]
    train_config["criterion_type"] = ["Focal"]
    train_config["scheduler_patience"] = 3
    train_config["scheduler_factor"] = 0.1

    # --- Get the grid searched HP from w&b and apply them --- #

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

    # Convert strings to float when needed --> has to be adatpted each time
    train_config["dropout"] = float(train_config["dropout"])
    train_config["learning_rate"] = [float(train_config["learning_rate_1"])]
    train_config["weight_decay"] = float(train_config["weight_decay"])
    train_config["siamese_prediction_loss_ratio"] = float(train_config["siamese_prediction_loss_ratio"])
    train_config["arteries_prediction_loss_ratio"] = float(train_config["arteries_prediction_loss_ratio"])
    train_config["SGD_momentum"] = float(train_config["SGD_momentum"])
    train_config["focal_alpha"] = float(train_config["focal_alpha"])
    train_config["focal_gamma"] = float(train_config["focal_gamma"])
    train_config["focal_reduction"] = train_config["focal_reduction"]

    # --- Do the training with this HP config --- #
    nb_cv = train_config["nb_cv"]
    wandb.init(project="dl_mi_pred_CNN", config=train_config)
    all_perf = []
    for i_cv in range(0,nb_cv):
        print("\n\nCross validation {}/{}\n\n".format(i_cv, nb_cv-1)) 
        perf = train_valid(train_config, device, cv_split=[i_cv, nb_cv], grid=True)
        all_perf.append(perf)


    # --- Log the obtained perf to W&B --- #
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
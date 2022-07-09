"""
Aim: run the training of a network manually chose
Author: Ivan-Daniel Sievering for the LTS4 Lab (EPFL)
"""

# --- Settings --- #
# Libraries
from train_valid import train_valid
from configuration_dict import train_configuration_default
from network import PatientLevelDNN
import torch
import wandb

if __name__ == '__main__':

    # Constants and global variables
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- Train definition --- #
    train_config = train_configuration_default

    #train_config["dataset_ratio"] = 0.1
    
    #train_config["save_best_net"] = True
    #train_config["load_network"] = "saved_networks/08052022_160158/4_0.0.pt"
    
    train_config["patch_randomness"] = 0.1
    train_config["patch_size_l"] = [64, 64, 64, 64]
    train_config["nb_patch_l"] = [32, 64, 128, 128]

    train_config["balance_method"] = "undersample"
    train_config["gaussian_blur"] = 0.1
    train_config["normalise"] = False
    train_config["random_rotation"] = 0.1
    train_config["random_crop"] = 0.1

    train_config["network_class"] = PatientLevelDNN
    train_config["init"] = "Xavier Uniform"
    train_config["dropout"] = 0.2629837844791735
    
    train_config["SGD_momentum"] = 0.0

    train_config["n_epochs"] = 30
    train_config["batch_size"] = 7
    train_config["optimizer_type"] = ["SGD"]
    train_config["change_opti_and_crit_epochs"] = [-1]
    train_config["learning_rate"] = [0.001763784599544271]
    train_config["weight_decay"] = 0.0006825761471016621
    train_config["criterion_type"] = ["BCE"]
    train_config["siamese_prediction_loss_ratio"] = 0.0004127151816047347
    train_config["arteries_prediction_loss_ratio"] = 0.003832958063701048
    train_config["scheduler_patience"] = 3
    train_config["scheduler_factor"] = 0.1
    
    train_config["PESG_gamma"] = 500
    train_config["PESG_margin"] = 1.0
    train_config["PESG_imratio"] = 0.5
    # train_config["Compo_gamma"] = 500
    # train_config["Compo_margin"] = 1.0
    # train_config["Compo_imratio"] = 0.5
    # train_config["Compo_beta1"] = 0.9
    # train_config["Compo_beta2"] = 0.999

    # --- Launch the training and its record --- #
    wandb.init(config=train_config, project="test")
    train_valid(train_config, device, cv_split=None)

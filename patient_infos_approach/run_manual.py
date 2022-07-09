"""
Aim: run the training of a network manualy defined
Author: Ivan-Daniel Sievering for the LTS4 Lab (EPFL)
"""

# --- Settings --- #
# Libraries
from train_valid import train_valid
from configuration_dict import train_configuration_default
from network import PatientNet
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
    
    train_config["save_best_net"] = True
    #train_config["load_network"] = None

    train_config["nb_neur_per_hidden_layer"] = [50, 10]
    train_config["batch_norm"] = True
    
    train_config["balance_method"] = "oversample"
    train_config["patient_normalisation"] = True

    train_config["network_class"] = PatientNet
    train_config["init"] = "Kaiming Normal"
    train_config["dropout"] = 0.4698083344250593
    
    train_config["PESG_gamma"] = 495
    train_config["PESG_margin"] = 0.8182242792577308
    train_config["PESG_imratio"] = 0.5
    train_config["SGD_momentum"] = 0.2649568158735824

    train_config["n_epochs"] = 500
    train_config["batch_size"] = 32
    train_config["optimizer_type"] = ["SGD", "PESG"]
    train_config["change_opti_and_crit_epochs"] = [-1, 200]
    train_config["learning_rate"] = [0.0083711584721558, 0.06356389270728012]
    train_config["weight_decay"] = 0.0022510349792765486
    train_config["criterion_type"] = ["BCE", "AUC"]
    train_config["scheduler_patience"] = 25
    train_config["scheduler_factor"] = 0.1

    # --- Launch the training and its record --- #
    wandb.init(config=train_config, project="dl_mi_pred_patient")
    train_valid(train_config, device, cv_split=None)

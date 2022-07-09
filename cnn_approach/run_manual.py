"""
Aim: run the training of a network manually chose
Author: Ivan-Daniel Sievering for the LTS4 Lab (EPFL)
"""

# --- Settings --- #
# Libraries
import os
from train_valid import train_valid
from configuration_dict import train_configuration_default
from network import MiPredArteryLevel, MiPredArteryLevel_Or
import torch
# torch.cuda.empty_cache()
# os.environ['TF_ENABLE_ONEDNN_OPTS']="0" # for custom layers
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
    train_config["optimizer_type"] = ["PESG"]
    train_config["criterion_type"] = ["AUC"]
    train_config["learning_rate"] = [0.03294594655686273]
    train_config["scheduler_patience"] = 3
    train_config["scheduler_factor"] = 0.1
    
    train_config["SGD_momentum"] = 0.0
    
    train_config["PESG_gamma"] = 411
    train_config["PESG_imratio"] = 0.5
    train_config["PESG_margin"] = 0.8094985278402564
    
    train_config["dropout"] = 0.0058908445481855734

    train_config["siamese_prediction_loss_ratio"] = 0.0008309253546460818
    train_config["arteries_prediction_loss_ratio"] = 0.00194789849830916
    
    train_config["weight_decay"] = 0.007069669767325502

    # --- Launch the training and its record --- #
    wandb.init(config=train_config, project="dl_mi_pred_CNN")
    train_valid(train_config, device, cv_split=None)

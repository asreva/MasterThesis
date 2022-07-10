"""
Aim: run a cross validation search with the W&B API
Author: Ivan-Daniel Sievering
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

if __name__ == '__main__':
    # Constants and global variables
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == "cuda":
        torch.cuda.empty_cache()
    
    # --- Train definition --- #
    train_config = train_configuration_default

    #train_config["dataset_ratio"] = 1
    train_config["nb_cv"] = 5

    train_config["balance_method"] = "oversample"
    train_config["gaussian_blur"] = None
    train_config["normalise"] = False
    train_config["random_rotation"] = 0.25
    train_config["random_crop"] = 0.25
    train_config["random_color_modifs"] = 0.25

    train_config["network_class"] = MiPredArteryLevel_Or
    train_config["init"] = "Xavier Uniform"

    train_config["n_epochs"] = 20
    train_config["batch_size"] = 4
    train_config["optimizer_type"] = "SGD"
    train_config["criterion_type"] = "BCE"
    train_config["scheduler_patience"] = 3
    train_config["scheduler_factor"] = 0.1

    train_config["dropout"] = 0.2492727621137275

    train_config["siamese_prediction_loss_ratio"] = 0.03420992030432363
    train_config["arteries_prediction_loss_ratio"] = 0.05684324269530545
    train_config["learning_rate"] = 0.01203732270908834
    train_config["weight_decay"] = 0.09605094155386935

    # --- Set the w&b and launche this iteration of the train --- #

    # Run it
    nb_cv = train_config["nb_cv"]
    group_name = str(wandb.util.generate_id())
    for i_cv in range(0,nb_cv):
        print("\n\nCross validation {}/{}\n\n".format(i_cv, nb_cv)) 
        wandb.init(group='experiment-'+group_name, project="dl_mi_pred_CNN", config=train_config)
        train_valid(train_config, device, cv_split=[i_cv, nb_cv])
        wandb.finish()

"""
Aim: run the training of a network manually chose
Author: Ivan-Daniel Sievering for the LTS4 Lab (EPFL)
"""

# --- Settings --- #
# Libraries
from train_valid import train_valid
from configuration_dict import train_configuration_default
from network import MiPredArteryLevel_Or_with_patient
import torch
import wandb

if __name__ == '__main__':

    # Constants and global variables
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- Train definition --- #
    train_config = train_configuration_default

    # train_config["dataset_ratio"] = 0.1
    
    train_config["test"] = False
    train_config["save_best_net"] = True
    #train_config["load_network"] = ["saved_networks/best_cnn_network.pt", "saved_networks/best_ann_patient.pt"]

    train_config["balance_method"] = "undersample"
    train_config["gaussian_blur"] = None
    train_config["normalise"] = False
    train_config["normalise_patient"] = True
    train_config["random_rotation"] = 0.2
    train_config["random_crop"] = 0.2
    train_config["random_color_modifs"] = 0.2

    train_config["network_class"] = MiPredArteryLevel_Or_with_patient
    train_config["init"] = "Xavier Uniform"
    train_config["init_patient"] = "Kaiming Normal"
    train_config["batch_norm_patient"] = True

    train_config["n_epochs"] = 30
    train_config["batch_size"] = 4
    train_config["change_opti_and_crit_epochs"] = [-1]
    train_config["optimizer_type"] = ["PESG"]
    train_config["criterion_type"] = ["AUC"]
    train_config["learning_rate"] = [0.07707886664635955]
    train_config["scheduler_patience"] = 3
    train_config["scheduler_factor"] = 0.1
    
    train_config["dropout"] = 0.1918385192596236 
    train_config["dropout_patient_net"] = 0.3375476811147977 
    train_config["nb_neur_per_hidden_layer_patient"] = [50, 10]

    train_config["siamese_prediction_loss_ratio"] = 0.007362747247516069
    train_config["arteries_prediction_loss_ratio"] = 0.05930066068056261
    train_config["pred_from_patient_data_loss_ratio"] = 0.00642281123823404
    
    train_config["weight_decay"] = 0.0029691679390628956
    
    train_config["PESG_gamma"] = 595 
    train_config["PESG_imratio"] = 0.5
    train_config["PESG_margin"] = 0.9909603835012512

    # --- Launch the training and its record --- #
    wandb.init(config=train_config, project="dl_mi_pred_CNN")
    train_valid(train_config, device, cv_split=None)

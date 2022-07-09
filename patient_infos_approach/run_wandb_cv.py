"""
Aim: run the a cross validation with the W&B API
Author: Ivan-Daniel Sievering for the LTS4 Lab (EPFL)
"""

# --- Settings --- #
# Libraries
import sys
import argparse
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
    
    train_config["nb_cv"] = 5
    
    train_config["nb_neur_per_hidden_layer"] = [50, 10]
    train_config["batch_norm"] = True
    
    train_config["balance_method"] = "oversample"
    train_config["patient_normalisation"] = True

    train_config["network_class"] = PatientNet
    train_config["init"] = "Kaiming Normal"
    train_config["dropout"] = 0.4860521608728981
    
    train_config["PESG_imratio"] = 0.5
    train_config["PESG_margin"] = 0.9329802903122753
    train_config["PESG_gamma"] = 486

    train_config["n_epochs"] = 500
    train_config["batch_size"] = 32
    train_config["optimizer_type"] = ["PESG"]
    train_config["change_opti_and_crit_epochs"] = [-1]
    train_config["learning_rate"] = [0.027592236918177977]
    train_config["weight_decay"] = 0.00376060569238696
    train_config["criterion_type"] = ["AUC"]
    train_config["scheduler_patience"] = 25
    train_config["scheduler_factor"] = 0.1

    # --- Set the w&b and launche this iteration of the train --- #

    # Run it
    nb_cv = train_config["nb_cv"]
    group_name = str(wandb.util.generate_id()) # define the group to stock this CV
    for i_cv in range(0,nb_cv):
        print("\n\nCross validation {}/{}\n\n".format(i_cv, nb_cv-1)) 
        wandb.init(group='experiment-'+group_name, project="dl_mi_pred_patient", config=train_config)
        train_valid(train_config, device, cv_split=[i_cv, nb_cv])
        wandb.finish()

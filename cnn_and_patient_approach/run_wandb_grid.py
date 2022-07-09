"""
Aim: run the a grid search with the W&B API
Author: Ivan-Daniel Sievering for the LTS4 Lab (EPFL)
"""

# --- Settings --- #
# Libraries
import sys
import argparse
from train_valid import train_valid
from configuration_dict import train_configuration_default
from network import MiPredArteryLevel_Or_with_patient
import torch
import wandb
import numpy as np

if __name__ == '__main__':

    # Constants and global variables
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- Train definition --- #
    train_config = train_configuration_default

    train_config["load_network"] = ["saved_networks/best_cnn_network.pt", "saved_networks/best_ann_patient.pt"]
    
    #train_config["dataset_ratio"] = 1
    
    train_config["nb_cv"] = 5

    train_config["balance_method"] = "oversample"
    train_config["gaussian_blur"] = None
    train_config["normalise"] = False
    train_config["normalise_patient"] = True
    train_config["random_rotation"] = 0.2
    train_config["random_crop"] = 0.2
    train_config["random_color_modifs"] = 0.2

    train_config["network_class"] = MiPredArteryLevel_Or_with_patient
    train_config["init"] = None # we are loading network --> we don't need to init
    train_config["init_patient"] = None # we are loading network --> we don't need to init
    train_config["batch_norm_patient"] = True

    train_config["n_epochs"] = 20
    train_config["batch_size"] = 4
    train_config["change_opti_and_crit_epochs"] = [-1]
    train_config["optimizer_type"] = ["PESG"]
    train_config["criterion_type"] = ["AUC"]
    #train_config["learning_rate"] = [0.1]
    train_config["scheduler_patience"] = 3
    train_config["scheduler_factor"] = 0.1
    
    #train_config["PESG_gamma"] = 0.0
    #train_config["PESG_margin"] = 0.0
    train_config["PESG_imratio"] = 0.5
    
    #train_config["dropout"] = 0.2
    #train_config["dropout_patient_net"] = 0.4
    train_config["nb_neur_per_hidden_layer_patient"] = [50, 10]

    #train_config["siamese_prediction_loss_ratio"] = 0.01
    #train_config["arteries_prediction_loss_ratio"] = 0.0001
    #train_config["pred_from_patient_data_loss_ratio"] = 0.01
    
    #train_config["weight_decay"] = 0.0001

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

    # Convert strings to float values
    train_config["learning_rate"] = [float(train_config["learning_rate_1"])]
    train_config["PESG_gamma"] = float(train_config["PESG_gamma"])
    train_config["PESG_margin"] = float(train_config["PESG_margin"])
    train_config["dropout"] = float(train_config["dropout"])
    train_config["dropout_patient_net"] = float(train_config["dropout_patient_net"])
    train_config["siamese_prediction_loss_ratio"] = float(train_config["siamese_prediction_loss_ratio"])
    train_config["arteries_prediction_loss_ratio"] = float(train_config["arteries_prediction_loss_ratio"])
    train_config["pred_from_patient_data_loss_ratio"] = float(train_config["pred_from_patient_data_loss_ratio"])
    train_config["weight_decay"] = float(train_config["weight_decay"])

    # Run the train and log it
    nb_cv = train_config["nb_cv"]
    wandb.init(project="dl_mi_pred_CNN", config=train_config)
    all_perf = []
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
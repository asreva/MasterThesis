"""
Aim: Train and validate (or train and test) a network based on a training configuration (see configuration_dict.py)
Author: Ivan-Daniel Sievering for the LTS4 Lab (EPFL)
"""

# --- Settings --- #
#Libraries
from time import time
from datetime import datetime
import os
import sys
import torch 
import wandb

from datasets import get_data_loaders
from init_net import init_net
from execute_epoch import execute_one_epoch

# --- Functions --- #
def train_valid(train_configuration, device, cv_split=None, grid=False):
    """ 
        Aim: Train and validate a network based on a given train configuration
        
        Parameters:
            - train_configuration: dictionnary defining the parameters of the run (see configuration_dict.py)
            - device: device on which the operations will take place
            - cv_split: None to use the whole data else we are doing k fold crossvalidation and receive [idx, k] where idx is the idx of the current kfold and k the nb of folds
            - grid: if we are doing a grid or not, in case of a grid do not log directly the value to WANB API but accumulate it and then return in order to compute means over various iterations
    """
    
    # Create network and training strategy
    net, criterion_l, scheduler_l, optimizer_l = init_net(train_configuration)
    net.to(device)
    print("\nModel {} on CUDA {}\n".format(str(train_configuration["network_class"]), next(net.parameters()).is_cuda))
    print("Dataset will have the following balancing method: {}\n".format(train_configuration["balance_method"]))
        
    # Get dataloader from the dataset
    train_data_loader, valid_data_loader = get_data_loaders(train_configuration, cv_split)
        
    if grid:
        perf_accumulation = []
        
    # Take the learning strategy at epoch 0
    current_config = 0
    criterion, scheduler, optimizer = criterion_l[current_config], scheduler_l[current_config], optimizer_l[current_config]
    print(" \nTraining starts with:\n{}\n{}\n{}".format(criterion, scheduler, optimizer))    
    
    # Create the folder to save the best networks
    if train_configuration["save_best_net"]:
        folder_name = datetime.now().strftime("%d%m%Y_%H%M%S")
        os.mkdir("saved_networks/"+folder_name)
        best_valid_f1 = 0
    
    # Iterate in epochs
    for epoch in range(0, train_configuration["n_epochs"]):
        # At the right epoch, change the learning strategy
        if (current_config<len(train_configuration["change_strategy_epoch_l"])-1) and epoch == train_configuration["change_strategy_epoch_l"][current_config+1]:
            current_config += 1
            criterion = criterion_l[current_config]
            scheduler = scheduler_l[current_config]
            optimizer = optimizer_l[current_config]
            print("\nTraining changed to:\n{}\n{}\n{}\n".format(criterion, scheduler, optimizer))    
        
        # Train the network for 1 epoch
        train_start_time = time()
        train_perf_dict = execute_one_epoch(net, train_data_loader, train_configuration, optimizer, criterion, scheduler, device,
                                            modify_net=True, PESG=(train_configuration["optimizer_l"][current_config]=="PESG"))
        train_end_time = time()
        
        # Validate its performance
        valid_start_time = time()
        valid_perf_dict = execute_one_epoch(net, valid_data_loader, train_configuration, optimizer, criterion, scheduler, device,
                                            modify_net=False)
        valid_end_time = time()
        
        # If new best performance or last epoch, save the new network
        if train_configuration["save_best_net"]:
            if best_valid_f1 < valid_perf_dict["f1_valid"] or epoch == train_configuration["n_epochs"]-1:
                best_valid_f1 = valid_perf_dict["f1_valid"]
                print("New best f1 valid (or last epoch) with {}".format(best_valid_f1))
                torch.save(net.state_dict(), "saved_networks/"+folder_name+"/"+str(epoch)+"_"+str(best_valid_f1)+".pt")
        
        # Show summary of perfs
        print("\nEpoch {}/{}:".format(epoch, train_configuration["n_epochs"]-1))
        print("Train duration: {:.2f}s | Valid duration: {:.2f}s".format(train_end_time-train_start_time, valid_end_time-valid_start_time))
        print("Train loss {:.2f} | Valid loss {:.2f}".format(train_perf_dict["total_loss"].item(), valid_perf_dict["total_loss_valid"].item()))
        print("Train F1 {:.2f} | Valid F1 {:.2f}".format(train_perf_dict["f1"], valid_perf_dict["f1_valid"]))
        print("Train acc {:.2f} | Valid acc {:.2f}".format(train_perf_dict["accuracy"], valid_perf_dict["accuracy_valid"]))

        # Log or accumulate the results for the wandb API
        train_perf_dict.update(valid_perf_dict) # "Concatenate" train and valid dict for wandb API (only accepts 1 dict)
        if not grid:
            wandb.log(train_perf_dict)
        else:
            perf_accumulation.append(train_perf_dict)
            
    # If we are doing a grid, we accumulated the performance and return to main file to log it to wandb
    if grid:
        return perf_accumulation
        
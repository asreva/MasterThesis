"""
Aim: Train and valid a network based on a training configuration
Author: Ivan-Daniel Sievering for the LTS4 Lab (EPFL)
"""

# --- Settings --- #
#Libraries
import os
from datetime import datetime
from time import time
import torch

import wandb

from datasets import get_data_loaders
from network import init_net
from execute_epoch import execute_one_epoch

# --- Functions --- #
def train_valid(train_configuration, device, cv_split=None, grid=False):
    """ 
        Aim: Train and validate a network based on a given train configuration
        
        Parameters:
            - train_configuration: dictionnary defining the parameters of the run (see configuration_dict.py)
            - device: device on which the operation take place
            - cv_split: if the train-valid is not part of a crossCV validation use None, else indicates to which step (i.e. 0/1/...) of the CV the process is
            - grid: if we are doing a grid or not, in case of a grid do not log directly the value but accumulate it and then return in order to compute means over various iterations
    """
    
    # Create network
    net, criterion_l, scheduler_l, optimizer_l = init_net(train_configuration)
    net.to(device)
    print("\nModel on CUDA {}".format(next(net.parameters()).is_cuda))
        
    # Get dataloader from the dataset
    train_data_loader, valid_data_loader = get_data_loaders(train_configuration, cv_split)
    
    if grid:
        perf_accumulation = []
        
    criterion = criterion_l[0]
    scheduler = scheduler_l[0]
    optimizer = optimizer_l[0]
    print(criterion)
    
    current_config = 0
    
    if train_configuration["save_best_net"]:
        folder_name = datetime.now().strftime("%d%m%Y_%H%M%S")
        os.mkdir("saved_networks/"+folder_name)
        best_valid_f1 = 0
    
    # Iterate in epochs
    for epoch in range(0, train_configuration["n_epochs"]):
        if (current_config<len(train_configuration["change_opti_and_crit_epochs"])-1) and (epoch == train_configuration["change_opti_and_crit_epochs"][current_config+1]):
            print("Change loss and optimizer")
            current_config += 1
            criterion = criterion_l[current_config]
            scheduler = scheduler_l[current_config]
            optimizer = optimizer_l[current_config]
            print(criterion)
        
        # Train the network
        train_start_time = time()
        train_perf_dict = execute_one_epoch(net, train_data_loader, train_configuration, optimizer, criterion, scheduler, device,
                                            modify_net=True) 
        train_end_time = time()
        
        # Validate its performances
        valid_start_time = time()
        valid_perf_dict = execute_one_epoch(net, valid_data_loader, train_configuration, optimizer, criterion, scheduler, device,
                                            modify_net=False)
        valid_end_time = time()

        # Log and alccumulate the performances
        print("\nEpoch {}/{}:".format(epoch, train_configuration["n_epochs"]-1))
        print("Train duration: {:.2f}s | Valid duration: {:.2f}s".format(train_end_time-train_start_time, valid_end_time-valid_start_time))
        print("Train loss {:.2f} | Valid loss {:.2f}".format(train_perf_dict["total_loss"].item(), valid_perf_dict["total_loss_valid"].item()))
        print("Train F1 {:.2f} | Valid F1 {:.2f}".format(train_perf_dict["f1"], valid_perf_dict["f1_valid"]))
        print("Train acc {:.2f} | Valid acc {:.2f}".format(train_perf_dict["accuracy"], valid_perf_dict["accuracy_valid"]))

        train_perf_dict.update(valid_perf_dict)
        
        if not grid:
            wandb.log(train_perf_dict)
        else:
            perf_accumulation.append(train_perf_dict)
            
        if train_configuration["save_best_net"]:
            if best_valid_f1 < valid_perf_dict["f1_valid"] or epoch == train_configuration["n_epochs"]-1:
                best_valid_f1 = valid_perf_dict["f1_valid"]
                print("New best f1 valid (or last epoch) with {}".format(best_valid_f1))
                torch.save(net.state_dict(), "saved_networks/"+folder_name+"/"+str(epoch)+"_"+str(best_valid_f1)+".pt")
            
    if grid:
        return perf_accumulation
    else:
        return None
        
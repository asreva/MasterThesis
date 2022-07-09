"""
Aim: Implement one epoch of training a network to predict MI from patient data
Author: Ivan-Daniel Sievering for the LTS4 Lab (EPFL)
"""

# --- Settings --- #
#Libraries
import torch
import wandb
from loss import compute_full_loss
from metrics import compute_metrics
from time import time

# --- Functions --- #
def execute_one_epoch(net, data_loader, train_configuration, optimizer, criterion, scheduler, device, modify_net):
    """ 
        Aim: Apply a network during a whole epoch, the network may or not modify its weights if in training mode. Performance is recorder and returned.
        
        Parameters:
            - net: model to use
            - data_loader: dataloader to get data
            - train_configuration: dictionnary defining the parameters of the run (see configuration_dict.py)
            - optimizer: optimizer used to modify weights, if train enabled
            - scheduler: scheduler to reduce lr, if train enabled
            - device: device on which the operation take place
            - modify_net: if the network is on train mode or not
        
        Output: a dictionnary with the performances during the epoch and the loss evolution
    """
    
    prediction_l, target_l = [], []
    loss_acc, loss_crit_acc = 0, 0
    
    if modify_net:
        net.train()
    else:
        net.eval()
    
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        
        if modify_net:
            optimizer.zero_grad()
        
        pred = net(data)
        
        loss_crit = compute_full_loss(pred, target, criterion)
        loss = loss_crit
        
        if modify_net:
            loss.backward()
            optimizer.step()
        
        loss_crit_acc += loss_crit.data.detach() # make sure that we take data to not accumulate gradient in this operation
        loss_acc += loss.data.detach()
        
        prediction_l += (pred.flatten()>0.5).detach().flatten().tolist()
        target_l += torch.unsqueeze(target, 1).detach().flatten().tolist()
        
        # Remove all the used tensor from the GPU to avoid memory issues
        del pred
        del loss, loss_crit
        del data, target
        torch.cuda.empty_cache()  
    
    if modify_net and (scheduler is not None):
        scheduler.step(loss_acc)
        
    if not modify_net:
        postfix = "_valid"
    else:
        postfix = ""
    
    loss_dict = {}
    loss_dict["mi_loss"+postfix] = loss_crit_acc
    loss_dict["total_loss"+postfix] = loss_acc

    metrics_dict = compute_metrics(prediction_l, target_l, postfix)
    
    # adds to loss_dict the record from metrics_dict
    loss_dict.update(metrics_dict)

    return loss_dict
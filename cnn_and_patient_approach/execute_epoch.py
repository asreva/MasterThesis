"""
Aim: Implement one epoch from a network on data
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
        
        Output: a dictionnary with the performances during the epoch
    """
    
    prediction_l,target_l = [], []
    prediction_lad_l,target_lad_l = [], []
    prediction_lcx_l,target_lcx_l = [], []
    prediction_rca_l,target_rca_l = [], []
    prediction_patient_l = []
    loss_acc, loss_crit_acc, loss_siam_acc, loss_arteries_acc = 0, 0, 0, 0
    loss_lad_acc, loss_lcx_acc, loss_rca_acc = 0, 0, 0
    loss_patient_acc = 0
    
    if modify_net:
        net.train()
    else:
        net.eval()
    
    before_data_loader = time()
    for batch_idx, (data, patient_data, target) in enumerate(data_loader):

        data, patient_data, target = data.to(device), patient_data.to(device), target.to(device)
        
        if modify_net:
            optimizer.zero_grad()

        pred, lad_pred, lcx_pred, rca_pred, x_lad_pair, x_lcx_pair, x_rca_pair, patient_data_pred = net(data, patient_data)
        
        loss_crit, loss_arteries, loss_siam, loss_patient = compute_full_loss(pred, lad_pred, lcx_pred, rca_pred, x_lad_pair, x_lcx_pair, x_rca_pair, patient_data_pred, target, criterion, train_configuration)
        
        loss_arteries_total = loss_arteries[0] + loss_arteries[1] + loss_arteries[2]
        loss_siam_total = loss_siam[0] + loss_siam[1] + loss_siam[2]
        loss = loss_crit + loss_arteries_total +  loss_siam_total + loss_patient
        
        if modify_net:
            if train_configuration["optimizer_type"] == "PESG":
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            optimizer.step()
        
        loss_crit_acc += loss_crit.data.detach() # make sure that we take data to not accumulate gradient in this operation
        loss_arteries_acc += loss_arteries_total.data.detach()
        loss_siam_acc += loss_siam_total.data.detach()
        loss_lad_acc += loss_arteries[0].data.detach()
        loss_lcx_acc += loss_arteries[1].data.detach()
        loss_rca_acc += loss_arteries[2].data.detach()
        loss_patient_acc += loss_patient.data.detach()
        loss_acc += loss.data.detach()
        
        prediction_l += (pred.flatten()>0.5).detach().flatten().tolist()
        target_l += torch.unsqueeze(target[:, 3], 1).detach().flatten().tolist()
        
        prediction_lad_l += (lad_pred.flatten()>0.5).detach().flatten().tolist()
        target_lad_l += torch.unsqueeze(target[:, 0], 1).detach().flatten().tolist()

        prediction_lcx_l += (lcx_pred.flatten()>0.5).detach().flatten().tolist()
        target_lcx_l += torch.unsqueeze(target[:, 1], 1).detach().flatten().tolist()
        
        prediction_rca_l += (rca_pred.flatten()>0.5).detach().flatten().tolist()
        target_rca_l += torch.unsqueeze(target[:, 2], 1).detach().flatten().tolist()
        
        prediction_patient_l += (patient_data_pred.flatten()>0.5).detach().flatten().tolist()
        
        # Remove all the used tensor from the GPU to avoid memory issues
        del pred, lad_pred, lcx_pred, rca_pred
        del x_lad_pair, x_lcx_pair, x_rca_pair
        del loss, loss_crit, loss_arteries, loss_siam
        del loss_arteries_total, loss_siam_total
        del data, target
        del loss_patient
        torch.cuda.empty_cache()  
    
    if modify_net and (scheduler is not None):
        scheduler.step(loss_acc)
        
    if not modify_net:
        postfix = "_valid"
    else:
        postfix = ""
    
    loss_dict = {}
    loss_dict["mi_loss"+postfix] = loss_crit_acc
    loss_dict["mi_artery_loss"+postfix] = loss_arteries_acc
    loss_dict["siamese_loss"+postfix] = loss_siam_acc
    loss_dict["total_loss"+postfix] = loss_acc
    loss_dict["loss_lad"+postfix] = loss_lad_acc
    loss_dict["loss_lcx"+postfix] = loss_lcx_acc
    loss_dict["loss_rca"+postfix] = loss_rca_acc
    loss_dict["loss_patient"+postfix] = loss_patient_acc

    metrics_dict = compute_metrics(prediction_l, target_l, postfix)
    metrics_dict_lad = compute_metrics(prediction_lad_l, target_lad_l, "_lad"+postfix)
    metrics_dict_lcx = compute_metrics(prediction_lcx_l, target_lcx_l, "_lcx"+postfix)
    metrics_dict_rca = compute_metrics(prediction_rca_l, target_rca_l, "_rca"+postfix)
    metrics_dict_patient = compute_metrics(prediction_patient_l, target_l, "_patient"+postfix)
    
    # adds to loss_dict the record from metrics_dict
    loss_dict.update(metrics_dict) 
    loss_dict.update(metrics_dict_lad) 
    loss_dict.update(metrics_dict_lcx) 
    loss_dict.update(metrics_dict_rca) 
    loss_dict.update(metrics_dict_patient)

    return loss_dict
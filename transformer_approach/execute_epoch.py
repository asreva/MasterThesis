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
            - criterion: classification loss to use
            - device: device on which the operation take place
            - modify_net: if the network is on train mode or not
        
        Output: a dictionnary with the performances during the epoch
    """
    
    prediction_l,target_l = [], []
    prediction_lad_l,target_lad_l = [], []
    prediction_lcx_l,target_lcx_l = [], []
    prediction_rca_l,target_rca_l = [], []
    loss_acc, loss_crit_acc, loss_siam_acc, loss_arteries_acc = 0, 0, 0, 0
    loss_lad_acc, loss_lcx_acc, loss_rca_acc = 0, 0, 0
    
    if modify_net:
        net.train()
    else:
        net.eval()
    
    # For each batch
    for batch_idx, (available_arteries, data, target) in enumerate(data_loader):
        (lad, lcx, rca), target = data, target.to(device)
        
        lad[0], lad[1] = lad[0].to(device), lad[1].to(device)
        lcx[0], lcx[1] = lcx[0].to(device), lcx[1].to(device)
        rca[0], rca[1] = rca[0].to(device), rca[1].to(device)
        
        if modify_net:
            optimizer.zero_grad()

        pred, lad_pred, lcx_pred, rca_pred, x_lad_pair, x_lcx_pair, x_rca_pair = net((lad, lcx, rca))
        
        loss_crit, loss_arteries, loss_siam = compute_full_loss(pred, lad_pred, lcx_pred, rca_pred, x_lad_pair, x_lcx_pair, x_rca_pair, target, criterion, train_configuration["arteries_prediction_loss_ratio"], train_configuration["siamese_prediction_loss_ratio"], available_arteries)
        
        # In some situation the loss is just an int (missing artery) --> not a tensor --> no detach
        try:
            loss_lad_acc += loss_arteries[0].data.detach()
        except:
            try:
                loss_lad_acc += loss_arteries[0][0].data.detach()
            except:
                loss_lad_acc += loss_arteries[0]
        try:
            loss_lcx_acc += loss_arteries[1].data.detach()
        except:
            try:
                loss_lcx_acc += loss_arteries[1][0].data.detach()
            except:
                loss_lcx_acc += loss_arteries[1]
        try:
            loss_rca_acc += loss_arteries[2].data.detach()
        except:
            try:
                loss_rca_acc += loss_arteries[2][0].data.detach()
            except:
                loss_rca_acc += loss_arteries[2]
        
        loss_arteries_total = loss_arteries[0] + loss_arteries[1] + loss_arteries[2]
        loss_siam_total = loss_siam[0] + loss_siam[1] + loss_siam[2]
        loss = loss_crit + loss_arteries_total +  loss_siam_total
        
        # If training, modify the weights
        if modify_net:
            if train_configuration["optimizer_type"] == "PESG":
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            optimizer.step()
            for param in net.parameters():
                param.requires_grad = True
        
        try:
            loss_crit_acc += loss_crit.data.detach()
        except:
            loss_crit_acc += loss_crit[0].data.detach()
        try:
            loss_arteries_acc += loss_arteries_total.data.detach()
        except:
            loss_arteries_acc += loss_arteries_total[0].data.detach()
        try:
            loss_siam_acc += loss_siam_total.data.detach()
        except:
            loss_siam_acc += loss_siam_total[0].data.detach()
        try:
            loss_acc += loss.data.detach()
        except:
            loss_acc += loss[0].data.detach()
        
        prediction_l += (pred.flatten()>0.5).detach().flatten().tolist()
        target_l += torch.unsqueeze(target[:, 3], 1).detach().flatten().tolist()
        
        # Only take the predicitions for arteries that were really present
        for i_b in range(0, len(available_arteries)):
            if available_arteries[i_b, 0]:
                prediction_lad_l += (lad_pred[i_b].flatten()>0.5).detach().flatten().tolist()
                target_lad_l.append(target[i_b, 0].detach().cpu())
            if available_arteries[i_b, 1]:
                prediction_lcx_l += (lcx_pred[i_b].flatten()>0.5).detach().flatten().tolist()
                target_lcx_l.append(target[i_b, 1].detach().cpu())
            if available_arteries[i_b, 2]:
                prediction_rca_l += (rca_pred[i_b].flatten()>0.5).detach().flatten().tolist()
                target_rca_l.append(target[i_b, 2].detach().cpu())
        
        # Remove all the used tensor from the GPU to avoid memory issues
        del pred, lad_pred, lcx_pred, rca_pred
        del x_lad_pair, x_lcx_pair, x_rca_pair
        del loss, loss_crit, loss_arteries, loss_siam
        del loss_arteries_total, loss_siam_total
        del data, target
        torch.cuda.empty_cache()  
    
    if modify_net and (scheduler is not None):
        scheduler.step(loss_acc)
        
    if not modify_net:
        postfix = "_valid"
    else:
        postfix = ""
    
    # Create a dict of the losses and metrics at this epoch
    loss_dict = {}
    loss_dict["mi_loss"+postfix] = loss_crit_acc
    loss_dict["mi_artery_loss"+postfix] = loss_arteries_acc
    loss_dict["siamese_loss"+postfix] = loss_siam_acc
    loss_dict["total_loss"+postfix] = loss_acc
    loss_dict["loss_lad"+postfix] = loss_lad_acc
    loss_dict["loss_lcx"+postfix] = loss_lcx_acc
    loss_dict["loss_rca"+postfix] = loss_rca_acc
    
    #print("Total loss {}: mi loss {} + mi artery loss {} + siamese loss {}\n".format(loss_acc.cpu().numpy()[0], loss_crit_acc.cpu().numpy()[0], loss_arteries_acc, loss_siam_acc))

    metrics_dict = compute_metrics(prediction_l, target_l, postfix)
    metrics_dict_lad = compute_metrics(prediction_lad_l, target_lad_l, "_lad"+postfix)
    metrics_dict_lcx = compute_metrics(prediction_lcx_l, target_lcx_l, "_lcx"+postfix)
    metrics_dict_rca = compute_metrics(prediction_rca_l, target_rca_l, "_rca"+postfix)
    
    # adds to loss_dict the record from metrics_dict
    loss_dict.update(metrics_dict) 
    loss_dict.update(metrics_dict_lad) 
    loss_dict.update(metrics_dict_lcx) 
    loss_dict.update(metrics_dict_rca) 

    return loss_dict
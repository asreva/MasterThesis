"""
Aim: Implement one epoch from a network on data
Author: Ivan-Daniel Sievering for the LTS4 Lab (EPFL)
"""

# --- Settings --- #
#Libraries
import torch
import wandb
from loss import compute_total_loss
from metrics import compute_metrics
from time import time

# --- Functions --- #
def execute_one_epoch(net, data_loader, train_configuration, optimizer, criterion, scheduler, device, modify_net, PESG=False, track_time=False):
    """ 
        Aim: Apply a network during a whole epoch, the network may or not modify its weights if in training mode. Performance is recorder and returned.
        
        Parameters:
            - net: model to use
            - data_loader: dataloader to get data
            - train_configuration: dictionnary defining the parameters of the run (see configuration_dict.py)
            - optimizer: optimizer used to modify weights, if train enabled
            - citerion: how to compute the MI loss
            - scheduler: scheduler to reduce lr, if train enabled
            - device: device on which the operation take place
            - modify_net: if the network is on train mode or not
            - if we are currently using PESG optimiser (different backward loss)
            - track_time: to show or not the time spent on different steps
        
        Output: a dictionnary with the performances during the epoch
    """
    
    # Create list and accumulators for metrics computations, records and time computation
    data_duration, inference_duration, loss_duration, opti_duration, record_duration = 0, 0, 0, 0, 0
    prediction_l, target_l = [], []
    prediction_lad_l, target_lad_l = [], []
    prediction_lcx_l, target_lcx_l = [], []
    prediction_rca_l, target_rca_l = [], []
    prediction_patient_l = []
    loss_acc, loss_crit_acc, loss_siam_acc, loss_arteries_acc = 0, 0, 0, 0
    loss_lad_acc, loss_lcx_acc, loss_rca_acc = 0, 0, 0
    loss_patient_acc = 0
    
    # Allow or not the modification of the network
    if modify_net:
        net.train()
    else:
        net.eval()
    
    before_data_loader = time()
    # Iterate in batches
    for batch_idx, batch in enumerate(data_loader):
        (data, patient_data, target, available_arteries) = batch 
        patient_data = patient_data.to(device)
        target = target.to(device)
        (lad, lcx, rca) = data
        lad[0], lad[1] = lad[0].to(device), lad[1].to(device)
        lcx[0], lcx[1] = lcx[0].to(device), lcx[1].to(device)
        rca[0], rca[1] = rca[0].to(device), rca[1].to(device)
        data = ((lad[0], lad[1]), (lcx[0], lcx[1]), (rca[0], rca[1]))
        
        after_data_loader = time()
        data_duration += after_data_loader-before_data_loader
        
        # Reset gradient
        if modify_net:
            optimizer.zero_grad()
        
        # Infer the network predictions
        before_inference = time()
        pred, lad_pred, lcx_pred, rca_pred, x_lad_pair, x_lcx_pair, x_rca_pair, patient_data_pred = net(data, patient_data)
        
        after_inference = time()
        inference_duration += after_inference-before_inference
        
        # Compute the loss
        before_loss = time()
        total_loss, loss_mi, loss_arteries, loss_siam, loss_patient = compute_total_loss(pred, (lad_pred, lcx_pred, rca_pred), 
                                                                                         (x_lad_pair, x_lcx_pair, x_rca_pair), 
                                                                                         patient_data_pred, target, criterion, 
                                                                                         train_configuration, available_arteries)
        after_loss = time()
        loss_duration += after_loss-before_loss
        
        # Backpropagate the loss to the network
        before_opti = time()
        if modify_net:
            if PESG:
                total_loss.backward(retain_graph=True)
            else:
                total_loss.backward()
            optimizer.step()
        after_opti = time()
        opti_duration += after_opti-before_opti
        
        before_record = time()
        
        # Accumulate the loss (make sure to just take the value and not the tensor with grad, ...)
        loss_acc += total_loss.data.detach()
        loss_crit_acc += loss_mi.data.detach()
        loss_patient_acc += loss_patient.data.detach()
        # if torch.is_tensor(loss_arteries[0]): # may not be tensor if no example of this artery in this batch
        loss_lad_acc += loss_arteries[0].data.detach()
        loss_arteries_acc += loss_arteries[0].data.detach()
        loss_siam_acc += loss_siam[0].data.detach()
        # if torch.is_tensor(loss_arteries[1]): # may not be tensor if no example of this artery in this batch
        loss_lcx_acc += loss_arteries[1].data.detach()
        loss_arteries_acc += loss_arteries[1].data.detach()
        loss_siam_acc += loss_siam[1].data.detach()
        # if torch.is_tensor(loss_arteries[2]): # may not be tensor if no example of this artery in this batch
        loss_rca_acc += loss_arteries[2].data.detach()
        loss_arteries_acc += loss_arteries[2].data.detach()
        loss_siam_acc += loss_siam[2].data.detach()
            
        # Record the target and prediction (Only take the predicitions for arteries that were available)
        target_l += torch.unsqueeze(target[:, 3], 1).detach().flatten().tolist()
        prediction_l += (pred.flatten()>0.5).detach().flatten().tolist()
        prediction_patient_l += (patient_data_pred.flatten()>0.5).detach().flatten().tolist() 
        for i_b in range(0, len(target)):
            if (available_arteries is None) or (available_arteries[i_b, 0]): 
                prediction_lad_l += (lad_pred[i_b].flatten()>0.5).detach().flatten().tolist()
                target_lad_l.append(target[i_b, 0].detach().cpu())
            if (available_arteries is None) or (available_arteries[i_b, 1]):
                prediction_lcx_l += (lcx_pred[i_b].flatten()>0.5).detach().flatten().tolist()
                target_lcx_l.append(target[i_b, 1].detach().cpu())
            if (available_arteries is None) or (available_arteries[i_b, 2]):
                prediction_rca_l += (rca_pred[i_b].flatten()>0.5).detach().flatten().tolist()
                target_rca_l.append(target[i_b, 2].detach().cpu())
                
        # Remove all the used tensor from the GPU (useless I assume)
        del pred, lad_pred, lcx_pred, rca_pred
        del x_lad_pair, x_lcx_pair, x_rca_pair
        del total_loss, loss_mi, loss_arteries, loss_siam
        del data, target
        del loss_patient, patient_data
        del available_arteries
        # torch.cuda.empty_cache()  
        
        after_record = time()
        record_duration += after_record-before_record
        
        before_data_loader = time()
       
    # Scheduler step
    if modify_net and (scheduler is not None):
        scheduler.step(loss_acc)
        
    # Detailed time duration
    if track_time:
        total_duration_epoch = data_duration + inference_duration + loss_duration + opti_duration + record_duration
        print("\nTotal duration: {:.2f}".format(total_duration_epoch))
        print("Data loading duration: {:.2f}".format(data_duration))
        print("Inference duration: {:.2f}".format(inference_duration))
        print("Loss computation duration: {:.2f}".format(loss_duration))
        print("Optimisation duration: {:.2f}".format(opti_duration))
        print("Recording duration: {:.2f}\n".format(record_duration))
    
    
    # Put all the losses in a dict (for WANDB API)
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
    
    # Add the metrics to the same dict (for WANDB API)
    loss_dict.update(compute_metrics(prediction_l, target_l, postfix)) 
    loss_dict.update(compute_metrics(prediction_lad_l, target_lad_l, "_lad"+postfix)) 
    loss_dict.update(compute_metrics(prediction_lcx_l, target_lcx_l, "_lcx"+postfix)) 
    loss_dict.update(compute_metrics(prediction_rca_l, target_rca_l, "_rca"+postfix)) 
    loss_dict.update(compute_metrics(prediction_patient_l, target_l, "_patient"+postfix))

    # Return the performance of the epoch
    return loss_dict
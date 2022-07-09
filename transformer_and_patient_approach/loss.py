"""
Aim: Implement the custom losses and the total loss computation and aggregation
Author: Ivan-Daniel Sievering for the LTS4 Lab (EPFL)
"""

# --- Settings --- #
#Libraries
import torch
import torchvision

# --- Classes --- #
class ContrastiveLoss(torch.nn.Module):
    """
        Aim: Implement an euclidean distance loss between two groups of channels. Used in order to optimize siamese networks that receive the same information and should output the near features.
        
        Functions:
            - Init: initialise the loss
        
            - Forward: compute the loss
                - Parameters: 
                    - output_pair: tupple of (output_1, output_2), each one is a tensor containing the channels coming from one of the siamese
                - Output: the loss
    """
    
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, output_pair_1, output_pair_2):
        euclidean_distance = torch.nn.functional.pairwise_distance(output_pair_1, output_pair_2, keepdim=True)
        loss_contrastive = torch.mean(torch.pow(euclidean_distance, 2))
        return loss_contrastive
    
class FocalLoss(torch.nn.Module):
    """
        Aim: Implement the Focal loss, designed to work with imbalanced data for networks like RetinaNet.
             Implementation based on https://pytorch.org/vision/stable/_modules/torchvision/ops/focal_loss.html
        
        Functions:
            - Init: initialise the loss
                - Parameters (see paper for meaning):
                    - alpha 
                    - gamma
                    - reduction 
        
            - Forward: compute the loss
                - Parameters: 
                    - inputs: list of prediction
                    - targets: list of the target values

                - Output: the loss
    """
    
    def __init__(self, alpha, gamma, reduction):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        return torchvision.ops.focal_loss.sigmoid_focal_loss(inputs, targets, self.alpha, self.gamma, self.reduction)
    
# --- Functions --- #
def compute_global_MI_loss(pred, target, criterion):
    """ 
        Aim: Compute the loss from the global MI prediction
        
        Parameters:
            - pred: global MI prediction
            - target: true MI labels structure (also with artery pred)
            - criterion: criterion to compute the loss 
        
        Output: the MI los
    """

    patient_mi = torch.clone(torch.unsqueeze(target[:, 3], 1))
    return criterion(pred, patient_mi)

def compute_patient_data_MI_loss(pred, target, criterion):
    """ 
        Aim: Compute the loss from the patient data MI prediction
        
        Parameters:
            - pred: patient MI prediction
            - target: true MI labels structure (also with artery pred)
            - criterion: criterion to compute the loss 
        
        Output: the MI los
    """

    patient_mi = torch.clone(torch.unsqueeze(target[:, 3], 1))
    return criterion(pred, patient_mi)

def compute_arteries_MI_loss(arteries_pred, target, criterion, available_arteries, AUC_loss=False):
    """ 
        Aim: Compute the loss for the MI prediction of each artery
        
        Parameters:
            - arteries_pred: arteries MI prediction
            - target: true MI labels for each artery
            - criterion: criterion to compute the loss 
            - available_arteries: for each element of the batch, if the image of this artery was present or not (else, do no compute the loss)
            - AUC_loss: boolean to indicate if AUC loss, that returns a format a bit different
        
        Output: (lad_loss, lcx_loss, rca_loss), loss of each artery predicition
    """

    # Define accumulators
    loss_arteries_lad, loss_arteries_lcx, loss_arteries_rca = torch.as_tensor(0, dtype=torch.float32, device="cuda"), torch.as_tensor(0, dtype=torch.float32, device="cuda"), torch.as_tensor(0, dtype=torch.float32, device="cuda")
    
    # Extract the MI state for each artery
    patient_mi_lad = torch.clone(torch.unsqueeze(target[:, 0], 1))
    patient_mi_lcx = torch.clone(torch.unsqueeze(target[:, 1], 1))
    patient_mi_rca = torch.clone(torch.unsqueeze(target[:, 2], 1))
    
    # For each element, if the artery was present, add its loss.
    for i_b in range(0, len(target)):
        if (available_arteries is None) or (available_arteries[i_b, 0]):
            if not AUC_loss:
                loss_arteries_lad += criterion(arteries_pred[0][i_b], patient_mi_lad[i_b])
            else:
                loss_arteries_lad += criterion(arteries_pred[0][i_b], patient_mi_lad[i_b])[0]
        if (available_arteries is None) or (available_arteries[i_b, 1]):
            if not AUC_loss:
                loss_arteries_lcx += criterion(arteries_pred[1][i_b], patient_mi_lcx[i_b])
            else:
                loss_arteries_lcx += criterion(arteries_pred[1][i_b], patient_mi_lcx[i_b])[0]
        if (available_arteries is None) or (available_arteries[i_b, 2]):
            if not AUC_loss:
                loss_arteries_rca += criterion(arteries_pred[2][i_b], patient_mi_rca[i_b])
            else:
                loss_arteries_rca += criterion(arteries_pred[2][i_b], patient_mi_rca[i_b])[0]
    
    return (loss_arteries_lad, loss_arteries_lcx, loss_arteries_rca)
    
def compute_arteries_siamese_loss(images, available_arteries):
    """ 
        Aim: Compute the siamese loss for the output of the arteries (2 views --> 2 siamese nets)
        
        Parameters:
            - pred: the images that have been computed
            - available_arteries: for each element of the batch, if the image of this artery was present or not (else, do no compute the loss)
        
        Output: (lad_loss, lcx_loss, rca_loss), loss of each artery predicition
    """

    # Extract the data
    x_lad_pair, x_lcx_pair, x_rca_pair = images
    
    # Define the loss and the accumulators
    siamese_loss = ContrastiveLoss()
    loss_siam_lad, loss_siam_lcx, loss_siam_rca = torch.as_tensor(0, dtype=torch.float32, device="cuda"), torch.as_tensor(0, dtype=torch.float32, device="cuda"), torch.as_tensor(0, dtype=torch.float32, device="cuda")
    
    # For each element, if the artery was present, add its loss.
    for i_b in range(0, len(x_lad_pair[0])):
        if (available_arteries is None) or (available_arteries[i_b, 0]):
            loss_siam_lad += siamese_loss(x_lad_pair[0][i_b, :, :], x_lad_pair[1][i_b, :, :])
        if (available_arteries is None) or (available_arteries[i_b, 1]):
            loss_siam_lcx += siamese_loss(x_lcx_pair[0][i_b, :, :], x_lcx_pair[1][i_b, :, :])
        if (available_arteries is None) or (available_arteries[i_b, 2]):
            loss_siam_rca += siamese_loss(x_rca_pair[0][i_b, :, :], x_rca_pair[1][i_b, :, :])
            
    return (loss_siam_lad, loss_siam_lcx, loss_siam_rca)

def compute_total_loss(pred, arteries_pred, images_pairs, patient_data_pred,
                      target, criterion, train_configuration, available_arteries):
    """ 
        Aim: Compute the total loss of a network and aggregate the different losses w.r.t to the selected ratio
        
        Parameters:
            - pred: global MI prediction
            - arteries_pred (lad_pred, lcx_pred, rca_pred): prediction of MI at LAD/LCX/RCA
            - patient_data_pred; prediction based on the patient data
            - images_pairs (x_lad_paid, x_lcx_pair, x_rca_pair): output of each siamese block (LAD/LCX/RCA), each block outputs a tupple with output of view 1 and output of view 2
            - target: tensor with the MI state of the patient (global and artery level)
            - criterion: criterion to compute the loss for the MI prediction
            - train_configuration: dictionnary defining all the parameters (i.e. the ratio between losses, see train_configuration.py)
            - available_arteries: a list [bool, bool, bool] indicating which artery is available or not for this patient
        
        Output: the total loss, the global MI loss, the MI loss at artery level, the siamese loss, the MI loss from patient data
    """ 
    
    # Compute and scale all losses w.r.t. the ratio defined in the structure 
    global_mi_loss = compute_global_MI_loss(pred, target, criterion)
    
    patient_loss = compute_patient_data_MI_loss(patient_data_pred, target, criterion)*train_configuration["patient_data_loss_ratio"]
        
    (loss_arteries_lad, loss_arteries_lcx, loss_arteries_rca) = compute_arteries_MI_loss(arteries_pred, target, criterion, available_arteries, AUC_loss="AUC" in str(criterion))
    loss_arteries_lad, loss_arteries_lcx, loss_arteries_rca = loss_arteries_lad*train_configuration["arteries_pred_loss_ratio"], loss_arteries_lcx*train_configuration["arteries_pred_loss_ratio"], loss_arteries_rca*train_configuration["arteries_pred_loss_ratio"]
    
    (loss_siam_lad, loss_siam_lcx, loss_siam_rca) = compute_arteries_siamese_loss(images_pairs, available_arteries)
    loss_siam_lad, loss_siam_lcx, loss_siam_rca = loss_siam_lad*train_configuration["siamese_pred_loss_ratio"], loss_siam_lcx*train_configuration["siamese_pred_loss_ratio"], loss_siam_rca*train_configuration["siamese_pred_loss_ratio"]
    
    # Sum all the losses
    total_loss = global_mi_loss
    total_loss += patient_loss
    total_loss += loss_arteries_lad + loss_arteries_lcx + loss_arteries_rca
    total_loss += loss_siam_lad + loss_siam_lcx + loss_siam_rca
    
    return total_loss, global_mi_loss, (loss_arteries_lad, loss_arteries_lcx, loss_arteries_rca), (loss_siam_lad, loss_siam_lcx, loss_siam_rca), patient_loss
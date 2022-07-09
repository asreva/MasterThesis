"""
Aim: Implement the loss computation as well as custom losses
Author: Ivan-Daniel Sievering for the LTS4 Lab (EPFL)
"""

# --- Settings --- #
#Libraries
import torch
import torchvision

# --- Classes --- #
class ContrastiveLoss(torch.nn.Module):
    """
        Aim: Implement an euclidean distance loss between two groups of channels. Used in order to optimize siamese networks that receive the same representation.
        
        Functions:
            - Init: initialise the loss
        
            - Forward: compute the loss
                - Parameters: 
                    - output_pair: tupple of (output_1, output_2), each one is a tensor containing the channels coming from one of the siamese
                - Output: the loss
    """
    
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, output_pair):
        euclidean_distance = torch.nn.functional.pairwise_distance(output_pair[0], output_pair[1], keepdim=True)
        loss_contrastive = torch.mean(torch.pow(euclidean_distance, 2))
        return loss_contrastive
    
class FocalLoss(torch.nn.Module):
    """ 
        Aim: Define the class to compute the Focal loss
             SEE https://pytorch.org/vision/stable/_modules/torchvision/ops/focal_loss.html
        
        Parameters: HP of the focal loss
    """
    
    def __init__(self, alpha, gamma, reduction):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        return torchvision.ops.focal_loss.sigmoid_focal_loss(inputs, targets, self.alpha, self.gamma, self.reduction)
    
# --- Functions --- #
def compute_full_loss(pred, lad_pred, lcx_pred, rca_pred, x_lad_pair, x_lcx_pair, x_rca_pair, target, criterion, train_configruation):
    """ 
        Aim: Compute the full loss of the MiPredArteryLevel Model by summing the global MI loss, the MI at artery level loss and the siamese loss
        
        Parameters:
            - pred, lad_pred, lcx_pred, rca_pred: global MI prediction and prediction of MI at LAD/LCX/RCA
            - x_lad_paid, x_lcx_pair, x_rca_pair: output of each siamese block (LAD/LCX/RCA), each block outputs a tupple with output of view 1 and output of view 2
            - target: tensor with the MI state of the patient (global and artery level)
            - criterion: criterion to compute the loss for the MI prediction
            - arteries_prediction_loss_ratio: ratio btw artery MI prediction and global MI prediction losses
            - siamese_prediction_loss_ratio: ratio btw siamese loss and global MI predicition losses
        
        Output: the MI loss, the MI loss at artery level, the siamese loss
    """
    
    # Extract the MI state
    patient_mi_lad = torch.unsqueeze(target[:, 0], 1)
    patient_mi_lcx = torch.unsqueeze(target[:, 1], 1)
    patient_mi_rca = torch.unsqueeze(target[:, 2], 1)
    patient_mi = torch.unsqueeze(target[:, 3], 1)
    
    # Global MI loss
    loss_crit = criterion(pred, patient_mi)

    # Compute the artery MI loss es
    loss_arteries_lad = criterion(lad_pred, patient_mi_lad)*train_configruation["arteries_prediction_loss_ratio"]
    loss_arteries_lcx = criterion(lcx_pred, patient_mi_lcx)*train_configruation["arteries_prediction_loss_ratio"]
    loss_arteries_rca = criterion(rca_pred, patient_mi_rca)*train_configruation["arteries_prediction_loss_ratio"]

    # Compute the siamese losses
    siamese_loss = ContrastiveLoss()
    loss_siam_lad = siamese_loss(x_lad_pair)*train_configruation["siamese_prediction_loss_ratio"]
    loss_siam_lcx = siamese_loss(x_lcx_pair)*train_configruation["siamese_prediction_loss_ratio"]
    loss_siam_rca = siamese_loss(x_rca_pair)*train_configruation["siamese_prediction_loss_ratio"]

    return loss_crit, (loss_arteries_lad, loss_arteries_lcx, loss_arteries_rca) , (loss_siam_lad, loss_siam_lcx, loss_siam_rca)
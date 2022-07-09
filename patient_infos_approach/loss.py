"""
Aim: Implement the loss computation as well as custom losses
Author: Ivan-Daniel Sievering for the LTS4 Lab (EPFL)
"""

# --- Settings --- #
#Libraries
import torch
import torchvision

# --- Classes --- #
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
def compute_full_loss(pred, target, criterion):
    """ 
        Aim: Compute the total loss to predict MI from patient data
    """
    
    loss_crit = criterion(pred, target)

    return loss_crit
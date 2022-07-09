"""
Aim: Implement the network for the MI prediction from CNN on images and ANN on patient data
!!! network adapted for interpretability !!!
Author: Ivan-Daniel Sievering for the LTS4 Lab (EPFL)
"""

# --- Settings --- #
#Libraries
import sys
import torch
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
from loss import FocalLoss
from saved_networks.cnn_network import MiPredArteryLevel_Or
from saved_networks.ann_network import PatientNetOriginal

# --- Functions --- #
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def xavier_uniform_init(m):
    """ 
        Aim: Apply Xavier Uniform initialization to a layer Linar or Conv2d
        
        Parameters:
            - m: the layer
    """
    
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

def xavier_normal_init(m):
    """ 
        Aim: Apply Xavier Normal initialization to a layer Linar or Conv2d
        
        Parameters:
            - m: the layer
    """
    
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
            
def kaiming_uniform_init(m):
    """ 
        Aim: Apply He Uniform initialization to a layer Linar or Conv2d
        
        Parameters:
            - m: the layer
    """
    
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

def kaiming_normal_init(m):
    """ 
        Aim: Apply He Normal initialization to a layer Linar or Conv2d
        
        Parameters:
            - m: the layer
    """
    
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

# --- Classes --- #
class PatientLayerFeature(torch.nn.Module):
    def __init__(self, in_size, out_size, drop, batch_norm=False):
        super(PatientLayerFeature, self).__init__()
        
        self.batch_norm = batch_norm
        
        self.fc = torch.nn.Linear(in_size, out_size)
        if batch_norm:
            self.bn = torch.nn.BatchNorm1d(out_size)
        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(p=drop)
        
    def forward(self, x):
        x = self.fc(x)
        if self.batch_norm:
            x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)

        return x
    
class PatientLayerClassification(torch.nn.Module):
    def __init__(self, in_size, out_size, batch_norm=False):
        super(PatientLayerClassification, self).__init__()
        
        self.fc = torch.nn.Linear(in_size, out_size)
        
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = torch.nn.BatchNorm1d(out_size)
        
        self.sigmoid = torch.nn.Sigmoid()       
        
    def forward(self, x):
        x = self.fc(x)
        if self.batch_norm:
            x = self.bn(x)
        x = self.sigmoid(x)

        return x
    
class PatientNet(torch.nn.Module):

    def __init__(self, train_configuration):
        super(PatientNet, self).__init__()
        self.nb_hidden_layer = len(train_configuration["nb_neur_per_hidden_layer_patient"])-1
        
        self.layer_l = []
        self.layer_l.append(PatientLayerFeature(11, train_configuration["nb_neur_per_hidden_layer_patient"][0], train_configuration["dropout_patient_net"], train_configuration["batch_norm_patient"]))
        
        for i in range(0, self.nb_hidden_layer):
            self.layer_l.append(PatientLayerFeature(train_configuration["nb_neur_per_hidden_layer_patient"][i], train_configuration["nb_neur_per_hidden_layer_patient"][i+1], train_configuration["dropout_patient_net"], train_configuration["batch_norm_patient"]))
            
        self.layer_l = torch.nn.ModuleList(self.layer_l)
        
        self.pred_layer = PatientLayerClassification(train_configuration["nb_neur_per_hidden_layer_patient"][-1], 1, train_configuration["batch_norm_patient"])
  
    def forward(self, x):
        for layer in self.layer_l:
            x = layer(x)
   
        patient_pred = self.pred_layer(x)
        
        return x # , patient_pred

class SiameseArteryAnalysis_Or_patient(torch.nn.Module):
    """
        Aim: A block that extract features from the two views of an artery with ResNet18 siamese net. The block also predicts the MI at the artery level. 
        
        Functions:
            - Init: initialise the block
        
            - Forward: analyse the image
                - Parameters: 
                    - x1: view 1 of the artery (image+mask)
                    - x2: view 2 of the artery (image+mask)
                - Output: x1 (feature extraction of view 1), x2 (feature extraction of view2), pred (MI prediction in this artery)
    """
    
    def __init__(self, train_config, two_neur_out, artery_only_level, patient_data):
        super(SiameseArteryAnalysis_Or_patient, self).__init__()
        
        # avoid HTTP error 403 in some cases
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True 
        
        # Get resnet18
        resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        # Remove the head
        self.resnet18_without_head = torch.nn.Sequential(*(list(resnet18.children())[:-3]))
        # Convert 3 channels input to 2 channels input (mask + image)
        custom_conv = torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet18_without_head[0] = custom_conv
        
        # Define the classification layer
        if train_config["dropout"] is not None:
            self.drop = torch.nn.Dropout(p=train_config["dropout"])
        else:
            self.drop = None
        self.artery_pred = torch.nn.Linear(in_features=522, out_features=1, bias=True)
        
        self.artery_only_level = artery_only_level
        self.two_neur_out = two_neur_out
        
        self.patient_data = patient_data
        
    def forward(self, x):
        x1 = x[:, 0, :, :, :]
        x2 = x[:, 1, :, :, :]
        
        # Extract features in both image with the same net
        x1 = self.resnet18_without_head(x1)
        x2 = self.resnet18_without_head(x2)
        
        # Predict MI at artery level
        x = torch.cat((x1, x2), dim=1)
        x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=(1,1))
        x = torch.flatten(x, start_dim=1)
        if self.drop is not None:
            x = self.drop(x)
            
        x = torch.cat((x, self.patient_data), dim=1) # CHECK THIS LINE
        
            
        pred = self.artery_pred(x)
        pred = torch.sigmoid(pred)
        
        if self.two_neur_out and self.artery_only_level:
            pred_classes = torch.ones(1,2) # 1 per batch (interpretability) and 2 values
            pred_classes[0, 0] = 1-pred
            pred_classes[0, 1] = pred
            return pred_classes
        else:
            return pred

        return pred
    
class MiPredArteryLevel_Or_with_patient(torch.nn.Module):
    """
        Aim: A network that predicts global MI (and artery MI) based on three artery level blocks prediction (MAX). No common feature analysis is done there.
        
        Functions:
            - Init: initialise the block
                - Parameters:
                    - train_configuration: dictionnary defining the parameters of the run (see configuration_dict.py)
        
            - Forward: analyse the image
                - Parameters: 
                    - x: full input
                - Output: 
                    - pred, lad_pred, lcx_pred, rca_pred: global MI prediction and prediction of MI at LAD/LCX/RCA
                    - x_lad_paid, x_lcx_pair, x_rca_pair: output of each siamese block (LAD/LCX/RCA), each block outputs a tupple with output of view 1 and output of view 2 --> will be used to compute siamese loss
    """

    def __init__(self, train_config, two_neur_out, artery_only_level, patient_data):
        super(MiPredArteryLevel_Or_with_patient, self).__init__()
        
        # Construct the six artery analysis CNN block (3 artery)
        self.resnet_lad = SiameseArteryAnalysis_Or_patient(train_config, two_neur_out, artery_only_level, patient_data)
        self.resnet_lcx = SiameseArteryAnalysis_Or_patient(train_config, two_neur_out, artery_only_level, patient_data)
        self.resnet_rca = SiameseArteryAnalysis_Or_patient(train_config, two_neur_out, artery_only_level, patient_data)
        
        
        self.two_neur_out = two_neur_out
        
    def reset_classification_layers(self):
        torch.nn.init.xavier_uniform_(self.resnet_lad.artery_pred.weight)
        torch.nn.init.xavier_uniform_(self.resnet_lcx.artery_pred.weight)
        torch.nn.init.xavier_uniform_(self.resnet_rca.artery_pred.weight)
        
        self.resnet_lad.artery_pred.weight.data.fill_(0.01)
        self.resnet_lcx.artery_pred.weight.data.fill_(0.01)
        self.resnet_rca.artery_pred.weight.data.fill_(0.01)

    def forward(self, x):
        
        # Extract each view of each artery (each tensor is Cx2x1525x1524)
        x_lad = x[:, 0, :, :, :, :]
        x_lcx = x[:, 1, :, :, :, :]
        x_rca = x[:, 2, :, :, :, :]
        
        # Treat each view of each artery separatly and get prediction (each tensor is Cx256x96x96)
        lad_pred = self.resnet_lad(x_lad)
        lcx_pred = self.resnet_lcx(x_lcx)
        rca_pred = self.resnet_rca(x_rca)
           
        # Patient has MI if any of the artery has MI
        pred = torch.stack([lad_pred, lcx_pred, rca_pred], dim=1)
        pred = pred.max(dim=1).values # max return values and indices
        
        if self.two_neur_out:
            pred_classes = torch.ones(1,2) # 1 per batch (interpretability) and 2 values
            pred_classes[0, 0] = 1-pred
            pred_classes[0, 1] = pred
            return pred_classes
        else:
            return pred


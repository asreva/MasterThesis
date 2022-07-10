"""
Aim: Implement the network for the MI prediction from patient data
Author: Ivan-Daniel Sievering for the LTS4 Lab (EPFL)
"""

# --- Settings --- #
#Libraries
import sys
import torch
from torch import nn, einsum
from einops import rearrange, repeat
import torch.nn.functional as F
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
from loss import FocalLoss

# --- Functions --- #
def count_parameters(model):
    """ 
        Aim: Returns the number of trainable parameters
        
        Parameters:
            - model: the model
            
        Output: the number of parameters
    """
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
class PatientLayerFeature(nn.Module):
    """ 
        Aim: Define an ANN layer: FC (-> bn) -> ReLU -> dropout
        
        Parameters:
            - in_size, out_size: nb of features in/out
            - drop: dropout intensity
            - batch_norm: activte or not the dropout
    """
    
    def __init__(self, in_size, out_size, drop, batch_norm=False):
        super(PatientLayerFeature, self).__init__()
        
        self.batch_norm = batch_norm
        
        self.fc = nn.Linear(in_size, out_size)
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_size)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=drop)
        
    def forward(self, x):
        x = self.fc(x)
        if self.batch_norm:
            x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)

        return x
    
class PatientLayerClassification(nn.Module):
    """ 
        Aim: Define the classification ANN layer: FC (-> bn) -> Sigmoid
        
        Parameters:
            - in_size, out_size: nb of features in/out
            - batch_norm: activte or not the dropout
    """
    
    def __init__(self, in_size, out_size, batch_norm=False):
        super(PatientLayerClassification, self).__init__()
        
        self.fc = nn.Linear(in_size, out_size)
        
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_size)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc(x)
        if self.batch_norm:
            x = self.bn(x)
        x = self.sigmoid(x)

        return x

class PatientNet(nn.Module):
    """ 
        Aim: Define the ANN to predict MI from patient data
        
        Parameters:
            - train_configuration: dictionnary defining the parameters of the run (see configuration_dict.py)
    """

    def __init__(self, train_configuration):
        super(PatientNet, self).__init__()
        
        self.nb_hidden_layer = len(train_configuration["nb_neur_per_hidden_layer"])-1
        self.layer_l = []
        
        # Create first layer
        self.layer_l.append(PatientLayerFeature(11, train_configuration["nb_neur_per_hidden_layer"][0], train_configuration["dropout"], train_configuration["batch_norm"]))
        
        # Add hidden layers
        for i in range(0, self.nb_hidden_layer):
            self.layer_l.append(PatientLayerFeature(train_configuration["nb_neur_per_hidden_layer"][i], train_configuration["nb_neur_per_hidden_layer"][i+1], train_configuration["dropout"], train_configuration["batch_norm"]))
            
        # Add classification layer
        self.layer_l.append(PatientLayerClassification(train_configuration["nb_neur_per_hidden_layer"][-1], 1, train_configuration["batch_norm"]))
        
        self.layer_l = nn.ModuleList(self.layer_l)
        
    def forward(self, x):
        
        for layer in self.layer_l:
            x = layer(x)
        
        return x
    
def init_net(train_configuration):
    """ 
        Aim: Initialise a network and its optimisation (loss, optimiser ...) based on the given train configuration
        
        Parameters:
            - train_configuration: dictionnary defining the parameters of the run (see configuration_dict.py)
        
        Output: the network, the list of losses, the list of optimisers, the list of schedulers
    """
    
    # Create the network or load one
    if train_configuration["load_network"] is None:
        net = train_configuration["network_class"](train_configuration)
    else:
        print("Loading existing network {}".format(train_configuration["load_network"]))
        net = train_configuration["network_class"](train_configuration)
        net.load_state_dict(torch.load(train_configuration["load_network"]))
    print("Nb of parameters is {}".format(count_parameters(net)))
    
    criterion_l = []
    optimizer_l = []
    scheduler_l = []
    
    # For each criterion/loss add it in the list
    for i in range(len(train_configuration["criterion_type"])):
        # Choose criteraion
        if train_configuration["criterion_type"][i] == "BCE":
            criterion_l.append(torch.nn.BCELoss())
        elif train_configuration["criterion_type"][i] == "AUC":
            criterion_l.append(AUCMLoss(imratio=train_configuration["PESG_imratio"]))
        elif train_configuration["criterion_type"][i] == "Focal":
            criterion_l.append(FocalLoss(train_configuration["focal_alpha"], train_configuration["focal_gamma"], train_configuration["focal_reduction"]))
        else:
            print("Criterion not found. Exit code (network.py).")
            sys.exit()

        # Choose the optimizer and its scheduler 
        if train_configuration["optimizer_type"][i] == "SGD":
            optimizer_l.append(torch.optim.SGD(net.parameters(), lr=train_configuration["learning_rate"][i], weight_decay=train_configuration["weight_decay"], momentum=train_configuration["SGD_momentum"]))
            scheduler_l.append(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_l[i], mode='min', factor=train_configuration["scheduler_factor"], patience=train_configuration["scheduler_patience"], verbose=True))
        elif train_configuration["optimizer_type"][i] == "Adam":
            optimizer_l.append(torch.optim.Adam(net.parameters(), lr=train_configuration["learning_rate"][i], weight_decay=train_configuration["weight_decay"]))
            scheduler_l.append(None) # no scheduler for adam
        elif train_configuration["optimizer_type"][i] == "PESG":
            # imratio is percentage of positive cases, gamma and margin from their example
            optimizer_l.append(PESG(net, lr=train_configuration["learning_rate"][i], weight_decay=train_configuration["weight_decay"],
                             a=criterion_l[i].a, b=criterion_l[i].b, alpha=criterion_l[i].alpha, imratio=train_configuration["PESG_imratio"], gamma=train_configuration["PESG_gamma"], margin=train_configuration["PESG_margin"]))
            scheduler_l.append(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_l[i], mode='min', factor=train_configuration["scheduler_factor"], patience=train_configuration["scheduler_patience"], verbose=True))
        else:
            print("Optimizer not found. Exit code (network.py).")
            sys.exit()

    # Apply weights and biases initialisation if not loading an existing network
    if train_configuration["load_network"] is None:
        if train_configuration["init"] == "Xavier Uniform":
            net.apply(xavier_uniform_init)
        elif train_configuration["init"] == "Xavier Normal":
            net.apply(xavier_normal_init)
        elif train_configuration["init"] == "Kaiming Uniform":
            net.apply(kaiming_uniform_init)
        elif train_configuration["init"] == "Kaiming Normal":
            net.apply(kaiming_normal_init)
        else:
            print("Unknown Initialisation")
            sys.exit()

    return net, criterion_l, scheduler_l, optimizer_l
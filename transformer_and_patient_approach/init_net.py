"""
Aim: Implement the intialisation steps for a network (and its loss, optimizer, ...)
Author: Ivan-Daniel Sievering for the LTS4 Lab (EPFL)
"""

# --- Settings --- #
#Libraries
import sys
import torch
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
# from libauc.losses import CompositionalLoss
# from libauc.optimizers import PDSCA
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
            
def init_net(train_configuration):
    """ 
        Aim: Initialise a network and its optimisation (loss, optimiser ...) based on the given train configuration
        
        Parameters:
            - train_configuration: dictionnary defining the parameters of the run (see configuration_dict.py)
        
        Output: the network, the list of losses, the list of optimiser, the lsit of scheduler
    """
    
    # Create the network or load an existing one
    if train_configuration["load_network"] is None:
        net = train_configuration["network_class"](train_configuration)
    else:
        print("Loading existing network {}".format(train_configuration["load_network"]))
        net = train_configuration["network_class"](train_configuration)
        net.load_state_dict(torch.load(train_configuration["load_network"]))
        
    print("Nb of parameters is {}".format(count_parameters(net)))
    
    # Add iteratevly the different criterion, optimizer, ... that will be used
    criterion_l, optimizer_l, scheduler_l = [], [], []
    
    # For each change
    for i in range(len(train_configuration["change_strategy_epoch_l"])):
        # Choose criterion
        if train_configuration["criterion_l"][i] == "BCE":
            criterion_l.append(torch.nn.BCELoss())
        elif train_configuration["criterion_l"][i] == "AUC":
            criterion_l.append(AUCMLoss(imratio=train_configuration["PESG_imratio"]))
        elif train_configuration["criterion_l"][i] == "Focal":
            criterion_l.append(FocalLoss(train_configuration["focal_alpha"], train_configuration["focal_gamma"], train_configuration["focal_reduction"]))
        elif train_configuration["criterion_l"][i] == "CompositionalLoss":
            criterion_l.append(CompositionalLoss(imratio=train_configuration["Compo_imratio"]))
        else:
            print("Criterion not found. Exit code (network.py).")
            sys.exit()

        # Choose the optimizer and its scheduler 
        if train_configuration["optimizer_l"][i] == "SGD":
            optimizer_l.append(torch.optim.SGD(net.parameters(), lr=train_configuration["lr_l"][i], weight_decay=train_configuration["weight_decay"], momentum=train_configuration["SGD_momentum"]))
            scheduler_l.append(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_l[i], mode='min', factor=train_configuration["scheduler_factor"], patience=train_configuration["scheduler_patience"], verbose=True))

        elif train_configuration["optimizer_l"][i] == "Adam":
            optimizer_l.append(torch.optim.Adam(net.parameters(), lr=train_configuration["lr_l"][i], weight_decay=train_configuration["weight_decay"]))
            scheduler_l.append(None) # no scheduler for adam

        elif train_configuration["optimizer_l"][i] == "PESG":
            # imratio is percentage of positive cases, gamma and margin from their example
            optimizer_l.append(PESG(net, lr=train_configuration["lr_l"][i], weight_decay=train_configuration["weight_decay"],
                             a=criterion_l[i].a, b=criterion_l[i].b, alpha=criterion_l[i].alpha, imratio=train_configuration["PESG_imratio"], gamma=train_configuration["PESG_gamma"], margin=train_configuration["PESG_margin"]))
            scheduler_l.append(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_l[i], mode='min', factor=train_configuration["scheduler_factor"], patience=train_configuration["scheduler_patience"], verbose=True))
            
        elif train_configuration["optimizer_l"][i] == "PDSCA":
            # imratio is percentage of positive cases, gamma and margin from their example
            optimizer_l.append(PDSCA(net, lr=train_configuration["lr_l"][i], weight_decay=train_configuration["weight_decay"],
                             a=criterion_l[i].a, b=criterion_l[i].b, alpha=criterion_l[i].alpha, imratio=train_configuration["Compo_imratio"], gamma=train_configuration["Compo_gamma"], margin=train_configuration["Compo_margin"], beta1=train_configuration["Compo_beta1"], beta2=train_configuration["Compo_beta2"]))
            scheduler_l.append(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_l[i], mode='min', factor=train_configuration["scheduler_factor"], patience=train_configuration["scheduler_patience"], verbose=True))

        else:
            print("Optimizer not found. Exit code (network.py).")
            sys.exit()

    # Apply weights and biases initialisation if not loading an existing network
    if train_configuration["load_network"] is None:
        # Apply weights and biases initialisation
        if train_configuration["weights_init"] == "Xavier Uniform":
            net.apply(xavier_uniform_init)
        elif train_configuration["weights_init"] == "Xavier Normal":
            net.apply(xavier_normal_init)
        elif train_configuration["weights_init"] == "Kaiming Uniform":
            net.apply(kaiming_uniform_init)
        elif train_configuration["weights_init"] == "Kaiming Normal":
            net.apply(kaiming_normal_init)
        else:
            print("Unknown Initialisation (init.py)")
            sys.exit()

        # Apply weights and biases initialisation for the patient subnet (maybe different)
        if train_configuration["weights_init_patient"] == "Xavier Uniform":
            net.patient_net.apply(xavier_uniform_init)
        elif train_configuration["weights_init_patient"] == "Xavier Normal":
            net.patient_net.apply(xavier_normal_init)
        elif train_configuration["weights_init_patient"] == "Kaiming Uniform":
            net.patient_net.apply(kaiming_uniform_init)
        elif train_configuration["weights_init_patient"] == "Kaiming Normal":
            net.patient_net.apply(kaiming_normal_init)
        else:
            print("Unknown Initialisation (init.py)")
            sys.exit()

    return net, criterion_l, scheduler_l, optimizer_l
"""
Aim: Implement the CNN for the MI prediction from artery images
Author: Ivan-Daniel Sievering for the LTS4 Lab (EPFL)
"""

# --- Settings --- #
#Libraries
import sys
import torch
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
from loss import FocalLoss

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
class SiameseArteryAnalysis_Or(torch.nn.Module):
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
    
    def __init__(self, train_config=None):
        super(SiameseArteryAnalysis_Or, self).__init__()
        
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
        self.artery_pred = torch.nn.Linear(in_features=512, out_features=1, bias=True)
        
    def forward(self, x1, x2):
        # Extract features in both image with the same net
        x1 = self.resnet18_without_head(x1)
        x2 = self.resnet18_without_head(x2)
        
        # Predict MI at artery level
        x = torch.cat((x1, x2), dim=1)
        x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=(1,1))
        x = torch.flatten(x, start_dim=1)
        if self.drop is not None:
            x = self.drop(x)
        pred = self.artery_pred(x)
        pred = torch.sigmoid(pred)

        return x1, x2, pred
    
class SiameseArteryAnalysis(torch.nn.Module):
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
    
    def __init__(self, train_config=None):
        super(SiameseArteryAnalysis, self).__init__()
        
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
        self.artery_pred = torch.nn.Linear(in_features=512, out_features=1, bias=True)
        
    def forward(self, x1, x2):
        # Extract features in both image with the same net
        x1 = self.resnet18_without_head(x1)
        x2 = self.resnet18_without_head(x2)
        
        # Predict MI at artery level
        x = torch.cat((x1, x2), dim=1)
        x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=(1,1))
        x = torch.flatten(x, start_dim=1)
        if self.drop is not None:
            x = self.drop(x)
        pred = self.artery_pred(x)
        pred = torch.sigmoid(pred)

        return x1, x2, pred
    
class ResConvBlock(torch.nn.Module):
    """
        Aim: A block extract features from the concatenation of all the views of all the arteries. Convolutional layers with some skip connections and activation. 
        
        Functions:
            - Init: initialise the block
                - Parameters:
                    - nb_ch: number of channels of the input
        
            - Forward: analyse the image
                - Parameters: 
                    - x: full input
                - Output: x (the feature extracted input)
    """
    
    def __init__(self, nb_ch):
        super(ResConvBlock, self).__init__()
        
        # Parameters values based on layers of resnet 50
        self.conv2d_1 = torch.nn.Conv2d(nb_ch, nb_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batch_1 = torch.nn.BatchNorm2d(nb_ch, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d_2 = torch.nn.Conv2d(nb_ch, nb_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batch_2 = torch.nn.BatchNorm2d(nb_ch, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
    def forward(self, x):
        x_in = x
        
        # Extract features
        x = self.conv2d_1(x)
        x = self.batch_1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2d_2(x)
        x = self.batch_2(x)
        
        # Skip connect and activate
        x = x + x_in
        x = torch.nn.functional.relu(x)

        return x
    
class MiPredArteryLevel(torch.nn.Module):
    """
        Aim: A network that predicts global MI (and artery MI) based on three artery level blocks and feature extraction of their contatenation. 
        
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

    def __init__(self, train_config):
        super(MiPredArteryLevel, self).__init__()
        
        # Construct the six artery analysis CNN block (3 artery)
        self.resnet_lad = SiameseArteryAnalysis()
        self.resnet_lcx = SiameseArteryAnalysis()
        self.resnet_rca = SiameseArteryAnalysis()
        
        # construct the size reduction filter
        self.max_pool = torch.nn.MaxPool2d((5,5))
        
        # Construct the merged analysis CNN block
        self.res_conv_block = ResConvBlock(1536)
        
        # Dropout layer
        self.drop = torch.nn.Dropout(p=train_config["dropout"])
        
        # Construct the MI classification block
        self.classification = torch.nn.Linear(in_features=1536, out_features=1, bias=True)
    
    def reset_classification_layers(self):
        torch.nn.init.xavier_uniform_(self.classification.weight)
        torch.nn.init.xavier_uniform_(self.resnet_lad.artery_pred.weight)
        torch.nn.init.xavier_uniform_(self.resnet_lcx.artery_pred.weight)
        torch.nn.init.xavier_uniform_(self.resnet_rca.artery_pred.weight)
        
        self.classification.bias.data.fill_(0.01)
        self.resnet_lad.artery_pred.bias.data.fill_(0.01)
        self.resnet_lcx.artery_pred.bias.data.fill_(0.01)
        self.resnet_rca.artery_pred.bias.data.fill_(0.01)
    
    def forward(self, x):    
        # Extract each view of each artery (each tensor is Cx2x1525x1524)
        x_lad_1 = x[:, 0, 0, :, :, :]
        x_lad_2 = x[:, 0, 1, :, :, :]
        x_lcx_1 = x[:, 1, 0, :, :, :]
        x_lcx_2 = x[:, 1, 1, :, :, :]
        x_rca_1 = x[:, 2, 0, :, :, :]
        x_rca_2 = x[:, 2, 1, :, :, :]
        
        # Treat each view of each artery separatly and get prediction (each tensor is Cx256x96x96)
        x_lad_1, x_lad_2, lad_pred = self.resnet_lad(x_lad_1, x_lad_2)
        x_lcx_1, x_lcx_2, lcx_pred = self.resnet_lcx(x_lcx_1, x_lcx_2)
        x_rca_1, x_rca_2, rca_pred = self.resnet_rca(x_rca_1, x_rca_2)
           
        # Concatenate the analysis of each channel (the tensor is Cx1536x96x96)
        x_merged = torch.cat((x_lad_1, x_lad_2, x_lcx_1, x_lcx_2, x_rca_1, x_rca_2), dim=1)
        
        x_merged = self.max_pool(x_merged)
        
        # Analyse the concatenation (the tensor is Cx512x96x96 and then Cx2048x24x24)
        x_merged = self.res_conv_block(x_merged)
        
        # Make the prediciton (the tensor is Cx1)
        x_merged = torch.nn.functional.adaptive_avg_pool2d(x_merged, output_size=(1,1))
        x_merged = torch.flatten(x_merged, start_dim=1)
        x_merged = self.drop(x_merged)
        pred = self.classification(x_merged)
        pred = torch.sigmoid(pred)
        
        return pred, lad_pred, lcx_pred, rca_pred, \
                (x_lad_1, x_lad_2), (x_lcx_1, x_lcx_2), (x_rca_1, x_rca_2)
    
class MiPredArteryLevel_Or(torch.nn.Module):
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

    def __init__(self, train_config):
        super(MiPredArteryLevel_Or, self).__init__()
        
        # Construct the six artery analysis CNN block (3 artery)
        self.resnet_lad = SiameseArteryAnalysis_Or(train_config)
        self.resnet_lcx = SiameseArteryAnalysis_Or(train_config)
        self.resnet_rca = SiameseArteryAnalysis_Or(train_config)

    def forward(self, x):    
        # Extract each view of each artery (each tensor is Cx2x1525x1524)
        x_lad_1 = x[:, 0, 0, :, :, :]
        x_lad_2 = x[:, 0, 1, :, :, :]
        x_lcx_1 = x[:, 1, 0, :, :, :]
        x_lcx_2 = x[:, 1, 1, :, :, :]
        x_rca_1 = x[:, 2, 0, :, :, :]
        x_rca_2 = x[:, 2, 1, :, :, :]
        
        # Treat each view of each artery separatly and get prediction (each tensor is Cx256x96x96)
        x_lad_1, x_lad_2, lad_pred = self.resnet_lad(x_lad_1, x_lad_2)
        x_lcx_1, x_lcx_2, lcx_pred = self.resnet_lcx(x_lcx_1, x_lcx_2)
        x_rca_1, x_rca_2, rca_pred = self.resnet_rca(x_rca_1, x_rca_2)
           
        # Patient has MI if any of the artery has MI
        pred = torch.stack([lad_pred, lcx_pred, rca_pred], dim=1)
        pred = pred.max(dim=1).values # max return values and indices
        
        return pred, lad_pred, lcx_pred, rca_pred, \
                (x_lad_1, x_lad_2), (x_lcx_1, x_lcx_2), (x_rca_1, x_rca_2)

    
def init_net(train_configuration):
    """ 
        Aim: Initialise a network and its optimisation (optimiser, ...) based on the given train configuration
        
        Parameters:
            - train_configuration: dictionnary defining the parameters of the run (see configuration_dict.py)
        
        Output: the network, the train scheduler, the optimizer
    """
    
    # Create the network
    net = train_configuration["network_class"](train_configuration)
    print("Nb of parameters is {}".format(count_parameters(net)))
    
    criterion_l = []
    optimizer_l = []
    scheduler_l = []
    
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

    # Apply weights and biases initialisation
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
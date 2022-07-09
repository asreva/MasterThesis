"""
Aim: Implement the network for the MI prediction at patient level (based on artery blocks)
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
class PatientLayerFeature(torch.nn.Module):
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
    """ 
        Aim: Define the classification ANN layer: FC (-> bn) -> Sigmoid
        
        Parameters:
            - in_size, out_size: nb of features in/out
            - batch_norm: activte or not the dropout
    """
    
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
    """ 
        Aim: Define the ANN to predict MI from patient data
        
        Parameters:
            - train_configuration: dictionnary defining the parameters of the run (see configuration_dict.py)
    """

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
        
        return x, patient_pred

class SiameseArteryAnalysis_Or_patient(torch.nn.Module):
    """
        Aim: A block that extract features from the two views of an artery with ResNet18 siamese net. The block also predicts the MI at the artery level. The patient data is concatenated to the image features before prediction
        
        Functions:
            - Init: initialise the block
        
            - Forward: analyse the image
                - Parameters: 
                    - x1: view 1 of the artery (image+mask)
                    - x2: view 2 of the artery (image+mask)
                - Output: x1 (feature extraction of view 1), x2 (feature extraction of view2), pred (MI prediction in this artery)
    """
    
    def __init__(self, train_config=None):
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
        
    def forward(self, x1, x2, x_patient):
        # Extract features in both image with the same net
        x1 = self.resnet18_without_head(x1)
        x2 = self.resnet18_without_head(x2)
        
        # Predict MI at artery level
        x = torch.cat((x1, x2), dim=1)
        x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=(1,1))
        x = torch.flatten(x, start_dim=1)
        if self.drop is not None:
            x = self.drop(x)
        x = torch.cat((x, x_patient), dim=1) # CHECK THIS LINE
        pred = self.artery_pred(x)
        pred = torch.sigmoid(pred)

        return x1, x2, pred
    
class MiPredArteryLevel_Or_with_patient(torch.nn.Module):
    """
        Aim: A network that predicts global MI (and artery MI) based on three artery level blocks prediction (MAX). No common feature analysis is done there. It also considers the patient information (concatenated to the image features)
        
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
        super(MiPredArteryLevel_Or_with_patient, self).__init__()
        
        # Construct the six artery analysis CNN block (3 artery)
        self.resnet_lad = SiameseArteryAnalysis_Or_patient(train_config)
        self.resnet_lcx = SiameseArteryAnalysis_Or_patient(train_config)
        self.resnet_rca = SiameseArteryAnalysis_Or_patient(train_config)
        
        # prediction from patient data
        self.patient_net = PatientNet(train_config)

    def forward(self, x, x_patient):
        x_patient, patient_pred = self.patient_net(x_patient)
        
        # Extract each view of each artery (each tensor is Cx2x1525x1524)
        x_lad_1 = x[:, 0, 0, :, :, :]
        x_lad_2 = x[:, 0, 1, :, :, :]
        x_lcx_1 = x[:, 1, 0, :, :, :]
        x_lcx_2 = x[:, 1, 1, :, :, :]
        x_rca_1 = x[:, 2, 0, :, :, :]
        x_rca_2 = x[:, 2, 1, :, :, :]
        
        # Treat each view of each artery separatly and get prediction (each tensor is Cx256x96x96)
        x_lad_1, x_lad_2, lad_pred = self.resnet_lad(x_lad_1, x_lad_2, x_patient)
        x_lcx_1, x_lcx_2, lcx_pred = self.resnet_lcx(x_lcx_1, x_lcx_2, x_patient)
        x_rca_1, x_rca_2, rca_pred = self.resnet_rca(x_rca_1, x_rca_2, x_patient)
           
        # Patient has MI if any of the artery has MI
        pred = torch.stack([lad_pred, lcx_pred, rca_pred], dim=1)
        pred = pred.max(dim=1).values # max return values and indices
        
        return pred, lad_pred, lcx_pred, rca_pred, \
                (x_lad_1, x_lad_2), (x_lcx_1, x_lcx_2), (x_rca_1, x_rca_2), patient_pred

    
def load_state_dict_pretrained_to_net(self, state_dict):
    """
        Aim: apply a state dict to a network (and handle impossible match)
             from https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113
    """
 
    own_state = self.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
             continue
        if isinstance(param, torch.nn.parameter.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        try:
            own_state[name].copy_(param)
        except RuntimeError as e:
            print("State dict {} from pretrained cannot be loaded into new network (network.py).\nError: {}".format(name, e))
            
    
def init_net(train_configuration):
    """ 
        Aim: Initialise a network and its optimisation (optimiser, ...) based on the given train configuration
        
        Parameters:
            - train_configuration: dictionnary defining the parameters of the run (see configuration_dict.py)
        
        Output: the network, the train scheduler, the optimizer
    """
    
    # Create the network
    if train_configuration["load_network"] is None:
        net = train_configuration["network_class"](train_configuration)
    else:
        if not isinstance(train_configuration["load_network"], list): # if not a list we load the entire network
            print("Loading existing network {}".format(train_configuration["load_network"]))
            net = train_configuration["network_class"](train_configuration)
            net.load_state_dict(torch.load(train_configuration["load_network"]))
        else: # if its a list, the first network of the list corresponds to a raw CNN and the second to a raw ANN
            try:
                print("Loading existing base networks {} and {}".format(train_configuration["load_network"][0], train_configuration["load_network"][1]))
                net = train_configuration["network_class"](train_configuration)
                cnn_pretrained_state_dict = torch.load(train_configuration["load_network"][0])
                ann_pretrained_state_dict = torch.load(train_configuration["load_network"][1])

                load_state_dict_pretrained_to_net(net.patient_net, ann_pretrained_state_dict)
                load_state_dict_pretrained_to_net(net, cnn_pretrained_state_dict)
                
            except:
                    print("Loading of pretrained ANN and CNN failed (network.py).")
                    sys.exit()
                    
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
    
    if train_configuration["load_network"] is None:
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

        if train_configuration["init_patient"] == "Xavier Uniform":
            net.patient_net.apply(xavier_uniform_init)
        elif train_configuration["init_patient"] == "Xavier Normal":
            net.patient_net.apply(xavier_normal_init)
        elif train_configuration["init_patient"] == "Kaiming Uniform":
            net.patient_net.apply(kaiming_uniform_init)
        elif train_configuration["init_patient"] == "Kaiming Normal":
            net.patient_net.apply(kaiming_normal_init)
        else:
            print("Unknown Initialisation")
            sys.exit()

    return net, criterion_l, scheduler_l, optimizer_l
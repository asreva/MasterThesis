"""
Aim: Implement the ANN networks for the MI prediction with patient data
Author: Ivan-Daniel Sievering for the LTS4 Lab (EPFL)
"""

# --- Settings --- #
#Libraries
import sys
import torch

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

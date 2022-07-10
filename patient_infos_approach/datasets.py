"""
Aim: Implement the dataset loader for the MI prediction from patient data
Author: Ivan-Daniel Sievering for the LTS4 Lab (EPFL)
"""

# --- Settings --- #
#Libraries
import os
import sys
from time import time
import random
import pickle
import numpy as np
from sklearn.model_selection import train_test_split 
import pandas as pd

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

# Constants and global variables
PATH_TO_PATIENT_DF = "patient_mi_data/full_mi_patient_data_no_na.csv"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
# --- Classes --- #
class PatientDataset():
    """ 
    Aim: Define a PyTorch dataset class to get the information and labels of patients
    
    Parameters of constructor:
        - df: dataframe of the dataset
        - train_configuration: global structure that defines the training (see configuration_dict.py)
        - test: if the data is the training or testing data
        - train_data_info: if testing data and using normalisation, provide the means and std of the training data to apply normalisation
    """
    
    def __init__(self, df, train_configuration, test, train_data_info=None):
        self.df = df
        
        # if training data, save the std and mean of columns to apply normalisation on testing data
        if test==False: 
            age_mean, age_std = df["age"].mean(), df["age"].std()
            bmi_mean, bmi_std = df["bmi"].mean(), df["bmi"].std()
            eGFR_bln_mean, eGFR_bln_std = df["eGFR_bln"].mean(), df["eGFR_bln"].std()
            
            # save train info for testing dataset
            self.patient_data_info = [[age_mean, age_std],[bmi_mean, bmi_std],[eGFR_bln_mean, eGFR_bln_std]]
            
        # if testing data and normalisation enabled, load the train mean and std
        elif test and train_configuration["patient_normalisation"]: 
            age_mean, age_std = train_data_info[0]
            bmi_mean, bmi_std = train_data_info[1]
            eGFR_bln_mean, eGFR_bln_std = train_data_info[2]
        
        # normalise the dataset
        if train_configuration["patient_normalisation"]: 
            df["age"] = (df["age"]-age_mean)/age_std
            df["bmi"] = (df["bmi"]-bmi_mean)/bmi_std
            df["eGFR_bln"] = (df["eGFR_bln"]-eGFR_bln_mean)/eGFR_bln_std
            
        self.X = df.drop(columns=["mi"])
        self.y = df["mi"]
        self.idx_sjid = df.index.tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):        
        data = torch.tensor(self.X.iloc[idx].values).float()
        target = torch.tensor(self.y.iloc[idx]).float().unsqueeze(0)
        sjid = self.idx_sjid[idx] # not used for now, will be used with images
        
        return data, target
                    
# --- Main --- #
def get_data_loaders(train_configuration, cv_split):
    """ 
    Aim: Return train and validation dataloaders
    
    Parameters of constructor:
        - train_configuration: global structure that defines the training (see configuration_dict.py)
        - cv_split: None to use the whole data else we are doing k fold crossvalidation and receive [idx, k] where idx is the idx of the current kfold and k the nb of folds
                    
    Output: training and validation dataloaders
    """
    
    # Load the dataframe containing the patient names and extract them (as well as their related MI state)
    df = pd.read_csv(PATH_TO_PATIENT_DF)
    df = df.set_index("sjid")
    df["mi"] = df["mi"].astype(float)
    df = df.drop(columns=['nb_mi', 'time_to_mi', 'death_mi', 'time_to_death_mi', 'revasc', 'nb_revasc', 'time_to_revasc'])
    df = df.drop(columns=["grace_calc", "lvef_comb"])
    
    # Extract testing data with fixed seed
    df_train_valid, df_test = train_test_split(df, test_size=train_configuration["train_test_ratio"], random_state=train_configuration["seed"])
    
    if cv_split is None:
        # If no cross validation is used, just extract randomly a part of the data
        df_train, df_valid = train_test_split(df_train_valid, test_size=train_configuration["train_test_ratio"])
    else:
        # If cross validation extract the nb of split and the current index
        index_split, nb_split = cv_split
        
        # Separate by mi
        df_train_valid_mi = df_train_valid[df_train_valid["mi"] == 1]
        df_train_valid_nomi = df_train_valid[df_train_valid["mi"] == 0]
        
        # Separate train-valid MI patient for this CV
        valid_size_mi = len(df_train_valid_mi)//nb_split
        df_valid_mi = df_train_valid_mi.iloc[index_split*valid_size_mi:(index_split+1)*valid_size_mi]
        df_train_mi = pd.concat([df_train_valid_mi[:index_split*valid_size_mi], df_train_valid_mi[(index_split+1)*valid_size_mi:]])
        
        # Separate train-valid non MI patient for this CV
        valid_size_nomi = len(df_train_valid_nomi)//nb_split
        df_valid_nomi = df_train_valid_nomi.iloc[index_split*valid_size_nomi:(index_split+1)*valid_size_nomi]
        df_train_nomi = pd.concat([df_train_valid_nomi[:index_split*valid_size_nomi], df_train_valid_nomi[(index_split+1)*valid_size_nomi:]])
        
        # Get the complete datasets and shuffle them
        df_train = pd.concat([df_train_mi, df_train_nomi]).sample(frac=1)
        df_valid = pd.concat([df_valid_mi, df_valid_nomi]).sample(frac=1)
        
    if train_configuration["balance_method"] == "no": # No balance method, just create the dataloader
        patient_train_dataset = PatientDataset(df_train, train_configuration, test=False, train_data_info=None)
        patient_test_dataset = PatientDataset(df_test, train_configuration, test=True, train_data_info=patient_train_dataset.patient_data_info)
        
        train_data_loader = torch.utils.data.DataLoader(patient_train_dataset, batch_size=train_configuration["batch_size"], shuffle=True, persistent_workers=True, num_workers=os.cpu_count(), pin_memory=True)
        valid_data_loader = torch.utils.data.DataLoader(patient_test_dataset, batch_size=train_configuration["batch_size"], shuffle=True, persistent_workers=True, num_workers=os.cpu_count(), pin_memory=True)
        
    # Undersampling, take only subset of non MI and remerged both datasets before creating the dataloaders
    elif train_configuration["balance_method"] == "undersample":
        df_train_mi = df_train[df_train["mi"] == 1]
        df_train_no_mi = df_train[df_train["mi"] == 0]
        df_train_no_mi_reduced = df_train_no_mi.sample(len(df_train_mi))
        df_train_undersampled = pd.concat((df_train_mi, df_train_no_mi_reduced))
        
        patient_train_dataset = PatientDataset(df_train, train_configuration, test=False, train_data_info=None)
        patient_test_dataset = PatientDataset(df_test, train_configuration, test=True, train_data_info=patient_train_dataset.patient_data_info)
        
        train_data_loader = torch.utils.data.DataLoader(patient_train_dataset, batch_size=train_configuration["batch_size"], shuffle=True, persistent_workers=True, num_workers=os.cpu_count(), pin_memory=True)
        valid_data_loader = torch.utils.data.DataLoader(patient_test_dataset, batch_size=train_configuration["batch_size"], shuffle=True, persistent_workers=True, num_workers=os.cpu_count(), pin_memory=True)
        
    # oversampling, compute the vector of probability depending on the class (MI-nonMI) and then apply the sampler to the dataloader
    elif train_configuration["balance_method"] == "oversample":   
        df_train = df_train.sample(frac=1) # we can not shuffle after because the sampler needs to keep the order
        nb_non_mi = len(df_train[df_train["mi"] == 0])
        nb_mi = len(df_train[df_train["mi"] == 1])
        
        # Assign to each sample its apparition proba (proportionnal to its class)
        tmp_df = df_train.copy()
        tmp_df["sample_proba"] = 0
        try:
            tmp_df.loc[tmp_df["mi"] == 1, "sample_proba"] = 1./nb_mi
            tmp_df.loc[tmp_df["mi"] == 0, "sample_proba"] = 1./nb_non_mi
        except:
            print("The train dataset does not contain one of the two classes, impossible to define the WeightedRandomSampler.")
            sys.exit()
            
        # Define the sample w.r.t. the probabilities
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=tmp_df["sample_proba"].to_numpy(), replacement=True, 
                                                                 num_samples=len(df_train))
        
        patient_train_dataset = PatientDataset(df_train, train_configuration, test=False, train_data_info=None)
        patient_test_dataset = PatientDataset(df_test, train_configuration, test=True, train_data_info=patient_train_dataset.patient_data_info)
        
        # Create the dataloader with the sampler for training
        train_data_loader = torch.utils.data.DataLoader(patient_train_dataset, batch_size=train_configuration["batch_size"], sampler=sampler, persistent_workers=True, num_workers=os.cpu_count(), pin_memory=True)
        valid_data_loader = torch.utils.data.DataLoader(patient_test_dataset, batch_size=train_configuration["batch_size"], shuffle=True, persistent_workers=True, num_workers=os.cpu_count(), pin_memory=True)
        
    print("Nb of elements in train_data_loader {}".format(len(train_data_loader)*train_configuration["batch_size"]))
    print("Nb of MI in train data {}".format(len(df_train[df_train["mi"]==1])))
    print("Nb of elements in valid_data_loader {}".format(len(valid_data_loader)*train_configuration["batch_size"]))
    print("Nb of MI in valid data {}".format(len(df_valid[df_valid["mi"]==1])))
        
    return train_data_loader, valid_data_loader
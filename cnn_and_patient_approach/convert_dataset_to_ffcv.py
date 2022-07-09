"""
Aim: From a "classic" dataset from PyTorch create a FFCV optimised dataframe
Author: Ivan-Daniel Sievering for the LTS4 Lab (EPFL)
"""

# --- Settings --- #
#Libraries
import os
import pandas as pd
import numpy as np

import torch
from sklearn.model_selection import train_test_split 

from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField, FloatField, BytesField

# Constants and global variables
PATH_TO_DF_INFO = "csv_with_infos/df_patient_names_MIstate.csv"
PATH_TO_IMG_FOLDER = "../cnn_approach/patient_tensor_bigger"
PATH_TO_BETON = "beton_files/"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == "cuda":
    torch.cuda.empty_cache()

# --- Classes --- #
class PatientImageDataset():
    """
    Aim: creates a dataset with the CNN images and the patient data
                
    """
    
    def __init__(self, df, basepath, device):
        self.df = df
        self.basepath = basepath
        self.mi = torch.tensor(df["patient_mi"], dtype=torch.float)
        self.mi_lad = torch.tensor(df["mi_lad"], dtype=torch.float)
        self.mi_lcx = torch.tensor(df["mi_lcx"], dtype=torch.float)
        self.mi_rca = torch.tensor(df["mi_rca"], dtype=torch.float)
        self.device = device
        # Get the patient name from the df index
        self.patient_name = df.index
        self.patient_sjid = df["sjid"]
        
        self.other_patient_data = df.copy().reset_index()
        self.other_patient_data = self.other_patient_data[['sex', 'age', 'bmi', 'diabetes_hist', 'smoking_bln', 'hypertension_hist', 'cholesterolemia_hist', 'prev_CVD', 'eGFR_bln', 'resuscitation', 'chf_code_proc']]
        
    def __len__(self):
        return len(self.mi)

    def __getitem__(self,idx):  
        
        # Create MI info tensor
        all_mi_patient = torch.stack((self.mi_lad[idx], self.mi_lcx[idx], self.mi_rca[idx], self.mi[idx]), 0)
        
        # Load MI imgs tensor
        all_img_patient = torch.load(os.path.join(self.basepath,self.patient_name[idx],"patient_tensor.pt"))
        
        # Create patient infos tensor
        all_patient_data = torch.from_numpy(self.other_patient_data.loc[idx].values)
              
        all_img_patient = all_img_patient.numpy().astype('float32')
        all_mi_patient = all_mi_patient.numpy().astype('float32')
        all_patient_data = all_patient_data.numpy().astype('float32')
        
        return (all_img_patient, all_patient_data, all_mi_patient)

def create_ffcv_dataset(df, name):
    # Save the df
    df.to_csv(PATH_TO_BETON+name+".csv")
    
    # Creat the "standard" dataframe
    my_dataset  = PatientImageDataset(df, PATH_TO_IMG_FOLDER, device=device)
    
    # Define the writer
    writer = DatasetWriter(
        PATH_TO_BETON+name+".beton", # path to the folder
        { # description of the elements contained in the dataset
            'images': NDArrayField(shape=(3,2,2,1524,1524), dtype=np.dtype('float32')), # the 3 arteries 2 views and image+mask images of 1524*1524 resolution
            'patient_data': NDArrayField(shape=(11,), dtype=np.dtype('float32')), # the 11 selected patient feature
            'labels': NDArrayField(shape=(4,), dtype=np.dtype('float32')) # the MI state of the patient for the 3 arteries + the global state
        }, 
        num_workers=os.cpu_count(), # number of workers to assign for the computer
        page_size=2**27 # I don't know what it is, but else it crashes --> set it at the minimum value that makes it workes (needs to be power of 2)
    )

    # Write the dataset
    writer.from_indexed_dataset(my_dataset)
    
# --- Main --- #
if __name__ == '__main__':
    # load the dataframe
    dl_artery_level_df= pd.read_csv(PATH_TO_DF_INFO, index_col=0)
    
    # Separate test and train
    train_valid_df, test_df = train_test_split(dl_artery_level_df, test_size=0.2)
        
    # Save train and test
    create_ffcv_dataset(test_df, "test_data")
    create_ffcv_dataset(train_valid_df, "train_valid_data")
    
    # Preparate for balancing
    train_valid_df_mi = train_valid_df[train_valid_df["patient_mi"] == 1]
    train_valid_df_no_mi = train_valid_df[train_valid_df["patient_mi"] == 0]
    print("NB MI {}".format(len(train_valid_df_mi)))
    print("NB NO MI {}".format(len(train_valid_df_no_mi)))
    
    # undersample
    train_valid_df_no_mi_undersampled = train_valid_df_no_mi.sample(len(train_valid_df_mi))
    train_valid_undersampled = pd.concat([train_valid_df_no_mi_undersampled, train_valid_df_mi])
    train_valid_undersampled = train_valid_undersampled.sample(frac=1)
    print("NB UNDER {}".format(len(train_valid_undersampled)))
    create_ffcv_dataset(train_valid_undersampled, "train_valid_data_undersampled")
        
    # oversampled
    train_valid_df_mi_oversampled = train_valid_df_mi.sample(len(train_valid_df_no_mi), replace=True)
    train_valid_oversampled = pd.concat([train_valid_df_mi_oversampled, train_valid_df_no_mi])
    train_valid_oversampled = train_valid_oversampled.sample(frac=1)
    print("NB OVER {}".format(len(train_valid_oversampled)))
    create_ffcv_dataset(train_valid_oversampled, "train_valid_data_oversampled")
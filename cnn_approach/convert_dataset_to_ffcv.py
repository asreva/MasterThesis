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
PATH_TO_DF_INFO = "data/dl_artery_level_df_bigger.csv"
PATH_TO_IMG_FOLDER = "data/patient_tensor_bigger"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == "cuda":
    torch.cuda.empty_cache()

# --- Classes --- #
class PatientImageDataset():
    """
        Aim: Implement the dataset containing all the patient's images and it's MI record vector. 
             Each patient has 12 images (3 arteries * 2 views * 2 representation (image + mask)) 
             and 4 target value (had MI, had MI in LAD, had MI in LCX, had MI in RCA).
        
        Functions:
            - Init: initialise the datast
                - Parameters:
                    - df: the dataframe containing the data
                    - basepath: position in the folders to find the correct folder
                    - device: on which device the dataset has to work
        
            - __len__: get length of the df
            
            - __getitem__: get the desired patient instance from the dataset (images will be preprocessed and extended)
                - Parameters:
                    - idx: index of the patient to get
                - Output
                    - all_img_patient: a tensor containing all the images from this patient
                    - all_mi_patient: a tensor containing all the target of the patient
                
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
        
    def __len__(self):
        return len(self.mi)

    def __getitem__(self,idx):
        # Create MI tensor
        all_mi_patient = torch.stack((self.mi_lad[idx], self.mi_lcx[idx], self.mi_rca[idx], self.mi[idx]), 0)
        
        # Load MI imgs tensor
        all_img_patient = torch.load(os.path.join(self.basepath,self.patient_name[idx],"patient_tensor.pt"))
              
        all_img_patient = all_img_patient.numpy().astype('float32')
        all_mi_patient = all_mi_patient.numpy().astype('float32')
        
        return (all_img_patient, all_mi_patient)

def create_ffcv_dataset(df, name):
    # Save the df
    df.to_csv(name+".csv")
    
    # Creat the "standard" dataframe
    my_dataset  = PatientImageDataset(df, PATH_TO_IMG_FOLDER, device=device)
    
    # Define the writer
    writer = DatasetWriter(
        name+".beton", # path to the folder
        { # description of the elements contained in the dataset
            'images': NDArrayField(shape=(3,2,2,1524,1524), dtype=np.dtype('float32')), # the 3 arteries 2 views and image+mask images of 1524*1524 resolution
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
    create_ffcv_dataset(train_valid_df, "train_valid_data")
    create_ffcv_dataset(test_df, "test_data")
    
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
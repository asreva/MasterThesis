"""
Aim: Implement the datasetloader for the MI prediction from patches
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

import cv2
from scipy import interpolate
from scipy.ndimage import rotate
from skimage.morphology import skeletonize
from skimage.morphology import binary_closing
from skimage.filters import threshold_otsu
from skimage.filters import frangi

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

# Constants and global variables
IMGS_BASEPATH = "transformer_patient_torch_tensors/"
PATH_TO_TRANSFORMER_DF = "transformer_df.pkl"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Functions --- #
def extract_patches(img, img_mask, patch_size, nb_patch, rnd=0, test=False):
    """ 
        Aim: Extract patches on a box based on the detected centerline
        
        Parameters:
            - img: the box (HxW)
            - img_mask: the centerline detected on the box (HxW)
            - patch_size: the size of the patch to extract (L)
            - nb_patch: the number of patches to extract on the box (N)
            - rnd: the probability to extract a box not on the centerline
        
        Output: all the extracted patches (NxLxL)
    """
    
    # Always extract the same patches
    if test:
        random.seed(42)
        np.random.seed(42)
    
    patches = []
    
    img_shape = img.shape
    
    # If the patch is too big, extract a smaller one
    reduction_ratio = 1
    while patch_size >= img_shape[0]//2 or patch_size >= img_shape[1]//2:
        patch_size //= 2
        reduction_ratio *= 2
    
    # Do not extract on the border
    img_mask[:, :int(patch_size/2)] = 0
    img_mask[:int(patch_size/2), :] = 0
    img_mask[int(-patch_size/2):, :] = 0
    img_mask[:, int(-patch_size/2):] = 0
    
    # Select uniformly (+ some noise) the position of the extracted patches
    x_line = np.linspace(start=patch_size//2+1, stop=img_shape[1]-patch_size-1, num=nb_patch)
    if test == False:
        x_line += (patch_size//2-1)*(np.random.rand(nb_patch))
            
    # For each patch
    for i in range(0, nb_patch):
        # Select the x position
        width_pos = int(x_line[i])
        
        # Get the centerline distribution along this x position
        proba_line = img_mask[:,width_pos].numpy() # for some reason choice does not work with tensor
        
        # If true, randomly select the y position 
        if random.random() < rnd:
            height_pos = patch_size//2+1+int((img_shape[0]-patch_size-1)*random.random())
        # Else select the y position along the possible centerline
        else:
            # If no pixel is centerline, select randomly
            if np.sum(proba_line)  == 0:
                height_pos = patch_size//2+1+int((img_shape[0]-patch_size-1)*random.random())
            else:
                proba_line /= np.sum(proba_line)
                height_pos = np.random.choice(img_shape[0], p=proba_line)
        
        # Extract the patch from the image
        extracted_patch = img[int(height_pos-patch_size/2):int(height_pos+patch_size/2), 
                              int(width_pos-patch_size/2):int(width_pos+patch_size/2)]
        
        # If we took a smaller patch than expected make it bigger by interpolation
        if reduction_ratio != 1:
            resizer = transforms.Resize(extracted_patch.shape[0]*reduction_ratio)
            extracted_patch = resizer(extracted_patch.unsqueeze(0).unsqueeze(0))[0,0,:,:]
        
        patches.append(extracted_patch)
        
    random.seed(None)
    np.random.seed(None)
        
    return patches

def extend_images(img, img_mask, train_configuration, test=False):
    """ 
        Aim: Extend the image of the box and its centerline
        
        Parameters:
            - img: the box (HxW)
            - img_mask: the centerline detected on the box (HxW)
            - train_configuration: training structure (see configuration_dict.py), contains the proba of extension
            - test: if we are in the test data loader, if true only apply normalisation and no extension
        
        Output: the extended box and centerline image
    """
    
    # Some operations need RGB images
    img = img.repeat(3, 1, 1)
    img_mask = img_mask.repeat(3, 1, 1)
    
    if train_configuration["normalise"]:
        mean, std = 0.44531356896770125, 0.2692461874154524 # https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
        img = TF.normalize(img, [mean], [std])
        img_mask = TF.normalize(img_mask, [mean], [std])

    if test is False: # only extend images when training
        if train_configuration["random_crop"] is not None:
            if random.random() < train_configuration["random_crop"]:
                x, y, h, w = transforms.RandomResizedCrop.get_params(img, ratio=(0.9,1.1), scale=(0.9, 1))
                img = TF.resized_crop(img, x, y, h, w, size=(img.shape[1], img.shape[2]))
                img_mask = TF.resized_crop(img_mask, x, y, h, w, size=(img.shape[1], img.shape[2]))

        if train_configuration["gaussian_blur"] is not None:
            if random.random() < train_configuration["gaussian_blur"]:
                sigma = transforms.GaussianBlur.get_params(0.001, 0.01)
                img = TF.gaussian_blur(img, kernel_size=(3, 3), sigma=sigma)
                img_mask = TF.gaussian_blur(img_mask, kernel_size=(3, 3), sigma=sigma)

        if train_configuration["random_rotation"] is not None:
            if random.random() < train_configuration["random_rotation"]:
                angle = transforms.RandomRotation.get_params(degrees=[-2,2])
                img = TF.rotate(img, angle)
                img_mask = TF.rotate(img_mask, angle)


    img = img[0, :, :]
    img_mask = img_mask[0, :, :]
                
    return img, img_mask
    
# --- Classes --- #
class AttentionDataset():
    """ 
        Aim: Impelement the dataset class to get the images as series of patches following the arteries blood
        
        Attributes:
            - df: the dataframe containing the patient names 
            - patient_names: the list of all the patient names
            - basepath: the path to the images
            - train_configuration: training structure (see configuration_dict.py)
            - patch_size_l: the list of the size of the patches to extract on each kind of box
            - nb_patch_l: the list of the number of patches to extract for each kind of box
            - rnd: the probability to extract a box not on the centerline
            - test: if the dataset if for training or testing
        
        Functions:
            - __init__: initialise the dataset
            - __len__: get the dataset length
            - __getitem__: get a item of the dataset
    """
    
    def __init__(self, df, basepath, train_configuration, test):
        """ 
            Aim: Define the dataset

            Parameters:
                - df: the dataframe containing the patient names 
                - patient_names: the list of all the patient names
                - basepath: the path to the images
                - train_configuration: training structure (see configuration_dict.py)
                - test: if the dataset if for training or testing
        """
    
        self.df = df
        self.patient_names = df.index.get_level_values('patient_name').to_series().unique()
        self.basepath = basepath
        
        self.train_configuration = train_configuration
        self.patch_size_l = train_configuration["patch_size_l"]
        self.nb_patch_l = train_configuration["nb_patch_l"]
        self.rnd=train_configuration["patch_randomness"]
        
        self.test = test
        
    def __len__(self):
        """ 
            Aim: Return length of the dataset

            Output: the dataset length
        """
        return len(self.patient_names)

    def __getitem__(self,idx):
        """ 
            Aim: Get an item of the dataset

            Parameters:
                - idx: index of the item that we want

            Output: all the information about the patient:
                - available_arteries: [bool, bool, bool], indicating which arteries are available for this patient
                - (lad, lcx, rca) a tuple with for each artery a 2-sized tuple (2 views) containing the N patches of the artery
                - mi: the mi state of the patient [mi_lad, mi_lcx, mi_rca, mi_global]
        """

        patient_name = self.patient_names[idx]
        
        available_arteries = torch.load(self.basepath+"/"+patient_name+"/available_arteries.pt")
        mi = torch.load(self.basepath+"/"+patient_name+"/global_mi.pt")
        
        all_patches = [[[] for k in range(0,2)] for i in range(0,3)]
          
        for i_artery, artery in enumerate(["lad", "lcx", "rca"]):
            for i_view, view in enumerate(["view1", "view2"]):
                if available_arteries[i_artery]:
                    for i_sect, section in enumerate(["magenta", "yellow", "green", "brown"]):
                        
                        box = torch.load(self.basepath+"/"+patient_name+"/"+artery+"_"+view+"_"+section+"_box.pt")
                        centerline = torch.load(self.basepath+"/"+patient_name+"/"+artery+"_"+view+"_"+section+"_centerline.pt")
                        
                        box, centerline = extend_images(box, centerline, self.train_configuration, test=self.test)
                        
                        if not all_patches[i_artery][i_view]: # create the list
                            all_patches[i_artery][i_view] = extract_patches(box,  centerline, 
                                                                 self.patch_size_l[i_sect], self.nb_patch_l[i_sect], 
                                                                 rnd=self.rnd, test=self.test)
                            
                        else: # append to the list
                            all_patches[i_artery][i_view] += extract_patches(box,  centerline, 
                                                                 self.patch_size_l[i_sect], self.nb_patch_l[i_sect], 
                                                                 rnd=self.rnd, test=self.test)

                    all_patches[i_artery][i_view] = torch.stack(all_patches[i_artery][i_view], dim=0)
                else:
                    # If the artery does not exist, replace with zeros (!!! here we assume constant patch size along boxes))
                    tot_len = 0
                    for size in self.nb_patch_l:
                        tot_len += size
                    all_patches[i_artery][i_view] = torch.zeros((tot_len, self.patch_size_l[0], self.patch_size_l[0]))
                       
        lad = (all_patches[0][0].unsqueeze(1), 
               all_patches[0][1].unsqueeze(1))
        lcx = (all_patches[1][0].unsqueeze(1), 
               all_patches[1][1].unsqueeze(1))        
        rca = (all_patches[2][0].unsqueeze(1),
               all_patches[2][1].unsqueeze(1))
        
        return available_arteries, (lad, lcx, rca), mi
                    
# --- Main --- #
def get_data_loaders(train_configuration, cv_split):
    """ 
        Aim: Get the dataloaders

        Parameters:
            - train_configuration: training structure (see configuration_dict.py)
            - cv_split: None if cross validation not used, else a tuple with (index of the split, total number of splits to do) 

        Output: the training and the validation data loaders
    """
    
    # Load the dataframe containing the patient names and extract them (as welle as their related MI state)
    pkl_file = open(PATH_TO_TRANSFORMER_DF, 'rb')
    df = pickle.load(pkl_file)
    df = df.reset_index().groupby("patient_name").agg({"target_box":pd.Series.max})
    
    # Extract testing data with fixed seed
    df_train_valid, df_test = train_test_split(df, test_size=train_configuration["train_test_ratio"], random_state=train_configuration["seed"])
    
    if cv_split is None:
        # If no cross validation is used, just extract randomly a part of the data
        df_train, df_valid = train_test_split(df_train_valid, test_size=train_configuration["train_test_ratio"])
    else:
        # If cross validation extract the nb of split and the current index
        index_split, nb_split = cv_split
        
        # Separate by mi
        df_train_valid_mi = df_train_valid[df_train_valid["target_box"] == 1]
        df_train_valid_nomi = df_train_valid[df_train_valid["target_box"] == 0]
        
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
        
    # No balance method, just create the dataloader
    if train_configuration["balance_method"] == "no":
        train_data_loader = torch.utils.data.DataLoader(AttentionDataset(df_train, IMGS_BASEPATH, train_configuration, test=False), batch_size=train_configuration["batch_size"], shuffle=True, persistent_workers=True, num_workers=os.cpu_count(), pin_memory=True)
        valid_data_loader = torch.utils.data.DataLoader(AttentionDataset(df_valid, IMGS_BASEPATH, train_configuration, test=True), batch_size=train_configuration["batch_size"], shuffle=True, persistent_workers=True, num_workers=os.cpu_count(), pin_memory=True)
        
    # Undersampling, take only subset of non MI and remerged both datasets before creating the dataloaders
    elif train_configuration["balance_method"] == "undersample":
        df_train_mi = df_train[df_train["target_box"] == 1]
        df_train_no_mi = df_train[df_train["target_box"] == 0]
        df_train_no_mi_reduced = df_train_no_mi.sample(len(df_train_mi))
        df_train_undersampled = pd.concat((df_train_mi, df_train_no_mi_reduced))
        
        train_data_loader = torch.utils.data.DataLoader(AttentionDataset(df_train_undersampled, IMGS_BASEPATH, train_configuration, test=False), batch_size=train_configuration["batch_size"], shuffle=True, persistent_workers=True, num_workers=os.cpu_count(), pin_memory=True)
        valid_data_loader = torch.utils.data.DataLoader(AttentionDataset(df_valid, IMGS_BASEPATH, train_configuration, test=True), batch_size=train_configuration["batch_size"], shuffle=True, persistent_workers=True, num_workers=os.cpu_count(), pin_memory=True)
        
    # oversampling, compute the vector of probability depending on the class (MI-nonMI) and then apply the sampler to the dataloader
    elif train_configuration["balance_method"] == "oversample":   
        df_train = df_train.sample(frac=1) # we can not shuffle after because the sampler needs to keep the order
        nb_non_mi = len(df_train[df_train["target_box"] == 0])
        nb_mi = len(df_train[df_train["target_box"] == 1])
        
        # Assign to each sample its apparition proba (proportionnal to its class)
        tmp_df = df_train.copy()
        tmp_df["sample_proba"] = 0
        try:
            tmp_df.loc[tmp_df["target_box"] == 1, "sample_proba"] = 1./nb_mi
            tmp_df.loc[tmp_df["target_box"] == 0, "sample_proba"] = 1./nb_non_mi
        except:
            print("The train dataset does not contain one of the two classes, impossible to define the WeightedRandomSampler.")
            sys.exit()
            
        # Define the sample w.r.t. the probabilities
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=tmp_df["sample_proba"].to_numpy(), replacement=True, 
                                                                 num_samples=len(df_train))
        
        # Create the dataloader with the sampler for training
        train_data_loader = torch.utils.data.DataLoader(AttentionDataset(df_train, IMGS_BASEPATH, train_configuration, test=False), batch_size=train_configuration["batch_size"], sampler=sampler, persistent_workers=True, num_workers=os.cpu_count(), pin_memory=True)
        valid_data_loader = torch.utils.data.DataLoader(AttentionDataset(df_valid, IMGS_BASEPATH, train_configuration, test=True), batch_size=train_configuration["batch_size"], shuffle=True, persistent_workers=True, num_workers=os.cpu_count(), pin_memory=True)
        
    print("Nb of elements in train_data_loader {}".format(len(train_data_loader)*train_configuration["batch_size"]))
    print("Nb of MI in train data {}".format(len(df_train[df_train["target_box"]==1])))
    print("Nb of elements in valid_data_loader {}".format(len(valid_data_loader)*train_configuration["batch_size"]))
    print("Nb of MI in valid data {}".format(len(df_valid[df_valid["target_box"]==1])))
        
    return train_data_loader, valid_data_loader
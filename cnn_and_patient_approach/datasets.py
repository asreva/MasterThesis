"""
Aim: Implement the datasetloader for the MI prediction from CNN images and patient data
Author: Ivan-Daniel Sievering for the LTS4 Lab (EPFL)
"""

# --- Settings --- #
#Libraries
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split 
import pandas as pd

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from ffcv.loader import OrderOption
from ffcv.loader import Loader
from ffcv.fields.decoders import BytesDecoder, NDArrayDecoder
from ffcv.transforms import ToTensor, ToDevice

# Constants and global variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATH_TO_DATA = 'beton_files/train_valid_data'
PATH_TO_DATA_OVERSAMPLED = 'beton_files/train_valid_data_oversampled'
PATH_TO_DATA_TEST = 'beton_files/test_data'

# --- Classes --- #
class ExtendPatientImages(torch.nn.Module):
    """
        Aim: Implement a class to apply the dataset extension on the images. Needed to be implemented into FFCV. Works per BATCH

        Functions:
            - Init: initialise the data extender
                - Parameters:
                    - train_configuration: dictionnary defining the parameters of the run (see configuration_dict.py)
                    - test: if this dataset class will be used to train or test a network (not same data extensions)

            - forward: modify the batch of patient iamges
                - Parameters:
                    - imgs_tensor: the tensor of patient images of shape [BATCH_SIZE, NB_ARTERY, NB_VIEWS, NB_IMG_VERSION (mask+original), IMG_WIDTH, IMG_HEIGHT] --> [4, 3, 2, 2, 1524, 1524]
                - Output
                    - imgs_tensor: the modified tensor

    """
    def __init__(self, train_configuration, test):
        super(ExtendPatientImages, self).__init__()
        self.train_configuration = train_configuration
        self.test = test
        
    def forward(self, imgs_tensor):
        train_configuration = self.train_configuration
        test = self.test
        
        for i_batch in range(0,imgs_tensor.shape[0]): # iterate in the batch
            for i_artery in range(0,2+1): # iterate in arteries
                for i_view in range(0,1+1): # iterate in view
                    # Take the specific image and its mask
                    img = imgs_tensor[i_batch, i_artery, i_view, 0, :, :]
                    img_mask = imgs_tensor[i_batch, i_artery, i_view, 1, :, :]
                    
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
                                x, y, h, w = transforms.RandomResizedCrop.get_params(img, ratio=(0.75,1.25), scale=(0.8, 1))
                                img = TF.resized_crop(img, x, y, h, w, size=(1524, 1524))
                                img_mask = TF.resized_crop(img_mask, x, y, h, w, size=(1524, 1524))

                        if train_configuration["gaussian_blur"] is not None:
                            if random.random() < train_configuration["gaussian_blur"]:
                                sigma = transforms.GaussianBlur.get_params(25, 50)
                                img = TF.gaussian_blur(img, kernel_size=(25, 25), sigma=sigma)
                                img_mask = TF.gaussian_blur(img_mask, kernel_size=(25, 25), sigma=sigma)

                        if train_configuration["random_rotation"] is not None:
                            if random.random() < train_configuration["random_rotation"]:
                                angle = transforms.RandomRotation.get_params(degrees=[-30,30])
                                img = TF.rotate(img, angle)
                                img_mask = TF.rotate(img_mask, angle)

                        if train_configuration["random_color_modifs"] is not None:
                            if random.random() < train_configuration["random_color_modifs"]:
                                fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
                                                                    transforms.ColorJitter.get_params(brightness=(0.8, 1.2), contrast=(0.8, 1.2), 
                                                                                                      saturation=(0.8, 1.2), hue=(-0.2, 0.2))

                                # to understand, see https://discuss.pytorch.org/t/typeerror-tuple-object-is-not-callable/55075/5
                                for fn_id in fn_idx:
                                    if fn_id == 0 and brightness_factor is not None:
                                        img = TF.adjust_brightness(img, brightness_factor)
                                        img_mask = TF.adjust_brightness(img_mask, brightness_factor)
                                    elif fn_id == 1 and contrast_factor is not None:
                                        img = TF.adjust_contrast(img, contrast_factor)
                                        img_mask = TF.adjust_contrast(img_mask, contrast_factor)
                                    elif fn_id == 2 and saturation_factor is not None:
                                        img = TF.adjust_saturation(img, saturation_factor)
                                        img_mask = TF.adjust_saturation(img_mask, saturation_factor)
                                    elif fn_id == 3 and hue_factor is not None:
                                        img = TF.adjust_hue(img, hue_factor)
                                        img_mask = TF.adjust_hue(img_mask, hue_factor)

                                
                    img = img[0, :, :].unsqueeze(0)
                    img_mask = img_mask[0, :, :].unsqueeze(0)
                        
                    imgs_tensor[i_batch, i_artery, i_view, 0, :, :] = img
                    imgs_tensor[i_batch, i_artery, i_view, 1, :, :] = img_mask
                
        return imgs_tensor
    
class NormalisePatientDate(torch.nn.Module):
    """
    Aim: normalise the patient data 
    """
    
    def __init__(self, normalise_patient):
        super(NormalisePatientDate, self).__init__()
        self.norm = normalise_patient
        
    def forward(self, patient_data):
        if self.norm:
            for i in range(0, patient_data.shape[0]):
                patient_data[i][1] -= 62.488857029506136
                patient_data[i][1] /= 12.621293377407468
                patient_data[i][2] -= 27.2068571635655
                patient_data[i][2] /= 4.439795166397547
                patient_data[i][8] -= 83.73087065151759
                patient_data[i][8] /= 24.1583841403525
                     
        return patient_data

def get_data_loaders(train_configuration, cv_split):
    """ 
        Aim: Generate train and validation dataset
        
        Parameters:
            - train_configuration: dictionnary defining the parameters of the run (see configuration_dict.py)
            - cv_split: if the train-valid is not part of a crossCV validation use None, else indicates to which step (i.e. 0/1/...) of the CV the process is
            
        Output:
            - (train_data_loader, valid_data_loader): training and validation dataset
    """
    
    # Get the data file and df indicating the mi state of the entries
    train_valid_df = pd.read_csv(PATH_TO_DATA+".csv")
    test_df = pd.read_csv(PATH_TO_DATA_TEST+".csv")
    ffcv_path = PATH_TO_DATA+".beton"
    ffcv_path_train = PATH_TO_DATA+".beton"
    trainvalid_indices = np.arange(0, len(train_valid_df)) # FFCV only accept indices and not keys
    
    
    print("{} train-valid data".format(len(trainvalid_indices)))
    
    # Extract train and valid dataset
    if train_configuration["balance_method"] == "no":
        if cv_split is None:
            train_indices, valid_indices = train_test_split(trainvalid_indices, test_size=train_configuration["train_test_ratio"])
            valid_size = len(valid_indices)
        else:
            index_split, nb_split = cv_split
            valid_size = len(trainvalid_indices)//nb_split
            valid_indices = trainvalid_indices[index_split*valid_size:(index_split+1)*valid_size]
            train_indices = np.append(trainvalid_indices[:index_split*valid_size], trainvalid_indices[(index_split+1)*valid_size:])
            
        train_df = train_valid_df.iloc[train_indices]
        valid_df = train_valid_df.iloc[valid_indices]
        
    elif train_configuration["balance_method"] == "undersample":
        train_valid_mi_df = train_valid_df[train_valid_df["patient_mi"]==1] 
        train_valid_no_mi_df = train_valid_df[train_valid_df["patient_mi"]==0]
        
        if cv_split is None:
            train_mi_df, _ = train_test_split(train_valid_mi_df, test_size=train_configuration["train_test_ratio"])
            train_no_mi_df = train_valid_no_mi_df.sample(len(train_mi_df))
        else:
            index_split, nb_split = cv_split
            mi_size = len(train_valid_mi_df)//nb_split
            no_mi_size = len(train_valid_no_mi_df)//nb_split
            
            train_mi_df = pd.concat([train_valid_mi_df.iloc[:index_split*mi_size], 
                                    train_valid_mi_df.iloc[(index_split+1)*mi_size:]])
            
            train_no_mi_df = pd.concat([train_valid_no_mi_df.iloc[:index_split*no_mi_size], 
                                    train_valid_no_mi_df.iloc[(index_split+1)*no_mi_size:]]).sample(len(train_mi_df))
        
        train_df = pd.concat([train_no_mi_df, train_mi_df]).sample(frac=1)
        valid_df = train_valid_df.drop(train_df.index)
        valid_df_mi = valid_df[valid_df["patient_mi"]==1].sample(frac=1)
        valid_df = valid_df.sample(int(len(train_valid_df)*train_configuration["train_test_ratio"]))
        # Force at least one patient with MI
        if len(valid_df[valid_df["patient_mi"]==1]) == 0:
            valid_df.iloc[0] = valid_df_mi.iloc[0]
        
        train_indices = train_df.index
        valid_indices = valid_df.index
        
    elif train_configuration["balance_method"] == "oversample":
        ffcv_path_train = PATH_TO_DATA_OVERSAMPLED+".beton"
        
        if train_configuration["test"]:
            valid_df = test_df
        elif cv_split is None:
            _, valid_df = train_test_split(train_valid_df, test_size=train_configuration["train_test_ratio"])
        else:
            index_split, nb_split = cv_split
            valid_split_size = len(train_valid_df)//nb_split
            
            valid_df = train_valid_df[index_split*valid_split_size:(index_split+1)*valid_split_size]
        
        train_valid_df_oversampled = pd.read_csv(PATH_TO_DATA_OVERSAMPLED+".csv")
        train_df_oversampled = train_valid_df_oversampled[~train_valid_df_oversampled["patient_name"].isin(valid_df["patient_name"])]
        
        train_df = train_df_oversampled
        train_indices = train_df.index
        
        if not train_configuration["test"]:
            valid_indices = test_df.index
        else:
            valid_indices = None

    # Extract part of it to make tests
    if train_configuration["dataset_ratio"] < 1:
        _, train_indices = train_test_split(train_indices, test_size=train_configuration["dataset_ratio"]) 
        _, valid_indices = train_test_split(valid_indices, test_size=train_configuration["dataset_ratio"]) 
        _, valid_df = train_test_split(valid_df, test_size=train_configuration["dataset_ratio"]) 
        _, train_df = train_test_split(train_df, test_size=train_configuration["dataset_ratio"]) 
    
        print("Nb of common patients between valid and train is {}".format(len(list(set(train_df["patient_name"]) & set(valid_df["patient_name"])))))
    
    print("{} MI in validation.".format(sum(valid_df["patient_mi"])))
    print("{} MI in train.".format(sum(train_df["patient_mi"])))
        
        
    if train_configuration["test"]:
        ffcv_path_valid = PATH_TO_DATA_TEST+".beton"
    else:
        ffcv_path_valid = ffcv_path
        
    # Get dataloader from the dataset
    train_data_loader = Loader(ffcv_path_train,
                batch_size=train_configuration["batch_size"],
                num_workers=os.cpu_count(),
                order=OrderOption.QUASI_RANDOM,
                indices=train_indices,
                pipelines={
                  'images': [NDArrayDecoder(), ToTensor(), ExtendPatientImages(train_configuration, False), ToDevice(device, non_blocking=True)],
                  'patient_data': [NDArrayDecoder(), ToTensor(), NormalisePatientDate(train_configuration["normalise_patient"]), ToDevice(device, non_blocking=True)],
                  'label': [BytesDecoder(), ToTensor(), ToDevice(device, non_blocking=True)]
                },
                batches_ahead=10,
                recompile=True
                )
    
    valid_data_loader = Loader(ffcv_path_valid,
                batch_size=train_configuration["batch_size"],
                num_workers=os.cpu_count(),
                order=OrderOption.QUASI_RANDOM,
                pipelines={
                  'images': [NDArrayDecoder(), ToTensor(), ExtendPatientImages(train_configuration, True), ToDevice(device, non_blocking=True)],
                  'patient_data': [NDArrayDecoder(), ToTensor(), NormalisePatientDate(train_configuration["normalise_patient"]), ToDevice(device, non_blocking=True)],
                  'label': [BytesDecoder(), ToTensor(), ToDevice(device, non_blocking=True)]
                },    
                indices=valid_indices,
                batches_ahead=10,
                recompile=True
                              )
    print("Nb of elements in train_data_loader {}".format(len(train_data_loader)*train_configuration["batch_size"]))
    print("Nb of elements in valid_data_loader {}".format(len(valid_data_loader)*train_configuration["batch_size"]))
    
    return train_data_loader, valid_data_loader
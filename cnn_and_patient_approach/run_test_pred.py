"""
Aim: Apply a given CNN and patient data network to a dataset to evaluate its performance
Author: Ivan-Daniel Sievering for the LTS4 Lab (EPFL)
"""

# --- Libraries --- #
import os 
import pickle as pkl
import torch

from ffcv.loader import OrderOption
from ffcv.loader import Loader
from ffcv.fields.decoders import BytesDecoder, NDArrayDecoder
from ffcv.transforms import ToTensor, ToDevice

from configuration_dict import train_configuration_default
from network import MiPredArteryLevel_Or_with_patient, load_state_dict_pretrained_to_net
from datasets import NormalisePatientDate

# --- Constants and parameters --- #
PATH_TO_NETWORK = "saved_networks/best_cnn_and_pat.pt"
PATH_TO_DATA_TEST = 'beton_files/test_data.beton'
PATH_TO_DATA_TRAIN = 'beton_files/train_valid_data.beton'
PRED_ON = PATH_TO_DATA_TEST

# --- Model definition --- #
train_config = train_configuration_default

train_config["load_network"] = ["saved_networks/best_cnn_and_pat.pt"]

train_config["balance_method"] = "no"
train_config["gaussian_blur"] = None
train_config["normalise"] = False
train_config["normalise_patient"] = True
train_config["random_rotation"] = 0.2
train_config["random_crop"] = 0.2
train_config["random_color_modifs"] = 0.2

train_config["network_class"] = MiPredArteryLevel_Or_with_patient
train_config["init"] = "Xavier Uniform"
train_config["init_patient"] = "Kaiming Normal"
train_config["batch_norm_patient"] = False

train_config["batch_size"] = 1

train_config["dropout"] = 0.1918385192596236 
train_config["dropout_patient_net"] = 0.3375476811147977 
train_config["nb_neur_per_hidden_layer_patient"] = [50, 10]

net = MiPredArteryLevel_Or_with_patient(train_config).cuda()
load_state_dict_pretrained_to_net(net, torch.load(PATH_TO_NETWORK))

# --- Dataset definition --- #
data_loader_test = Loader(PRED_ON,
    batch_size=1,
    num_workers=os.cpu_count(),
    order=OrderOption.SEQUENTIAL,
    pipelines={
      'images': [NDArrayDecoder(), ToTensor(), ToDevice("cuda", non_blocking=True)],
      'patient_data': [NDArrayDecoder(), ToTensor(), NormalisePatientDate(True), ToDevice("cuda", non_blocking=True)],
      'label': [BytesDecoder(), ToTensor(), ToDevice("cuda", non_blocking=True)]
    },    
    batches_ahead=10,
    recompile=True
)

# --- Evaluate the model on the dataset --- #
y, y_pred = [], []
y_lad, y_lad_pred = [], []
y_lcx, y_lcx_pred = [], []
y_rca, y_rca_pred = [], []

for idx, (data, patient_data, target) in enumerate(data_loader_test):    
    print(idx)
    
    y.append(target[:,-1].tolist())
    y_lad.append(target[:,0].tolist())
    y_lcx.append(target[:,1].tolist())
    y_rca.append(target[:,2].tolist())
    
    pred = net(data.cuda(), patient_data.cuda())
    
    y_pred.append(pred[0].tolist()[0])
    y_lad_pred.append(pred[1].tolist()[0])
    y_lcx_pred.append(pred[2].tolist()[0])
    y_rca_pred.append(pred[3].tolist()[0])
    
    del data, patient_data, target
    torch.cuda.empty_cache()

# --- Save the performance and labels --- #
with open('test/y_train.pkl', 'wb') as f:
    pkl.dump(y, f)
    
with open('test/y_lad_train.pkl', 'wb') as f:
    pkl.dump(y_lad, f)

with open('test/y_lcx_train.pkl', 'wb') as f:
    pkl.dump(y_lcx, f)
    
with open('test/y_rca_train.pkl', 'wb') as f:
    pkl.dump(y_rca, f)
    
with open('test/y_pred_train.pkl', 'wb') as f:
    pkl.dump(y_pred, f)
    
with open('test/y_lad_pred_train.pkl', 'wb') as f:
    pkl.dump(y_lad_pred, f)

with open('test/y_lcx_pred_train.pkl', 'wb') as f:
    pkl.dump(y_lcx_pred, f)
    
with open('test/y_rca_pred_train.pkl', 'wb') as f:
    pkl.dump(y_rca_pred, f)

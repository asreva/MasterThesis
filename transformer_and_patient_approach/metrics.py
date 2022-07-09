"""
Aim: Implement evaluation functions
Author: Ivan-Daniel Sievering for the LTS4 Lab (EPFL)
"""

# --- Settings --- #
#Libraries
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

# --- Functions --- #
def compute_metrics(predicition_list, target_list, name_extension):
    """ 
        Aim: Compute the metrics comparing the prediction and targets, returns the result as a dictionnary
        
        Parameters:
            - predicition_list: list of the predictions made by the algorithm
            - target_list: list of true labels
            - name_extension: extension to add to the metric name in the dictionnary (ex: to indicate that it is the validation metric)
        
        Output: metric dictionnary
    """
    
    metrics_dict = {}
    
    metrics_dict["accuracy"+name_extension] = accuracy_score(target_list, predicition_list)
    metrics_dict["f1"+name_extension] = f1_score(target_list, predicition_list, zero_division=0)
    metrics_dict["recall"+name_extension] = recall_score(target_list, predicition_list, zero_division=0)
    metrics_dict["precision"+name_extension] = precision_score(target_list, predicition_list, zero_division=0)
    metrics_dict["specificity"+name_extension] = recall_score(target_list, predicition_list, pos_label=0)
    try: # The AUC computation may fail if a class is not present in one of the vectors -> consider 0 in this case
        metrics_dict["auc_roc"+name_extension] =  roc_auc_score(target_list, predicition_list)
        precision, recall, _ = precision_recall_curve(target_list, predicition_list)
        metrics_dict["precision_recall_auc"+name_extension] = auc(recall, precision)
    except:
        print("Missing elements in a class for {}, impossible to compute some metrics.".format(name_extension))
        metrics_dict["auc_roc"+name_extension] = 0
        metrics_dict["precision_recall_auc"+name_extension] = 0
    
    return metrics_dict
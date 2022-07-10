# Master Thesis
Master Thesis (LTS4@EPFL): A multi-modal Deep Learning approach for Myocardial Infarction prediction

## Summary
The work aims to train networks that predict future MI (up to five years) from XCA images (X-ray Coronary Angiography) and patient data. Different strategies have been explored:
- Classical Machine Learning algorithms to patient data
- CNN to XCA images (and patient data)
- Transformers to XCA (and patient information)

![general_scheme](https://user-images.githubusercontent.com/56682743/178145916-d7465d83-7e62-41d2-bea8-fad79b56fe3d.png)

The results obtained are not sufficient for real-world application but are exciting, as the networks reach performances comparable to an interventional cardiologist. The CNN obtained the higher performance, but the Transformer approach can still be improved and seems more promising in the long run.

The report (MasterProject_report.pdf) details all the theory and the performance.

## Libraries and Kubernetes
The Kubernetes image used is: jupyter/data science-notebook:latest

The "non-standard" libraries to download are:
- FFCV: https://pypi.org/project/ffcv/ (fast data loading for CNN, comes inside of a conda environment)
- Scikit-image: https://scikit-image.org/docs/dev/install.html (to deal with images)
- LibAUC: https://libauc.org/get-started/ (take the latest version for AUC-based loss)
- Einops: https://pypi.org/project/einops/ (to use Einstein operators)
- WanbB (W&B): https://pypi.org/project/wandb/ (API to record performance)
- Captum: https://captum.ai/docs/getting_started (for model interpretability)

## Code organisation
### Main folders
  - patient_infos_approach: implementation of the classical ML applied to patient data
  - cnn_approach: implementation of the CNN network applied to XCA
  - cnn_and_patient_approach: implementation of the CNN network applied to XCA and patient data
  - transformer_approach: implementation of the Transformer network applied to XCA
  - transformer_and_patient_approach: implementation of the Transformer network applied to XCA and patient data
  - final_testing: testing dataset and performance of the different approaches on it
    
### Code in folders
Most of the folders contain similar files:
  - configuration_dict.py: a skeleton of the structure that defines how the network will be trained
  - dataset.py: defines the preprocessing steps and the dataloaders that will provide the data
  - execute_epoch.py: defines a complete epoch loop (with or without gradient descent)
  - loss.py: defines the losses used for the network
  - metrics.py: defines the metric used for evaluation
  - network.py: define the networks available (in some folders, there may be various network structures, and some may be used only for interpretability analysis (indicated in their name))
  - train_valid.py: apply a whole training and evaluation procedure to a network
  - run_manual.py: run a given scenario defined by a modified instance of the skeleton defined in configuration_dict.py
  - run_wandb_cv.py: run a cross-validation evaluation of a given scenario defined by a modified instance of the skeleton defined in configuration_dict.py; results are handled through the W&B API
  - run_wandb_grid.py: run a grid search with cross-validation of a given scenario defined by a modified instance of the skeleton defined in configuration_dict.py, results and HP exploration through the W&B API
  
### Quickstart examples
#### Run a simple network training and evaluation
1. Go to the folder of the architecture (for example, patient_info_approach)
2. Open run_manual.py
3. Fill the parameters of the skeleton dictionary defined in configuration_dict.py; for example, define the number of epochs by changing the value of the "n_epochs" key. The parameters are defined in configuration_dict.py
4. Execute run_manual.py (log into W&B the first time you run the code)
5. Wait and grab a coffee
6. Enjoy the results on the W&B API or in the console :)

#### Run a network cross validation
1. Go to the folder of the architecture (for example, patient_info_approach)
2. Open run_wandb_cv.py
3. Fill the parameters of the skeleton dictionary defined in configuration_dict.py and set the "nb_cv" key to the number of folds. For example, define the number of epochs by changing the value of the "n_epochs" key. The parameters are defined in configuration_dict.py
4. Execute run_wandb_cv.py (log into W&B the first time you run the code)
5. Wait and grab a coffee
6. Enjoy the results on the W&B API :)

#### Run a network gird search with cross-validation
1. Go to the folder of the architecture (for example, patient_info_approach)
2. Open run_wandb_grid.py
3. Fill the non-grid searched parameters of the skeleton dictionary defined in configuration_dict.py and set "nb_cv" to the number of folds. For example, define the number of epochs by changing the value of the "n_epochs" key. The parameters are defined in configuration_dict.py
4. For the grid searched parameters, they will be received as strings; convert them in their proper format (int/float/...) after receiving them. W&B doesn't support a list as an HP: the (ugly) workaround is to have a value for each entry of the list and then recreate the list in the code
5. Go to the W&B API and create a sweep with the parameters to grid search (https://docs.wandb.ai/guides/sweeps)
6. Execute the W&B command to launch the agent (log into W&B the first time you run the code)
7. Wait and grab a coffee
8. Enjoy the results on the W&B API :)

## Models Specificities
Each file has some specific Jupyter Notebooks and Python files; they are presented below.

### patient_infos_approach
create_mi_patient_datasets.ipynb creates the dataset of patient data used in this work. predict_mi_from_patient_data.ipynb test different classical ML methods on this dataset.

### cnn_approach
The create_img_and_df_artery_level_from_pierre.ipynb creates the dataset used by the CNN networks (image+mask of each artery for each patient). create_img_and_df_artery_level_custom.ipynb was an attempt to do it without using the dataframe from the previous work (https://github.com/Vuillecard/CVD_predicition), but it was not successful.

convert_dataset_to_ffcv.py convert the previous dataset to FFCV (compressed files) to speed the loading time. utils_pierre.py is only used by create_img_and_df_artery_level_custom.ipynb.

### cnn_and_patient_approach
The interpretability.ipynb implements the interpretability analysis of the network. 

run_test_pred.py evaluates a network on the testing dataset. 

convert_dataset_to_ffcv.py convert the dataset to FFCV (compressed files) to speed the loading time. network_interpretability.py and network_interpretability_tsne.py are two networks used only for interpretability. 

### transformer_approach
create_transformer_df_and_data.ipynb creates the dataset of oriented boxes with centerline. patch_extraction_method_tests.ipynb analyses how to extract patches from the image and its centerline. From the two last notebooks, create_transformer_torch_dataset.ipynb defines the PyTorch dataset to use further (with dataset extension).

The df_patient_names_to_sjid.csv converts the patient name to its SJID identifier and transformer_df.pkl contains the dataframe of patients.

### transformer_and_patient_approach
The interpretability.ipynb implements the interpretability analysis of the network. 

The code in init_net.py is contained in networks.py for the others approaches. In this approach, each network version has its file instead of having them all in networks.py. Some are for interpretability only.

### Final testing
It contains the testing images and test_data_eval.ipynb that evaluates the networks on this dataset and some other milestones (like the cardiologist).

## Others
- The results are available on W&B API: https://wandb.ai/asreva/dl_mi_pred_patient, https://wandb.ai/asreva/dl_mi_pred_CNN and https://wandb.ai/asreva/dl_mi_pred_transformers
- All the references, results and analysis are available in the report (MasterProject_report.pdf)
- The last Master Thesis working on this challenge is available at: https://github.com/Vuillecard/CVD_predicition
- For remarks or comments, you can send me an email at ivan-daniel.sievering@outlook.com

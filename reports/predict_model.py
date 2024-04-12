'''
Créé le 12/03/2024

@author: Equipe DS Jan 2024
12->>models/predict_model.py << 

'''
#####
# Imports 
#####
from tensorflow.keras.callbacks import TensorBoard, Callback, EarlyStopping,CSVLogger,LearningRateScheduler# Callbacks

## Import des modules utiles
import os
from sklearn.model_selection import train_test_split
import time
import numpy as np
import pandas as pd

## Import des modules de logging et timestamping
from datetime import datetime # timestamping
import logging
## Import du fichier de configuration pour des informations sur les Logs
from config import paths,infolog
## Import des modules de configuration

## Import des librairies
from src.utils import utils_models as um

## 0 - Gestion des logs
## Gestion des logs
# récupération du chemin projet
main_path = paths["main_path"]
# récupération du chemin des logs
log_folder = infolog["logs_folder"]
# récupération du nom des fichiers logs
logfile_name = infolog["logfile_name"]
logfile_path = os.path.join(main_path,log_folder,logfile_name)
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Création d'un nouveau fichier et renommage de l'ancien fichier s'il existe
if os.path.exists(logfile_path):
    os.rename(logfile_path, f"{logfile_path}.{current_time}")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Créer un gestionnaire de logs pour un fichier
file_handler = logging.FileHandler(logfile_name)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
# Créer un logger pour le module main
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Créer un gestionnaire de logs pour la console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
# Ajouter les gestionnaires de logs au logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def evaluate_model(model,X_eval,y_eval,num_classes,classes):# X_eval et y_eval
    
    """
    Evaluation du modèle et envoi des metriques
    
    Args:
        - model: Le modèle à évaluer
        - X: les images
        - y: Les cibles d'évaluation
        - num_classes: Le nombre de classe qui définit la binarité ou non de la classification et donc le calcul des metriques
        - classes: Dictionnaire de mapping des classes
    Returns: 
        - Metriques, 
        - confusion matrix dataframe
        - classification report  dataframe
        Confusion matrix et Classification report seront stockés dans des fichiers par MLFlow 
        Metriques seront stockées dans le RUN
    """

    #1- Récupération de metriques supplémentaires    
    #accuracy,loss=um.evaluaee(model,X_eval,y_eval)   
    if num_classes==1:
        metrics_dict=um.bin_get_prediction_metrics(model,X_eval,y_eval)
    else:
        metrics_dict=um.multi_get_prediction_metrics(model,X_eval,y_eval)
    
    sensitivities=metrics_dict["Sensitivity - Recall"]
    specificities=metrics_dict["Specificity"]
    conf_matrix=metrics_dict["Confusion Matrix"]
    class_report=metrics_dict["Classification report"] 
    
    final_metrics={}
    final_metrics["Accuracy"]=metrics_dict["Accuracy"]
    final_metrics["Recall"]=metrics_dict["Recall"]
    final_metrics["F1-score"]=metrics_dict["F1-score"]
    
    for i, (sensitivity, specificity) in enumerate(zip(sensitivities, specificities)):
            final_metrics[f'Recall sensitivity_class_{i}'] = sensitivity
            final_metrics[f'Recall specificity_class_{i}'] = specificity
    
    logger.debug(f"Accuracy_Score: {final_metrics['Accuracy']}")
    logger.debug(f"Recall: {final_metrics['Recall']}")
    logger.debug(f"F1-score: {final_metrics['F1-score']}" )
    logger.debug(f"Sensibilités - Recall: {sensitivities}")
    logger.debug(f"Spécificités: {specificities}")
    logger.debug(f"Matrice de confusion {conf_matrix}")
    logger.debug(f"Rapport de classification {class_report}")
        
    # construction du dictionnaire des metriques
    
    # Confusion matrix en format dataframe
    conf_matrix_df = pd.DataFrame(conf_matrix, index=[i for i in classes], columns=[i for i in classes])  #Etiquettes
    class_report_df = pd.DataFrame(class_report).transpose()
    
    return final_metrics,conf_matrix_df,class_report_df
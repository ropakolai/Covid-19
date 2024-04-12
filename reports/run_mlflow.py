'''
Créé le 28 mars 2024

@author: Equipe DS Jan 2024

Fonction MLFlow pour l'exécution des experiments

'''
#####
# Imports 
#####
# MLFlow
import mlflow

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
from config import paths,infolog,hyperparams_list
from config import experiments_sm,dataset_sm_hlo,dataset_sm
from config import experiments_cv,dataset_cv_hlo,dataset_cv
from config import experiments_mc,dataset_mc
from config import experiments_3c,dataset_3c

## Import des fichiers pythons de traitement du modele
from src.utils import utils_preprocess as up
from src.utils import utils_models as um
from src.models import build_model as build
from src.models import train_model as train
from src.models import predict_model as predict

## import to_categorical pour le one hot encoding
from keras.utils import to_categorical

## FIN DES IMPORTS

# récupération du chemin projet
main_path = paths["main_path"]
# récupération du répertoire des données
dataset_folder=paths["datasets_folder"]

## Gestion des logs
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

artifact_path = paths["artifact_path"] 

# Chaque fonction d'experiment récupére les datasets associés à la classification

def run_experiment_MC(hyperparams): #multiclasse
    """
    Fonction d'exécution des experiments pour une classification multiple
    """
    logger.debug("-----------run_experiment_MC------------")
    
    ## 1 - Récupération des informations sur les données à utiliser
    # récupération du dataset
    dataset_path=os.path.join(dataset_folder,dataset_mc["current_dataset_path"])
    # Initialisation du nom de répertoire dédié à la classification
    classif_folder=os.path.join(main_path,paths["model_outputs"],experiments_mc["classif_folder"])
    ## 1b - Création de tous les répertoires et sous répertoires liés au type de classification
    # classif_folder: sous-répertoire principal de la classification
    logger.debug(f"Création du répertoire de classification s'il 'existe pas : {classif_folder}")
    os.makedirs(os.path.dirname(classif_folder), exist_ok=True)
    # history
    history_subfolder=os.path.join(classif_folder,paths["model_history_outputs"])
    logger.debug(f"Création du répertoire d'history s'il 'existe pas : {history_subfolder}")
    os.makedirs(os.path.dirname(history_subfolder), exist_ok=True)
    # training plots
    training_plot_subfolder=os.path.join(classif_folder,paths["model_training_plot_outputs"])
    logger.debug(f"Création du répertoire de training_plot s'il 'existe pas : {training_plot_subfolder}")
    os.makedirs(os.path.dirname(training_plot_subfolder), exist_ok=True)
    # tuning history
    tuning_history_subfolder=os.path.join(classif_folder,paths["tuning_history_outputs"])
    logger.debug(f"Création du répertoire de tuning_history s'il 'existe pas : {tuning_history_subfolder}")
    os.makedirs(os.path.dirname(tuning_history_subfolder), exist_ok=True)
    # reports
    reports_subfolder=os.path.join(classif_folder,paths["reports_outputs"])
    logger.debug(f"Création du répertoire de report s'il 'existe pas : {reports_subfolder}")
    os.makedirs(os.path.dirname(reports_subfolder), exist_ok=True)
    
    # reports
    heatmap_subfolder=os.path.join(classif_folder,paths["heatmap_output"])
    logger.debug(f"Création du répertoire de heatmap s'il 'existe pas : {heatmap_subfolder}")
    os.makedirs(os.path.dirname(reports_subfolder), exist_ok=True)
    
    # models
    models_files_subfolder=os.path.join(classif_folder,"models_files")
    logger.debug(f"Création du répertoire de report s'il 'existe pas : {models_files_subfolder}")
    os.makedirs(os.path.dirname(models_files_subfolder), exist_ok=True)
    # récupération du nom du fichier de description (fichier json)
    dataset_desc=os.path.join(dataset_path,dataset_mc["dataset_desc"])
    classes_dic=experiments_mc["classes_dic"]
    dataset_tagname=dataset_mc["tag_name"]
    logger.debug("dataset_path")
    logger.debug(dataset_path)
    logger.debug("Dictionnaire")
    logger.debug(f"{classes_dic}")
    
    ## Split des données en données d'entrainement (qui seront splittés en données entrainement/validation) et données d'évaluation
    ## Pour cela, nous prevoyons 20% de volume supplémentaire (ex 750 au lieu de 600)
    logger.debug("Split des données X_train, y_train, X_eva,l y_eval")
    X_train, y_train, X_eval, y_eval,dic=up.get_data_from_parent_directory_labelled(dataset_path, classes_dic, hyperparams['img_size'],hyperparams['img_dim'],hyperparams['archi'])
    ## Traiter le cas multiclasse dans les run MC et 3classes - one hot encoding
    #logger.debug("---Multiclasse - One hot encoding----")
    #y_train = to_categorical(y_train,  hyperparams['num_classes'])
    #y_eval = to_categorical(y_eval, hyperparams['num_classes'])

    logger.debug("Directory to Label ")
    logger.debug(f"{dic}")
    mlflow.set_experiment(experiments_mc["experiment_name"])
    ##2- Lancement du RUN
    with mlflow.start_run(run_name=experiments_mc["run_name"]) as run:
        # récupération de l'id du run
        run_id = run.info.run_id
        logger.debug(f"Run ID {run_id}")
        # 1ers tags liés au Dataset
        mlflow.set_tag("dataset_name",dataset_tagname )
        mlflow.log_artifact(dataset_desc,"dataset_info")
        #mlflow.log_param("Dataset", dataset_tagname)
        mlflow.log_artifacts(dataset_path)
        mlflow.log_params(hyperparams)
        
        logger.debug("Data path")
        logger.debug("Hyperparamètres")
        logger.debug(str(hyperparams))
        
        # 2 - Initialisation d'informations supplémentaire pour le projet
        directory=os.path.join(tuning_history_subfolder,f"tuning_{hyperparams['archi']}_{run_id}")
        project_name=(f"tuning_{hyperparams['archi']}_{run_id}")
        logger.debug(f"Tuning Directory {directory}")
        logger.debug(f"Tuning project_name {project_name}")
        # 3 - Construction du modèle via Keras Tuning
        best_model = build.tuner_randomsearch(hyperparams,run_id,directory,project_name,X_train,y_train)
        
        # 4 - Entrainement du modèle
        trained_model,metrics,history=train.train_model(best_model,hyperparams,X_train,y_train,run_id)
        # 5 - Evaluation du modèle et génération de metriques supplémentaires
        final_metrics,conf_matrix_df,class_report_df=predict.evaluate_model(trained_model,X_eval,y_eval,hyperparams["num_classes"],classes_dic)
        # mise à jour de metrics avec les métriques supplementaire d'evaluation
        metrics.update(final_metrics)
        mlflow.log_metrics(metrics)
        
        # 5 - Sauvegarde des éléments générés : modèle / history / training plots / confusion matrix / classification report
        model_filename = f"model_{hyperparams['archi']}_{run_id}.keras"
        history_file_name=f"history_{hyperparams['archi']}_{run_id}.json"
        um.save_model(trained_model,os.path.join(models_files_subfolder,model_filename))
        history_filepath=um.save_history_json(history,os.path.join(history_subfolder,history_file_name))
        um.generate_training_plots_from_json(history_filepath,training_plot_subfolder,run_id)
        conf_matrix_df.to_csv(os.path.join(reports_subfolder,f"model_conf_m_{hyperparams['archi']}_{run_id}.csv"))
        um.plot_and_save_conf_matrix(conf_matrix_df, classes_dic, reports_subfolder,f"model_conf_m_plot_{hyperparams['archi']}_{run_id}.png")
        class_report_df.to_csv(os.path.join(reports_subfolder,f"model_class_report_{hyperparams['archi']}_{run_id}.csv"))
        um.plot_and_save_classification_report(class_report_df, reports_subfolder,f"model_class_report_{hyperparams['archi']}_{run_id}.png")
        
        '''
        ## Gradcam
        if hyperparams['archi'] in ['VGG16','VGG19']:
            logger.debug(f"Interprétabilité pour {hyperparams['archi']}")
            directory_path=os.path.join(main_path,paths["test_images"])
            um.gradcam_process_images(trained_model,f"{hyperparams['archi']}_{run_id}",directory_path,heatmap_subfolder,hyperparams['img_size'],hyperparams['num_classes'],hyperparams["last_conv_layer"])
        '''
        
def run_experiment_sain_malade(hyperparams): # binaire    
    """
    Fonction d'exécution des experiments pour une classification Binaire Sain/Malade
    """
    logger.debug("-----------run_experiment_sain_malade------------")
    
    ## 1 - Récupération des informations sur les données à utiliser
    # Initialisation du nom de répertoire dédié à la classification
    classif_folder=os.path.join(main_path,paths["model_outputs"],experiments_sm["classif_folder"])
    ## 1b - Création de tous les répertoires et sous répertoires liés au type de classification
    # classif_folder: sous-répertoire principal de la classification
    logger.debug(f"Création du répertoire de classification s'il 'existe pas : {classif_folder}")
    os.makedirs(os.path.dirname(classif_folder), exist_ok=True)
    # history
    history_subfolder=os.path.join(classif_folder,paths["model_history_outputs"])
    logger.debug(f"Création du répertoire d'history s'il n'existe pas : {history_subfolder}")
    os.makedirs(os.path.dirname(history_subfolder), exist_ok=True)
    # training plots
    training_plot_subfolder=os.path.join(classif_folder,paths["model_training_plot_outputs"])
    logger.debug(f"Création du répertoire de training_plot s'il n'existe pas : {training_plot_subfolder}")
    os.makedirs(os.path.dirname(training_plot_subfolder), exist_ok=True)
    # tuning history
    tuning_history_subfolder=os.path.join(classif_folder,paths["tuning_history_outputs"])
    logger.debug(f"Création du répertoire de tuning_history s'il n'existe pas : {tuning_history_subfolder}")
    os.makedirs(os.path.dirname(tuning_history_subfolder), exist_ok=True)
    # reports
    reports_subfolder=os.path.join(classif_folder,paths["reports_outputs"])
    logger.debug(f"Création du répertoire de report s'il 'existe pas : {reports_subfolder}")
    os.makedirs(os.path.dirname(reports_subfolder), exist_ok=True)
    
    # reports figures
    reports_figures=os.path.join(reports_subfolder,paths["reports_figures"])
    logger.debug(f"Création du répertoire de report digures s'il n'existe pas : {reports_figures}")
    os.makedirs(os.path.dirname(reports_figures), exist_ok=True)
    
    # Heatmaps
    heatmap_subfolder=os.path.join(reports_subfolder,paths["heatmap_output"])
    logger.debug(f"Création du répertoire de report heatmap s'il 'existe pas : {heatmap_subfolder}")
    os.makedirs(os.path.dirname(reports_subfolder), exist_ok=True)
    
    # models
    models_files_subfolder=os.path.join(classif_folder,"models_files")
    logger.debug(f"Création du répertoire de s'il 'existe pas : {models_files_subfolder}")
    os.makedirs(os.path.dirname(models_files_subfolder), exist_ok=True)
    
    # récupération du nom du fichier de description (fichier json)
    classes_dic=experiments_sm["classes_dic"]
    # récupération du dataset en fonction avec ou Sans Lung Opacity
    if hyperparams["data_lo_hlo"]=="HLO":   
        dataset_path=os.path.join(dataset_folder,dataset_sm_hlo["current_dataset_path"])
        dataset_desc=os.path.join(dataset_path,dataset_sm_hlo["dataset_desc"])
        dataset_tagname=dataset_sm_hlo["tag_name"]     
    else:
        dataset_path=os.path.join(dataset_folder,dataset_sm["current_dataset_path"])
        dataset_desc=os.path.join(dataset_path,dataset_sm["dataset_desc"])
        dataset_tagname=dataset_sm["tag_name"]              
    logger.debug("Dataset path")
    logger.debug(f"{dataset_path}")
    logger.debug("Dataset desc")
    logger.debug(f"{dataset_desc}")
    logger.debug("Dataset tagname")
    logger.debug(f"{dataset_tagname}")
    logger.debug("Dictionnaire")
    logger.debug(f"{classes_dic}")
    
    ## Split des données en données d'entrainement (qui seront splittés en données entrainement/validation) et données d'évaluation
    ## Pour cela, nous prevoyons 20% de volume supplémentaire (ex 750 au lieu de 600)
    logger.debug("Split des données X_train, y_train, X_eva,l y_eval")
    X_train, y_train, X_eval, y_eval,dic=up.get_data_from_parent_directory_labelled(dataset_path, classes_dic, hyperparams['img_size'],hyperparams['img_dim'],hyperparams['archi'])
    logger.debug("Directory to Label ")
    logger.debug(f"{dic}")
    mlflow.set_experiment(experiments_sm["experiment_name"])
    ##2- Lancement du RUN
    with mlflow.start_run(run_name=experiments_sm["run_name"]) as run:
        # récupération de l'id du run
        run_id = run.info.run_id
        logger.debug(f"Run ID {run_id}")
        # 1ers tags liés au Dataset
        mlflow.set_tag("dataset_name",dataset_tagname )
        mlflow.log_artifact(dataset_desc,"dataset_info")
        mlflow.log_param("Dataset", dataset_tagname)
        mlflow.log_artifacts(dataset_path)
        mlflow.log_params(hyperparams)
        
        logger.debug("Hyperparamètres")
        logger.debug(str(hyperparams))
    
        # 2 - Initialisation d'informations supplémentaire pour le projet
        directory=os.path.join(tuning_history_subfolder,f"tuning_{hyperparams['archi']}_{run_id}")
        project_name=(f"tuning_{hyperparams['archi']}_{run_id}")
        logger.debug(f"Tuning Directory {directory}")
        logger.debug(f"Tuning project_name {project_name}")
        # 3 - Construction du modèle via Keras Tuning
        best_model = build.tuner_randomsearch(hyperparams,run_id,directory,project_name,X_train,y_train)

        # 4 - Entrainement du modèle
        trained_model,metrics,history=train.train_model(best_model,hyperparams,X_train,y_train,run_id)
        # 5 - Evaluation du modèle et génération de metriques supplémentaires
        final_metrics,conf_matrix_df,class_report_df=predict.evaluate_model(trained_model,X_eval,y_eval,hyperparams["num_classes"],classes_dic)
        # mise à jour de metrics avec les métriques supplementaire d'evaluation
        metrics.update(final_metrics)
        mlflow.log_metrics(metrics)
        
        # 5 - Sauvegarde des éléments générés : modèle / history / training plots / confusion matrix / classification report
        model_filename = f"model_{hyperparams['archi']}_{run_id}.keras"
        history_file_name=f"history_{hyperparams['archi']}_{run_id}.json"
        um.save_model(trained_model,os.path.join(models_files_subfolder,model_filename))
        history_filepath=um.save_history_json(history,os.path.join(history_subfolder,history_file_name))
        um.generate_training_plots_from_json(history_filepath,training_plot_subfolder,run_id)
        conf_matrix_df.to_csv(os.path.join(reports_subfolder,f"model_conf_m_{hyperparams['archi']}_{run_id}.csv"))
        um.plot_and_save_conf_matrix(conf_matrix_df, classes_dic, reports_figures,f"model_conf_m_plot_{hyperparams['archi']}_{run_id}.png")
        class_report_df.to_csv(os.path.join(reports_subfolder,f"model_class_report_{hyperparams['archi']}_{run_id}.csv"))
        um.plot_and_save_classification_report(class_report_df, reports_figures,f"model_class_report_{hyperparams['archi']}_{run_id}.png")
        
        '''
        ## Gradcam
        if hyperparams['archi'] in ['VGG16','VGG19']:
            logger.debug(f"Interprétabilité pour {hyperparams['archi']}")
            directory_path=os.path.join(main_path,paths["test_images"])
            um.gradcam_process_images(trained_model,f"{hyperparams['archi']}_{run_id}",directory_path,heatmap_subfolder,hyperparams['img_size'],hyperparams['num_classes'],hyperparams["last_conv_layer"])
        '''
            
def run_experiment_covid_pascovid(hyperparams): # binaire
    """
    Fonction d'exécution des experiments pour une classification Binaire Sain/Malade
    """
    logger.debug("-----------run_experiment_covid_pascovid------------")
    
    ## 1 - Récupération des informations sur les données à utiliser
    # Initialisation du nom de répertoire dédié à la classification
    classif_folder=os.path.join(main_path,paths["model_outputs"],experiments_cv["classif_folder"])
    ## 1b - Création de tous les répertoires et sous répertoires liés au type de classification
    # classif_folder: sous-répertoire principal de la classification
    logger.debug(f"Création du répertoire de classification s'il 'existe pas : {classif_folder}")
    os.makedirs(os.path.dirname(classif_folder), exist_ok=True)
    # history
    history_subfolder=os.path.join(classif_folder,paths["model_history_outputs"])
    logger.debug(f"Création du répertoire d'history s'il 'existe pas : {history_subfolder}")
    os.makedirs(os.path.dirname(history_subfolder), exist_ok=True)
    # training plots
    training_plot_subfolder=os.path.join(classif_folder,paths["model_training_plot_outputs"])
    logger.debug(f"Création du répertoire de training_plot s'il 'existe pas : {training_plot_subfolder}")
    os.makedirs(os.path.dirname(training_plot_subfolder), exist_ok=True)
    # tuning history
    tuning_history_subfolder=os.path.join(classif_folder,paths["tuning_history_outputs"])
    logger.debug(f"Création du répertoire de tuning_history s'il 'existe pas : {tuning_history_subfolder}")
    os.makedirs(os.path.dirname(tuning_history_subfolder), exist_ok=True)
    # reports
    reports_subfolder=os.path.join(classif_folder,paths["reports_outputs"])
    logger.debug(f"Création du répertoire de report s'il 'existe pas : {reports_subfolder}")
    os.makedirs(os.path.dirname(reports_subfolder), exist_ok=True)
    
    # reports
    heatmap_subfolder=os.path.join(classif_folder,paths["heatmap_output"])
    logger.debug(f"Création du répertoire de heatmap s'il 'existe pas : {heatmap_subfolder}")
    os.makedirs(os.path.dirname(reports_subfolder), exist_ok=True)
    
    # reports figures
    reports_figures=os.path.join(reports_subfolder,paths["reports_figures"])
    logger.debug(f"Création du répertoire de report digures s'il n'existe pas : {reports_figures}")
    os.makedirs(os.path.dirname(reports_figures), exist_ok=True)
    
    # models
    models_files_subfolder=os.path.join(classif_folder,"models_files")
    logger.debug(f"Création du répertoire de report s'il 'existe pas : {models_files_subfolder}")
    os.makedirs(os.path.dirname(models_files_subfolder), exist_ok=True)
    
    # récupération du nom du fichier de description (fichier json)
    classes_dic=experiments_cv["classes_dic"]
    # récupération du dataset en fonction avec ou Sans Lung Opacity
    if hyperparams["data_lo_hlo"]=="HLO":   
        dataset_path=os.path.join(dataset_folder,dataset_cv_hlo["current_dataset_path"])
        dataset_desc=os.path.join(dataset_path,dataset_cv_hlo["dataset_desc"])
        dataset_tagname=dataset_cv_hlo["tag_name"]     
    else:
        dataset_path=os.path.join(dataset_folder,dataset_cv["current_dataset_path"])
        dataset_desc=os.path.join(dataset_path,dataset_cv["dataset_desc"])
        dataset_tagname=dataset_cv["tag_name"]              
    logger.debug("Dataset path")
    logger.debug(f"{dataset_path}")
    logger.debug("Dataset desc")
    logger.debug(f"{dataset_desc}")
    logger.debug("Dataset tagname")
    logger.debug(f"{dataset_tagname}")
    logger.debug("Dictionnaire classes_dic")
    logger.debug(f"{classes_dic}")
    ## Split des données en données d'entrainement (qui seront splittés en données entrainement/validation) et données d'évaluation
    ## Pour cela, nous prevoyons 20% de volume supplémentaire (ex 750 au lieu de 600)
    logger.debug("Split des données X_train, y_train, X_eva,l y_eval")
    X_train, y_train, X_eval, y_eval,dic=up.get_data_from_parent_directory_labelled(dataset_path, classes_dic, hyperparams['img_size'],hyperparams['img_dim'],hyperparams['archi'])
    logger.debug("Directory to Label ")
    logger.debug(f"{dic}")
    mlflow.set_experiment(experiments_cv["experiment_name"])
    ##2- Lancement du RUN
    with mlflow.start_run(run_name=experiments_cv["run_name"]) as run:
        # récupération de l'id du run
        run_id = run.info.run_id
        logger.debug(f"Run ID {run_id}")
        # 1ers tags liés au Dataset
        mlflow.set_tag("dataset_name",dataset_tagname )
        mlflow.log_artifact(dataset_desc,"dataset_info")
        mlflow.log_param("Dataset", dataset_tagname)
        mlflow.log_artifacts(dataset_path)
        mlflow.log_params(hyperparams)
        
        logger.debug("Data path")
        logger.debug("Hyperparamètres")
        logger.debug(str(hyperparams))
        
        # 2 - Initialisation d'informations supplémentaire pour le projet
        directory=os.path.join(tuning_history_subfolder,f"tuning_{hyperparams['archi']}_{run_id}")
        project_name=(f"tuning_{hyperparams['archi']}_{run_id}")
        logger.debug(f"Tuning Directory {directory}")
        logger.debug(f"Tuning project_name {project_name}")
        # 3 - Construction du modèle via Keras Tuning
        best_model = build.tuner_randomsearch(hyperparams,run_id,directory,project_name,X_train,y_train)

        # 4 - Entrainement du modèle
        trained_model,metrics,history=train.train_model(best_model,hyperparams,X_train,y_train,run_id)
        # 5 - Evaluation du modèle et génération de metriques supplémentaires
        final_metrics,conf_matrix_df,class_report_df=predict.evaluate_model(trained_model,X_eval,y_eval,hyperparams["num_classes"],classes_dic)
        # mise à jour de metrics avec les métriques supplementaire d'evaluation
        metrics.update(final_metrics)
        mlflow.log_metrics(metrics)
        
        # 5 - Sauvegarde des éléments générés : modèle / history / training plots / confusion matrix / classification report
        model_filename = f"model_{hyperparams['archi']}_{run_id}.keras"
        history_file_name=f"history_{hyperparams['archi']}_{run_id}.json"
        um.save_model(trained_model,os.path.join(models_files_subfolder,model_filename))
        history_filepath=um.save_history_json(history,os.path.join(history_subfolder,history_file_name))
        um.generate_training_plots_from_json(history_filepath,training_plot_subfolder,run_id)
        conf_matrix_df.to_csv(os.path.join(reports_subfolder,f"model_conf_m_{hyperparams['archi']}_{run_id}.csv"))
        um.plot_and_save_conf_matrix(conf_matrix_df, classes_dic, reports_figures,f"model_conf_m_plot_{hyperparams['archi']}_{run_id}.png")
        class_report_df.to_csv(os.path.join(reports_subfolder,f"model_class_report_{hyperparams['archi']}_{run_id}.csv"))
        um.plot_and_save_classification_report(class_report_df, reports_figures,f"model_class_report_{hyperparams['archi']}_{run_id}.png")
        
        '''
        ## Gradcam
        if hyperparams['archi'] in ['VGG16','VGG19']:
            logger.debug(f"Interprétabilité pour {hyperparams['archi']}")
            directory_path=os.path.join(main_path,paths["test_images"])
            um.gradcam_process_images(trained_model,f"{hyperparams['archi']}_{run_id}",directory_path,heatmap_subfolder,hyperparams['img_size'],hyperparams['num_classes'],hyperparams["last_conv_layer"])
        '''
            
def run_experiment_3classes(hyperparams): # multiclasse
    """
    Fonction d'exécution des experiments pour une classification multiple
    """
    logger.debug("-----------run_experiment_3classes------------")
    ## 1 - Récupération des informations sur les données à utiliser
    # récupération du dataset
    dataset_path=os.path.join(dataset_folder,dataset_3c["current_dataset_path"])
    # Initialisation du nom de répertoire dédié à la classification
    classif_folder=os.path.join(main_path,paths["model_outputs"],experiments_3c["classif_folder"])
    ## 1b - Création de tous les répertoires et sous répertoires liés au type de classification
    # classif_folder: sous-répertoire principal de la classification
    logger.debug(f"Création du répertoire de classification s'il 'existe pas : {classif_folder}")
    os.makedirs(os.path.dirname(classif_folder), exist_ok=True)
    # history
    history_subfolder=os.path.join(classif_folder,paths["model_history_outputs"])
    logger.debug(f"Création du répertoire d'history s'il 'existe pas : {history_subfolder}")
    os.makedirs(os.path.dirname(history_subfolder), exist_ok=True)
    # training plots
    training_plot_subfolder=os.path.join(classif_folder,paths["model_training_plot_outputs"])
    logger.debug(f"Création du répertoire de training_plot s'il 'existe pas : {training_plot_subfolder}")
    os.makedirs(os.path.dirname(training_plot_subfolder), exist_ok=True)
    # tuning history
    tuning_history_subfolder=os.path.join(classif_folder,paths["tuning_history_outputs"])
    logger.debug(f"Création du répertoire de tuning_history s'il 'existe pas : {tuning_history_subfolder}")
    os.makedirs(os.path.dirname(tuning_history_subfolder), exist_ok=True)
    # reports
    reports_subfolder=os.path.join(classif_folder,paths["reports_outputs"])
    logger.debug(f"Création du répertoire de report s'il 'existe pas : {reports_subfolder}")
    os.makedirs(os.path.dirname(reports_subfolder), exist_ok=True)
    
    # reports
    heatmap_subfolder=os.path.join(classif_folder,paths["heatmap_output"])
    logger.debug(f"Création du répertoire de heatmap s'il 'existe pas : {heatmap_subfolder}")
    os.makedirs(os.path.dirname(reports_subfolder), exist_ok=True)
    
    # reports figures
    reports_figures=os.path.join(reports_subfolder,paths["reports_figures"])
    logger.debug(f"Création du répertoire de report digures s'il n'existe pas : {reports_figures}")
    os.makedirs(os.path.dirname(reports_figures), exist_ok=True)
    
    # models
    models_files_subfolder=os.path.join(classif_folder,"models_files")
    logger.debug(f"Création du répertoire de report s'il 'existe pas : {models_files_subfolder}")
    os.makedirs(os.path.dirname(models_files_subfolder), exist_ok=True)
    # récupération du nom du fichier de description (fichier json)
    dataset_desc=os.path.join(dataset_path,dataset_3c["dataset_desc"])
    classes_dic=experiments_3c["classes_dic"]
    dataset_tagname=dataset_3c["tag_name"]
    logger.debug("Dataset Path")
    logger.debug(dataset_path)
    logger.debug("Dictionnaire")
    logger.debug(f"{classes_dic}")
    
    ## Split des données en données d'entrainement (qui seront splittés en données entrainement/validation) et données d'évaluation
    ## Pour cela, nous prevoyons 20% de volume supplémentaire (ex 750 au lieu de 600)
    logger.debug("Split des données X_train, y_train, X_eva,l y_eval")
    X_train, y_train, X_eval, y_eval,dic=up.get_data_from_parent_directory_labelled(dataset_path, classes_dic, hyperparams['img_size'],hyperparams['img_dim'],hyperparams['archi'])
    ## Traiter le cas multiclasse dans les run MC et 3classes - one hot encoding
    #logger.debug("---Multiclasse - One hot encoding----")
    #y_train = to_categorical(y_train,  hyperparams['num_classes'])
    #y_eval = to_categorical(y_eval, hyperparams['num_classes'])

    logger.debug("Directory to Label ")
    logger.debug(f"{dic}")
    mlflow.set_experiment(experiments_3c["experiment_name"])
    ##2- Lancement du RUN
    with mlflow.start_run(run_name=experiments_3c["run_name"]) as run:
        # récupération de l'id du run
        run_id = run.info.run_id
        logger.debug(f"Run ID {run_id}")
        # 1ers tags liés au Dataset
        mlflow.set_tag("dataset_name",dataset_tagname )
        mlflow.log_artifact(dataset_desc,"dataset_info")
        mlflow.log_param("Dataset", dataset_tagname)
        mlflow.log_artifacts(dataset_path)
        mlflow.log_params(hyperparams)
        
        logger.debug("Data path")
        logger.debug("Hyperparamètres")
        logger.debug(str(hyperparams))
        
        # 2 - Initialisation d'informations supplémentaire pour le projet
        directory=os.path.join(tuning_history_subfolder,f"tuning_{hyperparams['archi']}_{run_id}")
        project_name=(f"tuning_{hyperparams['archi']}_{run_id}")
        logger.debug(f"Tuning Directory {directory}")
        logger.debug(f"Tuning project_name {project_name}")
        # 3 - Construction du modèle via Keras Tuning
        best_model = build.tuner_randomsearch(hyperparams,run_id,directory,project_name,X_train,y_train)

        # 4 - Entrainement du modèle
        trained_model,metrics,history=train.train_model(best_model,hyperparams,X_train,y_train,run_id)
        # 5 - Evaluation du modèle et génération de metriques supplémentaires
        final_metrics,conf_matrix_df,class_report_df=predict.evaluate_model(trained_model,X_eval,y_eval,hyperparams["num_classes"],classes_dic)
        # mise à jour de metrics avec les métriques supplementaire d'evaluation
        metrics.update(final_metrics)
        mlflow.log_metrics(metrics)
        
        # 5 - Sauvegarde des éléments générés : modèle / history / training plots / confusion matrix / classification report
        model_filename = f"model_{hyperparams['archi']}_{run_id}.keras"
        history_file_name=f"history_{hyperparams['archi']}_{run_id}.json"
        um.save_model(trained_model,os.path.join(models_files_subfolder,model_filename))
        history_filepath=um.save_history_json(history,os.path.join(history_subfolder,history_file_name))
        um.generate_training_plots_from_json(history_filepath,training_plot_subfolder,run_id)
        conf_matrix_df.to_csv(os.path.join(reports_subfolder,f"model_conf_m_{hyperparams['archi']}_{run_id}.csv"))
        um.plot_and_save_conf_matrix(conf_matrix_df, classes_dic, reports_figures,f"model_conf_m_plot_{hyperparams['archi']}_{run_id}.png")
        class_report_df.to_csv(os.path.join(reports_subfolder,f"model_class_report_{hyperparams['archi']}_{run_id}.csv"))
        um.plot_and_save_classification_report(class_report_df, reports_figures,f"model_class_report_{hyperparams['archi']}_{run_id}.png")
        
        '''
        ## Gradcam
        if hyperparams['archi'] in ['VGG16','VGG19']:
            logger.debug(f"Interprétabilité pour {hyperparams['archi']}")
            directory_path=os.path.join(main_path,paths["test_images"])
            um.gradcam_process_images(trained_model,f"{hyperparams['archi']}_{run_id}",directory_path,heatmap_subfolder,hyperparams['img_size'],hyperparams['num_classes'],hyperparams["last_conv_layer"])
        '''
            
## Fonction main principale
if __name__ == "__main__":
    for hyperparams in hyperparams_list:
        if hyperparams["num_classes"]==1: #binaire
            run_experiment_sain_malade(hyperparams) # Binaire
            run_experiment_covid_pascovid(hyperparams) # Binaire
        elif hyperparams["num_classes"]==4: # multiple  
            run_experiment_MC(hyperparams) # Multiclasse
        elif hyperparams["num_classes"]==3: # 3 classes
            run_experiment_3classes(hyperparams) # Multiclasse
        
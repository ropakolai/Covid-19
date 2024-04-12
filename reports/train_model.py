'''
Créé le 12/03/2024

@author: Equipe DS Jan 2024
@summary: Fonctions de construction et entrainement des modèles
Prend en argument les hyperparamètres des fichiers de configuration
Les étapes:
1- >>features/build_features.py<< Preprocessing: Utilisation de  pour le sampling et preprocessing
2- >>models/build_model.py<< Définir un bloc de construction pour chaque architecture de modèle (option Dégel ou non de couches)
3- >>models/build_model.py<< Identifier les spécificités entre Binaire et Multiple dans la construction du modèle et la visualisation du résultat
4- >>models/build_model.py<< Identifier les paramètres à ajuster pour Keras_tuner
5- >>models/build_model.py<< Définir les conditions de EarlyStopping
6- >>models/build_model.py<< Lancer l'execution sur X epochs avec Hyperband
7- >>models/build_model.py<< Récupération du meilleur modèle

8- >>models/train_model.py<< Entrainement du modèle sur le nombre d'epoch
9- >>models/train_model.py<< Récupération du modèle entrainé
10->>models/train_model.py<< Récupération des métriques
11->>models/train_model.py<< Métriques avancées: Récupération du nom de la dernière couche de convolution pour affichage de GradCam
12->>models/predict_model.py << 
13->>visualization/visualize.py<< Affichage de la Matrice de Confusion avec pour objectif métier COVID de minimiser les Faux Negatifs COVID

Les modèles
- LeNet Baseline
- ResNet50
- VGG16
- VGG19 (Idéalement en Feature extraction)
- EfficientNetB0
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

## Import des modules de logging et timestamping
from datetime import datetime # timestamping
import logging
## Import du fichier de configuration pour des informations sur les Logs
from config import paths,infolog
## Import des modules de configuration

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


##
# 1 - Entrainement du modèle
# Ici il sera basé sur le meilleur modèle identifié par Keras Tuner
##

def train_model(model,ml_hp,X,y,run_id):
    """
    Entrainement du modèle
    Args: 
    - model: Le modele construit pour entrainement
    - ml_hp: Hyperparamètres de MLFlow
    - X,y : Les données d'entrainement
    - run_id : Pour le nommage des fichiers de callback (CSVLogger)
    Returns: Modèle, Métriques (de base, les autres seront calculées dans MLFlow), History, Temps d'exécution
    """
    logger.debug("--------- train_model ---------")
    # 1 - Récupération de hyperparamètres MLFlow pour l'entrainement du modèle
    # max_epochs
    max_epochs=ml_hp['max_epochs']
    
    # 2 - Split des données pour entrainement
    X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2, random_state=1234)
    logger.debug("Infos X, y")
    logger.debug(f"taille X {len(X)}")
    logger.debug(f"taille y {len(y)}")
    logger.debug("Infos SPLIT ")
    logger.debug(f"taille X train {len(X_train)}")
    logger.debug(f"taille y train {len(y_train)}")
    logger.debug(f"taille X val {len(X_val)}")
    logger.debug(f"taille y val {len(y_val)}")
    logger.debug("Infos Shape X, y")
    logger.debug(f"taille X {X.shape}")
    logger.debug(f"taille y {y.shape}")
    logger.debug("Infos Shape SPLIT ")
    logger.debug(f"taille X train {X_train.shape}")
    logger.debug(f"taille y train {y_train.shape}")
    logger.debug(f"taille X val {X_val.shape}")
    logger.debug(f"taille y val {y_val.shape}")
    ## 2 - Callbacks: Early Stopping, CSV Logger, LearningRateScheduler. Patience 10
    # Définir l'arrêt anticipé
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
    main_models_log=os.path.join(main_path,infolog["training_log_path"])
    csv_logger = CSVLogger(os.path.join(main_models_log,f"modeL_training_log_{run_id}.csv"), append=True, separator=';')
    # ajustement du taux d'apprentissage
    # Fonction pour définir le taux d'apprentissage, ici on régle le learning rate au bout de 10 epoch
    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * np.exp(-0.1)
    lr_scheduler = LearningRateScheduler(scheduler, verbose=1)    
    
    
    debut = time.time()  # Enregistre le temps avant l'exécution de la fonction
    ## Entrainement du modèle
    history = model.fit(X_train, y_train,
                         epochs=max_epochs,
                         validation_data=(X_val, y_val), callbacks=[early_stopping,lr_scheduler,csv_logger])
    
    
    
    fin = time.time()  # Enregistre le temps avant l'exécution de la fonction
    temps_execution = round(fin - debut,2)/60
    logger.debug(f"Model training time {temps_execution} min")
    metrics = {
    'accuracy': max(history.history['accuracy']),
    'val_accuracy': max(history.history['val_accuracy']),
    'loss': max(history.history['loss']),
    'val_loss': max(history.history['val_loss']),
    }
    
    return model,metrics,history

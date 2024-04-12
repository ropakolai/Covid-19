'''
Créé le 27 mars 2024

@author: Equipe DS Jan 2024
2- >>models/build_model.py<< Définir un bloc de construction pour chaque architecture de modèle (option Dégel ou non de couches)
3- >>models/build_model.py<< Identifier les spécificités entre Binaire et Multiple dans la construction du modèle et la visualisation du résultat
4- >>models/build_model.py<< Identifier les paramètres à ajuster pour Keras_tuner
5- >>models/build_model.py<< Définir les conditions de EarlyStopping
6- >>models/build_model.py<< Lancer l'execution sur X epochs avec RandomSearch
7- >>models/build_model.py<< Récupération du meilleur modèle

'''
#####
# Imports 
#####
### Import des modèles
# VGG16
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as pp_vgg16 
# VGG19
from tensorflow.keras.applications.vgg19 import VGG19 # VGG16
from tensorflow.keras.applications.vgg19 import preprocess_input as pp_vgg19
# ResNet50
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as pp_resnet50
# EfficientNetB0
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input as pp_efficientnet

## Import des modules liées au réseau de neurone
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization,Flatten,Dropout,MaxPooling2D,Conv2D # couches etc.
from tensorflow.keras.models import Model,Sequential # modèles
from tensorflow.keras.optimizers import Adam,SGD # Optimiseurs
from tensorflow.keras.losses import BinaryCrossentropy # Fonctions de perte
from tensorflow.keras.regularizers import l2 # Régularizers
from tensorflow.keras.callbacks import TensorBoard, Callback, EarlyStopping,CSVLogger,LearningRateScheduler# Callbacks

## Import keras tuner
from kerastuner.tuners import RandomSearch, Hyperband

## Modules de métriques
from tensorflow.keras.metrics import Accuracy # Métriques
from sklearn.metrics import recall_score, f1_score, confusion_matrix,accuracy_score,classification_report

## Import des modules utiles
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

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
# 1 - Fonction build_model qui sera utilisée par Keras Tuner et qui renvoit vers la fonction spécifique à l'architecture
##
def build_model(hp, ml_hp):
    """
    ###
    ## Fonction build_model prend en argument les hyperparams et redirige l'exécution à une fonction dédiée au modèle
    ## En effet, la configuration des modèles pré-entrainés diverge en particulier sur les hyperparamètres et en dégel des couches
    ## Le but est aussi de pouvoir enrichir les fonctions du travail de chaque membre de l'équipe
    ###
    
    Args:
    ml_hp (List): Un Liste d'hyperparamètres MLFlow contenant des informations.

    Returns:
    keras model: Un modèle keras.

    Raises:
    ValueError: Si le dictionnaire est vide ou si les valeurs ne sont pas des nombres.
    """
    logging.debug("---------build_model------------")
    archi=ml_hp['archi']
    archi=archi.lower()
    if archi=='lenet':
        return build_model_lenet(hp, ml_hp)
            
    elif archi=='vgg16':
        return build_model_vgg16(hp, ml_hp)
        
    elif archi=='vgg19':
        return build_model_vgg19(hp, ml_hp)
        
    elif archi=='resnet50':
        return build_model_resnet50(hp, ml_hp)
        
    elif archi=='efficientnetb0':
        return build_model_efficientnetb0(hp, ml_hp)
     
    else:
        logging.debug(f"Erreur - Nom d'architecture du modèle incorrecte - {archi}")
        return None

##
# 1 fonction par architecture - LeNet
##
def build_model_lenet(hp, ml_hp):
    """
    Construit un modèle d'architecture LeNet pour la classification d'image (binaire ou multi-classes)
    Utilise à la fois des hyperparamètres utiles pour keras tuner, et des hyperparamètres généraux d'un fichier de configuration
    Ce fichier de configuration contiendra des informations tel que: 
    - archi: qui définit l'architecture à construire
    - img_size: qui définit la taille de l'image à utiliser dans la construction du modèle
    - img_dim: Dimensions (RGB 3 ou Niveau de Gris 1)
    - ETC

    Args:
        hp: objet HyperParameters de Keras Tuner pour optimiser les hyperparamètres du modèle.
        ml_hp: objet HyperParameters de MLFlow
        
    Returns:
        modèle Keras compilé avec les hyperparamètres optimisés.
    """
    logger.debug("---------build_model_lenet------------")
    
    # 1a - Récupération de hyperparamètres MLFlow
    # A terme nous pouvons aussi y inclure la liste des valeurs pour keras tuner 
    archi = ml_hp["archi"] # nom de l'architecture
    img_size =  ml_hp["img_size"] # dimensions de l'image 
    img_dim =  ml_hp["img_dim"] # niveau de couleur  (RGB 3 ou Niveau de Gris 1)
    num_classes = ml_hp["num_classes"] # niveau de couleur  (RGB 3 ou Niveau de Gris 1)
    hidden_layers_activation = ml_hp["hl_activation"] # Fonction d'activation des hidden layers, nous utiliserons toujours 'relu' pour l'instant
    
    # 1b - Initialisation de variables supplémentaire sur la bases des hyperparamètres mlflow_archive
    shape = (img_size,img_size,img_dim) # shape
    ## particularité selon la classification : fonction de perte
    ## Note : Le one_hot_encoding sera appliqué dans la fonction principale de MLFlow (preprocessing)
    if num_classes == 1: # classification binaire
        logger.debug("--- CLASSIFICATION BINAIRE ------")
        loss_function = 'binary_crossentropy'
        output_activation="sigmoid"
    else:
        logger.debug("--- CLASSIFICATION MULTIPLE ------")
        loss_function = 'sparse_categorical_crossentropy'
        output_activation="softmax"
        
    ## LOGGING
    logger.debug("--- Hyperparamètres ---")
    logger.debug(f"Archi = {archi}")
    logger.debug(f"img_size = {img_size}")
    logger.debug(f"img_dim = {img_dim}")
    logger.debug(f"shape = {shape}")
    logger.debug(f"num_classes = {num_classes}")
    logger.debug(f"hidden_layers_activation = {hidden_layers_activation}")
    logger.debug(f"loss_function = {loss_function}")
    logger.debug(f"output_activation = {output_activation}")
    
    # 2a - Définition les hyperparamètres à optimiser à l'aide de l'objet HyperParameters de Keras Tuner
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]) # 0.01 / 0.001 et 0.0001
    units = hp.Int('units', min_value=32, max_value=512, step=32)
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
    l2_lambda = hp.Choice('l2_lambda', values=[1e-3, 1e-4, 1e-5])  #Hyperparamètre pour la régularisation L2
    num_dropout_layers = hp.Int('num_dropout_layers', min_value=1, max_value=5)
    logger.debug("--- Définition des HP à optimiser ---")
    logger.debug(f"learning_rate = {learning_rate}")
    logger.debug(f"units = {units}")
    logger.debug(f"dropout_rate = {dropout_rate}")
    logger.debug(f"l2_lambda = {l2_lambda}")
    logger.debug(f"num_dropout_layers = {num_dropout_layers}")
    
    # 3 - Construction du modèle de base
    model = Sequential()
    model.add(Conv2D(filters = 30,                   # Nombre de filtres
                    kernel_size = (5, 5),            # Shape du kernel
                    input_shape = shape,       # Shape de l'entrée
                    activation = hidden_layers_activation, kernel_regularizer=l2(l2_lambda)))            # Fonction d'activation
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(filters = 16,                    
                    kernel_size = (3, 3),
                    activation = hidden_layers_activation,  kernel_regularizer=l2(l2_lambda)))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(rate = dropout_rate))
    model.add(Flatten())
    model.add(Dense(units = 128, activation = hidden_layers_activation, kernel_regularizer=l2(l2_lambda)))
    
    # 4 - Dégel de couches (Non applicable)
    
    # 6 - Ajout de couches de Dropout
    for _ in range(num_dropout_layers):  # Ajoutez jusqu'à 5 couches de dropout
        model.add(Dropout(dropout_rate))
        
    # 7 - Couche de sortie 
    model.add(Dense(units = num_classes, activation = output_activation))
    
    # 8 - Finalisation de la création du modèle
    
    #logger.debug(f"Model summary EfficientNetB0")
    #logger.debug(model.summary())
    # 9 - Compilation du modèle avec les hyperparamètres et les fonctions de perte lié au type de classification
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=loss_function, # binary_crossentropy ou sparse_categorical_crossentropy
                  metrics=['accuracy'])
    
    return model
##
# 1 fonction par architecture - VGG16
##
def build_model_vgg16(hp, ml_hp):
    """
    Construit un modèle d'architecture VGG16 pour la classification d'image (binaire ou multi-classes)
    Utilise à la fois des hyperparamètres utiles pour keras tuner, et des hyperparamètres généraux d'un fichier de configuration
    Ce fichier de configuration contiendra des informations tel que: 
    - archi: qui définit l'architecture à construire
    - img_size: qui définit la taille de l'image à utiliser dans la construction du modèle
    - img_dim: Dimensions (RGB 3 ou Niveau de Gris 1)
    - ETC

    Args:
        hp: objet HyperParameters de Keras Tuner pour optimiser les hyperparamètres du modèle.
        ml_hp: objet HyperParameters de MLFlow
        
    Returns:
        modèle Keras compilé avec les hyperparamètres optimisés.
    """
    logger.debug("---------build_model_vgg16------------")
    
    # 1a - Récupération de hyperparamètres MLFlow
    # A terme nous pouvons aussi y inclure la liste des valeurs pour keras tuner 
    archi = ml_hp["archi"] # nom de l'architecture
    img_size =  ml_hp["img_size"] # dimensions de l'image 
    img_dim =  ml_hp["img_dim"] # niveau de couleur  (RGB 3 ou Niveau de Gris 1)
    num_classes = ml_hp["num_classes"] # niveau de couleur  (RGB 3 ou Niveau de Gris 1)
    hidden_layers_activation = ml_hp["hl_activation"] # Fonction d'activation des hidden layers, nous utiliserons toujours 'relu' pour l'instant
    
    # 1b - Initialisation de variables supplémentaire sur la bases des hyperparamètres mlflow_archive
    shape = (img_size,img_size,img_dim) # shape
    ## particularité selon la classification : fonction de perte
    ## Note : Le one_hot_encoding sera appliqué dans la fonction principale de MLFlow (preprocessing)
    if num_classes == 1: # classification binaire
        logger.debug("--- CLASSIFICATION BINAIRE ------")
        loss_function = 'binary_crossentropy'
        output_activation="sigmoid"
    else:
        logger.debug("--- CLASSIFICATION MULTIPLE ------")
        loss_function = 'sparse_categorical_crossentropy'
        output_activation="softmax"
        
    ## LOGGING
    logger.debug("--- Hyperparamètres ---")
    logger.debug(f"Archi = {archi}")
    logger.debug(f"img_size = {img_size}")
    logger.debug(f"img_dim = {img_dim}")
    logger.debug(f"shape = {shape}")
    logger.debug(f"num_classes = {num_classes}")
    logger.debug(f"hidden_layers_activation = {hidden_layers_activation}")
    logger.debug(f"loss_function = {loss_function}")
    logger.debug(f"output_activation = {output_activation}")
    
    # 2 - Définition les hyperparamètres à optimiser à l'aide de l'objet HyperParameters de Keras Tuner
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]) # 0.01 / 0.001 et 0.0001
    units = hp.Int('units', min_value=32, max_value=512, step=32)
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
    l2_lambda = hp.Choice('l2_lambda', values=[1e-3, 1e-4, 1e-5])  #Hyperparamètre pour la régularisation L2
    num_dropout_layers = hp.Int('num_dropout_layers', min_value=1, max_value=5)

    logger.debug("--- Définition des HP à optimiser ---")
    logger.debug(f"learning_rate = {learning_rate}")
    logger.debug(f"units = {units}")
    logger.debug(f"dropout_rate = {dropout_rate}")
    logger.debug(f"l2_lambda = {l2_lambda}")
    logger.debug(f"num_dropout_layers = {num_dropout_layers}")
    
    # 3 - Chargement du modèle de base (si applicable)
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=shape)

    # 4 - Dégel de couches (si applicable)
    for layer in base_model.layers:
        if 'block5' in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False
      
    # 5 - Ajout de couches Fully Connected
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(units, activation=hidden_layers_activation, kernel_regularizer=l2(l2_lambda))(x)  # Utilisation de la régularisation L2

    # 6 - Ajout de couches de Dropout
    for _ in range(num_dropout_layers):  # Ajoutez jusqu'à 5 couches de dropout
        x = Dropout(dropout_rate)(x)
        
    # 7 - Couche de sortie 
    output = Dense(num_classes, activation=output_activation)(x)
    
    # 8 - Finalisation de la création du modèle
    model = Model(inputs=base_model.input, outputs=output)
    
    logger.debug(f"Base model Input {base_model.input}")
    logger.debug(f"Model Output {output}")
    #logger.debug(f"Model summary VGG16")
    #logger.debug(model.summary())
    # 9 - Compilation du modèle avec les hyperparamètres et les fonctions de perte lié au type de classification
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=loss_function, # binary_crossentropy ou sparse_categorical_crossentropy
                  metrics=['accuracy'])
    return model

##
# 1 fonction par architecture - VGG19
##
def build_model_vgg19(hp, ml_hp):
    """
    Construit un modèle d'architecture VGG19 pour la classification d'image (binaire ou multi-classes)
    Utilise à la fois des hyperparamètres utiles pour keras tuner, et des hyperparamètres généraux d'un fichier de configuration
    Ce fichier de configuration contiendra des informations tel que: 
    - archi: qui définit l'architecture à construire
    - img_size: qui définit la taille de l'image à utiliser dans la construction du modèle
    - img_dim: Dimensions (RGB 3 ou Niveau de Gris 1)
    - ETC

    Args:
        hp: objet HyperParameters de Keras Tuner pour optimiser les hyperparamètres du modèle.
        ml_hp: objet HyperParameters de MLFlow
        
    Returns:
        modèle Keras compilé avec les hyperparamètres optimisés.
    """
    logger.debug("---------build_model_vgg19------------")
    
    # 1a - Récupération de hyperparamètres MLFlow
    # A terme nous pouvons aussi y inclure la liste des valeurs pour keras tuner 
    archi = ml_hp["archi"] # nom de l'architecture
    img_size =  ml_hp["img_size"] # dimensions de l'image 
    img_dim =  ml_hp["img_dim"] # niveau de couleur  (RGB 3 ou Niveau de Gris 1)
    num_classes = ml_hp["num_classes"] # niveau de couleur  (RGB 3 ou Niveau de Gris 1)
    hidden_layers_activation = ml_hp["hl_activation"] # Fonction d'activation des hidden layers, nous utiliserons toujours 'relu' pour l'instant
    
    # 1b - Initialisation de variables supplémentaire sur la bases des hyperparamètres mlflow_archive
    shape = (img_size,img_size,img_dim) # shape
    ## particularité selon la classification : fonction de perte
    ## Note : Le one_hot_encoding sera appliqué dans la fonction principale de MLFlow (preprocessing)
    if num_classes == 1: # classification binaire
        logger.debug("--- CLASSIFICATION BINAIRE ------")
        loss_function = 'binary_crossentropy'
        output_activation="sigmoid"
    else:
        logger.debug("--- CLASSIFICATION MULTIPLE ------")
        loss_function = 'sparse_categorical_crossentropy'
        output_activation="softmax"
        
    ## LOGGING
    logger.debug("--- Hyperparamètres ---")
    logger.debug(f"Archi = {archi}")
    logger.debug(f"img_size = {img_size}")
    logger.debug(f"img_dim = {img_dim}")
    logger.debug(f"shape = {shape}")
    logger.debug(f"num_classes = {num_classes}")
    logger.debug(f"hidden_layers_activation = {hidden_layers_activation}")
    logger.debug(f"loss_function = {loss_function}")
    logger.debug(f"output_activation = {output_activation}")
    
    # 2 - Définition les hyperparamètres à optimiser à l'aide de l'objet HyperParameters de Keras Tuner
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]) # 0.01 / 0.001 et 0.0001
    units = hp.Int('units', min_value=32, max_value=512, step=32)
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
    l2_lambda = hp.Choice('l2_lambda', values=[1e-3, 1e-4, 1e-5])  #Hyperparamètre pour la régularisation L2
    num_dropout_layers = hp.Int('num_dropout_layers', min_value=1, max_value=5)

    logger.debug("--- Définition des HP à optimiser ---")
    logger.debug(f"learning_rate = {learning_rate}")
    logger.debug(f"units = {units}")
    logger.debug(f"dropout_rate = {dropout_rate}")
    logger.debug(f"l2_lambda = {l2_lambda}")
    logger.debug(f"num_dropout_layers = {num_dropout_layers}")
    
    # 3 - Chargement du modèle de base (si applicable)
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=shape)
    
    # 4 - Dégel de couches (si applicable)
    for layer in base_model.layers:
        if 'block5' in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False
      
    # 5 - Ajout de couches Fully Connected
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(units, activation=hidden_layers_activation, kernel_regularizer=l2(l2_lambda))(x)  # Utilisation de la régularisation L2

    # 6 - Ajout de couches de Dropout
    for _ in range(num_dropout_layers):  # Ajoutez de couches de dropout
        x = Dropout(dropout_rate)(x)
        
    # 7 - Couche de sortie 
    output = Dense(num_classes, activation=output_activation)(x)
    
    # 8 - Finalisation de la création du modèle
    model = Model(inputs=base_model.input, outputs=output)
    
    logger.debug(f"Base model Input {base_model.input}")
    logger.debug(f"Model Output {output}")
    #logger.debug(f"Model summary VGG16")
    #logger.debug(model.summary())
    # 9 - Compilation du modèle avec les hyperparamètres et les fonctions de perte lié au type de classification
    model.compile(optimizer=SGD(learning_rate=learning_rate),
                  loss=loss_function, # binary_crossentropy ou sparse_categorical_crossentropy
                  metrics=['accuracy'])
    return model

##
# 1 fonction par architecture - ResNet50
##
def build_model_resnet50(hp, ml_hp):
    """
    Construit un modèle d'architecture ResNet50 pour la classification d'image (binaire ou multi-classes)
    Utilise à la fois des hyperparamètres utiles pour keras tuner, et des hyperparamètres généraux d'un fichier de configuration
    Ce fichier de configuration contiendra des informations tel que: 
    - archi: qui définit l'architecture à construire
    - img_size: qui définit la taille de l'image à utiliser dans la construction du modèle
    - img_dim: Dimensions (RGB 3 ou Niveau de Gris 1)
    - ETC

    Args:
        hp: objet HyperParameters de Keras Tuner pour optimiser les hyperparamètres du modèle.
        ml_hp: objet HyperParameters de MLFlow
        
    Returns:
        modèle Keras compilé avec les hyperparamètres optimisés.
    """
    logger.debug("---------build_model_resnet50------------")
    
    # 1a - Récupération de hyperparamètres MLFlow
    # A terme nous pouvons aussi y inclure la liste des valeurs pour keras tuner 
    archi = ml_hp["archi"] # nom de l'architecture
    img_size =  ml_hp["img_size"] # dimensions de l'image 
    img_dim =  ml_hp["img_dim"] # niveau de couleur  (RGB 3 ou Niveau de Gris 1)
    num_classes = ml_hp["num_classes"] # niveau de couleur  (RGB 3 ou Niveau de Gris 1)
    hidden_layers_activation = ml_hp["hl_activation"] # Fonction d'activation des hidden layers, nous utiliserons toujours 'relu' pour l'instant
    
    # 1b - Initialisation de variables supplémentaire sur la bases des hyperparamètres mlflow_archive
    shape = (img_size,img_size,img_dim) # shape
    ## particularité selon la classification : fonction de perte
    ## Note : Le one_hot_encoding sera appliqué dans la fonction principale de MLFlow (preprocessing)
    if num_classes == 1: # classification binaire
        logger.debug("--- CLASSIFICATION BINAIRE ------")
        loss_function = 'binary_crossentropy'
        output_activation="sigmoid"
    else:
        logger.debug("--- CLASSIFICATION MULTIPLE ------")
        loss_function = 'sparse_categorical_crossentropy'
        output_activation="softmax"
        
    ## LOGGING
    logger.debug("--- Hyperparamètres ---")
    logger.debug(f"Archi = {archi}")
    logger.debug(f"img_size = {img_size}")
    logger.debug(f"img_dim = {img_dim}")
    logger.debug(f"shape = {shape}")
    logger.debug(f"num_classes = {num_classes}")
    logger.debug(f"hidden_layers_activation = {hidden_layers_activation}")
    logger.debug(f"loss_function = {loss_function}")
    logger.debug(f"output_activation = {output_activation}")
    
    # 2a - Définition les hyperparamètres à optimiser à l'aide de l'objet HyperParameters de Keras Tuner
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]) # 0.01 / 0.001 et 0.0001
    units = hp.Int('units', min_value=32, max_value=512, step=32)
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
    l2_lambda = hp.Choice('l2_lambda', values=[1e-3, 1e-4, 1e-5])  #Hyperparamètre pour la régularisation L2
    num_dropout_layers = hp.Int('num_dropout_layers', min_value=1, max_value=5)
    logger.debug("--- Définition des HP à optimiser ---")
    logger.debug(f"learning_rate = {learning_rate}")
    logger.debug(f"units = {units}")
    logger.debug(f"dropout_rate = {dropout_rate}")
    logger.debug(f"l2_lambda = {l2_lambda}")
    logger.debug(f"num_dropout_layers = {num_dropout_layers}")
    
    
    # 3 - Chargement du modèle de base (si applicable)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=shape)
    
    # 4 - Pas de Dégel de couches pour ResNet50
    base_model.trainable = False
         
    # 5 - Ajout de couches Fully Connected
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(units, activation=hidden_layers_activation, kernel_regularizer=l2(l2_lambda))(x)  # Utilisation de la régularisation L2
    
    # 6 - Ajout de couches de Dropout
    for _ in range(num_dropout_layers):  # Ajoutez jusqu'à 5 couches de dropout
        x = Dropout(dropout_rate)(x)
        
    # 7 - Couche de sortie 
    output = Dense(num_classes, activation=output_activation)(x)
    
    # 8 - Finalisation de la création du modèle
    model = Model(inputs=base_model.input, outputs=output)
    
    logger.debug(f"Base model Input {base_model.input}")
    logger.debug(f"Model Output {output}")
    #logger.debug(f"Model summary ResNet50")
    #logger.debug(model.summary())
    # 9 - Compilation du modèle avec les hyperparamètres et les fonctions de perte lié au type de classification
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=loss_function, # binary_crossentropy ou categorical_crossentropy
                  metrics=['accuracy'])
    
    return model


##
# 1 fonction par architecture - EfficientNetB0
##
def build_model_efficientnetb0(hp, ml_hp):
    """
    Construit un modèle d'architecture EfficientNetB0 pour la classification d'image (binaire ou multi-classes)
    Utilise à la fois des hyperparamètres utiles pour keras tuner, et des hyperparamètres généraux d'un fichier de configuration
    Ce fichier de configuration contiendra des informations tel que: 
    - archi: qui définit l'architecture à construire
    - img_size: qui définit la taille de l'image à utiliser dans la construction du modèle
    - img_dim: Dimensions (RGB 3 ou Niveau de Gris 1)
    - ETC

    Args:
        hp: objet HyperParameters de Keras Tuner pour optimiser les hyperparamètres du modèle.
        ml_hp: objet HyperParameters de MLFlow
        
    Returns:
        modèle Keras compilé avec les hyperparamètres optimisés.
    """
    logger.debug("---------build_model_efficientnetb0------------")
    
    # 1a - Récupération de hyperparamètres MLFlow
    # A terme nous pouvons aussi y inclure la liste des valeurs pour keras tuner 
    archi = ml_hp["archi"] # nom de l'architecture
    img_size =  ml_hp["img_size"] # dimensions de l'image 
    img_dim =  ml_hp["img_dim"] # niveau de couleur  (RGB 3 ou Niveau de Gris 1)
    num_classes = ml_hp["num_classes"] # niveau de couleur  (RGB 3 ou Niveau de Gris 1)
    hidden_layers_activation = ml_hp["hl_activation"] # Fonction d'activation des hidden layers, nous utiliserons toujours 'relu' pour l'instant
    
    # 1b - Initialisation de variables supplémentaire sur la bases des hyperparamètres mlflow_archive
    shape = (img_size,img_size,img_dim) # shape
    ## particularité selon la classification : fonction de perte
    ## Note : Le one_hot_encoding sera appliqué dans la fonction principale de MLFlow (preprocessing)
    if num_classes == 1: # classification binaire
        logger.debug("--- CLASSIFICATION BINAIRE ------")
        loss_function = 'binary_crossentropy'
        output_activation="sigmoid"
    else:
        logger.debug("--- CLASSIFICATION MULTIPLE ------")
        loss_function = 'sparse_categorical_crossentropy'
        output_activation="softmax"
        
    ## LOGGING
    logger.debug("--- Hyperparamètres ---")
    logger.debug(f"Archi = {archi}")
    logger.debug(f"img_size = {img_size}")
    logger.debug(f"img_dim = {img_dim}")
    logger.debug(f"shape = {shape}")
    logger.debug(f"num_classes = {num_classes}")
    logger.debug(f"hidden_layers_activation = {hidden_layers_activation}")
    logger.debug(f"loss_function = {loss_function}")
    logger.debug(f"output_activation = {output_activation}")
    
    # 2a - Définition les hyperparamètres à optimiser à l'aide de l'objet HyperParameters de Keras Tuner
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]) # 0.01 / 0.001 et 0.0001
    units = hp.Int('units', min_value=32, max_value=512, step=32)
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
    l2_lambda = hp.Choice('l2_lambda', values=[1e-3, 1e-4, 1e-5])  #Hyperparamètre pour la régularisation L2
    num_dropout_layers = hp.Int('num_dropout_layers', min_value=1, max_value=5)
    logger.debug("--- Définition des HP à optimiser ---")
    logger.debug(f"learning_rate = {learning_rate}")
    logger.debug(f"units = {units}")
    logger.debug(f"dropout_rate = {dropout_rate}")
    logger.debug(f"l2_lambda = {l2_lambda}")
    logger.debug(f"num_dropout_layers = {num_dropout_layers}")
    
    # 2b - Spécifique à l'architecture du modèle
    dropout_connect_rate = hp.Float('dropout_connect_rate', min_value=0.2, max_value=0.4, step=0.1)
    logger.debug(f"dropout_connect_rate = {dropout_connect_rate}")
    
    # 3 - Chargement du modèle de base (si applicable)
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=shape)

    # 4 - Dégel de couches (si applicable)
    for layer in base_model.layers[-20:]:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = True
                
    # 5a - Ajout de couches Fully Connected
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(units, activation=hidden_layers_activation, kernel_regularizer=l2(l2_lambda))(x)  # Utilisation de la régularisation L2

    # 5b - Spécificité architecture
    x = BatchNormalization()(x)
    
    # 6 - Ajout de couches de Dropout
    for _ in range(num_dropout_layers):  # Ajoutez jusqu'à 5 couches de dropout
        x = Dropout(dropout_rate)(x)
        
    # 7 - Couche de sortie 
    output = Dense(num_classes, activation=output_activation)(x)
    
    # 8 - Finalisation de la création du modèle
    model = Model(inputs=base_model.input, outputs=output)
    
    logger.debug(f"Base model Input {base_model.input}")
    logger.debug(f"Model Output {output}")
    #logger.debug(f"Model summary EfficientNetB0")
    #logger.debug(model.summary())
    # 9 - Compilation du modèle avec les hyperparamètres et les fonctions de perte lié au type de classification
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=loss_function, # binary_crossentropy ou sparse categorical_crossentropy
                  metrics=['accuracy'])
    
    return model


##
# Fonction de recherche des hyperparametre
##
def tuner_hyperband(ml_hp,run_id,directory,project_name,X,y):
    logger.debug("---------tuner_hyperband------------")
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
    # 1 - Récupération de hyperparamètres MLFlow pour keras tuner
    # les données, directory et project_name seront fournis par la fonction mlflow_archive car dépendra du contexte projet et du run ID
    # max_epochs et facto
    # A terme nous pouvons aussi y inclure la liste des valeurs pour keras tuner 
    max_epochs=ml_hp['max_epochs']
    factor = ml_hp['factor']
    archi= ml_hp['archi']
    
    ## 2 - Callbacks: Early Stopping, CSV Logger, LearningRateScheduler. Patience 10
    # Définir l'arrêt anticipé
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
    main_models_log=os.path.join(main_path,infolog["training_log_path"])
    csv_logger = CSVLogger(os.path.join(main_models_log,f"hb_tuning_training_log_{archi}_{run_id}.csv"), append=True, separator=';')
    # ajustement du taux d'apprentissage
    # Fonction pour définir le taux d'apprentissage, ici on régle le learning rate au bout de 10 epoch
    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * np.exp(-0.1)
    lr_scheduler = LearningRateScheduler(scheduler, verbose=1)    
    
    # Créer un tuner Keras pour optimiser les hyperparamètres du modèle avec Hyperband
    tuner = Hyperband(
        hypermodel=lambda hp: build_model(hp=hp, ml_hp=ml_hp),
        objective='val_accuracy',
        max_epochs=max_epochs,
        factor=factor,
        directory=directory,
        project_name=project_name
    )
    
    
    # Rechercher les meilleurs hyperparamètres en utilisant le tuner Keras avec Hyperband
    debut = time.time()  # Enregistre le temps avant l'exécution de la fonction
    tuner.search(X_train, y_train, validation_data=(X_val, y_val), callbacks=[early_stopping,lr_scheduler,csv_logger])
    fin = time.time()  # Enregistre le temps avant l'exécution de la fonction
    temps_execution = round(fin - debut,2)/60
    logger.debug(f"Keras Tuning time {temps_execution} min")
    
    # Obtenir le meilleur modèle trouvé par Keras Tuner avec Hyperband
    best_hp = tuner.get_best_hyperparameters()[0]
    logger.debug(f"Best Parameters {best_hp}")
    # Construire le modèle avec les meilleurs hyperparamètres trouvés
    best_model = tuner.hypermodel.build(best_hp)
    
    return best_model


##
# Fonction de recherche des hyperparametre
##
def tuner_randomsearch(ml_hp,run_id,directory,project_name,X,y):
    logger.debug("---------tuner_randomsearch------------")
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
    logger.debug("X train")
    logger.debug(f"{X_train}")    
    logger.debug("y train")
    logger.debug(f"{y_train}")
    logger.debug("X val")
    logger.debug(f"{X_val}")
    logger.debug("X val")
    logger.debug(f"{y_val}")

    # 1 - Récupération de hyperparamètres MLFlow pour keras tuner
    # les données, directory et project_name seront fournis par la fonction mlflow_archive car dépendra du contexte projet et du run ID
    # max_epochs et facto
    # A terme nous pouvons aussi y inclure la liste des valeurs pour keras tuner 
    max_epochs=ml_hp['max_epochs']
    num_trials = ml_hp['num_trials']
    archi= ml_hp['archi']
    logger.debug(f"max_epochs {max_epochs}")
    logger.debug(f"num_trials {num_trials}")
    logger.debug(f"archi {archi}")
    ## 2 - Callbacks: Early Stopping, CSV Logger, LearningRateScheduler. Patience 10
    # Définir l'arrêt anticipé
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
    main_models_log=os.path.join(main_path,infolog["training_log_path"])
    csv_logger = CSVLogger(os.path.join(main_models_log,f"rs_tuning_training_log_{archi}_{run_id}.csv"), append=True, separator=';')
    # ajustement du taux d'apprentissage
    # Fonction pour définir le taux d'apprentissage, ici on régle le learning rate au bout de 10 epoch
    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * np.exp(-0.1)
    lr_scheduler = LearningRateScheduler(scheduler, verbose=1)    
    
    # Créer un tuner Keras pour optimiser les hyperparamètres du modèle avec Hyperband
    tuner = RandomSearch(
        hypermodel=lambda hp: build_model(hp=hp, ml_hp=ml_hp),
        objective='val_accuracy',
        max_trials=num_trials,
        directory=directory,
        project_name=project_name
    )
    
    # Rechercher les meilleurs hyperparamètres en utilisant le tuner Keras avec Hyperband
    debut = time.time()  # Enregistre le temps avant l'exécution de la fonction
    tuner.search(X_train, y_train, validation_data=(X_val, y_val), callbacks=[early_stopping,lr_scheduler,csv_logger])
    fin = time.time()  # Enregistre le temps avant l'exécution de la fonction
    temps_execution = round(fin - debut,2)/60
    logger.debug(f"Keras Tuning time {temps_execution} min")
    
    # Obtenir le meilleur modèle trouvé par Keras Tuner avec Hyperband
    best_hp = tuner.get_best_hyperparameters()[0]
    logger.debug(f"Best Parameters {best_hp}")
    # Construire le modèle avec les meilleurs hyperparamètres trouvés
    best_model = tuner.hypermodel.build(best_hp)
    
    return best_model

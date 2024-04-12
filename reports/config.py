MLFLOW_TRACKING_URI = 'http://127.0.0.1:8080'

paths = {
#    "main_path": ".",
    "datasets_folder": "data/raw/datasets",
    "model_outputs": "data/processed/models/mlflow",
    "best_models_outputs": "data/processed/models/mlflow/best_models",
    "model_training_plot_outputs":"training_plot",
    "model_history_outputs":"history",
    "tuning_history_outputs":"tuning_history",
    "reports_outputs":"reports",
    "reports_figures":"figures",
    "artifact_path" : "COVID19-Project",
    "test_images":"data/raw/test_images",
    "heatmap_output":"heatmaps"
}

# experiment sain/malade
experiments_sm = {
    "experiment_name": "Classe_Sain_Malade",
    "run_name": "Sain_Malade",
    "classes":["Sain","Malade"],
    "classes_dic":{0: 'Sain', 1: 'Malade'},
    "classif_folder":"classif_sm"
}
# Dataset : utilise le répertoire datasets_folder de path 
dataset_sm_hlo = { 
    "tag_name": "120 HLO Sain/Malade",
    "current_dataset_path": "Sain_Malade_120_Dataset_HorsLO",
    "dataset_desc": "Sain_Malade_HLO_120_Metadata.json",
}
dataset_sm = { 
    "tag_name": "120 Sain/Malade",
    "current_dataset_path": "Sain_Malade_120_Dataset_All",
    "dataset_desc": "Sain_Malade_120_Metadata.json",
}
# experiment Covid_Pas_Covid
experiments_cv = {
    "experiment_name": "Classe_Covid_PasCovid",
    "run_name": "Covid_PasCovid",
    "classes":["COVID","PAS_COVID"],
    "classes_dic":{0: 'PAS_COVID', 1: 'COVID'},
    "classif_folder":"classif_cv"
}
# Dataset : utilise le répertoire datasets_folder de path 
dataset_cv_hlo = { 
    "tag_name": "120 HLO Covid/Pas Covid",
    "current_dataset_path": "Covid_PasCovid_120_Dataset_HorsLO",
    "dataset_desc": "Covid_PasCovid_HLO_120_Metadata.json",
}
dataset_cv = { 
    "tag_name": "120 Covid/Pas Covid",
    "current_dataset_path": "Covid_PasCovid_120_Dataset_All",
    "dataset_desc": "Covid_PasCovid_120_Metadata.json",
}
# experiment Multiclasse
experiments_mc = {
    "experiment_name": "Classe_Multi_Classes",
    "run_name": "MC",
    "classes":["COVID","Lung_Opacity","Normal","Viral_Pneumonia"],
    "classes_dic":{0: 'Viral_Pneumonia', 1: 'Lung_Opacity', 2: 'Normal', 3: 'COVID'},
    "classif_folder":"classif_mc"
}
# Dataset : utilise le répertoire datasets_folder de path 

dataset_mc = { 
    "tag_name": "120 Multiclasse",
    "current_dataset_path": "Covid-19_MC_120_Dataset_All",
    "dataset_desc": "Covid_MC_120_Metadata.json"
}

# experiment 3 classes
experiments_3c = {
    "experiment_name": "Classe_3_Classes",
    "run_name": "trois_C",
    "classes":["COVID","Normal","Viral_Pneumonia"],
    "classes_dic":{0: 'Viral_Pneumonia', 1: 'Normal', 2: 'COVID'},
    "classif_folder":"classif_3c"
}
# Dataset : utilise le répertoire datasets_folder de path 
dataset_3c = { 
    "tag_name": "120 TroisC",
    "current_dataset_path": "Covid-19_3C_120_Dataset_HLO",
    "dataset_desc": "Covid_3C_120_HLO_Metadata.json"
}

# Configuration de la journalisation
infolog = {
    "logs_folder": "logs",
    "logfile_name": "mlflow.log",
    "training_log_path":"logs/training_logs"
}

hyperparams_list = [
    ### VGG16
    # Binaire Sain/Malade Covid/pas Covid - Toutes les données, 120 images
    {"archi": "VGG16", "img_size": 224, "img_dim": 3, "num_classes": 1, "hl_activation": "relu", "max_epochs": 11, "factor": 3, "num_trials": 1, "data_lo_hlo": "LO","last_conv_layer":"block5_conv3"},
    # Binaire Sain/Malade Covid/pas Covid - Hors Lung Opacity, 120 images
#    {"archi": "VGG16", "img_size": 224, "img_dim": 3, "num_classes": 1, "hl_activation": "relu", "max_epochs": 11, "factor": 3, "num_trials": 1, "data_lo_hlo": "HLO","last_conv_layer":"block5_conv3"},
    # Classification multiple: 4 classes
#    {"archi": "VGG16", "img_size": 224, "img_dim": 3, "num_classes": 4, "hl_activation": "relu", "max_epochs": 11, "factor": 3, "num_trials": 1, "data_lo_hlo": "NA","last_conv_layer":"block5_conv3"},
    # Classification multiple : 3 classes hors Lung Opacity
#    {"archi": "VGG16", "img_size": 224, "img_dim": 3, "num_classes": 3, "hl_activation": "relu", "max_epochs": 11, "factor": 3, "num_trials": 1, "data_lo_hlo": "NA","last_conv_layer":"block5_conv3"},
    
    ### VGG19
    # Binaire Sain/Malade Covid/pas Covid - Toutes les données, 120 images
    {"archi": "VGG19", "img_size": 224, "img_dim": 3, "num_classes": 1, "hl_activation": "relu", "max_epochs": 11, "factor": 3, "num_trials": 1, "data_lo_hlo": "LO","last_conv_layer":"block5_conv4"},
    # Binaire Sain/Malade Covid/pas Covid - Hors Lung Opacity, 120 images
#    {"archi": "VGG19", "img_size": 224, "img_dim": 3, "num_classes": 1, "hl_activation": "relu", "max_epochs": 11, "factor": 3, "num_trials": 1, "data_lo_hlo": "HLO","last_conv_layer":"block5_conv4"},
    # Classification multiple: 4 classes
#    {"archi": "VGG19", "img_size": 224, "img_dim": 3, "num_classes": 4, "hl_activation": "relu", "max_epochs": 11, "factor": 3, "num_trials": 1, "data_lo_hlo": "NA","last_conv_layer":"block5_conv4"},
    # Classification multiple : 3 classes hors Lung Opacity
#    {"archi": "VGG19", "img_size": 224, "img_dim": 3, "num_classes": 3, "hl_activation": "relu", "max_epochs": 11, "factor": 3, "num_trials": 1, "data_lo_hlo": "NA","last_conv_layer":"top_block5_conv4conv"},
    
    ### ResNet50
    # Binaire Sain/Malade Covid/pas Covid - Toutes les données, 120 images
    {"archi": "ResNet50", "img_size": 224, "img_dim": 3, "num_classes": 1, "hl_activation": "relu", "max_epochs": 11, "factor": 3, "num_trials": 1, "data_lo_hlo": "LO","last_conv_layer":"NA"},
    # Binaire Sain/Malade Covid/pas Covid - Hors Lung Opacity, 120 images
#    {"archi": "ResNet50", "img_size": 224, "img_dim": 3, "num_classes": 1, "hl_activation": "relu", "max_epochs": 11, "factor": 3, "num_trials": 1, "data_lo_hlo": "HLO","last_conv_layer":"NA"},
    # Classification multiple: 4 classes
#    {"archi": "ResNet50", "img_size": 224, "img_dim": 3, "num_classes": 4, "hl_activation": "relu", "max_epochs": 11, "factor": 3, "num_trials": 1, "data_lo_hlo": "NA","last_conv_layer":"NA"},
    # Classification multiple : 3 classes hors Lung Opacity
#    {"archi": "ResNet50", "img_size": 224, "img_dim": 3, "num_classes": 3, "hl_activation": "relu", "max_epochs": 11, "factor": 3, "num_trials": 1, "data_lo_hlo": "NA","last_conv_layer":"NA"},
 
     ### LeNet
    # Binaire Sain/Malade Covid/pas Covid - Toutes les données, 120 images
    {"archi": "LeNet", "img_size": 224, "img_dim": 1, "num_classes": 1, "hl_activation": "relu", "max_epochs": 11, "factor": 3, "num_trials": 1, "data_lo_hlo": "LO","last_conv_layer":"top_conv"},
    # Binaire Sain/Malade Covid/pas Covid - Hors Lung Opacity, 120 images
#    {"archi": "LeNet", "img_size": 224, "img_dim": 1, "num_classes": 1, "hl_activation": "relu", "max_epochs": 11, "factor": 3, "num_trials": 1, "data_lo_hlo": "HLO","last_conv_layer":"top_conv"},
    # Classification multiple: 4 classes
#    {"archi": "LeNet", "img_size": 224, "img_dim": 1, "num_classes": 4, "hl_activation": "relu", "max_epochs": 11, "factor": 3, "num_trials": 1, "data_lo_hlo": "NA","last_conv_layer":"top_conv"},
    # Classification multiple : 3 classes hors Lung Opacity
#    {"archi": "LeNet", "img_size": 224, "img_dim": 1, "num_classes": 3, "hl_activation": "relu", "max_epochs": 11, "factor": 3, "num_trials": 1, "data_lo_hlo": "NA","last_conv_layer":"top_conv"},
    
    ### EfficientNetB0
    # Binaire Sain/Malade Covid/pas Covid - Toutes les données, 120 images
    {"archi": "EfficientNetB0", "img_size": 224, "img_dim": 3, "num_classes": 1, "hl_activation": "relu", "max_epochs": 11, "factor": 3, "num_trials": 1, "data_lo_hlo": "LO","last_conv_layer":"top_conv"},
    # Binaire Sain/Malade Covid/pas Covid - Hors Lung Opacity, 120 images
#    {"archi": "EfficientNetB0", "img_size": 224, "img_dim": 3, "num_classes": 1, "hl_activation": "relu", "max_epochs": 11, "factor": 3, "num_trials": 1, "data_lo_hlo": "HLO","last_conv_layer":"top_conv"},
    # Classification multiple: 4 classes
#    {"archi": "EfficientNetB0", "img_size": 224, "img_dim": 3, "num_classes": 4, "hl_activation": "relu", "max_epochs": 11, "factor": 3, "num_trials": 1, "data_lo_hlo": "NA","last_conv_layer":"top_conv"},
    # Classification multiple : 3 classes hors Lung Opacity
#    {"archi": "EfficientNetB0", "img_size": 224, "img_dim": 3, "num_classes": 3, "hl_activation": "relu", "max_epochs": 11, "factor": 3, "num_trials": 1, "data_lo_hlo": "NA","last_conv_layer":"top_conv"},  
]
import pandas as pd
import os
from configparser import ConfigParser
import glob
import re
import sys


def drop_bp_cords(dataFrame, inifile):
    config = ConfigParser()
    configFile = str(inifile)
    config.read(configFile)
    project_path = config.get('General settings', 'project_path')
    bodyparthListPath = os.path.join(project_path, 'logs', 'measures', 'pose_configs', 'bp_names', 'project_bp_names.csv')
    poseConfigDf = pd.read_csv(bodyparthListPath, header=None)
    poseConfigList = list(poseConfigDf[0])
    columnHeaders = []
    for bodypart in poseConfigList:
        colHead1, colHead2, colHead3 = (bodypart + '_x', bodypart + '_y', bodypart + '_p')
        columnHeaders.extend((colHead1, colHead2, colHead3))
    dataFrame = dataFrame.drop(columnHeaders, axis=1)

    return dataFrame

def define_bp_drop_down(configini):
    config = ConfigParser()
    configFile = str(configini)
    config.read(configFile)
    animalno = config.getint('General settings', 'animal_no')
    # get list
    bpcsv = (os.path.join(os.path.dirname(configini), 'logs', 'measures', 'pose_configs', 'bp_names',
                          'project_bp_names.csv'))
    bplist = []
    with open(bpcsv) as f:
        for row in f:
            bplist.append(row)
    bplist = list(map(lambda x: x.replace('\n', ''), bplist))

    if animalno != 1:
        animal1bp = [f for f in bplist if '_1' in f]
        animal2bp = [f for f in bplist if '_2' in f]
        return animal1bp,animal2bp
    else:
        animal1bp = bplist
        return animal1bp,['No body parts']

def define_movement_cols(noAnimals):
    if noAnimals == 1:
        columnNames = ['Video', 'Frames processed', 'Total movement', 'Mean velocity', 'Median velocity']
    if noAnimals == 2:
        columnNames = ['Video', 'Frames processed', 'Total movement animal 1', 'Mean velocity animal 1', 'Median velocity animal 1', 'Total movement animal 2', 'Mean velocity animal 2', 'Median velocity animal 2', 'Mean animal distance', 'Median animal distance']
    print(columnNames)
    return columnNames

def bodypartConfSchematic():
    optionsBaseListImagesPath = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), 'pose_configurations', 'schematics')
    optionsBaseListNamesPath = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), 'pose_configurations', 'configuration_names', 'pose_config_names.csv')
    optionsBaseNameList = pd.read_csv(optionsBaseListNamesPath, header=None)
    optionsBaseNameList = list(optionsBaseNameList[0])
    optionsBaseNameList.append('Create pose config...')
    optionsBasePhotosList = glob.glob(optionsBaseListImagesPath + '/*.png')
    optionsBasePhotosList.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    return optionsBaseNameList, optionsBasePhotosList


def GenerateMetaDataFileHeaders():
    metaDataHeaders = ["Classifier_name", "RF_criterion", "RF_max_features", "RF_min_sample_leaf",
     "RF_n_estimators", "compute_feature_permutation_importance",
     "generate_classification_report", "generate_example_decision_tree",
     "generate_features_importance_bar_graph", "generate_features_importance_log",
     "generate_precision_recall_curves", "generate_rf_model_meta_data_file",
     "generate_sklearn_learning_curves", "learning_curve_data_splits",
     "learning_curve_k_splits", "n_feature_importance_bars",
     "over_sample_ratio", "over_sample_setting", "train_test_size", "under_sample_ratio",
     "under_sample_setting"]

    return metaDataHeaders

def getBpNames(inifile):
    Xcols, Ycols, Pcols = ([],[],[])
    config = ConfigParser()
    configFile = str(inifile)
    config.read(configFile)
    project_path = config.get('General settings', 'project_path')
    bodyparthListPath = os.path.join(project_path, 'logs', 'measures', 'pose_configs', 'bp_names', 'project_bp_names.csv')
    poseConfigDf = pd.read_csv(bodyparthListPath, header=None)
    poseConfigList = list(poseConfigDf[0])
    for bodypart in poseConfigList:
        colHead1, colHead2, colHead3 = (bodypart + '_x', bodypart + '_y', bodypart + '_p')
        Xcols.append(colHead1)
        Ycols.append(colHead2)
        Pcols.append(colHead3)
    return Xcols, Ycols, Pcols

def getBpHeaders(inifile):
    colHeads = []
    config = ConfigParser()
    configFile = str(inifile)
    config.read(configFile)
    project_path = config.get('General settings', 'project_path')
    bodyparthListPath = os.path.join(project_path, 'logs', 'measures', 'pose_configs', 'bp_names','project_bp_names.csv')
    poseConfigDf = pd.read_csv(bodyparthListPath, header=None)
    poseConfigList = list(poseConfigDf[0])
    for bodypart in poseConfigList:
        colHead1, colHead2, colHead3 = (bodypart + '_x', bodypart + '_y', bodypart + '_p')
        colHeads.append(colHead1)
        colHeads.append(colHead2)
        colHeads.append(colHead3)
    print(poseConfigList)

    return colHeads




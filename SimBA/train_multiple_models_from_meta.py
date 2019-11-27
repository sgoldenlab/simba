from configparser import ConfigParser
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from yellowbrick.classifier import ClassificationReport
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.tree import export_graphviz
from subprocess import call
import matplotlib.pyplot as plt
import pickle
import csv
from tabulate import tabulate

def train_multimodel(configini):
    pd.options.mode.chained_assignment = None

    configFile = configini
    config = ConfigParser()
    config.read(configFile)
    modelDir = config.get('SML settings', 'model_dir')
    modelSavePath = os.path.join(modelDir, 'validations')
    if not os.path.exists(modelSavePath):
        os.makedirs(modelSavePath)
    ensemble_evaluations_out = os.path.join(modelSavePath, 'model_evaluations')
    if not os.path.exists(ensemble_evaluations_out):
        os.makedirs(ensemble_evaluations_out)
    modelPath = os.path.join(modelSavePath, 'model_files')
    if not os.path.exists(modelPath):
        os.makedirs(modelPath)
    model_nos = config.getint('SML settings', 'No_targets')
    data_folder = config.get('create ensemble settings', 'data_folder')
    train_test_size = config.getfloat('create ensemble settings', 'train_test_size')
    targetFrame = pd.DataFrame()
    features = pd.DataFrame()

    def generateClassificationReport(clf, class_names, classifierName, saveFileNo):
        try:
            visualizer = ClassificationReport(clf, classes=class_names, support=True)
            visualizer.score(data_test, target_test)
            visualizerPath = os.path.join(ensemble_evaluations_out, str(classifierName) + '_' + str(saveFileNo) + '_classificationReport.png')
            visualizer.poof(outpath=visualizerPath)
        except KeyError:
            print(('Warning, not enough data for classification report: ') + str(classifierName))

    def generateFeatureImportanceLog(importances, classifierName, saveFileNo):
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
        feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
        feature_importance_list = [('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
        feature_importance_list_varNm = [i.split(':' " ", 3)[1] for i in feature_importance_list]
        feature_importance_list_importance = [i.split(':' " ", 3)[2] for i in feature_importance_list]
        log_df = pd.DataFrame()
        log_df['Feature_name'] = feature_importance_list_varNm
        log_df['Feature_importance'] = feature_importance_list_importance
        logPath = os.path.join(ensemble_evaluations_out, str(classifierName) + '_' + str(saveFileNo) + '_feature_importance_log.csv')
        log_df.to_csv(logPath)
        return log_df

    def generateFeatureImportanceBarGraph(log_df, N_feature_importance_bars, classifierName, saveFileNo):
        log_df['Feature_importance'] = log_df['Feature_importance'].apply(pd.to_numeric)
        log_df['Feature_name'] = log_df['Feature_name'].map(lambda x: x.lstrip('+-').rstrip('Importance'))
        log_df = log_df.head(N_feature_importance_bars)
        ax = log_df.plot.bar(x='Feature_name', y='Feature_importance', legend=False, rot=90, fontsize=6)
        figName = str(classifierName) + '_' + str(saveFileNo) + '_feature_bars.png'
        figSavePath = os.path.join(ensemble_evaluations_out, figName)
        plt.tight_layout()
        plt.savefig(figSavePath, dpi=600)
        plt.close('all')

    def generateExampleDecisionTree(estimator, classifierName, saveFileNo):
        dot_name = os.path.join(ensemble_evaluations_out, str(classifierName) + '_' + str(saveFileNo) + '_tree.dot')
        file_name = os.path.join(ensemble_evaluations_out, str(classifierName) + '_' + str(saveFileNo) + '_tree.pdf')
        export_graphviz(estimator, out_file=dot_name, filled=True, rounded=True, special_characters=False, impurity=False,
                        class_names=class_names, feature_names=features.columns)
        commandStr = ('dot ' + str(dot_name) + ' -T pdf -o ' + str(file_name) + ' -Gdpi=600')
        call(commandStr, shell=True)

    def generateMetaData(metaDataList, classifierName, saveFileNo):
        metaDataFn = str(classifierName) + '_' + str(saveFileNo) + '_meta.csv'
        metaDataFolder = os.path.join(modelSavePath, 'meta_data')
        if not os.path.exists(metaDataFolder):
            os.makedirs(metaDataFolder)
        metaDataPath = os.path.join(metaDataFolder, metaDataFn)
        metaDataHeaders = ["Classifier_name", "Ensamble_method", "Under_sampling_setting", "Under_sampling_ratio", "Over_sampling_method", "Over_sampling_ratio", "Estimators", "Max_features", "RF_criterion", "RF_min_sample_leaf", "Feature_list"]
        with open(metaDataPath, 'w', newline='') as f:
            out_writer = csv.writer(f)
            out_writer.writerow(metaDataHeaders)
            out_writer.writerow(metaDataList)

    #READ IN THE META FILES
    meta_files_folder = config.get('create ensemble settings', 'meta_files_folder')
    metaFilesList = []
    for i in os.listdir(meta_files_folder):
        if i.__contains__(".meta"):
            metaFile = os.path.join(meta_files_folder, i)
            metaFilesList.append(metaFile)
    print('Total number of models to be created: ' + str(len(metaFilesList)))

    loopy = 0
    for i in metaFilesList:
        loopy+=1
        currMetaFile = pd.read_csv(i, index_col=False)
        classifierName = currMetaFile['Classifier_name'].iloc[0]
        saveFileNo = (len(os.listdir(modelPath)) + 1)

        #READ IN DATA FOLDER AND REMOVE ALL NON-FEATURE VARIABLES (POP DLC COORDINATE DATA AND TARGET DATA)
        features = pd.DataFrame()
        df = pd.DataFrame()
        for i in os.listdir(data_folder):
            if i.__contains__(".csv"):
                currentFn = os.path.join(data_folder, i)
                df = pd.read_csv(currentFn)
                features = features.append(df, ignore_index=True)
        features = features.loc[:, ~features.columns.str.contains('^Unnamed')]
        frame_number = features.pop('frames').values
        video_number = features.pop('video_no').values
        targetFrame = features.pop(classifierName).values
        features = features.fillna(0)
        features = features.drop(["scorer", "Ear_left_1_x", "Ear_left_1_y", "Ear_left_1_p", "Ear_right_1_x", "Ear_right_1_y", "Ear_right_1_p", "Nose_1_x", "Nose_1_y", "Nose_1_p", "Center_1_x", "Center_1_y", "Center_1_p", "Lat_left_1_x", "Lat_left_1_y",
                    "Lat_left_1_p", "Lat_right_1_x", "Lat_right_1_y", "Lat_right_1_p", "Tail_base_1_x", "Tail_base_1_y", "Tail_base_1_p", "Tail_end_1_x", "Tail_end_1_y", "Tail_end_1_p", "Ear_left_2_x",
                    "Ear_left_2_y", "Ear_left_2_p", "Ear_right_2_x", "Ear_right_2_y", "Ear_right_2_p", "Nose_2_x", "Nose_2_y", "Nose_2_p", "Center_2_x", "Center_2_y", "Center_2_p", "Lat_left_2_x", "Lat_left_2_y",
                    "Lat_left_2_p", "Lat_right_2_x", "Lat_right_2_y", "Lat_right_2_p", "Tail_base_2_x", "Tail_base_2_y", "Tail_base_2_p", "Tail_end_2_x", "Tail_end_2_y", "Tail_end_2_p"], axis=1)
        target_names = []
        loop=1

        for i in range(model_nos):
            currentModelNames = 'target_name_' + str(loop)
            currentModelNames = config.get('SML settings', currentModelNames)
            if currentModelNames != classifierName:
                target_names.append(currentModelNames)
            loop+=1
        loop = 0
        for i in range(len(target_names)):
            currentModelName = target_names[i]
            features.pop(currentModelName).values
        class_names = class_names = ['Not_' + classifierName, classifierName]
        feature_list = list(features.columns)

        under_sample_setting = currMetaFile['Under_sampling_setting'].iloc[0]
        under_sample_ratio = currMetaFile['Under_sampling_ratio'].iloc[0]
        over_sample_setting = currMetaFile['Over_sampling_setting'].iloc[0]
        over_sample_ratio = currMetaFile['Over_sampling_ratio'].iloc[0]
        model_to_run = currMetaFile['Ensamble_method'].iloc[0]
        RF_n_estimators = currMetaFile['Estimators'].iloc[0]
        RF_criterion = currMetaFile['RF_criterion'].iloc[0]
        RF_min_sample_leaf = currMetaFile['RF_min_sample_leaf'].iloc[0]
        RF_max_features = currMetaFile['Max_features'].iloc[0]

        #PRINT INFORMATION TABLE ON THE MODEL BEING CREATED
        print('MODEL ' + str(loopy) + str(' settings'))
        tableView = [["Ensemble method", model_to_run], ["Estimators (trees)", RF_n_estimators], ["Max features", RF_max_features], ["Under sampling setting", under_sample_setting], ["Under sampling ratio", under_sample_ratio], ["Over sampling setting", over_sample_setting], ["Under sampling ratio", under_sample_ratio], ["criterion", RF_criterion], ["Min sample leaf", RF_min_sample_leaf]]
        headers = ["Setting", "value"]
        print(tabulate(tableView, headers, tablefmt="grid"))

        # IF SET BY USER - PERFORM UNDERSAMPLING AND OVERSAMPLING IF SET BY USER
        data_train, data_test, target_train, target_test = train_test_split(features, targetFrame, test_size=train_test_size)
        if under_sample_setting == 'Random undersample':
            print('Performing undersampling..')
            trainDf = data_train
            trainDf[classifierName] = target_train
            targetFrameRows = trainDf.loc[trainDf[classifierName] == 1]
            nonTargetFrameRows = trainDf.loc[trainDf[classifierName] == 0]
            nontargetFrameRowsSize = int(len(targetFrameRows) * under_sample_ratio)
            nonTargetFrameRows = nonTargetFrameRows.sample(nontargetFrameRowsSize, replace=False)
            trainDf = pd.concat([targetFrameRows, nonTargetFrameRows])
            target_train = trainDf.pop(classifierName).values
            data_train = trainDf
        if over_sample_setting == 'SMOTEENN':
            print('Performing SMOTEEN oversampling..')
            smt = SMOTEENN(sampling_strategy=over_sample_ratio)
            data_train, target_train = smt.fit_sample(data_train, target_train)
        if over_sample_setting == 'SMOTE':
            print('Performing SMOTE oversampling..')
            smt = SMOTE(sampling_strategy=over_sample_ratio)
            data_train, target_train = smt.fit_sample(data_train, target_train)
        print(type(data_train))
        # RUN THE DECISION ENSEMBLE SET BY THE USER
        #run random forest
        if model_to_run == 'RF':
            clf = RandomForestClassifier(n_estimators=RF_n_estimators, max_features=RF_max_features, n_jobs=-1, criterion=RF_criterion, min_samples_leaf=RF_min_sample_leaf, bootstrap=True, verbose=1)
            clf.fit(data_train, target_train)
            clf_pred = clf.predict(data_test)
            print("Accuracy " + str(classifierName) + ' model:', metrics.accuracy_score(target_test, clf_pred))

            #RUN RANDOM FOREST EVALUATIONS
            generate_example_decision_tree = config.get('create ensemble settings', 'generate_example_decision_tree')
            if generate_example_decision_tree == 'yes':
                estimator = clf.estimators_[3]
                generateExampleDecisionTree(estimator, classifierName, saveFileNo)

            generate_classification_report = config.get('create ensemble settings', 'generate_classification_report')
            if generate_classification_report == 'yes':
                generateClassificationReport(clf, class_names, classifierName, saveFileNo)

            generate_features_importance_log = config.get('create ensemble settings', 'generate_features_importance_log')
            if generate_features_importance_log == 'yes':
                importances = list(clf.feature_importances_)
                log_df = generateFeatureImportanceLog(importances, classifierName, saveFileNo)

            generate_features_importance_bar_graph = config.get('create ensemble settings', 'generate_features_importance_bar_graph')
            N_feature_importance_bars = config.getint('create ensemble settings', 'N_feature_importance_bars')
            if generate_features_importance_bar_graph == 'yes':
                generateFeatureImportanceBarGraph(log_df, N_feature_importance_bars, classifierName, saveFileNo)

            # SAVE MODEL META DATA
            RF_meta_data = config.get('create ensemble settings', 'RF_meta_data')
            if RF_meta_data == 'yes':
                metaDataList = [classifierName, model_to_run, under_sample_setting, under_sample_ratio, over_sample_setting, over_sample_ratio, RF_n_estimators, RF_max_features, RF_criterion, RF_min_sample_leaf, class_names, feature_list]
                generateMetaData(metaDataList, classifierName, saveFileNo)

        #SAVE MODEL
        modelfn = str(classifierName) + '_' + str(saveFileNo) + '.sav'
        modelFileSavePath = os.path.join(modelPath, modelfn)
        pickle.dump(clf, open(modelFileSavePath, 'wb'))
        print('Classifier ' + str(modelfn) + ' saved @' + str(modelPath))
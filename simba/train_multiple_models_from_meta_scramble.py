import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=DeprecationWarning)
from configparser import ConfigParser, MissingSectionHeaderError
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from yellowbrick.classifier import ClassificationReport
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_curve
from sklearn.tree import export_graphviz
from subprocess import call
import matplotlib.pyplot as plt
import pickle
import csv
from tabulate import tabulate
import eli5
from eli5.sklearn import PermutationImportance
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from drop_bp_cords import drop_bp_cords, GenerateMetaDataFileHeaders

def train_multimodel(configini):
    pd.options.mode.chained_assignment = None
    configFile = configini
    config = ConfigParser()
    try:
        config.read(configFile)
    except MissingSectionHeaderError:
        print('ERROR:  Not a valid project_config file. Please check the project_config.ini path.')
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
    features = pd.DataFrame()

    def generateClassificationReport(clf, class_names, classifierName, saveFileNo):
        try:
            visualizer = ClassificationReport(clf, classes=class_names, support=True)
            visualizer.score(data_test, target_test)
            visualizerPath = os.path.join(ensemble_evaluations_out, str(classifierName) + '_' + str(saveFileNo) + '_classificationReport.png')
            visualizer.poof(outpath=visualizerPath, clear_figure=True)
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
        print('Meta data file saved.')
        metaDataPath = os.path.join(metaDataFolder, metaDataFn)
        metaDataHeaders = GenerateMetaDataFileHeaders()

        with open(metaDataPath, 'w', newline='') as f:
            out_writer = csv.writer(f)
            out_writer.writerow(metaDataHeaders)
            out_writer.writerow(metaDataList)

    def computePermutationImportance(data_test, target_test, clf):
        perm = PermutationImportance(clf, random_state=1).fit(data_test, target_test)
        permString = (eli5.format_as_text(eli5.explain_weights(perm, feature_names=data_test.columns.tolist())))
        permString = permString.split('\n', 9)[-1]
        all_rows = permString.split("\n")
        all_cols = [row.split(' ') for row in all_rows]
        all_cols.pop(0)
        fimp = [row[0] for row in all_cols]
        errot = [row[2] for row in all_cols]
        name = [row[4] for row in all_cols]
        dfvals = pd.DataFrame(list(zip(fimp, errot, name)), columns=['A', 'B', 'C'])
        fname = os.path.join(ensemble_evaluations_out, str(classifierName) + '_permutations_importances.csv')
        dfvals.to_csv(fname, index=False)

    def LearningCurve(features, targetFrame, shuffle_splits, dataset_splits):
        cv = ShuffleSplit(n_splits=shuffle_splits, test_size=train_test_size, random_state=0)
        model = RandomForestClassifier(n_estimators=RF_n_estimators, max_features=RF_max_features, n_jobs=-1, criterion=RF_criterion, min_samples_leaf=RF_min_sample_leaf, bootstrap=True, verbose=1)
        train_sizes, train_scores, test_scores = learning_curve(model, features, targetFrame, cv=cv, scoring='f1',shuffle=True, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, dataset_splits))
        train_sizes = np.linspace(0.01, 1.0, dataset_splits)
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        learningCurve_df = pd.DataFrame()
        learningCurve_df['Fraction Train Size'] = train_sizes
        learningCurve_df['Train_mean_f1'] = train_mean
        learningCurve_df['Test_mean_f1'] = test_mean
        learningCurve_df['Train_std_f1'] = train_std
        learningCurve_df['Test_std_f1'] = test_std
        fname = os.path.join(ensemble_evaluations_out, str(classifierName) + '_learning_curve.csv')
        learningCurve_df.to_csv(fname, index=False)

    #READ IN THE META FILES
    meta_files_folder = config.get('create ensemble settings', 'meta_files_folder')
    metaFilesList = []
    for i in os.listdir(meta_files_folder):
        if i.__contains__("meta"):
            metaFile = os.path.join(meta_files_folder, i)
            metaFilesList.append(metaFile)
    print('# of models to be created: ' + str(len(metaFilesList)))
    loopy = 0

    # READ IN DATA FOLDER AND REMOVE ALL NON-FEATURE VARIABLES (POP DLC COORDINATE DATA AND TARGET DATA)
    features = pd.DataFrame()
    print('Reading in ' + str(len(os.listdir(data_folder))) + ' annotated files...')

    for p in os.listdir(data_folder):
        if (".csv") in p:
            currentFn = os.path.join(data_folder, p)
            df = pd.read_csv(currentFn)
            features = features.append(df, ignore_index=True)
    features = features.loc[:, ~features.columns.str.contains('^Unnamed')]
    baseFeatureFrame = features.drop(["scorer"], axis=1, errors='ignore')
    try:
        baseFeatureFrame = baseFeatureFrame.drop(['video_no', 'frames'], axis=1)
    except KeyError:
        pass


    for i in metaFilesList:
        loopy+=1
        features = baseFeatureFrame.copy()
        currMetaFile = pd.read_csv(i, index_col=False)
        classifierName = currMetaFile['Classifier_name'].iloc[0]
        saveFileNo = (len(os.listdir(modelPath)) + 1)
        totalTargetframes = features[classifierName].sum()
        try:
            targetFrame = features.pop(classifierName).values
        except KeyError:
            print('Error: the dataframe does not contain any target annotations. Please check the csv files in the project_folder/csv/target_inserted folder')
            break
        features = features.fillna(0)
        features = drop_bp_cords(features, configini)
        target_names = []
        loop=1

        for bb in range(model_nos):
            currentModelNames = 'target_name_' + str(loop)
            currentModelNames = config.get('SML settings', currentModelNames)
            if currentModelNames != classifierName:
                target_names.append(currentModelNames)
            loop+=1
        for ss in range(len(target_names)):
            currentModelName = target_names[ss]
            features.pop(currentModelName).values
        class_names = class_names = ['Not_' + classifierName, classifierName]
        feature_list = list(features.columns)
        try:
            under_sample_setting = currMetaFile['under_sample_setting'].iloc[0]
            under_sample_ratio = currMetaFile['under_sample_ratio'].iloc[0]
            over_sample_setting = currMetaFile['over_sample_setting'].iloc[0]
            over_sample_ratio = currMetaFile['over_sample_ratio'].iloc[0]
            model_to_run = 'RF' #currMetaFile['Ensamble_method'].iloc[0]
            RF_n_estimators = currMetaFile['RF_n_estimators'].iloc[0]
            RF_criterion = currMetaFile['RF_criterion'].iloc[0]
            RF_min_sample_leaf = currMetaFile['RF_min_sample_leaf'].iloc[0]
            RF_max_features = currMetaFile['RF_max_features'].iloc[0]
            train_test_size = currMetaFile['train_test_size'].iloc[0]
        except KeyError:
            print('ERROR: The config file containing the RF hyperparameters is not right. Check the csv files in the project_folder/configs.')

        #PRINT INFORMATION TABLE ON THE MODEL BEING CREATED
        print('MODEL ' + str(loopy) + str(' settings'))
        tableView = [["Model name", classifierName], ["Ensemble method", model_to_run], ["Estimators (trees)", RF_n_estimators], ["Max features", RF_max_features], ["Under sampling setting", under_sample_setting], ["Under sampling ratio", under_sample_ratio], ["Over sampling setting", over_sample_setting], ["Over sampling ratio", over_sample_ratio], ["criterion", RF_criterion], ["Min sample leaf", RF_min_sample_leaf]]
        headers = ["Setting", "value"]
        print(tabulate(tableView, headers, tablefmt="grid"))
        print('# ' + str(len(features.columns)) + ' features.')

        # IF SET BY USER - PERFORM UNDERSAMPLING AND OVERSAMPLING IF SET BY USER
        data_train, data_test, target_train, target_test = train_test_split(features, targetFrame, test_size=train_test_size)
        trainDf = data_train
        trainDf[classifierName] = target_train
        print('# of ' + str(classifierName) + ' frames in dataset: ' + str(totalTargetframes))
        if under_sample_setting == 'Random undersample':
            print('Performing undersampling...')
            targetFrameRows = trainDf.loc[trainDf[classifierName] == 1]
            nonTargetFrameRows = trainDf.loc[trainDf[classifierName] == 0]
            nontargetFrameRowsSize = int(len(targetFrameRows) * under_sample_ratio)
            nonTargetFrameRows = nonTargetFrameRows.sample(nontargetFrameRowsSize, replace=False)
            trainDf = pd.concat([targetFrameRows, nonTargetFrameRows])
            target_train = trainDf.pop(classifierName).values
            data_train = trainDf
        if under_sample_setting != 'Random undersample':
            target_train = trainDf.pop(classifierName).values
            under_sample_ratio = 'NaN'
        if over_sample_setting == 'SMOTEENN':
            print('Performing SMOTEEN oversampling...')
            over_sample_ratio = config.getfloat('create ensemble settings', 'over_sample_ratio')
            smt = SMOTEENN(sampling_strategy=over_sample_ratio)
            data_train, target_train = smt.fit_sample(data_train, target_train)
        if over_sample_setting == 'SMOTE':
            print('Performing SMOTE oversampling...')
            over_sample_ratio = config.getfloat('create ensemble settings', 'over_sample_ratio')
            smt = SMOTE(sampling_strategy=over_sample_ratio)
            data_train, target_train = smt.fit_sample(data_train, target_train)
        if (over_sample_setting != 'SMOTEENN') or (over_sample_setting != 'SMOTE'):
            over_sample_ratio = 'NaN'
        data_train = data_train.sample(frac=1).reset_index(drop=True)


        # RUN THE DECISION ENSEMBLE SET BY THE USER
        #run random forest
        if model_to_run == 'RF':
            clf = RandomForestClassifier(n_estimators=RF_n_estimators, max_features=RF_max_features, n_jobs=-1, criterion=RF_criterion, min_samples_leaf=RF_min_sample_leaf, bootstrap=True, verbose=1)
            print('Training model ' + str(loopy) + '...')
            try:
                clf.fit(data_train, target_train)
            except ValueError:
                print('ERROR: The model contains an incompatible array. This may happen when trying to train a model with 0 examples of the behavior of interest')
            except ModuleNotFoundError:
                print('ERROR: ModuleNotFoundError. This can happen with incompatible versions of NumPy.')

            #RUN RANDOM FOREST EVALUATIONS
            generate_example_decision_tree = currMetaFile['generate_example_decision_tree'].iloc[0]
            if generate_example_decision_tree == 'yes':
                print('Generating example decision tree using graphviz...')
                estimator = clf.estimators_[3]
                generateExampleDecisionTree(estimator, classifierName, saveFileNo)

            generate_classification_report = currMetaFile['generate_classification_report'].iloc[0]
            if generate_classification_report == 'yes':
                print('Generating yellowbrick classification report...')
                generateClassificationReport(clf, class_names, classifierName, saveFileNo)

            generate_features_importance_log =currMetaFile['generate_features_importance_log'].iloc[0]
            if generate_features_importance_log == 'yes':
                print('Generating feature importance log...')
                importances = list(clf.feature_importances_)
                log_df = generateFeatureImportanceLog(importances, classifierName, saveFileNo)

            generate_features_importance_bar_graph = currMetaFile['generate_features_importance_bar_graph'].iloc[0]
            N_feature_importance_bars = currMetaFile['n_feature_importance_bars'].iloc[0]
            if generate_features_importance_bar_graph == 'yes':
                print('Generating feature importance bar graph...')
                generateFeatureImportanceBarGraph(log_df, N_feature_importance_bars, classifierName, saveFileNo)

            compute_permutation_importance = currMetaFile['compute_feature_permutation_importance'].iloc[0]
            if compute_permutation_importance == 'yes':
                print('Calculating permutation importances...')
                computePermutationImportance(data_test, target_test, clf)

            generate_precision_recall_curve = currMetaFile['generate_precision_recall_curves'].iloc[0]
            if generate_precision_recall_curve == 'yes':
                print('Calculating precision recall curves...')
                precisionRecallDf = pd.DataFrame()
                probabilities = clf.predict_proba(data_test)[:, 1]
                precision, recall, thresholds = precision_recall_curve(target_test, probabilities, pos_label=1)
                precisionRecallDf['precision'] = precision
                precisionRecallDf['recall'] = recall
                thresholds = list(thresholds)
                thresholds.insert(0, 0.00)
                precisionRecallDf['thresholds'] = thresholds
                PRCpath = os.path.join(ensemble_evaluations_out, str(classifierName) + '_precision_recall.csv')
                precisionRecallDf.to_csv(PRCpath)

            generate_learning_curve = currMetaFile['generate_sklearn_learning_curves'].iloc[0]
            shuffle_splits = currMetaFile['learning_curve_k_splits'].iloc[0]
            dataset_splits = currMetaFile['learning_curve_data_splits'].iloc[0]
            if generate_learning_curve == 'yes':
                print('Calculating learning curves...')
                LearningCurve(features, targetFrame, shuffle_splits, dataset_splits)

            # SAVE MODEL META DATA
            RF_meta_data = currMetaFile['generate_rf_model_meta_data_file'].iloc[0]
            if RF_meta_data == 'yes':
                print('Generating model meta data csv...')
                metaDataList = [classifierName, RF_criterion, RF_max_features, RF_min_sample_leaf, RF_n_estimators,
                                compute_permutation_importance, generate_classification_report,
                                generate_example_decision_tree, generate_features_importance_bar_graph,
                                generate_features_importance_log,
                                generate_precision_recall_curve, RF_meta_data, generate_learning_curve, dataset_splits,
                                shuffle_splits,
                                N_feature_importance_bars, over_sample_ratio, over_sample_setting, train_test_size,
                                under_sample_ratio, under_sample_setting]
                generateMetaData(metaDataList, classifierName, saveFileNo)


        #SAVE MODEL
        modelfn = str(classifierName) + '_' + str(saveFileNo) + '.sav'
        modelFileSavePath = os.path.join(modelPath, modelfn)
        pickle.dump(clf, open(modelFileSavePath, 'wb'))
        print('Classifier ' + str(modelfn) + ' saved in ' + str('models/validations/model_files ') + 'folder')
    print('Models generated. The models/evaluation files are in models/validations folders')
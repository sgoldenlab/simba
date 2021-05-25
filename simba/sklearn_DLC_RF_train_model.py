import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=DeprecationWarning)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
from configparser import ConfigParser, NoOptionError
import os
import warnings
from sklearn.tree import export_graphviz
from subprocess import call
from yellowbrick.classifier import ClassificationReport
import matplotlib.pyplot as plt
import xgboost as xgb
from imblearn.combine import SMOTEENN
import datetime

def RF_trainmodel(configini):

    warnings.simplefilter("ignore")

    configFile = str(configini)
    config = ConfigParser()
    config.read(configFile)
    modelDir = config.get('SML settings', 'model_dir')
    modelDir_out = os.path.join(modelDir, 'generated_models')
    if not os.path.exists(modelDir_out):
        os.makedirs(modelDir_out)
    tree_evaluations_out = os.path.join(modelDir_out, 'model_evaluations')
    if not os.path.exists(tree_evaluations_out):
        os.makedirs(tree_evaluations_out)
    model_nos = config.getint('SML settings', 'No_targets')
    csv_dir = config.get('General settings', 'csv_path')
    csv_dir_in = os.path.join(csv_dir, 'targets_inserted')
    N_feature_importance = config.getint('RF settings', 'Feature_importances_to_plot')
    under_sample_correction_value = config.getfloat('RF settings', 'under_sample_correction_value')
    n_estimators = config.getint('RF settings', 'n_estimators')
    max_features = config.get('RF settings', 'max_features')
    max_depth = config.getint('RF settings', 'max_depth')
    ensemble_method = config.get('RF settings', 'ensemble_method')
    n_jobs = config.getint('RF settings', 'n_jobs')
    criterion = config.get('RF settings', 'criterion')
    min_samples_leaf = config.getint('RF settings', 'min_samples_leaf')
    test_size = config.getfloat('RF settings', 'test_size')

    loop = 1
    model_paths = []
    target_names = []
    filesFound = []
    features = pd.DataFrame()
    modelFrame = pd.DataFrame()

    log_path = config.get('General settings', 'project_path')
    log_path = os.path.join(log_path, 'logs')

    ########### GET MODEL PATHS AND NAMES ###########
    for i in range(model_nos):
        currentModelPaths = 'model_path_' + str(loop)
        currentModelNames = 'target_name_' + str(loop)
        currentModelPaths = config.get('SML settings', currentModelPaths)
        currentModelNames = config.get('SML settings', currentModelNames)
        model_paths.append(currentModelPaths)
        target_names.append(currentModelNames)
        loop += 1
    loop = 0

    ########### FIND CSV FILES AND CONSOLIDATE ###########
    print('consolidating csvs....')
    for i in os.listdir(csv_dir_in):
        if i.__contains__(".csv"):
            currentFn = os.path.join(csv_dir_in, i)
            print(currentFn)
            df = pd.read_csv(currentFn)
            features = features.append(df, ignore_index=True)
    features = features.loc[:, ~features.columns.str.contains('^Unnamed')]

    ########### REMOVE TARGET VALUES, IF THEY EXSIST ##########
    frame_number = features.pop('frames').values
    video_number = features.pop('video_no').values
    try:
        for i in range(model_nos):
            currentModelName = target_names[i]
            modelFrame[currentModelName] = features.pop(currentModelName).values
    except KeyError:
        print('No target data found in input data... ')
    features = features.fillna(0)

    ########## REMOVE COORDINATE DATA ###########################################
    featuresDf = features.drop(
        ["scorer", "Ear_left_1_x", "Ear_left_1_y", "Ear_left_1_p", "Ear_right_1_x", "Ear_right_1_y", "Ear_right_1_p",
         "Nose_1_x", "Nose_1_y", "Nose_1_p", "Center_1_x", "Center_1_y", "Center_1_p", "Lat_left_1_x", "Lat_left_1_y",
         "Lat_left_1_p", "Lat_right_1_x", "Lat_right_1_y", "Lat_right_1_p", "Tail_base_1_x", "Tail_base_1_y",
         "Tail_base_1_p", "Tail_end_1_x", "Tail_end_1_y", "Tail_end_1_p", "Ear_left_2_x",
         "Ear_left_2_y", "Ear_left_2_p", "Ear_right_2_x", "Ear_right_2_y", "Ear_right_2_p", "Nose_2_x", "Nose_2_y",
         "Nose_2_p", "Center_2_x", "Center_2_y", "Center_2_p", "Lat_left_2_x", "Lat_left_2_y",
         "Lat_left_2_p", "Lat_right_2_x", "Lat_right_2_y", "Lat_right_2_p", "Tail_base_2_x", "Tail_base_2_y",
         "Tail_base_2_p", "Tail_end_2_x", "Tail_end_2_y", "Tail_end_2_p"], axis=1)

    feature_list = list(featuresDf.columns)

    ########## STANDARDIZE DATA ###########################################
    # scaler = MinMaxScaler()
    # scaled_values = scaler.fit_transform(featuresDf)
    # featuresDf.loc[:,:] = scaled_values

    ########################## CREATE MODEL FOR EACH TARGET ###################################################
    for i in range(model_nos):
        currTargetName = target_names[i]
        featuresDf[currTargetName] = modelFrame[modelFrame.columns[i]]
        targetFrameRows = featuresDf.loc[featuresDf[currTargetName] == 1]
        nonTargetFrameRows = featuresDf.loc[featuresDf[currTargetName] == 0]

        # SPLIT THE DATA UP IN TRAINING AND TESTING
        if "attack_prediction" in currTargetName:
            # sample = int((len(targetFrameRows) * 12))
            # nonTargetFrameRows = nonTargetFrameRows.sample(sample, replace=False)
            trainFrame = pd.concat([targetFrameRows, nonTargetFrameRows])
            trainFrame = trainFrame.sample(frac=1)
            target = trainFrame.pop(currTargetName).values
            featuresDf = featuresDf.drop([currTargetName], axis=1)
            data_train, data_test, target_train, target_test = train_test_split(trainFrame, target, test_size=test_size)
            print('SMOTE skipped')
        elif "anogenital_prediction" in currTargetName:
            sample = int((len(targetFrameRows) * 1.5))
            nonTargetFrameRows = nonTargetFrameRows.sample(sample, replace=False)
            trainFrame = pd.concat([targetFrameRows, nonTargetFrameRows])
            trainFrame = trainFrame.sample(frac=1)
            target = trainFrame.pop(currTargetName).values
            featuresDf = featuresDf.drop([currTargetName], axis=1)
            data_train, data_test, target_train, target_test = train_test_split(trainFrame, target, test_size=test_size)
            # print('Applying SMOTE for anogenital...')
            # smt = SMOTEENN(sampling_strategy=1)
            # data_train, target_train = smt.fit_sample(data_train, target_train)
        elif "tail_rattle" in currTargetName:
            sample = int((len(targetFrameRows) * 2.5))
            nonTargetFrameRows = nonTargetFrameRows.sample(sample, replace=False)
            trainFrame = pd.concat([targetFrameRows, nonTargetFrameRows])
            trainFrame = trainFrame.sample(frac=1)
            target = trainFrame.pop(currTargetName).values
            featuresDf = featuresDf.drop([currTargetName], axis=1)
            data_train, data_test, target_train, target_test = train_test_split(trainFrame, target, test_size=test_size)
            # print('Applying SMOTE for tail rattle...')
            # smt = SMOTEENN(sampling_strategy=1)
            # data_train, target_train = smt.fit_sample(data_train, target_train)
        elif "pursuit_prediction" in currTargetName:
            sample = int((len(targetFrameRows) * 2.5))
            nonTargetFrameRows = nonTargetFrameRows.sample(sample, replace=False)
            trainFrame = pd.concat([targetFrameRows, nonTargetFrameRows])
            trainFrame = trainFrame.sample(frac=1)
            target = trainFrame.pop(currTargetName).values
            featuresDf = featuresDf.drop([currTargetName], axis=1)
            data_train, data_test, target_train, target_test = train_test_split(trainFrame, target, test_size=test_size)
            # print('Applying SMOTE for pursuit...')
            # smt = SMOTEENN(sampling_strategy=1)
            # data_train, target_train = smt.fit_sample(data_train, target_train)
        elif "lateral_threat" in currTargetName:
            sample = int((len(targetFrameRows) * 2.5))
            nonTargetFrameRows = nonTargetFrameRows.sample(sample, replace=False)
            trainFrame = pd.concat([targetFrameRows, nonTargetFrameRows])
            trainFrame = trainFrame.sample(frac=1)
            target = trainFrame.pop(currTargetName).values
            featuresDf = featuresDf.drop([currTargetName], axis=1)
            data_train, data_test, target_train, target_test = train_test_split(trainFrame, target, test_size=test_size)
            # print('Applying SMOTE lateral threat...')
            # smt = SMOTEENN(sampling_strategy=1)
            # data_train, target_train = smt.fit_sample(data_train, target_train)
        else:
            sample = int((len(targetFrameRows) * under_sample_correction_value))
            nonTargetFrameRows = nonTargetFrameRows.sample(sample, replace=False)
            trainFrame = pd.concat([targetFrameRows, nonTargetFrameRows])
            trainFrame = trainFrame.sample(frac=1)
            target = trainFrame.pop(currTargetName).values
            featuresDf = featuresDf.drop([currTargetName], axis=1)
            data_train, data_test, target_train, target_test = train_test_split(trainFrame, target, test_size=test_size)

        # RANDOM FORREST
        if ensemble_method == 'RF':
            clf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, n_jobs=n_jobs,
                                         criterion=criterion, min_samples_leaf=min_samples_leaf, bootstrap=True,
                                         verbose=1)
            clf.fit(data_train, target_train)
            clf_pred = clf.predict(data_test)
            print("Accuracy " + str(currTargetName) + ' model:', metrics.accuracy_score(target_test, clf_pred))

        if ensemble_method == 'GBC':
            clf = GradientBoostingClassifier(max_depth=max_depth, n_estimators=n_estimators, learning_rate=0.1,
                                             max_features='sqrt', verbose=1)
            clf.fit(data_train, target_train)
            clf_pred = clf.predict(data_test)
            print(str(currTargetName) + str(" Accuracy train: ") + str(clf.score(data_train, target_train)))
            print(str(currTargetName) + str(" Accuracy test: ") + str(clf.score(data_test, target_test)))

        if ensemble_method == 'XGB':
            data_train = xgb.DMatrix(data_train, target)
            data_test = xgb.DMatrix(data_test)
            clf = xgb.XGBClassifier(max_depth=max_depth, min_child_weight=1, learning_rate=0.1,
                                    n_estimators=n_estimators, silent=0, objective='binary:logistic', max_delta_step=0,
                                    subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=0,
                                    scale_pos_weight=1, seed=1, missing=None, verbosity=3)
            clf.fit(data_train, target_train, verbose=True)
            clf_pred = clf.predict(data_test)
            print(str(currTargetName) + str(" Accuracy train: ") + str(clf.score(data_train, target_train)))
            print(str(currTargetName) + str(" Accuracy test: ") + str(clf.score(data_test, target_test)))

        # SAVE MODELS
        modelfn = str(currTargetName) + '.sav'
        modelPath = os.path.join(modelDir_out, modelfn)
        pickle.dump(clf, open(modelPath, 'wb'))

        # VISUALIZE A SINGLE TREE
        print('Generating model evaluations...')
        if ensemble_method == 'RF':
            estimator = clf.estimators_[3]
            dot_name = os.path.join(tree_evaluations_out, str(currTargetName) + '_tree.dot')
            file_name = os.path.join(tree_evaluations_out, str(currTargetName) + '_tree.pdf')
            class_names = ['Not_' + currTargetName, currTargetName]
            export_graphviz(estimator, out_file=dot_name, filled=True, rounded=True, special_characters=False,
                            impurity=False, class_names=class_names, feature_names=trainFrame.columns)
            commandStr = ('dot ' + str(dot_name) + ' -T pdf -o ' + str(file_name) + ' -Gdpi=600')
            call(commandStr, shell=True)

            ################ VISUALIZE CLASSIFICATION REPORT ##################################
            try:
                visualizer = ClassificationReport(clf, classes=class_names, support=True)
                visualizer.fit(data_train, target_train)
                visualizer.score(data_test, target_test)
                visualizerPath = os.path.join(tree_evaluations_out, str(currTargetName) + '_classificationReport.png')
                g = visualizer.poof(outpath=visualizerPath)
            except KeyError:
                print(('Warning, not enough data for ') + str(currTargetName))

        ################ FEATURE IMPORTANCE LOG  ##################################
        importances = list(clf.feature_importances_)
        feature_importances = [(feature, round(importance, 2)) for feature, importance in
                               zip(feature_list, importances)]
        feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
        feature_importance_list = [('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
        feature_importance_list_varNm = [i.split(':' " ", 3)[1] for i in feature_importance_list]
        feature_importance_list_importance = [i.split(':' " ", 3)[2] for i in feature_importance_list]

        log_df = pd.DataFrame()
        log_df['Feature_name'] = feature_importance_list_varNm
        log_df['Feature_importance'] = feature_importance_list_importance
        logPath = os.path.join(log_path, str(currTargetName) + '.csv')
        log_df.to_csv(logPath)

        ################ FEATURE IMPORTANCE BAR GRAPH #################################
        log_df['Feature_importance'] = log_df['Feature_importance'].apply(pd.to_numeric)
        log_df['Feature_name'] = log_df['Feature_name'].map(lambda x: x.lstrip('+-').rstrip('Importance'))
        log_df = log_df.head(N_feature_importance)
        ax = log_df.plot.bar(x='Feature_name', y='Feature_importance', legend=False, rot=90, fontsize=6)
        figName = str(currTargetName) + '_feature_bars.png'
        figSavePath = os.path.join(tree_evaluations_out, figName)
        plt.tight_layout()
        plt.savefig(figSavePath, dpi=600)
        plt.close('all')
        print('Train model complete.')

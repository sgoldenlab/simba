import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=DeprecationWarning)
from configparser import ConfigParser, MissingSectionHeaderError
import os
import pandas as pd
from dtreeviz.trees import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from yellowbrick.classifier import ClassificationReport
from xgboost.sklearn import XGBClassifier
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.tree import export_graphviz
from subprocess import call
import pickle
import csv
import warnings
import numpy as np
import eli5
from eli5.sklearn import PermutationImportance
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import precision_recall_curve
import graphviz
import graphviz.backend
from dtreeviz.shadow import *
from sklearn import tree
from drop_bp_cords import drop_bp_cords, GenerateMetaDataFileHeaders
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, cross_validate

def trainmodel2(inifile):
    configFile = str(inifile)
    config = ConfigParser()
    try:
        config.read(configFile)
    except MissingSectionHeaderError:
        print('ERROR:  Not a valid project_config file. Please check the project_config.ini path.')
    modelDir = config.get('SML settings', 'model_dir')
    modelDir_out = os.path.join(modelDir, 'generated_models')
    if not os.path.exists(modelDir_out):
        os.makedirs(modelDir_out)
    tree_evaluations_out = os.path.join(modelDir_out, 'model_evaluations')
    if not os.path.exists(tree_evaluations_out):
        os.makedirs(tree_evaluations_out)
    try:
        model_nos = config.getint('SML settings', 'No_targets')
        data_folder = config.get('create ensemble settings', 'data_folder')
        model_to_run = config.get('create ensemble settings', 'model_to_run')
        classifierName = config.get('create ensemble settings', 'classifier')
        train_test_size = config.getfloat('create ensemble settings', 'train_test_size')
        pose_estimation_body_parts = config.get('create ensemble settings', 'pose_estimation_body_parts')
    except ValueError:
        print('ERROR: Project_config.ini contains errors in the [create ensemble settings] or [SML settings] sections. Please check the project_config.ini file.')
    log_path = config.get('General settings', 'project_path')

    log_path = os.path.join(log_path, 'logs')
    features = pd.DataFrame()

    def generateClassificationReport(clf, class_names, rounds):
        try:
            print(rounds)
            visualizer = ClassificationReport(clf, classes=class_names, support=True)
            visualizer.score(data_test, target_test)
            visualizerPath = os.path.join(tree_evaluations_out, str(classifierName) + str(rounds) + '_classificationReport.png')
            g = visualizer.poof(outpath=visualizerPath, clear_figure=True)
        except KeyError:
            print(('Warning, not enough data for classification report: ') + str(classifierName))

    def generateFeatureImportanceLog(importances):
        feature_importances = [(feature, round(importance, 2)) for feature, importance in
                               zip(feature_list, importances)]
        feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
        feature_importance_list = [('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
        feature_importance_list_varNm = [i.split(':' " ", 3)[1] for i in feature_importance_list]
        feature_importance_list_importance = [i.split(':' " ", 3)[2] for i in feature_importance_list]
        log_df = pd.DataFrame()
        log_df['Feature_name'] = feature_importance_list_varNm
        log_df['Feature_importance'] = feature_importance_list_importance
        savePath = os.path.join(tree_evaluations_out, str(classifierName) + '_feature_importance_log.csv')
        log_df.to_csv(savePath)
        return log_df

    def generateFeatureImportanceBarGraph(log_df, N_feature_importance_bars):
        log_df['Feature_importance'] = log_df['Feature_importance'].apply(pd.to_numeric)
        log_df['Feature_name'] = log_df['Feature_name'].map(lambda x: x.lstrip('+-').rstrip('Importance'))
        log_df = log_df.head(N_feature_importance_bars)
        ax = log_df.plot.bar(x='Feature_name', y='Feature_importance', legend=False, rot=90, fontsize=6)
        figName = str(classifierName) + '_feature_bars.png'
        figSavePath = os.path.join(tree_evaluations_out, figName)
        plt.ylabel('Feature_importance (mean decrease impurity)')
        plt.tight_layout()
        plt.savefig(figSavePath, dpi=600)
        plt.close('all')

    def generateExampleDecisionTree(estimator):
        dot_name = os.path.join(tree_evaluations_out, str(classifierName) + '_tree.dot')
        file_name = os.path.join(tree_evaluations_out, str(classifierName) + '_tree.pdf')
        export_graphviz(estimator, out_file=dot_name, filled=True, rounded=True, special_characters=False,
                        impurity=False,
                        class_names=class_names, feature_names=data_train.columns)
        commandStr = ('dot ' + str(dot_name) + ' -T pdf -o ' + str(file_name) + ' -Gdpi=600')
        call(commandStr, shell=True)

    def generateMetaData(metaDataList):
        metaDataFn = str(classifierName) + '_meta.csv'
        metaDataPath = os.path.join(modelDir_out, metaDataFn)
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
        fname = os.path.join(tree_evaluations_out, str(classifierName) + '_permutations_importances.csv')
        dfvals.to_csv(fname, index=False)

    def LearningCurve(features, targetFrame, shuffle_splits, dataset_splits):
        newDataTargets = np.concatenate((target_train, target_test), axis=0)
        newDataFeatures = np.concatenate((data_train, data_test), axis=0)
        cv = ShuffleSplit(n_splits=shuffle_splits, test_size=train_test_size, random_state=0)
        model = RandomForestClassifier(n_estimators=RF_n_estimators, max_features=RF_max_features, n_jobs=-1, criterion=RF_criterion, min_samples_leaf=RF_min_sample_leaf, bootstrap=True, verbose=0)
        train_sizes, train_scores, test_scores = learning_curve(model, newDataFeatures, newDataTargets, cv=cv, scoring='f1', shuffle=True, n_jobs=-1, verbose=1, train_sizes=np.linspace(0.01, 1.0, dataset_splits))
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
        fname = os.path.join(tree_evaluations_out, str(classifierName) + '_learning_curve.csv')
        learningCurve_df.to_csv(fname, index=False)

    def dviz_classification_visualization(data_train, target_train, classifierName):
        clf = tree.DecisionTreeClassifier(max_depth=5, random_state=666)
        clf.fit(data_train, target_train)
        svg_tree = dtreeviz(clf, data_train, target_train, target_name=classifierName, feature_names=data_train.columns, orientation="TD", class_names=[classifierName, 'not_' + classifierName], fancy=True, histtype='strip', X=None, label_fontsize=12, ticks_fontsize=8, fontname="Arial")
        fname = os.path.join(tree_evaluations_out, str(classifierName) + 'fancy_decision_tree_example.svg')
        svg_tree.save(fname)

        print(rounds)
        # READ IN DATA FOLDER AND REMOVE ALL NON-FEATURE VARIABLES (POP DLC COORDINATE DATA AND TARGET DATA)
        print('Reading in ' + str(len(os.listdir(data_folder))) + ' annotated files...')
        for i in os.listdir(data_folder):
            if i.__contains__(".csv"):
                currentFn = os.path.join(data_folder, i)
                df = pd.read_csv(currentFn, index_col=0)
                features = features.append(df, ignore_index=True)
                print(features)
        features = features.loc[:, ~features.columns.str.contains('^Unnamed')]
        features = features.drop(["scorer"], axis=1, errors='ignore')
        totalTargetframes = features[classifierName].sum()
        try:
            targetFrame = features.pop(classifierName).values
        except KeyError:
            print('Error: the dataframe does not contain any target annotations. Please check the csv files in the project_folder/csv/target_inserted folder')
        features = features.fillna(0)
        features = drop_bp_cords(features, inifile)
        target_names = []
        loop = 1
        for i in range(model_nos):
            currentModelNames = 'target_name_' + str(loop)
            currentModelNames = config.get('SML settings', currentModelNames)
            if currentModelNames != classifierName:
                target_names.append(currentModelNames)
            loop += 1
        print('# of models to be created: 1')

        for i in range(len(target_names)):
            currentModelName = target_names[i]
            features.pop(currentModelName).values
        class_names = class_names = ['Not_' + classifierName, classifierName]
        feature_list = list(features)
        print('# of features in dataset: ' + str(len(feature_list)))

        # IF SET BY USER - PERFORM UNDERSAMPLING AND OVERSAMPLING IF SET BY USER
        data_train, data_test, target_train, target_test = train_test_split(features, targetFrame, test_size=train_test_size)
        under_sample_setting = config.get('create ensemble settings', 'under_sample_setting')
        over_sample_setting = config.get('create ensemble settings', 'over_sample_setting')
        trainDf = data_train
        trainDf[classifierName] = target_train
        targetFrameRows = trainDf.loc[trainDf[classifierName] == 1]
        print('# of ' + str(classifierName) + ' frames in dataset: ' + str(totalTargetframes))
        trainDf = trainDf.sample(frac=1).reset_index(drop=True)
        if under_sample_setting == 'Random undersample':
            print('Performing undersampling...')
            under_sample_ratio = config.getfloat('create ensemble settings', 'under_sample_ratio')
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
        #target_train = np.random.shuffle(target_train)



        # RUN THE DECISION ENSEMBLE SET BY THE USER
        # run random forest
        if model_to_run == 'RF':
            print('Training model ' + str(classifierName) + '...')
            RF_n_estimators = config.getint('create ensemble settings', 'RF_n_estimators')
            RF_max_features = config.get('create ensemble settings', 'RF_max_features')
            RF_criterion = config.get('create ensemble settings', 'RF_criterion')
            RF_min_sample_leaf = config.getint('create ensemble settings', 'RF_min_sample_leaf')
            clf = RandomForestClassifier(n_estimators=RF_n_estimators, max_features=RF_max_features, n_jobs=-1,
                                         criterion=RF_criterion, min_samples_leaf=RF_min_sample_leaf, bootstrap=True,
                                         verbose=1)
            try:
                clf.fit(data_train, target_train)
            except ValueError:
                print('ERROR: The model contains a faulty array. This may happen when trying to train a model with 0 examples of the behavior of interest')

            # predictions = clf.predict_proba(data_test)
            # data_test['probability'] = predictions[:, 1]
            # data_test['prediction'] = np.where(data_test['probability'] > 0.499999, 1, 0)
            # print(data_test['prediction'].sum())





            scoring = ['precision', 'recall', 'f1']
            newDataTargets = np.concatenate((target_train, target_test), axis=0)
            # #newDataTargets = np.where((newDataTargets == 0) | (newDataTargets == 1), newDataTargets ** 1, newDataTargets)
            # newDataFeatures = np.concatenate((data_train, data_test), axis=0)
            # #newDataFeatures = np.where((newDataFeatures == 0) | (newDataFeatures == 1), newDataFeatures ** 1, newDataFeatures)
            # cv = ShuffleSplit(n_splits=5, test_size=train_test_size)
            # results = cross_validate(clf, newDataFeatures, newDataTargets, cv=cv, scoring=scoring)
            # results = pd.DataFrame.from_dict(results)
            # crossValresultsFname = os.path.join(tree_evaluations_out, str(classifierName) + '_cross_val_100.csv')
            # results.to_csv(crossValresultsFname)

            # #RUN RANDOM FOREST EVALUATIONS
            # compute_permutation_importance = config.get('create ensemble settings', 'compute_permutation_importance')
            # if compute_permutation_importance == 'yes':
            #     print('Calculating permutation importances...')
            #     computePermutationImportance(data_test, target_test, clf)
            #
            # generate_learning_curve = config.get('create ensemble settings', 'generate_learning_curve')
            # if generate_learning_curve == 'yes':
            #     shuffle_splits = config.getint('create ensemble settings', 'LearningCurve_shuffle_k_splits')
            #     dataset_splits = config.getint('create ensemble settings', 'LearningCurve_shuffle_data_splits')
            #     print('Calculating learning curves...')
            #     LearningCurve(features, targetFrame, shuffle_splits, dataset_splits)
            # if generate_learning_curve != 'yes':
            #     shuffle_splits = 'NaN'
            #     dataset_splits = 'NaN'

            # generate_precision_recall_curve = config.get('create ensemble settings', 'generate_precision_recall_curve')
            # if generate_precision_recall_curve == 'yes':
            #     print('Calculating precision recall curve...')
            #     precisionRecallDf = pd.DataFrame()
            #     probabilities = clf.predict_proba(data_test)[:, 1]
            #     precision, recall, thresholds = precision_recall_curve(target_test, probabilities, pos_label=1)
            #     precisionRecallDf['precision'] = precision
            #     precisionRecallDf['recall'] = recall
            #     thresholds = list(thresholds)
            #     thresholds.insert(0, 0.00)
            #     precisionRecallDf['thresholds'] = thresholds
            #     PRCpath = os.path.join(tree_evaluations_out, str(classifierName) + '_precision_recall.csv')
            #     precisionRecallDf.to_csv(PRCpath)
            #
            # generate_example_decision_tree = config.get('create ensemble settings', 'generate_example_decision_tree')
            # if generate_example_decision_tree == 'yes':
            #     print('Generating example decision tree using graphviz...')
            #     estimator = clf.estimators_[3]
            #     generateExampleDecisionTree(estimator)

            generate_classification_report = config.get('create ensemble settings', 'generate_classification_report')
            if generate_classification_report == 'yes':
                print('Generating yellowbrick classification report...')
                generateClassificationReport(clf, class_names, rounds)

            # generate_features_importance_log = config.get('create ensemble settings', 'generate_features_importance_log')
            # if generate_features_importance_log == 'yes':
            #     print('Generating feature importance log...')
            #     importances = list(clf.feature_importances_)
            #     log_df = generateFeatureImportanceLog(importances)
            #
            # generate_features_importance_bar_graph = config.get('create ensemble settings', 'generate_features_importance_bar_graph')
            # if generate_features_importance_bar_graph == 'yes':
            #     N_feature_importance_bars = config.getint('create ensemble settings', 'N_feature_importance_bars')
            #     print('Generating feature importance bar graph...')
            #     generateFeatureImportanceBarGraph(log_df, N_feature_importance_bars)
            # if generate_features_importance_bar_graph != 'yes':
            #     N_feature_importance_bars = 'NaN'

            # generate_example_decision_tree_fancy = config.get('create ensemble settings','generate_example_decision_tree_fancy')
            # if generate_example_decision_tree_fancy == 'yes':
            #     print('Generating fancy decision tree example...')
            #     dviz_classification_visualization(data_train, target_train, classifierName)

            # SAVE MODEL META DATA
            RF_meta_data = config.get('create ensemble settings', 'RF_meta_data')
            if RF_meta_data == 'yes':
                metaDataList = [classifierName, RF_criterion, RF_max_features, RF_min_sample_leaf, RF_n_estimators,
                                compute_permutation_importance, generate_classification_report,
                                generate_example_decision_tree, generate_features_importance_bar_graph,
                                generate_features_importance_log,
                                generate_precision_recall_curve, RF_meta_data, generate_learning_curve, dataset_splits,
                                shuffle_splits, N_feature_importance_bars, over_sample_ratio, over_sample_setting, train_test_size,
                                under_sample_ratio, under_sample_ratio]
                generateMetaData(metaDataList)

        # run gradient boost model
        if model_to_run == 'GBC':
            GBC_n_estimators = config.getint('create ensemble settings', 'GBC_n_estimators')
            GBC_max_features = config.get('create ensemble settings', 'GBC_max_features')
            GBC_max_depth = config.getint('create ensemble settings', 'GBC_max_depth')
            GBC_learning_rate = config.getfloat('create ensemble settings', 'GBC_learning_rate')
            GBC_min_sample_split = config.getint('create ensemble settings', 'GBC_min_sample_split')
            clf = GradientBoostingClassifier(max_depth=GBC_max_depth, n_estimators=GBC_n_estimators,
                                             learning_rate=GBC_learning_rate, max_features=GBC_max_features,
                                             min_samples_split=GBC_min_sample_split, verbose=1)
            clf.fit(data_train, target_train)
            clf_pred = clf.predict(data_test)
            print(str(classifierName) + str(" Accuracy train: ") + str(clf.score(data_train, target_train)))

            generate_example_decision_tree = config.get('create ensemble settings', 'generate_example_decision_tree')
            if generate_example_decision_tree == 'yes':
                estimator = clf.estimators_[3, 0]
                generateExampleDecisionTree(estimator)

            generate_classification_report = config.get('create ensemble settings', 'generate_classification_report')
            if generate_classification_report == 'yes':
                generateClassificationReport(clf, class_names)

            generate_features_importance_log = config.get('create ensemble settings', 'generate_features_importance_log')
            if generate_features_importance_log == 'yes':
                importances = list(clf.feature_importances_)
                log_df = generateFeatureImportanceLog(importances)

            generate_features_importance_bar_graph = config.get('create ensemble settings',
                                                                'generate_features_importance_bar_graph')
            N_feature_importance_bars = config.getint('create ensemble settings', 'N_feature_importance_bars')
            if generate_features_importance_bar_graph == 'yes':
                generateFeatureImportanceBarGraph(log_df, N_feature_importance_bars)

        # run XGboost
        if model_to_run == 'XGB':
            XGB_n_estimators = config.getint('create ensemble settings', 'XGB_n_estimators')
            XGB_max_depth = config.getint('create ensemble settings', 'GBC_max_depth')
            XGB_learning_rate = config.getfloat('create ensemble settings', 'XGB_learning_rate')
            clf = XGBClassifier(max_depth=XGB_max_depth, min_child_weight=1, learning_rate=XGB_learning_rate,
                                n_estimators=XGB_n_estimators,
                                silent=0, objective='binary:logistic', max_delta_step=0, subsample=1, colsample_bytree=1,
                                colsample_bylevel=1, reg_alpha=0, reg_lambda=0, scale_pos_weight=1, seed=1, missing=None,
                                verbosity=3)
            clf.fit(data_train, target_train, verbose=True)

        # SAVE MODEL
        modelfn = str(classifierName) + '.sav'
        modelPath = os.path.join(modelDir_out, modelfn)
        pickle.dump(clf, open(modelPath, 'wb'))
        print('Classifier ' + str(classifierName) + ' saved @ ' + str('models/generated_models ') + 'folder')
        print('Evaluation files are in models/generated_models/model_evaluations folders')




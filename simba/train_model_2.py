import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=DeprecationWarning)
from configparser import ConfigParser, MissingSectionHeaderError, NoSectionError,NoOptionError
import os, glob
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
import shap
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
from simba.drop_bp_cords import drop_bp_cords, GenerateMetaDataFileHeaders
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, cross_validate
from simba.rw_dfs import *
from sklearn.feature_selection import RFECV
from simba.shap_calcs import shap_summary_calculations
import time
# import timeit


def trainmodel2(inifile):
    configFile = str(inifile)
    config = ConfigParser()
    try:
        config.read(configFile)
    except MissingSectionHeaderError:
        print('ERROR:  Not a valid project_config file. Please check the project_config.ini path.')
    modelDir = config.get('SML settings', 'model_dir')
    modelDir_out = os.path.join(modelDir, 'generated_models')
    projectPath = config.get('General settings', 'project_path')
    csv_dir_in, csv_dir_out = os.path.join(projectPath, 'csv', 'targets_inserted'), os.path.join(projectPath,'csv', 'machine_results')
    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'
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
    except ValueError:
        print('ERROR: Project_config.ini contains errors in the [create ensemble settings] or [SML settings] sections. Please check the project_config.ini file.')
    features = pd.DataFrame()

    def generateClassificationReport(clf, class_names):
        try:
            visualizer = ClassificationReport(clf, classes=class_names, support=True)
            visualizer.score(data_test, target_test)
            visualizerPath = os.path.join(tree_evaluations_out, str(classifierName) + '_classificationReport.png')
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

    def generateShapLog(trainingSet, target_train, feature_list, classifierName, shap_target_present_no,shap_target_absent_no, inifile):
        print('Calculating SHAP scores for ' + str(len(trainingSet)) +' observations...')
        trainingSet[classifierName] = target_train
        targetTrainSet = trainingSet.loc[trainingSet[classifierName] == 1]
        nonTargetTrain = trainingSet.loc[trainingSet[classifierName] == 0]
        try:
            targetsForShap = targetTrainSet.sample(shap_target_present_no, replace=False)
        except ValueError:
            print('Could not find ' + str(shap_target_present_no) + ' in dataset. Taking the maximum instead (' + str(len(targetTrainSet)) + ')')
            targetsForShap = targetTrainSet.sample(len(targetTrainSet), replace=False)
        nontargetsForShap = nonTargetTrain.sample(shap_target_absent_no, replace=False)
        shapTrainingSet = pd.concat([targetsForShap, nontargetsForShap])
        targetValFrame = shapTrainingSet.pop(classifierName).values
        explainer = shap.TreeExplainer(clf, data=None, model_output='raw', feature_perturbation='tree_path_dependent')
        expected_value = explainer.expected_value[1]
        outputDfRaw = pd.DataFrame(columns=feature_list)
        shapHeaders = feature_list.copy()
        shapHeaders.extend(('Expected_value', 'Sum', 'Prediction_probability', str(classifierName)))
        outputDfShap = pd.DataFrame(columns=shapHeaders)
        counter = 0
        for frame in range(len(shapTrainingSet)):
            currInstance = shapTrainingSet.iloc[[frame]]
            shap_values = explainer.shap_values(currInstance, check_additivity=False)
            shapList = list(shap_values[0][0])
            shapList = [x * -1 for x in shapList]
            shapList.append(expected_value)
            shapList.append(sum(shapList))
            probability = clf.predict_proba(currInstance)
            shapList.append(probability[0][1])
            shapList.append(targetValFrame[frame])
            currRaw = list(shapTrainingSet.iloc[frame])
            outputDfRaw.loc[len(outputDfRaw)] = currRaw
            outputDfShap.loc[len(outputDfShap)] = shapList
            counter += 1
            print('SHAP calculated for: ' + str(counter) + '/' + str(len(shapTrainingSet)) + ' frames')
        outputDfShap.to_csv(os.path.join(tree_evaluations_out, 'SHAP_values_' + str(classifierName) + '.csv'))
        print('Creating SHAP summary statistics...')
        shap_summary_calculations(inifile, outputDfShap, classifierName, expected_value, tree_evaluations_out)
        outputDfRaw.to_csv(os.path.join(tree_evaluations_out, 'RAW_SHAP_feature_values_' + str(classifierName) + '.csv'))
        print('All SHAP data saved in project_folder/models/evaluations directory')

    def perf_RFCVE(projectPath, RFCVE_CVs, RFCVE_step_size, clf, data_train, target_train, feature_list):
        selector = RFECV(estimator=clf, step=RFCVE_step_size, cv=RFCVE_CVs, scoring='f1', verbose=1)
        selector = selector.fit(data_train, target_train)
        selectorSupport = selector.support_.tolist()
        trueIndex = np.where(selectorSupport)
        trueIndex = list(trueIndex[0])
        selectedFeatures = [feature_list[i] for i in trueIndex]
        selectedFeaturesDf = pd.DataFrame(selectedFeatures, columns=['Selected_features'])
        savePath = os.path.join(tree_evaluations_out, 'RFECV_selected_features_' + str(classifierName) + '.csv')
        selectedFeaturesDf.to_csv(savePath)
        print('Recursive feature elimination results stored in ' + str(savePath))

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

    # READ IN DATA FOLDER AND REMOVE ALL NON-FEATURE VARIABLES (POP DLC COORDINATE DATA AND TARGET DATA)
    print('Reading in ' + str(len(glob.glob(data_folder + '/*.' + wfileType))) + ' annotated files...')
    filesFound = glob.glob(csv_dir_in + '/*.' + wfileType)
    for file in filesFound:
        df = read_df(file, wfileType)
        df = df.dropna(axis=0, how='all')
        features = features.append(df, ignore_index=True)
    try:
        features = features.set_index('scorer')
    except KeyError:
        pass
    features = features.loc[:, ~features.columns.str.contains('^Unnamed')]
    totalTargetframes = features[classifierName].sum()
    try:
        targetFrame = features.pop(classifierName).values
    except KeyError:
        print('Error: the dataframe does not contain any target annotations. Please check the csv files in the project_folder/csv/target_inserted folder')
    features = features.fillna(0)
    try:
        features = drop_bp_cords(features, inifile)
    except KeyError:
        print('Could not drop bodypart coordinates, bodypart coordinates missing in dataframe')
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

    if under_sample_setting == 'Random undersample':
        try:
            print('Performing undersampling...')
            under_sample_ratio = config.getfloat('create ensemble settings', 'under_sample_ratio')
            nonTargetFrameRows = trainDf.loc[trainDf[classifierName] == 0]
            nontargetFrameRowsSize = int(len(targetFrameRows) * under_sample_ratio)
            nonTargetFrameRows = nonTargetFrameRows.sample(nontargetFrameRowsSize, replace=False)
            trainDf = pd.concat([targetFrameRows, nonTargetFrameRows])
            target_train = trainDf.pop(classifierName).values
            data_train = trainDf
        except ValueError:
            print('Undersampling failed: the undersampling ratio for the model is likely too high - there are not enough non-events too sample. Fix this by decreasing the undersampling ratio.')
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
        except Exception as e:
            print(e)
            print('ERROR: The model contains a faulty array. This may happen when trying to train a model with 0 examples of the behavior of interest')

        # #RUN RANDOM FOREST EVALUATIONS
        compute_permutation_importance = config.get('create ensemble settings', 'compute_permutation_importance')
        if compute_permutation_importance == 'yes':
            print('Calculating permutation importances...')
            computePermutationImportance(data_test, target_test, clf)

        generate_learning_curve = config.get('create ensemble settings', 'generate_learning_curve')
        if generate_learning_curve == 'yes':
            shuffle_splits = config.getint('create ensemble settings', 'LearningCurve_shuffle_k_splits')
            dataset_splits = config.getint('create ensemble settings', 'LearningCurve_shuffle_data_splits')
            print('Calculating learning curves...')
            LearningCurve(features, targetFrame, shuffle_splits, dataset_splits)
        if generate_learning_curve != 'yes':
            shuffle_splits = 'NaN'
            dataset_splits = 'NaN'

        generate_precision_recall_curve = config.get('create ensemble settings', 'generate_precision_recall_curve')
        if generate_precision_recall_curve == 'yes':
            print('Calculating precision recall curve...')
            precisionRecallDf = pd.DataFrame()
            probabilities = clf.predict_proba(data_test)[:, 1]
            precision, recall, thresholds = precision_recall_curve(target_test, probabilities, pos_label=1)
            precisionRecallDf['precision'] = precision
            precisionRecallDf['recall'] = recall
            thresholds = list(thresholds)
            thresholds.insert(0, 0.00)
            precisionRecallDf['thresholds'] = thresholds
            PRCpath = os.path.join(tree_evaluations_out, str(classifierName) + '_precision_recall.csv')
            precisionRecallDf.to_csv(PRCpath)

        generate_example_decision_tree = config.get('create ensemble settings', 'generate_example_decision_tree')
        if generate_example_decision_tree == 'yes':
            print('Generating example decision tree using graphviz...')
            estimator = clf.estimators_[3]
            generateExampleDecisionTree(estimator)

        generate_classification_report = config.get('create ensemble settings', 'generate_classification_report')
        if generate_classification_report == 'yes':
            print('Generating yellowbrick classification report...')
            generateClassificationReport(clf, class_names)

        # generate_features_importance_log = config.get('create ensemble settings', 'generate_features_importance_log')
        # if generate_features_importance_log == 'yes':

        generate_features_importance_bar_graph = config.get('create ensemble settings', 'generate_features_importance_bar_graph')
        if generate_features_importance_bar_graph == 'yes':
            print('Generating feature importance log...')
            importances = list(clf.feature_importances_)
            log_df = generateFeatureImportanceLog(importances)
            generate_features_importance_log = 'yes'
            try:
                N_feature_importance_bars = config.getint('create ensemble settings', 'N_feature_importance_bars')
            except ValueError:
                print('SimBA could not find how many feature importance bars you want to generate. Double check that you entered an integer in the "Genereate Feature Importance Bar Graph" entry box"')
            print('Generating feature importance bar graph...')
            generateFeatureImportanceBarGraph(log_df, N_feature_importance_bars)
        if generate_features_importance_bar_graph != 'yes':
            N_feature_importance_bars = 'NaN'
            generate_features_importance_log = 'no'

        generate_example_decision_tree_fancy = config.get('create ensemble settings','generate_example_decision_tree_fancy')
        if generate_example_decision_tree_fancy == 'yes':
            print('Generating fancy decision tree example...')
            dviz_classification_visualization(data_train, target_train, classifierName)

        try:
            shap_scores_input = config.get('create ensemble settings', 'generate_shap_scores')
        except NoOptionError:
            shap_scores_input = 'no'
        if shap_scores_input == 'yes':
            shap_target_present_no = config.getint('create ensemble settings', 'shap_target_present_no')
            shap_target_absent_no = config.getint('create ensemble settings', 'shap_target_absent_no')
            generateShapLog(data_train, target_train, feature_list, classifierName, shap_target_present_no,shap_target_absent_no, inifile)

        try:
            RFECV_setting = config.get('create ensemble settings', 'perform_RFECV')
        except NoOptionError:
            RFECV_setting = 'no'
        if RFECV_setting == 'yes':
            perf_RFCVE(projectPath, 3, 5, clf, data_train, target_train, feature_list)

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



    # SAVE MODEL
    modelfn = str(classifierName) + '.sav'
    modelPath = os.path.join(modelDir_out, modelfn)
    pickle.dump(clf, open(modelPath, 'wb'))
    print('Classifier ' + str(classifierName) + ' saved @ ' + str('models/generated_models ') + 'folder')
    print('Evaluation files are in models/generated_models/model_evaluations folders')
    # stop = timeit.default_timer()
    # execution_time = stop - startTime
    # print(execution_time)
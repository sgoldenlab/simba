import os
import numpy as np
import pandas as pd
from simba.rw_dfs import read_df
from simba.misc_tools import get_fn_ext
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import precision_recall_curve
from copy import deepcopy
from sklearn.tree import export_graphviz
from subprocess import call
from yellowbrick.classifier import ClassificationReport
import shap
from simba.drop_bp_cords import GenerateMetaDataFileHeaders
from tabulate import tabulate
import pickle
from simba.shap_calcs import shap_summary_calculations
from simba.read_config_unit_tests import (check_int, check_str, check_float, read_config_entry, check_file_exist_and_readable)


from dtreeviz.trees import tree, dtreeviz

import matplotlib.pyplot as plt
plt.switch_backend('agg')

def read_all_files_in_folder(file_paths, file_type, classifier_names=None):
    df_concat = pd.DataFrame()
    for file in file_paths:
        _, vid_name, _ = get_fn_ext(file)
        df = read_df(file, file_type).dropna(axis=0, how='all').fillna(0)
        if classifier_names != None:
            for clf_name in classifier_names:
                if not clf_name in df.columns:
                    raise ValueError('Data for video {} does not contain any annotations for behavior {}'.format(vid_name, clf_name))
                else:
                    df_concat = pd.concat([df_concat, df], axis=0)
        else:
            df_concat = pd.concat([df_concat, df], axis=0)
    try:
        df_concat = df_concat.set_index('scorer')
    except KeyError:
        pass
    if len(df_concat) == 0:
        raise ValueError('ANNOTATION ERROR: SimBA found 0 annotated frames in the project_folder/csv/targets_inserted directory')
    df_concat = df_concat.loc[:, ~df_concat.columns.str.contains('^Unnamed')]

    return df_concat.reset_index(drop=True)


def read_in_all_model_names_to_remove(config, model_cnt, clf_name):
    annotation_cols_to_remove = []
    for model_no in range(model_cnt):
        model_name = config.get('SML settings', 'target_name_' + str(model_no+1))
        if model_name != clf_name:
            annotation_cols_to_remove.append(model_name)
    return annotation_cols_to_remove

def delete_other_annotation_columns(df, annotations_lst):
    for a_col in annotations_lst:
        df = df.drop([a_col], axis=1)
    return df

def split_df_to_x_y(df, clf_name):
    df = deepcopy(df)
    y = df.pop(clf_name)
    return df, y

def random_undersampler(x_train, y_train, sample_ratio):
    print('Performing under-sampling...')
    data_df = pd.concat([x_train, y_train], axis=1)
    present_df, absent_df = data_df[data_df[y_train.name] == 1], data_df[data_df[y_train.name] == 0]
    ratio_n = int(len(present_df) * sample_ratio)
    if len(absent_df) < ratio_n:
        raise ValueError('UNDER SAMPLING ERROR: The under-sample ratio of {} in classifier {} demands {} '
                         'behavior-absent annotations. This is more than the number of behavior-absent annotations in '
                         'the entire dataset ({}). Please annotate more images or decrease the under-sample ratio.'.format(str(sample_ratio), y_train.name, str(ratio_n), str(len(absent_df))))

    data_df = pd.concat([present_df, absent_df.sample(n=ratio_n, replace=False)], axis=0)
    return split_df_to_x_y(data_df, y_train.name)

def smoteen_oversampler(x_train, y_train, sample_ratio):
    print('Performing SMOTEEN oversampling...')
    smt = SMOTEENN(sampling_strategy=sample_ratio)
    return smt.fit_sample(x_train, y_train)

def smote_oversampler(x_train, y_train, sample_ratio):
    print('Performing SMOTE oversampling...')
    smt = SMOTE(sampling_strategy=sample_ratio)
    return smt.fit_sample(x_train, y_train)

def calc_permutation_importance(x_test,
                                y_test,
                                clf,
                                feature_names,
                                clf_name,
                                save_dir,
                                save_file_no=None):
    print('Calculating feature permutation importances...')
    p_importances = permutation_importance(clf, x_test, y_test, n_repeats=10, random_state = 0)
    df = pd.DataFrame(np.column_stack([feature_names, p_importances.importances_mean, p_importances.importances_std]), columns=['FEATURE_NAME', 'FEATURE_IMPORTANCE_MEAN', 'FEATURE_IMPORTANCE_STDEV'])
    df = df.sort_values(by=['FEATURE_IMPORTANCE_MEAN'], ascending=False)
    if save_file_no != None:
        save_file_path = os.path.join(save_dir, clf_name + '_' + str(save_file_no+1) +'_permutations_importances.csv')
    else:
        save_file_path = os.path.join(save_dir, clf_name + '_permutations_importances.csv')
    df.to_csv(save_file_path, index=False)

def calc_learning_curve(x_y_df,
                  clf_name,
                  shuffle_splits,
                  dataset_splits,
                  tt_size,
                  rf_clf,
                  save_dir,
                  save_file_no=None):
    print('Calculating learning curves...')
    x_df, y_df = split_df_to_x_y(x_y_df, clf_name)
    cv = ShuffleSplit(n_splits=shuffle_splits, test_size=tt_size, random_state=0)
    train_sizes, train_scores, test_scores = learning_curve(rf_clf, x_df, y_df, cv=cv, scoring='f1',shuffle=True, n_jobs=-1, verbose=0, train_sizes=np.linspace(0.01, 1.0, dataset_splits))
    results_df = pd.DataFrame()
    results_df['FRACTION TRAIN SIZE'] = np.linspace(0.01, 1.0, dataset_splits)
    results_df['TRAIN_MEAN_F1'] = np.mean(train_scores, axis=1)
    results_df['TEST_MEAN_F1'] = np.mean(test_scores, axis=1)
    results_df['TRAIN_STDEV_F1'] = np.std(train_scores, axis=1)
    results_df['TEST_STDEV_F1'] = np.std(test_scores, axis=1)
    if save_file_no != None:
        save_file_path = os.path.join(save_dir, clf_name + '_' + str(save_file_no+1) +'_learning_curve.csv')
    else:
        save_file_path = os.path.join(save_dir, clf_name + '_learning_curve.csv')
    results_df.to_csv(save_file_path, index=False)

def calc_pr_curve(rf_clf,
                  x_df,
                  y_df,
                  clf_name,
                  save_dir,
                  save_file_no=None):
    print('Calculating PR curves...')
    p = rf_clf.predict_proba(x_df)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_df, p, pos_label=1)
    pr_df = pd.DataFrame()
    pr_df['PRECISION'] = precision
    pr_df['RECALL'] = recall
    pr_df['F1'] = 2 * pr_df['RECALL'] * pr_df['PRECISION'] / (pr_df['RECALL'] + pr_df['PRECISION'])
    thresholds = list(thresholds)
    thresholds.insert(0, 0.00)
    pr_df['DISCRIMINATION THRESHOLDS'] = thresholds
    if save_file_no != None:
        save_file_path = os.path.join(save_dir, clf_name + '_' + str(save_file_no+1) +'_pr_curve.csv')
    else:
        save_file_path = os.path.join(save_dir, clf_name + '_pr_curve.csv')
    pr_df.to_csv(save_file_path, index=False)

def create_example_dt(rf_clf,
                      clf_name,
                      feature_names,
                      class_names,
                      save_dir,
                      save_file_no=None):
    print('Visualizing example decision tree using graphviz...')
    estimator = rf_clf.estimators_[3]
    if save_file_no != None:
        dot_name = os.path.join(save_dir, str(clf_name) + '_' + str(save_file_no) + '_tree.dot')
        file_name = os.path.join(save_dir, str(clf_name) + '_' + str(save_file_no) +'_tree.pdf')
    else:
        dot_name = os.path.join(save_dir, str(clf_name) + '_tree.dot')
        file_name = os.path.join(save_dir, str(clf_name) + '_tree.pdf')
    export_graphviz(estimator, out_file=dot_name, filled=True, rounded=True, special_characters=False, impurity=False, class_names=class_names, feature_names=feature_names)
    command = ('dot ' + str(dot_name) + ' -T pdf -o ' + str(file_name) + ' -Gdpi=600')
    call(command, shell=True)


def create_clf_report(rf_clf,
                      x_df,
                      y_df,
                      class_names,
                      save_dir,
                      save_file_no=None):
    print('Creating classification report visualization...')
    try:
        visualizer = ClassificationReport(rf_clf, classes=class_names, support=True)
        visualizer.score(x_df, y_df)
        if save_file_no != None:
            save_path = os.path.join(save_dir, class_names[1] + '_' + str(save_file_no) + '_classification_report.png')
        else:
            save_path = os.path.join(save_dir, class_names[1] + '_classification_report.png')
        visualizer.poof(outpath=save_path, clear_figure=True)
    except KeyError:
        print('SIMBA WARNING: Not enough data to create classification report: {}'.format(class_names[1]))

def create_x_importance_log(rf_clf,
                            x_names,
                            clf_name,
                            save_dir,
                            save_file_no=None):
    print('Creating feature importance log...')
    importances = list(rf_clf.feature_importances_)
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(x_names, importances)]
    df = pd.DataFrame(feature_importances, columns=['FEATURE', 'FEATURE_IMPORTANCE']).sort_values(by=['FEATURE_IMPORTANCE'], ascending=False)
    if save_file_no != None:
        save_file_path = os.path.join(save_dir, clf_name + '_' + str(save_file_no) + '_feature_importance_log.csv')
    else:
        save_file_path = os.path.join(save_dir, clf_name + '_feature_importance_log.csv')
    df.to_csv(save_file_path, index=False)

def create_x_importance_bar_chart(rf_clf,
                                  x_names,
                                  clf_name,
                                  save_dir,
                                  n_bars,
                                  save_file_no=None):
    print('Creating feature importance bar chart...')
    create_x_importance_log(rf_clf, x_names, clf_name, save_dir)
    importances_df = pd.read_csv(os.path.join(save_dir, clf_name + '_feature_importance_log.csv'))
    importances_head = importances_df.head(n_bars)
    ax = importances_head.plot.bar(x='FEATURE', y='FEATURE_IMPORTANCE', legend=False, rot=90, fontsize=6)
    plt.ylabel("Feature importances' (mean decrease impurity)")
    plt.tight_layout()
    if save_file_no != None:
        save_file_path = os.path.join(save_dir, clf_name + '_' + str(save_file_no) + '_feature_importance_bar_graph.png')
    else:
        save_file_path = os.path.join(save_dir, clf_name + '_feature_importance_bar_graph.png')
    plt.savefig(save_file_path, dpi=600)
    plt.close('all')


def dviz_classification_visualization(x_train, y_train, clf_name, class_names, save_dir):
    clf = tree.DecisionTreeClassifier(max_depth=5, random_state=666)
    clf.fit(x_train, y_train)
    svg_tree = dtreeviz(clf, x_train, y_train, target_name=clf_name, feature_names=x_train.columns, orientation="TD", class_names=class_names, fancy=True, histtype='strip', X=None, label_fontsize=12, ticks_fontsize=8, fontname="Arial")
    save_path = os.path.join(save_dir, clf_name + '_fancy_decision_tree_example.svg')
    svg_tree.save(save_path)

def create_shap_log(ini_file_path,
                    rf_clf,
                    x_df,
                    y_df,
                    x_names,
                    clf_name,
                    cnt_present,
                    cnt_absent,
                    save_path,
                    save_file_no=None):
    print('Calculating SHAP values...')
    print(save_file_no)
    data_df = pd.concat([x_df, y_df], axis=1)
    target_df, nontarget_df = data_df[data_df[y_df.name] == 1], data_df[data_df[y_df.name] == 0]
    if len(target_df) < cnt_present:
        print('SHAP WARNING: Train data contains {} behavior-present annotations. This is less the number of frames you specified to calculate shap values for {}. SimBA will calculate shap scores for the {} behavior-present frames available'.format(str(len(target_df)), str(cnt_present)))
        cnt_present = len(target_df)
    if len(nontarget_df) < cnt_absent:
        print('SHAP WARNING: Train data contains {} behavior-absent annotations. This is less the number of frames you specified to calculate shap values for {}. SimBA will calculate shap scores for the {} behavior-absent frames available'.format(str(len(nontarget_df)), str(cnt_absent)))
        cnt_absent = len(nontarget_df)
    non_target_for_shap = nontarget_df.sample(cnt_absent, replace=False)
    targets_for_shap = target_df.sample(cnt_present, replace=False)
    shap_df = pd.concat([targets_for_shap, non_target_for_shap], axis=0)
    y_df = shap_df.pop(clf_name).values
    explainer = shap.TreeExplainer(rf_clf, data=None, model_output='raw', feature_perturbation='tree_path_dependent')
    expected_value = explainer.expected_value[1]
    out_df_raw = pd.DataFrame(columns=x_names)
    shap_headers = list(x_names)
    shap_headers.extend(('Expected_value', 'Sum', 'Prediction_probability', clf_name))
    out_df_shap = pd.DataFrame(columns=shap_headers)
    for cnt, frame in enumerate(range(len(shap_df))):
        frame_data = shap_df.iloc[[frame]]
        frame_shap = explainer.shap_values(frame_data, check_additivity=False)[1][0].tolist()
        frame_shap.extend((expected_value, sum(frame_shap), rf_clf.predict_proba(frame_data)[0][1], y_df[cnt]))
        out_df_raw.loc[len(out_df_raw)] = list(shap_df.iloc[frame])
        out_df_shap.loc[len(out_df_shap)] = frame_shap
        print('SHAP frame: {} / {}'.format(str(cnt+1), str(len(shap_df))))

    if save_file_no == None:
        out_df_shap.to_csv(os.path.join(save_path, 'SHAP_values_' + clf_name + '.csv'))
        out_df_raw.to_csv(os.path.join(save_path, 'RAW_SHAP_feature_values_' + clf_name + '.csv'))

    else:
        out_df_shap.to_csv(os.path.join(save_path, 'SHAP_values_' + '_' + str(save_file_no) + clf_name + '.csv'))
        out_df_raw.to_csv(os.path.join(save_path, 'RAW_SHAP_feature_values_' + '_' + str(save_file_no) + clf_name + '.csv'))
    shap_summary_calculations(ini_file_path, out_df_shap, clf_name, expected_value, save_path)


def print_machine_model_information(model_dict):
    table_view = [["Model name", model_dict['Classifier_name']], ["Ensemble method", 'RF'],
                 ["Estimators (trees)", model_dict['RF_n_estimators']], ["Max features", model_dict['RF_max_features']],
                 ["Under sampling setting", model_dict['under_sample_setting']], ["Under sampling ratio", model_dict['under_sample_ratio']],
                 ["Over sampling setting", model_dict['over_sample_setting']], ["Over sampling ratio", model_dict['over_sample_ratio']],
                 ["criterion", model_dict['RF_criterion']], ["Min sample leaf", model_dict['RF_min_sample_leaf']]]
    headers = ["Setting", "value"]
    print(tabulate(table_view, headers, tablefmt="grid"))

def create_meta_data_csv_training_one_model(meta_data_lst, clf_name, save_dir):
    print('Saving model meta data file...')
    save_path = os.path.join(save_dir, clf_name + '_meta.csv')
    out_df = pd.DataFrame(columns=GenerateMetaDataFileHeaders())
    out_df.loc[len(out_df)] = meta_data_lst
    out_df.to_csv(save_path)


def create_meta_data_csv_training_multiple_models(meta_data,
                                                  clf_name,
                                                  save_dir,
                                                  save_file_no=None):
    print('Saving model meta data file...')
    save_path = os.path.join(save_dir, clf_name + '_' + str(save_file_no) + '_meta.csv')
    out_df = pd.DataFrame.from_dict(meta_data, orient='index').T
    out_df.to_csv(save_path)

def save_rf_model(rf_clf,
                  clf_name,
                  save_dir,
                  save_file_no=None):
    if save_file_no != None:
        save_path = os.path.join(save_dir, clf_name + '_' + str(save_file_no) + '.sav')
    else:
        save_path = os.path.join(save_dir, clf_name + '.sav')
    pickle.dump(rf_clf, open(save_path, 'wb'))

def get_model_info(config=None, model_cnt=None):
    model_dict = {}
    for n in range(model_cnt):
        try:
            model_dict[n] = {}
            if config.get('SML settings', 'model_path_' + str(n+1)) == '':
                print('SIMBA WARNING: Skipping {} classifier analysis: no path set to model file'.format(str(config.get('SML settings', 'target_name_' + str(n + 1)))))
                continue
            model_dict[n]['model_path'] = config.get('SML settings', 'model_path_' + str(n+1))
            #check_file_exist_and_readable(model_dict[n]['model_path'])
            model_dict[n]['model_name'] = config.get('SML settings', 'target_name_' + str(n+1))
            check_str('model_name', model_dict[n]['model_name'])
            model_dict[n]['threshold'] = config.getfloat('threshold_settings', 'threshold_' + str(n+1))
            check_float('threshold', model_dict[n]['threshold'], min_value=0.0, max_value=1.0)
            model_dict[n]['minimum_bout_length'] = config.getfloat('Minimum_bout_lengths', 'min_bout_' + str(n+1))
            check_int('minimum_bout_length', model_dict[n]['minimum_bout_length'])

        except ValueError:
            print('SIMBA WARNING: Skipping {} classifier analysis: missing information (e.g., no discrimination threshold and/or minimum bout set in the project_config.ini'.format(str(config.get('SML settings', 'target_name_' + str(n+1)))))
    if len(model_dict.keys()) == 0:
        raise ValueError('There are no models with accurate data specified in the RUN MODELS menu. Speficy the model information to SimBA RUN MODELS menu to use them to analyze videos')
    else:
        return model_dict

def get_all_clf_names(config=None, target_cnt=None):
    model_names = []
    for i in range(target_cnt):
        entry_name = 'target_name_{}'.format(str(i+1))
        model_names.append(read_config_entry(config, 'SML settings', entry_name, data_type='str'))
    return model_names


def insert_column_headers_for_outlier_correction(data_df=None, new_headers=None, filepath=None):
    if len(new_headers) != len(data_df.columns):
        print('SIMBA ERROR: SimBA expects {} columns of data inside the files within project_folder/csv/input_csv directory. However, '
                         'within file {} file, SimBA found {} columns.'.format(str(len(new_headers)), filepath, str(len(data_df.columns))))
        raise ValueError
    else:
        data_df.columns = new_headers
        return data_df







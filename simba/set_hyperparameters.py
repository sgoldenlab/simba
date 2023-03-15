from simba.read_config_unit_tests import (read_config_file,
                                          read_config_entry,
                                          check_int,
                                          check_float,
                                          check_file_exist_and_readable,
                                          check_str)
from simba.tkinter_functions import (Entry_Box,
                                     FileSelect)
from simba.tkinter_functions import (hxtScrollbar,
                                     DropDownMenu)
from tkinter import *
import os, glob
import pandas as pd
import webbrowser
from simba.train_model_functions import get_all_clf_names
import trafaret as t


def check_int_or_acceptable_string(name: str,
                                   string_options: list = None,
                                   value=None):
    try:
        t.Int().check(value)
    except t.DataError as e:
        if value in string_options:
            pass
        else:
            print(
                'SIMBA ERROR: {} is set to {} which not an acceptable integer or acceptable string OPTIONS: {}'.format(
                    name, value, string_options))
            raise ValueError(
                'SIMBA ERROR: {} is set to {} which not an acceptable integer or acceptable string OPTIONS: {}'.format(
                    name, value, string_options))


class SetHyperparameterPopUp(object):
    def __init__(self,
                 config_path: str):

        self.get_expected_meta_dict_entry_keys()
        self.config_path, self.config = config_path, read_config_file(ini_path=config_path)
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.model_cnt = read_config_entry(config=self.config,section='SML settings', option='No_targets', data_type='int')
        self.clf_names = list(get_all_clf_names(config=self.config, target_cnt=self.model_cnt))
        self.main_frm = Toplevel()
        self.main_frm.minsize(400, 400)
        self.main_frm.wm_title("MACHINE MODEL SETTINGS")
        self.main_frm.lift()
        self.main_frm = Canvas(hxtScrollbar(self.main_frm))
        self.main_frm.pack(fill="both", expand=True)
        load_meta_data_frm = LabelFrame(self.main_frm, text='LOAD META DATA', font=('Helvetica', 12, 'bold'), pady=5, padx=5)
        self.selected_file = FileSelect(load_meta_data_frm, 'SELECT FILE', title='Select a meta (.csv) file', lblwidth='15')
        load_metadata_btn = Button(load_meta_data_frm, text='LOAD', command= lambda: self.load_meta_data(), fg='blue')

        link_1_lbl = Label(self.main_frm, text='[Click here to learn about the Hyperparameters]', cursor='hand2', fg='blue')
        link_1_lbl.bind('<Button-1>', lambda e: webbrowser.open_new('https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-7-train-machine-model'))

        choose_algo_frm = LabelFrame(self.main_frm, text='MACHINE MODEL',font=('Helvetica',12,'bold'), pady=5, padx=5)
        self.choose_algo_dropdown = DropDownMenu(choose_algo_frm, 'Choose algorithm:', ['RF','GBC','XGBoost'], '15')
        self.choose_algo_dropdown.setChoices('RF')

        hyperparameters_frm = LabelFrame(self.main_frm, text='HYPERPARAMETERS',font=('Helvetica',12,'bold'),pady=5,padx=5)
        self.n_estimators_entrybox = Entry_Box(hyperparameters_frm,'Estimators: ','25', validation='numeric')
        self.max_features_entrybox = Entry_Box(hyperparameters_frm,'Max features: ','25')
        self.criterion_entrybox = Entry_Box(hyperparameters_frm,'Criterion: ','25')
        self.test_size_entrybox = Entry_Box(hyperparameters_frm,'Test size: ','25')
        self.min_sample_leaf_entrybox = Entry_Box(hyperparameters_frm,'Min sample leaf: ','25', validation='numeric')
        self.undersample_setting_entrybox = Entry_Box(hyperparameters_frm, 'Under-sample setting', '25')
        self.undersample_ratio_entrybox = Entry_Box(hyperparameters_frm,'Under-sample ratio','25')
        self.oversample_setting_entrybox = Entry_Box(hyperparameters_frm, 'Over-sample setting', '25')
        self.oversample_ratio_entrybox = Entry_Box(hyperparameters_frm,'Over-sample ratio','25')

        self.evaluation_settings_frm = LabelFrame(self.main_frm, pady=5, padx=5, text='MODEL EVALUATION SETTINGS', font=('Helvetica', 12, 'bold'))
        self.learning_curve_k_splits_entry = Entry_Box(self.evaluation_settings_frm, 'Learning curve shuffle K splits', '25',status=DISABLED, validation='numeric')
        self.learning_curve_data_splits_entry = Entry_Box(self.evaluation_settings_frm, 'Learning curve shuffle data splits', '25',status=DISABLED, validation='numeric')
        self.n_feature_importance_bars_entry = Entry_Box(self.evaluation_settings_frm, 'Feature importance bars', '25', status=DISABLED, validation='numeric')
        self.shap_present = Entry_Box(self.evaluation_settings_frm,'# target present', '25',status=DISABLED, validation='numeric')
        self.shap_absent = Entry_Box(self.evaluation_settings_frm, '# target absent', '25', status=DISABLED, validation='numeric')

        self.create_meta_data_file_var = BooleanVar()
        create_meta_data_file_cb = Checkbutton(self.evaluation_settings_frm,text='Generate RF model meta data file',variable = self.create_meta_data_file_var)
        self.create_example_decision_tree_var = BooleanVar()
        create_example_decision_tree_cb = Checkbutton(self.evaluation_settings_frm, text='Create example decision rree (requires "graphviz")', variable=self.create_example_decision_tree_var)
        self.create_example_fancy_decision_tree_var = BooleanVar()
        create_example_fancy_decision_tree_cb = Checkbutton(self.evaluation_settings_frm, text='Create fancy example decision tree (requires "dtreeviz")', variable=self.create_example_fancy_decision_tree_var)
        self.create_clf_report_var = BooleanVar()
        create_clf_report_cb = Checkbutton(self.evaluation_settings_frm, text='Create classification report', variable=self.create_clf_report_var)
        self.create_feature_importance_log_var = BooleanVar()
        create_feature_importance_log_cb = Checkbutton(self.evaluation_settings_frm, text='Create feature importance log', variable=self.create_feature_importance_log_var)
        self.create_feature_importance_bar_graph_var = BooleanVar()
        create_feature_importance_bar_graph_cb = Checkbutton(self.evaluation_settings_frm, text='Create features importance bar graph', variable=self.create_feature_importance_bar_graph_var, command = lambda:self.activate_entry_boxes(self.create_feature_importance_bar_graph_var, self.n_feature_importance_bars_entry))
        self.compute_feature_permutation_importances_var = BooleanVar()
        compute_feature_permutation_importances_cb = Checkbutton(self.evaluation_settings_frm, text='Compute feature permutation importances (Note: CPU intensive)', variable=self.compute_feature_permutation_importances_var)
        self.compute_learning_curves_var = BooleanVar()
        compute_learning_curves_cb = Checkbutton(self.evaluation_settings_frm, text='Compute learning curves (Note: CPU intensive)', variable=self.compute_learning_curves_var, command = lambda:self.activate_entry_boxes(self.compute_learning_curves_var, self.learning_curve_data_splits_entry, self.learning_curve_k_splits_entry))
        self.compute_pr_curves_var = BooleanVar()
        compute_pr_curves_cb = Checkbutton(self.evaluation_settings_frm, text='Compute precision-recall curves', variable=self.compute_pr_curves_var)
        self.compute_shap_scores_var = BooleanVar()
        compute_shap_scores_cb = Checkbutton(self.evaluation_settings_frm,text='Calculate SHAP scores',variable=self.compute_shap_scores_var,command= lambda:self.activate_entry_boxes(self.compute_shap_scores_var,self.shap_present,self.shap_absent))

        select_model_frm = LabelFrame(self.main_frm, text='MODEL', font=('Helvetica', 12, 'bold'), pady=5, padx=5)
        self.model_name_dropdown = DropDownMenu(select_model_frm, 'Model name: ', self.clf_names, '15')
        self.model_name_dropdown.setChoices(self.clf_names[0])

        save_global_btn = Button(self.main_frm, text='SAVE SETTINGS INTO GLOBAL ENVIRONMENT', font=('Helvetica', 12, 'bold'),fg='blue', command=self.save_global_data)
        save_one_model_to_config = Button(self.main_frm, text='SAVE SETTINGS FOR SPECIFIC MODEL', font=('Helvetica', 12, 'bold'),fg='green' ,command=self.save_config_file)
        clear_cache_btn = Button(self.main_frm,text='CLEAR CACHE',font=('Helvetica', 12, 'bold'),fg='red',command = self.clear_cache)

        load_meta_data_frm.grid(row=0, sticky=W, pady=5, padx=5)
        self.selected_file.grid(row=0, column=0, sticky=W)
        load_metadata_btn.grid(row=1, column=0)
        choose_algo_frm.grid(row=1, sticky=W, pady=5)
        self.choose_algo_dropdown.grid(row=0, column=0, sticky=W)
        select_model_frm.grid(row=2, column=0, sticky=W)
        self.model_name_dropdown.grid(row=0, column=0, sticky=W)
        hyperparameters_frm.grid(row=3, sticky=W)
        self.n_estimators_entrybox.grid(row=1, sticky=W)
        self.max_features_entrybox.grid(row=2, sticky=W)
        self.criterion_entrybox.grid(row=3, sticky=W)
        self.test_size_entrybox.grid(row=4, sticky=W)
        self.min_sample_leaf_entrybox.grid(row=7, sticky=W)
        self.undersample_setting_entrybox.grid(row=8, sticky=W)
        self.undersample_ratio_entrybox.grid(row=9, sticky=W)
        self.oversample_setting_entrybox.grid(row=10, sticky=W)
        self.oversample_ratio_entrybox.grid(row=11, sticky=W)

        self.evaluation_settings_frm.grid(row=5, sticky=W)
        create_meta_data_file_cb.grid(row=0, sticky=W)
        create_example_decision_tree_cb.grid(row=1, sticky=W)
        create_example_fancy_decision_tree_cb.grid(row=2, sticky=W)
        create_clf_report_cb.grid(row=3, sticky=W)
        create_feature_importance_log_cb.grid(row=4, sticky=W)
        create_feature_importance_bar_graph_cb.grid(row=5, sticky=W)
        self.n_feature_importance_bars_entry.grid(row=6, sticky=W)
        compute_feature_permutation_importances_cb.grid(row=7, sticky=W)
        compute_learning_curves_cb.grid(row=8, sticky=W)
        self.learning_curve_k_splits_entry.grid(row=9, sticky=W)
        self.learning_curve_data_splits_entry.grid(row=10, sticky=W)
        compute_pr_curves_cb.grid(row=11, sticky=W)
        compute_shap_scores_cb.grid(row=12, sticky=W)
        self.shap_present.grid(row=13, sticky=W)
        self.shap_absent.grid(row=14, sticky=W)

        save_global_btn.grid(row=6, pady=5, sticky=NW)
        save_one_model_to_config.grid(row=7, sticky=NW)
        clear_cache_btn.grid(row=8, pady=5, sticky=NW)

    def activate_entry_boxes(self, box, *args):
        for entry in args:
            if box.get() == 0:
                entry.set_state(DISABLED)
            elif box.get() == 1:
                entry.set_state(NORMAL)

    def clear_cache(self):
        config_files = glob.glob(os.path.join(self.project_path, 'configs') + '/*.csv')
        for file_path in config_files:
            os.remove(file_path)
        print('SIMBA COMPLETE: {} cache model config file(s) deleted from {} folder'.format(str(len(config_files)), os.path.join(self.project_path, 'configs')))

    def load_meta_data(self):
        check_file_exist_and_readable(file_path=self.selected_file.file_path)
        try:
            self.meta_dict = pd.read_csv(self.selected_file.file_path, index_col=False).to_dict(orient='records')[0]
        except pd.errors.ParserError:
            print('SIMBA ERROR: {} is not a valid SimBA meta hyper-parameters file.'.format(self.selected_file.file_path))
            raise ValueError('SIMBA ERROR: {} is not a valid SimBA meta hyper-parameters file.'.format(self.selected_file.file_path))
        self.check_meta_data_integrity()
        self.populate_table()

    def save_global_data(self):
        self.global_unit_tests()
        self.config.set('create ensemble settings', 'model_to_run', self.choose_algo_dropdown.getChoices())
        self.config.set('create ensemble settings', 'RF_n_estimators', self.n_estimators_entrybox.entry_get)
        self.config.set('create ensemble settings', 'RF_max_features', self.max_features_entrybox.entry_get)
        self.config.set('create ensemble settings', 'RF_criterion', self.criterion_entrybox.entry_get)
        self.config.set('create ensemble settings', 'train_test_size', self.test_size_entrybox.entry_get)
        self.config.set('create ensemble settings', 'RF_min_sample_leaf', self.min_sample_leaf_entrybox.entry_get)
        self.config.set('create ensemble settings', 'under_sample_ratio', self.undersample_ratio_entrybox.entry_get)
        self.config.set('create ensemble settings', 'under_sample_setting',self.undersample_setting_entrybox.entry_get)
        self.config.set('create ensemble settings', 'over_sample_setting', self.oversample_setting_entrybox.entry_get)
        self.config.set('create ensemble settings', 'over_sample_ratio', self.oversample_ratio_entrybox.entry_get)
        self.config.set('create ensemble settings', 'classifier', self.model_name_dropdown.getChoices())
        self.config.set('create ensemble settings', 'RF_meta_data', str(self.create_meta_data_file_var.get()))
        self.config.set('create ensemble settings', 'generate_example_decision_tree', str(self.create_example_decision_tree_var.get()))
        self.config.set('create ensemble settings', 'generate_classification_report', str(self.create_clf_report_var.get()))
        self.config.set('create ensemble settings', 'generate_features_importance_log', str(self.create_feature_importance_log_var.get()))
        self.config.set('create ensemble settings', 'generate_features_importance_bar_graph', str(self.create_feature_importance_bar_graph_var.get()))
        self.config.set('create ensemble settings', 'N_feature_importance_bars', self.n_feature_importance_bars_entry.entry_get)
        self.config.set('create ensemble settings', 'compute_permutation_importance', str(self.compute_feature_permutation_importances_var.get()))
        self.config.set('create ensemble settings', 'generate_learning_curve', str(self.compute_learning_curves_var.get()))
        self.config.set('create ensemble settings', 'generate_precision_recall_curve', str(self.compute_pr_curves_var.get()))
        self.config.set('create ensemble settings', 'LearningCurve_shuffle_k_splits', self.learning_curve_k_splits_entry.entry_get)
        self.config.set('create ensemble settings', 'LearningCurve_shuffle_data_splits', self.learning_curve_data_splits_entry.entry_get)
        self.config.set('create ensemble settings', 'generate_example_decision_tree_fancy', str(self.create_example_fancy_decision_tree_var.get()))
        self.config.set('create ensemble settings', 'generate_shap_scores', str(self.compute_shap_scores_var.get()))
        self.config.set('create ensemble settings', 'shap_target_present_no', self.shap_present.entry_get)
        self.config.set('create ensemble settings', 'shap_target_absent_no',  self.shap_absent.entry_get)

        with open(self.config_path, 'w') as configfile:
            self.config.write(configfile)
        print('SIMBA COMPLETE: Model settings saved to global environment (project_config.ini)')

    def global_unit_tests(self):
        check_int(name='RF N Estimators', value=self.n_estimators_entrybox.entry_get)
        check_int_or_acceptable_string(name='RF Max features', value=self.max_features_entrybox.entry_get, string_options=['sqrt', 'log2'])
        check_str(name='RF Criterion', value=self.criterion_entrybox.entry_get, options=("gini", "entropy", "log_loss"))
        check_float(name='Train Test Size', value=self.test_size_entrybox.entry_get, max_value=1.0, min_value=0.00)
        check_int(name='RF Min sample leaf', value=self.min_sample_leaf_entrybox.entry_get, min_value=1)
        check_str(name='Under sample setting', value=self.undersample_setting_entrybox.entry_get.lower().strip(), options=('random undersample', '', 'none', 'nan'))
        if self.undersample_setting_entrybox.entry_get.lower().strip() == 'random undersample':
            check_float(name='Under sample ratio', value=self.undersample_ratio_entrybox.entry_get)
            if (self.undersample_ratio_entrybox.entry_get.lower().strip() == 'nan') or (self.undersample_ratio_entrybox.entry_get.lower().strip() == 'none'):
                print('SIMBA ERROR: Under sample setting is {}, but Under sample ratio is {}. If using undersampling, provide a Under sample ratio as a numeric value'.format(self.undersample_setting_entrybox.entry_get, self.undersample_ratio_entrybox.entry_get))
                raise ValueError('SIMBA ERROR: Under sample setting is {} but Under sample ratio is {}. If using undersampling, provide a Under sample ratio as a numeric value'.format(self.undersample_setting_entrybox.entry_get, self.undersample_ratio_entrybox.entry_get))
        check_str(name='Over sample setting', value=self.oversample_setting_entrybox.entry_get.lower().lower().strip(), options=('smote', 'smoteenn', '', 'none', 'nan'))
        if (self.oversample_setting_entrybox.entry_get.lower().lower().strip() == 'smote') or (self.oversample_setting_entrybox.entry_get.lower().lower().strip() == 'smoteenn'):
            check_float(name='Over sample ratio', value=self.undersample_ratio_entrybox.entry_get)
            if (self.oversample_ratio_entrybox.entry_get.lower().strip() == 'nan') or (self.oversample_ratio_entrybox.entry_get.lower().strip() == 'none'):
                print('SIMBA ERROR: Over sample setting is {} but Over sample ratio is {}. If using oversampling, provide a Over sample ratio as a numeric value'.format(self.oversample_setting_entrybox.entry_get, self.oversample_ratio_entrybox.entry_get))
                raise ValueError('SIMBA ERROR: Over sample setting is {} but Over sample ratio is {}. If using oversampling, provide a Over sample ratio as a numeric value'.format(self.oversample_setting_entrybox.entry_get, self.oversample_ratio_entrybox.entry_get))
        if self.compute_learning_curves_var.get():
            check_int(name='LearningCurve shuffle K splits', value=self.learning_curve_k_splits_entry.entry_get)
            check_int(name='LearningCurve shuffle Data splits', value=self.learning_curve_data_splits_entry.entry_get)
        if self.compute_shap_scores_var.get():
            check_int(name='SHAP # target present', value=self.shap_present.entry_get)
            check_int(name='SHAP # target absent', value=self.shap_absent.entry_get)
        if self.create_feature_importance_bar_graph_var.get():
            check_int(name='N feature importance bars', value=self.n_feature_importance_bars_entry.entry_get, min_value=1)

    def save_config_file(self):
        self.global_unit_tests()
        existing_file_cnt = 0
        for f in os.listdir(os.path.join(os.path.dirname(self.config_path), 'configs')):
            if f.__contains__('_meta') and f.__contains__(self.model_name_dropdown.getChoices()): existing_file_cnt += 1

        meta_dict = {'Classifier_name': self.model_name_dropdown.getChoices(),
                     'RF_n_estimators': self.n_estimators_entrybox.entry_get,
                     'RF_max_features': self.max_features_entrybox.entry_get,
                     'RF_criterion': self.criterion_entrybox.entry_get.lower().strip(),
                     'train_test_size': self.test_size_entrybox.entry_get,
                     'RF_min_sample_leaf': self.min_sample_leaf_entrybox.entry_get,
                     'under_sample_ratio': self.undersample_ratio_entrybox.entry_get,
                     'under_sample_setting': self.undersample_setting_entrybox.entry_get.lower().strip(),
                     'over_sample_ratio': self.undersample_ratio_entrybox.entry_get,
                     'over_sample_setting': self.undersample_ratio_entrybox.entry_get.lower().strip(),
                     'generate_rf_model_meta_data_file': self.create_meta_data_file_var.get(),
                     'generate_example_decision_tree': self.create_example_decision_tree_var.get(),
                     'generate_classification_report': self.create_clf_report_var.get(),
                     'generate_features_importance_log': self.create_feature_importance_log_var.get(),
                     'generate_features_importance_bar_graph': self.create_feature_importance_bar_graph_var.get(),
                     'n_feature_importance_bars': self.create_feature_importance_bar_graph_var.get(),
                     'compute_feature_permutation_importance': self.compute_feature_permutation_importances_var.get(),
                     'generate_sklearn_learning_curves': self.compute_learning_curves_var.get(),
                     'generate_precision_recall_curves': self.compute_pr_curves_var.get(),
                     'learning_curve_k_splits': self.learning_curve_k_splits_entry.entry_get,
                     'learning_curve_data_splits': self.learning_curve_data_splits_entry.entry_get,
                     'generate_shap_scores': self.compute_shap_scores_var.get(),
                     'shap_target_present_no': self.shap_present.entry_get,
                     'shap_target_absetn_no': self.shap_absent.entry_get}
        meta_df = pd.DataFrame(meta_dict, index=[0])
        save_name = os.path.join(os.path.dirname(self.config_path), 'configs', '{}_meta_{}.csv'.format(meta_dict['Classifier_name'], str(existing_file_cnt)))
        meta_df.to_csv(save_name, index=FALSE)
        print('SIMBA COMPLETE: Hyper-parameter config saved: {}'.format(os.path.basename(save_name)))

    def check_meta_data_integrity(self):
        self.meta_dict = {k.lower(): v for k, v in self.meta_dict.items()}
        for i in self.expected_meta_dict_entries:
            if i not in self.meta_dict.keys():
                print('SIMBA WARNING: The file does not contain an expected entry for {} parameter'.format(i))
                self.meta_dict[i] = None
            else:
                if type(self.meta_dict[i]) == str:
                    if self.meta_dict[i].lower().strip() == 'yes':
                        self.meta_dict[i] = True

    def populate_table(self):
        self.n_estimators_entrybox.entry_set(val=self.meta_dict['rf_n_estimators'])
        self.max_features_entrybox.entry_set(val=self.meta_dict['rf_max_features'])
        self.criterion_entrybox.entry_set(val=self.meta_dict['rf_criterion'])
        self.test_size_entrybox.entry_set(val=self.meta_dict['train_test_size'])
        self.min_sample_leaf_entrybox.entry_set(val=self.meta_dict['rf_min_sample_leaf'])
        self.undersample_ratio_entrybox.entry_set(val=self.meta_dict['under_sample_ratio'])
        self.undersample_setting_entrybox.entry_set(val=self.meta_dict['under_sample_setting'])
        self.oversample_setting_entrybox.entry_set(val=self.meta_dict['over_sample_setting'])
        self.oversample_ratio_entrybox.entry_set(val=self.meta_dict['over_sample_ratio'])
        self.model_name_dropdown.setChoices(choice=self.meta_dict['classifier_name'])
        self.create_meta_data_file_var.set(value=self.meta_dict['generate_rf_model_meta_data_file'])
        self.create_example_decision_tree_var.set(value=self.meta_dict['generate_example_decision_tree'])
        self.create_clf_report_var.set(value=self.meta_dict['generate_classification_report'])
        self.create_feature_importance_log_var.set(value=self.meta_dict['generate_features_importance_log'])
        self.create_feature_importance_bar_graph_var.set(value=self.meta_dict['generate_features_importance_bar_graph'])
        if self.create_feature_importance_bar_graph_var.get():
            self.activate_entry_boxes(self.create_feature_importance_bar_graph_var, self.n_feature_importance_bars_entry)
            self.n_feature_importance_bars_entry.entry_set(val=self.meta_dict['n_feature_importance_bars'])
        self.compute_feature_permutation_importances_var.set(value=self.meta_dict['compute_feature_permutation_importance'])
        self.compute_pr_curves_var.set(value=self.meta_dict['generate_precision_recall_curves'])
        self.compute_learning_curves_var.set(value=self.meta_dict['generate_sklearn_learning_curves'])
        if self.compute_learning_curves_var.get():
            self.activate_entry_boxes(self.compute_learning_curves_var, self.learning_curve_k_splits_entry)
            self.activate_entry_boxes(self.compute_learning_curves_var, self.learning_curve_data_splits_entry)
        self.learning_curve_k_splits_entry.entry_set(val=self.meta_dict['learning_curve_k_splits'])
        self.learning_curve_data_splits_entry.entry_set(val=self.meta_dict['learning_curve_data_splits'])
        try:
            self.create_example_fancy_decision_tree_var.set(value=self.meta_dict['generate_example_decision_tree_fancy'])
            self.compute_shap_scores_var.set(value=self.meta_dict['generate_shap_scores'])
            if self.compute_shap_scores_var.get():
                self.activate_entry_boxes(self.compute_shap_scores_var, self.shap_present)
                self.activate_entry_boxes(self.compute_shap_scores_var, self.shap_absent)
            self.shap_present.entry_set(val=self.meta_dict['shap_target_present_no'])
            self.shap_absent.entry_set(val=self.meta_dict['shap_target_absent_no'])
        except:
            pass
        self.global_unit_tests()

    def get_expected_meta_dict_entry_keys(self):
        self.expected_meta_dict_entries = ['classifier_name',
                                           'rf_n_estimators',
                                           'rf_max_features',
                                           'rf_criterion',
                                           'train_test_size',
                                           'rf_min_sample_leaf',
                                           'under_sample_ratio',
                                           'under_sample_setting',
                                           'over_sample_ratio',
                                           'over_sample_setting',
                                           'generate_rf_model_meta_data_file',
                                           'generate_example_decision_tree',
                                           'generate_classification_report',
                                           'generate_features_importance_log'
                                           'generate_features_importance_bar_graph',
                                           'n_feature_importance_bars',
                                           'compute_feature_permutation_importance',
                                           'generate_sklearn_learning_curves',
                                           'generate_precision_recall_curves',
                                           'learning_curve_k_splits',
                                           'learning_curve_data_splits',
                                           'generate_example_decision_tree_fancy',
                                           'generate_shap_scores',
                                           'shap_target_present_no',
                                           'shap_target_absent_no']


# test = SetHyperparameterPopUp(config_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini')
# test.main_frm.mainloop()

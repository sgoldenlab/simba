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
from simba.enums import (ReadConfig,
                         Formats)
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


test = SetHyperparameterPopUp(config_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini')
test.main_frm.mainloop()

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# class trainmachinemodel_settings:
#     def __init__(self,inifile):
#         self.configini = str(inifile)
#         # Popup window
#         trainmmsettings = Toplevel()
#         trainmmsettings.minsize(400, 400)
#         trainmmsettings.wm_title("Machine model settings")
#
#         trainmms = Canvas(hxtScrollbar(trainmmsettings))
#         trainmms.pack(expand=True,fill=BOTH)
#
#         #load metadata
#         load_data_frame = LabelFrame(trainmms, text='Load Metadata',font=('Helvetica',10,'bold'), pady=5, padx=5)
#         self.load_choosedata = FileSelect(load_data_frame,'File Select',title='Select a meta (.csv) file')
#         load_data = Button(load_data_frame, text = 'Load', command = self.load_RFvalues,fg='blue')
#
#         #link to github
#         label_git_hyperparameters = Label(trainmms,text='[Click here to learn about the Hyperparameters]',cursor='hand2',fg='blue')
#         label_git_hyperparameters.bind('<Button-1>',lambda e: webbrowser.open_new('https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-7-train-machine-model'))
#
#         #setting drop downs
#         label_mm = LabelFrame(trainmms, text='MACHINE MODEL',font=('Helvetica',10,'bold'), pady=5, padx=5)
#         label_choosemm = Label(label_mm, text='Choose machine model algorithm:')
#         options =['RF','GBC','Xboost']
#
#         self.var = StringVar()
#         self.var.set(options[0]) #set as default value
#
#         modeldropdown = OptionMenu(label_mm,self.var,*options)
#
#         self.meta_dict = {}
#         ## hyperparameter settings
#         label_settings = LabelFrame(trainmms, text='Hyperparameters',font=('Helvetica',10,'bold'),pady=5,padx=5)
#         self.settings = []
#         self.label_nestimators = Entry_Box(label_settings,'RF N Estimators','25')
#         self.label_maxfeatures = Entry_Box(label_settings,'RF Max features','25')
#         self.label_criterion = Entry_Box(label_settings,'RF Criterion','25')
#         self.label_testsize = Entry_Box(label_settings,'Train Test Size','25')
#         self.label_minsampleleaf = Entry_Box(label_settings,'RF Min sample leaf','25')
#         self.label_under_s_settings = Entry_Box(label_settings, 'Under sample setting', '25')
#         self.label_under_s_correctionvalue = Entry_Box(label_settings,'Under sample ratio','25')
#         self.label_over_s_settings = Entry_Box(label_settings, 'Over sample setting', '25')
#         self.label_over_s_ratio = Entry_Box(label_settings,'Over sample ratio','25')
#
#         self.settings = [self.label_nestimators, self.label_maxfeatures, self.label_criterion, self.label_testsize,
#                     self.label_minsampleleaf, self.label_under_s_correctionvalue,self.label_under_s_settings,
#                          self.label_over_s_ratio,self.label_over_s_settings]
#         ## model evaluation settings for checkboxes
#         self.label_settings_box = LabelFrame(trainmms,pady=5,padx=5,text='Model Evaluations Settings',font=('Helvetica',10,'bold'))
#         self.box1 = IntVar()
#         self.box2 = IntVar()
#         self.box3 = IntVar()
#         self.box4 = IntVar()
#         self.box5 = IntVar()
#         self.box6 = IntVar()
#         self.box7 = IntVar()
#         self.box8 = IntVar()
#         self.box9 = IntVar()
#         self.box10 = IntVar()
#
#         # model evaluations for entrybox
#         self.LC_ksplit = Entry_Box(self.label_settings_box, 'LearningCurve shuffle K splits', '25',status=DISABLED)
#         self.LC_datasplit = Entry_Box(self.label_settings_box, 'LearningCurve shuffle Data splits', '25',status=DISABLED)
#         self.label_n_feature_importance_bars = Entry_Box(self.label_settings_box, 'N feature importance bars', '25',status=DISABLED)
#         self.shap_present = Entry_Box(self.label_settings_box,'# target present', '25',status=DISABLED)
#         self.shap_absent = Entry_Box(self.label_settings_box, '# target absent', '25', status=DISABLED)
#         self.settings.extend([self.LC_ksplit, self.LC_datasplit, self.label_n_feature_importance_bars,self.shap_present,self.shap_absent])
#
#         def activate(box, *args):
#             for entry in args:
#                 if box.get() == 0:
#                     entry.set_state(DISABLED)
#                 elif box.get() == 1:
#                     entry.set_state(NORMAL)
#
#         checkbutton1 = Checkbutton(self.label_settings_box,text='Generate RF model meta data file',variable = self.box1)
#         checkbutton2 = Checkbutton(self.label_settings_box, text='Generate Example Decision Tree (requires "graphviz")', variable=self.box2)
#         checkbutton3 = Checkbutton(self.label_settings_box, text='Generate Fancy Example Decision Tree ("dtreeviz")', variable=self.box3)
#         checkbutton4 = Checkbutton(self.label_settings_box, text='Generate Classification Report', variable=self.box4)
#         # checkbutton5 = Checkbutton(self.label_settings_box, text='Generate Features Importance Log', variable=self.box5)
#         checkbutton6 = Checkbutton(self.label_settings_box, text='Generate Features Importance Bar Graph', variable=self.box6,
#                                    command = lambda:activate(self.box6, self.label_n_feature_importance_bars))
#         checkbutton7 = Checkbutton(self.label_settings_box, text='Compute Feature Permutation Importances (Note: CPU intensive)', variable=self.box7)
#         checkbutton8 = Checkbutton(self.label_settings_box, text='Generate Sklearn Learning Curves (Note: CPU intensive)', variable=self.box8,
#                                    command = lambda:activate(self.box8, self.LC_datasplit, self.LC_ksplit))
#         checkbutton9 = Checkbutton(self.label_settings_box, text='Generate Precision Recall Curves', variable=self.box9)
#         checkbutton10 = Checkbutton(self.label_settings_box,text='Calculate SHAP scores',variable=self.box10,command= lambda:activate(self.box10,self.shap_present,self.shap_absent))
#
#         self.check_settings = [checkbutton1, checkbutton2, checkbutton3, checkbutton4, checkbutton6,
#                                checkbutton7, checkbutton8, checkbutton9, checkbutton10]
#
#
#         # setting drop downs for modelname
#         configini = self.configini
#         config = ConfigParser()
#         config.read(configini)
#
#         number_of_model = config['SML settings'].getint('No_targets')
#
#         model_list = []
#         count = 1
#         for i in range(number_of_model):
#             a = str('target_name_' + str(count))
#             model_list.append(config['SML settings'].get(a))
#             count += 1
#
#         labelf_modelname = LabelFrame(trainmms,text='Model',font=('Helvetica',10,'bold'),pady=5,padx=5)
#         label_modelname = Label(labelf_modelname,text='Model name')
#
#         self.varmodel = StringVar()
#         self.varmodel.set(model_list[0])  # set as default value
#
#         model_name_dropdown = OptionMenu(labelf_modelname, self.varmodel, *model_list)
#
#         # button
#         button_settings_to_ini = Button(trainmms, text='Save settings into global environment', font=('Helvetica', 10, 'bold'),fg='blue', command=self.set_values)
#         button_save_meta = Button(trainmms, text='Save settings for specific model', font=('Helvetica', 10, 'bold'),fg='green' ,command=self.save_new)
#         button_remove_meta = Button(trainmms,text='Clear cache',font=('Helvetica', 10, 'bold'),fg='red',command = self.clearcache)
#
#         # organize
#         load_data_frame.grid(row=0, sticky=W, pady=5, padx=5)
#         self.load_choosedata.grid(row=0, column=0, sticky=W)
#         load_data.grid(row=1, column=0)
#
#         label_mm.grid(row=1, sticky=W, pady=5)
#         label_choosemm.grid(row=0, column=0, sticky=W)
#         modeldropdown.grid(row=0, column=1, sticky=W)
#
#         labelf_modelname.grid(row=2, sticky=W, pady=5)
#         label_modelname.grid(row=0, column=0, sticky=W)
#         model_name_dropdown.grid(row=0, column=1, sticky=W)
#
#         label_git_hyperparameters.grid(row=3,sticky=W)
#
#         label_settings.grid(row=4, sticky=W, pady=5)
#         self.label_nestimators.grid(row=1, sticky=W)
#         self.label_maxfeatures.grid(row=2, sticky=W)
#         self.label_criterion.grid(row=3, sticky=W)
#         self.label_testsize.grid(row=4, sticky=W)
#         self.label_minsampleleaf.grid(row=7, sticky=W)
#         self.label_under_s_settings.grid(row=8, sticky=W)
#         self.label_under_s_correctionvalue.grid(row=9, sticky=W)
#         self.label_over_s_settings.grid(row=10, sticky=W)
#         self.label_over_s_ratio.grid(row=11,sticky=W)
#
#         self.label_settings_box.grid(row=5,sticky=W)
#         checkbutton1.grid(row=0,sticky=W)
#         checkbutton2.grid(row=1,sticky=W)
#         checkbutton3.grid(row=2,sticky=W)
#         checkbutton4.grid(row=3,sticky=W)
#         # checkbutton5.grid(row=4,sticky=W)
#         checkbutton6.grid(row=5,sticky=W)
#         self.label_n_feature_importance_bars.grid(row=6, sticky=W)
#         checkbutton7.grid(row=7,sticky=W)
#         checkbutton8.grid(row=8, sticky=W)
#         self.LC_ksplit.grid(row=9, sticky=W)
#         self.LC_datasplit.grid(row=10, sticky=W)
#         checkbutton9.grid(row=11, sticky=W)
#         checkbutton10.grid(row=12,sticky=W)
#         self.shap_present.grid(row=13,sticky=W)
#         self.shap_absent.grid(row=14,sticky=W)
#
#         button_settings_to_ini.grid(row=6,pady=5)
#         button_save_meta.grid(row=7)
#         button_remove_meta.grid(row=8,pady=5)
#
#
#     def clearcache(self):
#         configs_dir = os.path.join(os.path.dirname(self.configini),'configs')
#         filelist = [f for f in os.listdir(configs_dir) if f.endswith('.csv')]
#         for f in filelist:
#             os.remove(os.path.join(configs_dir,f))
#             print(f,'deleted')
#
#     def load_RFvalues(self):
#
#         metadata = pd.read_csv(str(self.load_choosedata.file_path), index_col=False)
#         # metadata = metadata.drop(['Feature_list'], axis=1)
#         for m in metadata.columns:
#             self.meta_dict[m] = metadata[m][0]
#         print('Meta data file loaded')
#
#         for key in self.meta_dict:
#             cur_list = key.lower().split(sep='_')
#             # print(cur_list)
#             for i in self.settings:
#                 string = i.lblName.cget('text').lower()
#                 if all(map(lambda w: w in string, cur_list)):
#                     i.entry_set(self.meta_dict[key])
#             for k in self.check_settings:
#                 string = k.cget('text').lower()
#                 if all(map(lambda w: w in string, cur_list)):
#                     if self.meta_dict[key] == 'yes':
#                         k.select()
#                     elif self.meta_dict[key] == 'no':
#                         k.deselect()
#
#     def get_checkbox(self):
#         ### check box settings
#         if self.box1.get() == 1:
#             self.rfmetadata = 'yes'
#         else:
#             self.rfmetadata = 'no'
#
#         if self.box2.get() == 1:
#             self.generate_example_d_tree = 'yes'
#         else:
#             self.generate_example_d_tree = 'no'
#
#         if self.box3.get() == 1:
#             self.generate_example_decision_tree_fancy = 'yes'
#         else:
#             self.generate_example_decision_tree_fancy  = 'no'
#
#         if self.box4.get() == 1:
#             self.generate_classification_report = 'yes'
#         else:
#             self.generate_classification_report = 'no'
#
#         if self.box5.get() == 1:
#             self.generate_features_imp_log = 'yes'
#         else:
#             self.generate_features_imp_log = 'no'
#
#         if self.box6.get() == 1:
#             self.generate_features_bar_graph = 'yes'
#         else:
#             self.generate_features_bar_graph = 'no'
#         self.n_importance = self.label_n_feature_importance_bars.entry_get
#
#         if self.box7.get() == 1:
#             self.compute_permutation_imp = 'yes'
#         else:
#             self.compute_permutation_imp = 'no'
#
#         if self.box8.get() == 1:
#             self.generate_learning_c = 'yes'
#         else:
#             self.generate_learning_c = 'no'
#         self.learningcurveksplit = self.LC_ksplit.entry_get
#         self.learningcurvedatasplit = self.LC_datasplit.entry_get
#
#         if self.box9.get() == 1:
#             self.generate_precision_recall_c = 'yes'
#         else:
#             self.generate_precision_recall_c = 'no'
#
#         if self.box10.get() == 1:
#             self.getshapscores = 'yes'
#         else:
#             self.getshapscores = 'no'
#
#         self.shappresent = self.shap_present.entry_get
#         self.shapabsent = self.shap_absent.entry_get
#
#
#     def save_new(self):
#         self.get_checkbox()
#         meta_number = 0
#         for f in os.listdir(os.path.join(os.path.dirname(self.configini), 'configs')):
#             if f.__contains__('_meta') and f.__contains__(str(self.varmodel.get())):
#                 meta_number += 1
#
#         # for s in self.settings:
#         #     meta_df[s.lblName.cget('text')] = [s.entry_get]
#         new_meta_dict = {'RF_n_estimators': self.label_nestimators.entry_get,
#                          'RF_max_features': self.label_maxfeatures.entry_get, 'RF_criterion': self.label_criterion.entry_get,
#                          'train_test_size': self.label_testsize.entry_get, 'RF_min_sample_leaf': self.label_minsampleleaf.entry_get,
#                          'under_sample_ratio': self.label_under_s_correctionvalue.entry_get, 'under_sample_setting': self.label_under_s_settings.entry_get,
#                          'over_sample_ratio': self.label_over_s_ratio.entry_get, 'over_sample_setting': self.label_over_s_settings.entry_get,
#                          'generate_rf_model_meta_data_file': self.rfmetadata,
#                          'generate_example_decision_tree': self.generate_example_d_tree,'generate_classification_report':self.generate_classification_report,
#                          'generate_features_importance_log': self.generate_features_imp_log,'generate_features_importance_bar_graph':self.generate_features_bar_graph,
#                          'n_feature_importance_bars': self.n_importance,'compute_feature_permutation_importance':self.compute_permutation_imp,
#                          'generate_sklearn_learning_curves': self.generate_learning_c,
#                          'generate_precision_recall_curves':self.generate_precision_recall_c, 'learning_curve_k_splits':self.learningcurveksplit,
#                          'learning_curve_data_splits': self.learningcurvedatasplit,
#                          'generate_shap_scores':self.getshapscores,
#                          'shap_target_present_no':self.shappresent,
#                          'shap_target_absetn_no':self.shapabsent}
#         meta_df = pd.DataFrame(new_meta_dict, index=[0])
#         meta_df.insert(0, 'Classifier_name', str(self.varmodel.get()))
#
#         if currentPlatform == 'Windows':
#             output_path = os.path.dirname(self.configini) + "\\configs\\" + \
#                         str(self.varmodel.get())+ '_meta_' + str(meta_number) + '.csv'
#
#         if currentPlatform == 'Linux'or (currentPlatform == 'Darwin'):
#             output_path = os.path.dirname(self.configini) + "/configs/" + \
#                         str(self.varmodel.get())+ '_meta_' + str(meta_number) + '.csv'
#
#
#         print(os.path.basename(str(output_path)),'saved')
#
#         meta_df.to_csv(output_path, index=FALSE)
#
#     def set_values(self):
#         self.get_checkbox()
#         #### settings
#         model = self.var.get()
#         n_estimators = self.label_nestimators.entry_get
#         max_features = self.label_maxfeatures.entry_get
#         criterion = self.label_criterion.entry_get
#         test_size = self.label_testsize.entry_get
#         min_sample_leaf = self.label_minsampleleaf.entry_get
#         under_s_c_v = self.label_under_s_correctionvalue.entry_get
#         under_s_settings = self.label_under_s_settings.entry_get
#         over_s_ratio = self.label_over_s_ratio.entry_get
#         over_s_settings = self.label_over_s_settings.entry_get
#         classifier_settings = self.varmodel.get()
#
#
#         #export settings to config ini file
#         configini = self.configini
#         config = ConfigParser()
#         config.read(configini)
#
#         config.set('create ensemble settings', 'model_to_run', str(model))
#         config.set('create ensemble settings', 'RF_n_estimators', str(n_estimators))
#         config.set('create ensemble settings', 'RF_max_features', str(max_features))
#         config.set('create ensemble settings', 'RF_criterion', str(criterion))
#         config.set('create ensemble settings', 'train_test_size', str(test_size))
#         config.set('create ensemble settings', 'RF_min_sample_leaf', str(min_sample_leaf))
#         config.set('create ensemble settings', 'under_sample_ratio', str(under_s_c_v))
#         config.set('create ensemble settings', 'under_sample_setting', str(under_s_settings))
#         config.set('create ensemble settings', 'over_sample_ratio', str(over_s_ratio))
#         config.set('create ensemble settings', 'over_sample_setting', str(over_s_settings))
#         config.set('create ensemble settings', 'classifier',str(classifier_settings))
#         config.set('create ensemble settings', 'RF_meta_data', str(self.rfmetadata))
#         config.set('create ensemble settings', 'generate_example_decision_tree', str(self.generate_example_d_tree))
#         config.set('create ensemble settings', 'generate_classification_report', str(self.generate_classification_report))
#         config.set('create ensemble settings', 'generate_features_importance_log', str(self.generate_features_imp_log))
#         config.set('create ensemble settings', 'generate_features_importance_bar_graph', str(self.generate_features_bar_graph))
#         config.set('create ensemble settings', 'N_feature_importance_bars', str(self.n_importance))
#         config.set('create ensemble settings', 'compute_permutation_importance', str(self.compute_permutation_imp))
#         config.set('create ensemble settings', 'generate_learning_curve', str(self.generate_learning_c))
#         config.set('create ensemble settings', 'generate_precision_recall_curve', str(self.generate_precision_recall_c))
#         config.set('create ensemble settings', 'LearningCurve_shuffle_k_splits',str(self.learningcurveksplit))
#         config.set('create ensemble settings', 'LearningCurve_shuffle_data_splits',str(self.learningcurvedatasplit))
#         config.set('create ensemble settings', 'generate_example_decision_tree_fancy',str(self.generate_example_decision_tree_fancy))
#         config.set('create ensemble settings', 'generate_shap_scores',str(self.getshapscores))
#         config.set('create ensemble settings', 'shap_target_present_no', str(self.shappresent))
#         config.set('create ensemble settings', 'shap_target_absent_no', str(self.shapabsent))
#
#         with open(configini, 'w') as configfile:
#             config.write(configfile)
#
#         print('Settings exported to project_config.ini')
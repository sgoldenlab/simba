__author__ = "Simon Nilsson"

import ast
import os
import webbrowser
from tkinter import *
from typing import Union

import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        Entry_Box, FileSelect, SimbaButton,
                                        SimbaCheckbox, SimBADropDown,
                                        SimBALabel, hxtScrollbar)
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_int)
from simba.utils.enums import (ConfigKey, Dtypes, Formats, Keys, Links,
                               MLParamKeys, Options)
from simba.utils.errors import InvalidHyperparametersFileError
from simba.utils.printing import stdout_success, stdout_trash, stdout_warning
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_fn_ext, str_2_bool)


class MachineModelSettingsPopUp(PopUpMixin, ConfigReader):
    """
    GUI window for specifying ML model training parameters.
    """

    def __init__(self, config_path: Union[str, os.PathLike]):
        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        PopUpMixin.__init__(self, title="MACHINE MODEL SETTINGS", size=(50, 800), icon='equation_small')
        if not os.path.exists(self.configs_meta_dir): os.makedirs(self.configs_meta_dir)

        self.clf_options = Options.CLF_MODELS.value
        self.max_features_options = Options.CLF_MAX_FEATURES.value
        self.criterion_options = Options.CLF_CRITERION.value
        self.under_sample_options = Options.UNDERSAMPLE_OPTIONS.value
        self.over_sample_options = Options.OVERSAMPLE_OPTIONS.value
        self.class_weighing_options = Options.CLASS_WEIGHT_OPTIONS.value
        self.train_test_sizes_options = Options.CLF_TEST_SIZE_OPTIONS.value
        self.class_weights_options = list(range(1, 11, 1))
        load_meta_data_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="LOAD META-DATA",  icon_name=Keys.DOCUMENTATION.value, icon_link=Links.TRAIN_ML_MODEL.value,)
        self.select_config_file = FileSelect(load_meta_data_frm, "CONFIG PATH:")
        load_config_btn = SimbaButton(parent=load_meta_data_frm, txt="LOAD", img='load_blue',  txt_clr='blue', font=Formats.FONT_REGULAR.value, cmd=self.load_config)
        label_link = SimBALabel(parent=load_meta_data_frm, txt="[MODEL SETTINGS TUTORIAL]", txt_clr="blue", link=Links.TRAIN_ML_MODEL.value, cursor='hand2')

        machine_model_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="MACHINE MODEL ALGORITHM", icon_name='equation_small')
        self.machine_model_dropdown = SimBADropDown(parent=machine_model_frm, dropdown_options=[self.clf_options[0]], label="ALGORITHM:", label_width=30, dropdown_width=25, value=self.clf_options[0])

        behavior_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="BEHAVIOR", icon_name='decision_tree_small')
        self.behavior_name_dropdown = SimBADropDown(parent=behavior_frm, dropdown_options=self.clf_names, label="BEHAVIOR:", label_width=30, dropdown_width=25, value=self.clf_names[0])

        self.hyperparameters_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="HYPER-PARAMETERS", icon_name='settings')
        self.estimators_entrybox = Entry_Box(self.hyperparameters_frm,"RANDOM FOREST ESTIMATORS:", "30", validation="numeric", value="2000", entry_box_width=25, justify='center')
        self.max_features_dropdown = SimBADropDown(parent=self.hyperparameters_frm, dropdown_options=self.max_features_options, label="MAX FEATURES:", label_width=30, dropdown_width=25, value=self.max_features_options[0])

        self.criterion_dropdown = SimBADropDown(parent=self.hyperparameters_frm, dropdown_options=self.criterion_options, label="CRITERION:", label_width=30, dropdown_width=25, value=self.criterion_options[0])
        self.train_test_size_dropdown = SimBADropDown(parent=self.hyperparameters_frm, dropdown_options=self.train_test_sizes_options, label="TEST SIZE:", label_width=30, dropdown_width=25, value=0.2)
        self.train_test_type_dropdown = SimBADropDown(parent=self.hyperparameters_frm, dropdown_options=Options.TRAIN_TEST_SPLIT.value, label="TRAIN-TEST SPLIT TYPE:", label_width=30, dropdown_width=25, value=Options.TRAIN_TEST_SPLIT.value[0])
        self.min_sample_leaf_eb = Entry_Box(self.hyperparameters_frm, "MINIMUM SAMPLE LEAF:", "30", validation="numeric", value=1, entry_box_width=25, justify='center')
        self.under_sample_ratio_entrybox = Entry_Box(self.hyperparameters_frm, "UNDER SAMPLE RATIO:", "30", value=1.0, entry_box_width=25, justify='center', status=DISABLED)
        self.undersample_settings_dropdown = SimBADropDown(parent=self.hyperparameters_frm, dropdown_options=self.under_sample_options, label="UNDER SAMPLE SETTING:", label_width=30, dropdown_width=25, value='None', command=self.__switch_undersample_state)

        self.over_sample_ratio_entrybox = Entry_Box(self.hyperparameters_frm, "OVER SAMPLE RATIO:", "30", entry_box_width=25, justify='center', status=DISABLED, value=1.0)
        self.oversample_settings_dropdown = SimBADropDown(parent=self.hyperparameters_frm, dropdown_options=self.over_sample_options, label="OVER SAMPLE SETTING:", label_width=30, dropdown_width=25, value=Dtypes.NONE.value, command=self.__switch_oversample_state)
        self.class_weights_dropdown = SimBADropDown(parent=self.hyperparameters_frm, dropdown_options=self.class_weighing_options, label="CLASS WEIGHTS SETTINGS:", label_width=30, dropdown_width=25, value=Dtypes.NONE.value, command=self.__create_class_weight_table)

        self.evaluations_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="MODEL EVALUATION SETTINGS", icon_name='magnifying')
        self.create_clf_importance_bars_var = BooleanVar()
        self.calc_shap_scores_var = BooleanVar()
        self.learning_curve_var = BooleanVar()

        self.meta_data_file_cb, self.create_meta_data_file_var = SimbaCheckbox(parent=self.evaluations_frm, txt="MODEL META DATA FILE", txt_img='analyze_blue')
        self.decision_tree_graphviz_cb, self.create_example_decision_tree_graphviz_var = SimbaCheckbox(parent=self.evaluations_frm, txt='EXAMPLE DECISION TREE ("graphviz")', txt_img='decision_tree_blue')
        self.decision_tree_dtreeviz_cb, self.create_example_decision_tree_dtreeviz_var = SimbaCheckbox(parent=self.evaluations_frm, txt='EXAMPLE DECISION TREE ("dtreeviz")', txt_img='decision_tree_purple')
        self.clf_report_cb, self.create_clf_report_var = SimbaCheckbox(parent=self.evaluations_frm, txt="CLASSIFICATION REPORT", txt_img='four_squares')
        self.n_features_bars_entry_box = Entry_Box(self.evaluations_frm,"BAR GRAPH # FEATURES: ","30", status=DISABLED,validation="numeric", value=20, entry_box_width=15)
        self.bar_graph_cb, self.create_clf_importance_bars_var = SimbaCheckbox(parent=self.evaluations_frm, txt="FEATURE IMPORTANCE BAR GRAPH", txt_img='bar_graph_green', cmd=self.__switch_bar_graph_n_state)
        self.feature_permutation_cb, self.feature_permutation_importance_var = SimbaCheckbox(parent=self.evaluations_frm, txt="FEATURE PERMUTATION IMPORTANCE (NOTE: INTENSIVE)", txt_img='bar_graph_blue')
        self.learning_curve_k_splits_entry_box = Entry_Box(self.evaluations_frm, "LEARNING CURVE K SPLITS", "30", status=DISABLED, validation="numeric", value="5", entry_box_width=15)
        self.learning_curve_data_splits_entry_box = Entry_Box(self.evaluations_frm, "LEARNING CURVE DATA SPLITS", "30", status=DISABLED, validation="numeric", value="5", entry_box_width=15)
        self.learning_curve_cb, self.learning_curve_var = SimbaCheckbox(parent=self.evaluations_frm, txt="LEARNING CURVES (NOTE: INTENSIVE)", txt_img='line_chart_light_blue', cmd=self.__switch_learning_curve_states)

        self.create_pr_curve_cb, self.create_pr_curve_var = SimbaCheckbox(parent=self.evaluations_frm, txt="PRECISION RECALL CURVES", txt_img='line_chart_red')
        self.partial_dependency_cb, self.partial_dependency_var = SimbaCheckbox(parent=self.evaluations_frm, txt="PARTIAL DEPENDENCIES (Note: INTENSIVE)", txt_img='dependency')

        self.shap_present = Entry_Box( self.evaluations_frm, "SHAP TARGET PRESENT #", "30", status=DISABLED, validation="numeric", entry_box_width=15)
        self.shap_absent = Entry_Box( self.evaluations_frm, "SHAP TARGET ABSENT #", "30", status=DISABLED, validation="numeric", entry_box_width=15)
        self.shap_save_it_dropdown = SimBADropDown(parent=self.evaluations_frm, dropdown_options=[1, 10, 100, 1000, "ALL FRAMES"], label="SHAPE SAVE CADENCE:", label_width=30, dropdown_width=25, value="ALL FRAMES", state='disabled')
        self.shap_multiprocess_dropdown = SimBADropDown(parent=self.evaluations_frm, dropdown_options=["TRUE", "FALSE"], label="MULTI-PROCESS SHAP VALUES:", label_width=30, dropdown_width=25, value="FALSE", state='disabled', command=self.__change_shap_cadence_options)

        self.calculate_shap_scores_cb, self.calc_shap_scores_var = SimbaCheckbox(parent=self.evaluations_frm, txt="COMPUTE SHAP SCORES", txt_img='shap', cmd=self.__switch_shap_entry_states)

        self.save_frame = CreateLabelFrameWithIcon(parent=self.main_frm, header="SAVE", icon_name='save_small')
        save_global_btn = SimbaButton(parent=self.save_frame, txt="SAVE SETTINGS (GLOBAL ENVIRONMENT)", txt_clr='blue', img='save_blue', font=Formats.FONT_REGULAR.value, cmd=self.save_global)
        save_meta_btn = SimbaButton(parent=self.save_frame, txt="SAVE SETTINGS (SPECIFIC MODEL)", txt_clr='green', img='save_green', font=Formats.FONT_REGULAR.value, cmd=self.save_config)
        clear_cache_btn = SimbaButton(parent=self.save_frame, txt="CLEAR CACHE", txt_clr='red', img='trash_red', font=Formats.FONT_REGULAR.value, cmd=self.clear_cache)

        load_meta_data_frm.grid(row=0, column=0, sticky=NW)
        self.select_config_file.grid(row=0, column=0, sticky=NW)
        load_config_btn.grid(row=1, column=0, sticky=NW)
        label_link.grid(row=2, column=0, sticky=NW)

        machine_model_frm.grid(row=1, column=0, sticky=NW)
        self.machine_model_dropdown.grid(row=0, column=0, sticky=NW)

        behavior_frm.grid(row=2, column=0, sticky=NW)
        self.behavior_name_dropdown.grid(row=0, column=0, sticky=NW)

        self.hyperparameters_frm.grid(row=3, column=0, sticky=NW)
        self.estimators_entrybox.grid(row=0, column=0, sticky=NW)
        self.max_features_dropdown.grid(row=1, column=0, sticky=NW)
        self.criterion_dropdown.grid(row=2, column=0, sticky=NW)
        self.train_test_size_dropdown.grid(row=3, column=0, sticky=NW)
        self.train_test_type_dropdown.grid(row=4, column=0, sticky=NW)
        self.min_sample_leaf_eb.grid(row=5, column=0, sticky=NW)
        self.undersample_settings_dropdown.grid(row=6, column=0, sticky=NW)
        self.under_sample_ratio_entrybox.grid(row=7, column=0, sticky=NW)
        self.oversample_settings_dropdown.grid(row=8, column=0, sticky=NW)
        self.over_sample_ratio_entrybox.grid(row=9, column=0, sticky=NW)
        self.class_weights_dropdown.grid(row=10, column=0, sticky=NW)

        self.evaluations_frm.grid(row=4, column=0, sticky=NW)
        self.meta_data_file_cb.grid(row=0, column=0, sticky=NW)
        self.decision_tree_graphviz_cb.grid(row=1, column=0, sticky=NW)
        self.decision_tree_dtreeviz_cb.grid(row=2, column=0, sticky=NW)
        self.clf_report_cb.grid(row=3, column=0, sticky=NW)
        self.bar_graph_cb.grid(row=4, column=0, sticky=NW)
        self.n_features_bars_entry_box.grid(row=5, column=0, sticky=NW)
        self.feature_permutation_cb.grid(row=6, column=0, sticky=NW)
        self.learning_curve_cb.grid(row=7, column=0, sticky=NW)
        self.learning_curve_k_splits_entry_box.grid(row=8, column=0, sticky=NW)
        self.learning_curve_data_splits_entry_box.grid(row=9, column=0, sticky=NW)
        self.create_pr_curve_cb.grid(row=10, column=0, sticky=NW)
        self.partial_dependency_cb.grid(row=11, column=0, sticky=NW)
        self.calculate_shap_scores_cb.grid(row=12, column=0, sticky=NW)
        self.shap_present.grid(row=13, column=0, sticky=NW)
        self.shap_absent.grid(row=14, column=0, sticky=NW)
        self.shap_save_it_dropdown.grid(row=15, column=0, sticky=NW)
        self.shap_multiprocess_dropdown.grid(row=16, column=0, sticky=NW)

        self.save_frame.grid(row=5, column=0, sticky=NW)
        save_global_btn.grid(row=0, column=0, sticky=NW)
        save_meta_btn.grid(row=1, column=0, sticky=NW)
        clear_cache_btn.grid(row=2, column=0, sticky=NW)

        #self.main_frm.mainloop()

    def dropdown_switch_entry_box_state(self, box, dropdown):
        if dropdown.getChoices() != "None":
            box.set_state(NORMAL)
        else:
            box.set_state(DISABLED)

    def __switch_undersample_state(self, value):
        if value == 'None': self.under_sample_ratio_entrybox.set_state(DISABLED)
        else: self.under_sample_ratio_entrybox.set_state(NORMAL)

    def __switch_oversample_state(self, value):
        if value == 'None': self.over_sample_ratio_entrybox.set_state(DISABLED)
        else: self.over_sample_ratio_entrybox.set_state(NORMAL)

    def __switch_bar_graph_n_state(self):
        if self.create_clf_importance_bars_var.get(): self.n_features_bars_entry_box.set_state(NORMAL)
        else: self.n_features_bars_entry_box.set_state(DISABLED)

    def __switch_learning_curve_states(self):
        if self.learning_curve_var.get():
            self.learning_curve_k_splits_entry_box.set_state(NORMAL)
            self.learning_curve_data_splits_entry_box.set_state(NORMAL)
        else:
            self.learning_curve_k_splits_entry_box.set_state(DISABLED)
            self.learning_curve_data_splits_entry_box.set_state(DISABLED)

    def __change_shap_cadence_options(self, value):
        if str_2_bool(value):
            self.shap_save_it_dropdown.setChoices("ALL FRAMES")
            self.shap_save_it_dropdown.disable()
        else:
            self.shap_save_it_dropdown.enable()

    def __switch_shap_entry_states(self):
        value = self.calc_shap_scores_var.get()
        if value:
            self.shap_present.set_state(NORMAL)
            self.shap_absent.set_state(NORMAL)
            self.shap_save_it_dropdown.enable()
            self.shap_multiprocess_dropdown.enable()
        else:
            self.shap_present.set_state(DISABLED)
            self.shap_absent.set_state(DISABLED)
            self.shap_save_it_dropdown.disable()
            self.shap_multiprocess_dropdown.disable()


    def __create_class_weight_table(self, value: str):
        if hasattr(self, "class_weight_frm"):
            self.weight_present.destroy()
            self.weight_absent.destroy()
            self.class_weight_frm.destroy()

        if value == "custom":
            self.class_weight_frm = CreateLabelFrameWithIcon(parent=self.hyperparameters_frm, header="CLASS WEIGHTS", icon_name='weights')

            self.weight_present = SimBADropDown(parent=self.class_weight_frm, dropdown_options=self.class_weights_options, label=f"{self.behavior_name_dropdown.getChoices()} PRESENT:", label_width=30, dropdown_width=25, value=2)
            self.weight_absent = SimBADropDown(parent=self.class_weight_frm, dropdown_options=self.class_weights_options, label=f"{self.behavior_name_dropdown.getChoices()} ABSENT:", label_width=30, dropdown_width=25, value=1)
            self.class_weight_frm.grid(row=11, column=0, sticky=NW)
            self.weight_present.grid(row=0, column=0, sticky=NW)
            self.weight_absent.grid(row=1, column=0, sticky=NW)

    def __checks(self):
        check_int(name="Random forest estimators", value=self.estimators_entrybox.entry_get)
        check_int(name="Minimum sample leaf", value=self.min_sample_leaf_eb.entry_get)
        if self.undersample_settings_dropdown.getChoices() != "None":
            check_float(name="UNDER SAMPLE RATIO", value=self.under_sample_ratio_entrybox.entry_get, min_value=10e-6)
        if self.oversample_settings_dropdown.getChoices() != "None":
            check_float(name="OVER SAMPLE RATIO", value=self.over_sample_ratio_entrybox.entry_get, min_value=10e-6)
        if self.create_clf_importance_bars_var.get():
            check_int(name="# FEATURES", value=self.n_features_bars_entry_box.entry_get, min_value=1)
        if self.learning_curve_var.get():
            check_int(name="LEARNING CURVE K SPLITS", value=self.learning_curve_k_splits_entry_box.entry_get)
            check_int(name="LEARNING CURVE DATA SPLITS", value=self.learning_curve_data_splits_entry_box.entry_get)
        if self.calc_shap_scores_var.get():
            check_int(name="SHAP TARGET PRESENT", value=self.shap_present.entry_get, min_value=1)
            check_int(name="SHAP TARGET ABSENT", value=self.shap_absent.entry_get, min_value=1)

    def __get_variables(self):
        self.algorithm = self.machine_model_dropdown.getChoices()
        self.behavior_name = self.behavior_name_dropdown.getChoices()
        self.n_estimators = self.estimators_entrybox.entry_get
        self.max_features = self.max_features_dropdown.getChoices()
        self.criterion = self.criterion_dropdown.getChoices()
        self.test_size = self.train_test_size_dropdown.getChoices()
        self.train_test_type = self.train_test_type_dropdown.getChoices()
        self.min_sample_leaf = self.min_sample_leaf_eb.entry_get
        self.under_sample_setting = self.undersample_settings_dropdown.getChoices()
        self.under_sample_ratio = "NaN"
        if self.undersample_settings_dropdown.getChoices() != Dtypes.NONE.value:
            self.under_sample_ratio = self.under_sample_ratio_entrybox.entry_get
        self.over_sample_setting = self.oversample_settings_dropdown.getChoices()
        self.over_sample_ratio = "NaN"
        if self.oversample_settings_dropdown.getChoices() != Dtypes.NONE.value:
            self.over_sample_ratio = self.over_sample_ratio_entrybox.entry_get
        self.class_weight_method = self.class_weights_dropdown.getChoices()
        self.class_custom_weights = {}
        if self.class_weight_method == "custom":
            self.class_custom_weights[0] = self.weight_absent.getChoices()
            self.class_custom_weights[1] = self.weight_present.getChoices()

        self.meta_info_file = self.create_meta_data_file_var.get()
        self.example_graphviz = self.create_example_decision_tree_graphviz_var.get()
        self.example_dtreeviz = self.create_example_decision_tree_dtreeviz_var.get()
        self.clf_report = self.create_clf_report_var.get()
        self.clf_importance_bars = self.create_clf_importance_bars_var.get()
        self.clf_importance_bars_n = 0
        if self.clf_importance_bars:
            self.clf_importance_bars_n = self.n_features_bars_entry_box.entry_get
        self.permutation_importances = self.feature_permutation_importance_var.get()
        self.pr_curve = self.create_pr_curve_var.get()
        self.shap_scores_absent = 0
        self.shap_scores_present = 0
        self.shap_multiprocess = False
        self.shap_save_it = "ALL FRAMES"
        self.shap_scores = self.calc_shap_scores_var.get()
        if self.shap_scores:
            self.shap_scores_absent = self.shap_absent.entry_get
            self.shap_scores_present = self.shap_present.entry_get
            self.shap_save_it = self.shap_save_it_dropdown.getChoices()
            self.shap_multiprocess = self.shap_multiprocess_dropdown.getChoices()

        self.learning_curve = self.learning_curve_var.get()
        self.partial_dependency = self.partial_dependency_var.get()
        self.learning_curve_k_split = 0
        self.learning_curve_data_split = 0
        if self.learning_curve:
            self.learning_curve_k_split = (
                self.learning_curve_k_splits_entry_box.entry_get
            )
            self.learning_curve_data_split = (
                self.learning_curve_data_splits_entry_box.entry_get
            )

    def find_meta_file_cnt(self):
        self.meta_file_cnt = 0
        self.total_meta_files = find_files_of_filetypes_in_directory(
            directory=self.configs_meta_dir, extensions=[".csv"], raise_warning=False
        )
        for f in os.listdir(self.configs_meta_dir):
            if f.__contains__("_meta") and f.__contains__(str(self.behavior_name)):
                self.meta_file_cnt += 1

    def save_global(self):
        self.__checks()
        self.__get_variables()
        self.config.set(ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.MODEL_TO_RUN.value, self.algorithm)
        self.config.set(ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.RF_ESTIMATORS.value, str(self.n_estimators))
        self.config.set(ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.RF_MAX_FEATURES.value, str(self.max_features))
        self.config.set(ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.RF_CRITERION.value, self.criterion)
        self.config.set(ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.TT_SIZE.value, str(self.test_size))
        self.config.set(ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.TRAIN_TEST_SPLIT_TYPE.value, str(self.train_test_type))
        self.config.set(ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.MIN_LEAF.value, str(self.min_sample_leaf))
        self.config.set(ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.UNDERSAMPLE_RATIO.value, str(self.under_sample_ratio))
        self.config.set(ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.UNDERSAMPLE_SETTING.value, str(self.under_sample_setting))
        self.config.set(ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.OVERSAMPLE_RATIO.value, str(self.over_sample_ratio))
        self.config.set(ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.OVERSAMPLE_SETTING.value, str(self.over_sample_setting))
        self.config.set(ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.CLASSIFIER.value, self.behavior_name)
        self.config.set(ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.RF_META_DATA.value, str(self.meta_info_file))
        self.config.set(ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.EX_DECISION_TREE.value, str(self.example_graphviz))
        self.config.set(ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.CLF_REPORT.value, str(self.clf_report))
        self.config.set(ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.IMPORTANCE_LOG.value, str(self.clf_importance_bars))
        self.config.set(ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.IMPORTANCE_BAR_CHART.value, str(self.clf_importance_bars))
        self.config.set(ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.N_FEATURE_IMPORTANCE_BARS.value, str(self.clf_importance_bars_n))
        self.config.set(ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.PERMUTATION_IMPORTANCE.value, str(self.permutation_importances))
        self.config.set(ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.LEARNING_CURVE.value, str(self.learning_curve))
        self.config.set(ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.PRECISION_RECALL.value, str(self.pr_curve))
        self.config.set(ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.LEARNING_CURVE_K_SPLITS.value, str(self.learning_curve_k_split))
        self.config.set(ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.LEARNING_CURVE_DATA_SPLITS.value, str(self.learning_curve_data_split))
        self.config.set(ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.EX_DECISION_TREE_FANCY.value, str(self.example_dtreeviz),)
        self.config.set(ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.SHAP_SCORES.value, str(self.shap_scores))
        self.config.set(ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.SHAP_PRESENT.value,str(self.shap_scores_present))
        self.config.set(ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.SHAP_ABSENT.value, str(self.shap_scores_absent))
        self.config.set(ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.SHAP_SAVE_ITERATION.value, str(self.shap_save_it))
        self.config.set(ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.SHAP_MULTIPROCESS.value, str(self.shap_multiprocess))
        self.config.set(ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.PARTIAL_DEPENDENCY.value,str(self.partial_dependency))
        self.config.set(ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.CLASS_WEIGHTS.value, str(self.class_weight_method))
        self.config.set(ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.CLASS_CUSTOM_WEIGHTS.value, str(self.class_custom_weights))

        with open(self.config_path, "w") as f:
            self.config.write(f)

        stdout_success(msg=f"Global model settings saved in the {self.config_path}")

    def save_config(self):
        self.__checks()
        self.__get_variables()

        meta = {
            MLParamKeys.RF_ESTIMATORS.value: self.n_estimators,
            MLParamKeys.RF_MAX_FEATURES.value: self.max_features,
            MLParamKeys.RF_CRITERION.value: self.criterion,
            MLParamKeys.TT_SIZE.value: self.test_size,
            MLParamKeys.TRAIN_TEST_SPLIT_TYPE.value: self.train_test_type,
            MLParamKeys.MIN_LEAF.value: self.min_sample_leaf,
            MLParamKeys.UNDERSAMPLE_RATIO.value: self.under_sample_ratio,
            MLParamKeys.UNDERSAMPLE_SETTING.value: self.under_sample_setting,
            MLParamKeys.OVERSAMPLE_RATIO.value: self.over_sample_ratio,
            MLParamKeys.OVERSAMPLE_SETTING.value: self.over_sample_setting,
            MLParamKeys.RF_METADATA.value: self.meta_info_file,
            MLParamKeys.EX_DECISION_TREE.value: self.example_graphviz,
            MLParamKeys.CLF_REPORT.value: self.clf_report,
            MLParamKeys.IMPORTANCE_LOG.value: self.clf_importance_bars,
            MLParamKeys.IMPORTANCE_BAR_CHART.value: self.clf_importance_bars,
            MLParamKeys.N_FEATURE_IMPORTANCE_BARS.value: self.clf_importance_bars_n,
            MLParamKeys.PERMUTATION_IMPORTANCE.value: self.permutation_importances,
            MLParamKeys.LEARNING_CURVE.value: self.learning_curve,
            MLParamKeys.PRECISION_RECALL.value: self.pr_curve,
            MLParamKeys.LEARNING_CURVE_K_SPLITS.value: self.learning_curve_k_split,
            MLParamKeys.LEARNING_CURVE_DATA_SPLITS.value: self.learning_curve_data_split,
            MLParamKeys.SHAP_SCORES.value: self.shap_scores,
            MLParamKeys.SHAP_PRESENT.value: self.shap_scores_present,
            MLParamKeys.SHAP_ABSENT.value: self.shap_scores_absent,
            MLParamKeys.SHAP_SAVE_ITERATION.value: self.shap_save_it,
            MLParamKeys.SHAP_MULTIPROCESS.value: self.shap_multiprocess,
            MLParamKeys.PARTIAL_DEPENDENCY.value: self.partial_dependency,
            MLParamKeys.CLASS_WEIGHTS.value: self.class_weight_method,
            MLParamKeys.CLASS_CUSTOM_WEIGHTS.value: str(self.class_custom_weights)
        }

        meta_df = pd.DataFrame(meta, index=[0])
        meta_df.insert(0, "Classifier_name", self.behavior_name)
        self.find_meta_file_cnt()
        file_name = f"{self.behavior_name}_meta_{self.meta_file_cnt}.csv"
        save_path = os.path.join(self.configs_meta_dir, file_name)
        meta_df.to_csv(save_path, index=FALSE)
        stdout_success(msg=f"Hyper-parameter config saved ({str(len(self.total_meta_files)+1)} saved in {self.configs_meta_dir} folder).")

    def clear_cache(self):
        self.behavior_name = self.behavior_name_dropdown.getChoices()
        self.find_meta_file_cnt()
        for file_path in self.total_meta_files:
            os.remove(os.path.join(file_path))
            print(f"Deleted hyperparameters config {file_path} ...")
        stdout_trash(msg=f"{str(len(self.total_meta_files))} config files deleted")

    def check_meta_data_integrity(self):
        self.meta = {k.lower(): v for k, v in self.meta.items()}
        for i in self.expected_meta_dict_entries:
            if i not in self.meta.keys():
                stdout_warning(msg=f"The file does not contain an expected entry for {i} parameter")
                self.meta[i] = None
            else:
                if type(self.meta[i]) == str:
                    if self.meta[i].lower().strip() == "yes":
                        self.meta[i] = True

    def load_config(self):
        config_file_path = self.select_config_file.file_path
        _, config_name, _ = get_fn_ext(config_file_path)
        check_file_exist_and_readable(file_path=config_file_path)
        try:
            meta_df = pd.read_csv(config_file_path, index_col=False)
        except pd.errors.ParserError:
            raise InvalidHyperparametersFileError(msg=f"SIMBA ERROR: {config_name} is not a valid SimBA meta hyper-parameters file.")
        self.meta = {}
        for m in meta_df.columns:
            self.meta[m] = meta_df[m][0]
        self.get_expected_meta_dict_entry_keys()
        self.check_meta_data_integrity()
        self.behavior_name_dropdown.setChoices(self.meta[MLParamKeys.CLASSIFIER_NAME.value])
        self.estimators_entrybox.entry_set(val=self.meta[MLParamKeys.RF_ESTIMATORS.value])
        self.max_features_dropdown.setChoices(self.meta[MLParamKeys.RF_MAX_FEATURES.value])
        self.criterion_dropdown.setChoices(self.meta[MLParamKeys.RF_CRITERION.value])
        self.train_test_size_dropdown.setChoices(self.meta[MLParamKeys.TT_SIZE.value])
        self.min_sample_leaf_eb.entry_set(val=self.meta[MLParamKeys.MIN_LEAF.value])
        self.undersample_settings_dropdown.setChoices(self.meta[MLParamKeys.UNDERSAMPLE_SETTING.value])
        if self.undersample_settings_dropdown.getChoices() != Dtypes.NONE.value:
            self.under_sample_ratio_entrybox.entry_set(val=self.meta[MLParamKeys.UNDERSAMPLE_RATIO.value])
            self.under_sample_ratio_entrybox.set_state(NORMAL)
        else:
            self.under_sample_ratio_entrybox.set_state(DISABLED)
        self.oversample_settings_dropdown.setChoices(self.meta[MLParamKeys.OVERSAMPLE_SETTING.value])
        if self.oversample_settings_dropdown.getChoices() != Dtypes.NONE.value:
            self.over_sample_ratio_entrybox.entry_set(val=self.meta[MLParamKeys.OVERSAMPLE_RATIO.value])
            self.over_sample_ratio_entrybox.set_state(NORMAL)
        else:
            self.over_sample_ratio_entrybox.set_state(DISABLED)

        if self.meta[MLParamKeys.RF_METADATA.value]:
            self.create_meta_data_file_var.set(value=True)
        else:
            self.create_meta_data_file_var.set(value=False)
        if self.meta[MLParamKeys.EX_DECISION_TREE.value]:
            self.create_example_decision_tree_graphviz_var.set(value=True)
        else:
            self.create_example_decision_tree_graphviz_var.set(value=False)
        if self.meta[MLParamKeys.EX_DECISION_TREE.value]:
            self.create_example_decision_tree_graphviz_var.set(value=True)
        else:
            self.create_example_decision_tree_graphviz_var.set(value=False)
        if self.meta[MLParamKeys.CLF_REPORT.value]:
            self.create_clf_report_var.set(value=True)
        else:
            self.create_clf_report_var.set(value=False)
        if (self.meta[MLParamKeys.IMPORTANCE_LOG.value] or self.meta[MLParamKeys.IMPORTANCE_BAR_CHART.value]):
            self.create_clf_importance_bars_var.set(value=True)
            self.n_features_bars_entry_box.set_state(NORMAL)
            self.n_features_bars_entry_box.entry_set(val=self.meta[MLParamKeys.N_FEATURE_IMPORTANCE_BARS.value])
        else:
            self.create_clf_importance_bars_var.set(value=False)
            self.n_features_bars_entry_box.set_state(DISABLED)

        if self.meta[MLParamKeys.PERMUTATION_IMPORTANCE.value]:
            self.feature_permutation_importance_var.set(value=True)

        if self.meta[MLParamKeys.LEARNING_CURVE.value]:
            self.learning_curve_var.set(value=True)
            self.learning_curve_k_splits_entry_box.set_state(NORMAL)
            self.learning_curve_data_splits_entry_box.set_state(NORMAL)
            self.learning_curve_k_splits_entry_box.entry_set(val=self.meta[MLParamKeys.LEARNING_CURVE_K_SPLITS.value])
            self.learning_curve_data_splits_entry_box.entry_set(val=self.meta[MLParamKeys.LEARNING_CURVE_DATA_SPLITS.value])
        else:
            self.learning_curve_var.set(value=False)
            self.learning_curve_k_splits_entry_box.set_state(DISABLED)
            self.learning_curve_data_splits_entry_box.set_state(DISABLED)

        if self.meta[MLParamKeys.SHAP_SCORES.value]:
            self.calc_shap_scores_var.set(value=True)
            self.shap_present.set_state(NORMAL)
            self.shap_absent.set_state(NORMAL)
            self.shap_absent.set_state(NORMAL)
            self.shap_save_it_dropdown.enable()
            self.shap_multiprocess_dropdown.enable()
            self.shap_present.entry_set(val=self.meta[MLParamKeys.SHAP_PRESENT.value])
            self.shap_absent.entry_set(val=self.meta[MLParamKeys.SHAP_ABSENT.value])
            if MLParamKeys.SHAP_SAVE_ITERATION.value in self.meta.keys():
                self.shap_save_it_dropdown.setChoices(self.meta[MLParamKeys.SHAP_SAVE_ITERATION.value])
            else:
                self.shap_save_it_dropdown.setChoices("ALL FRAMES")
            if MLParamKeys.SHAP_MULTIPROCESS.value in self.meta.keys():
                self.shap_multiprocess_dropdown.setChoices(self.meta[MLParamKeys.SHAP_MULTIPROCESS.value])
        else:
            self.calc_shap_scores_var.set(value=False)
            self.shap_present.set_state(DISABLED)
            self.shap_absent.set_state(DISABLED)
            self.shap_save_it_dropdown.enable()
            self.shap_multiprocess_dropdown.disable()

        if MLParamKeys.TRAIN_TEST_SPLIT_TYPE.value in self.meta.keys():
            self.train_test_type_dropdown.setChoices(self.meta[MLParamKeys.TRAIN_TEST_SPLIT_TYPE.value])
        else:
            self.train_test_type_dropdown.setChoices(Options.TRAIN_TEST_SPLIT.value[0])
        if MLParamKeys.SHAP_SAVE_ITERATION.value in self.meta.keys():
            self.shap_save_it_dropdown.setChoices(self.meta[MLParamKeys.SHAP_SAVE_ITERATION.value])
        if MLParamKeys.SHAP_MULTIPROCESS.value in self.meta.keys():
            self.shap_multiprocess_dropdown.setChoices(self.meta[MLParamKeys.SHAP_MULTIPROCESS.value])
        if MLParamKeys.PARTIAL_DEPENDENCY.value in self.meta.keys():
            if self.meta[MLParamKeys.PARTIAL_DEPENDENCY.value] in Options.PERFORM_FLAGS.value:
                self.partial_dependency_var.set(value=True)
        else:
            self.shap_save_it_dropdown.setChoices(Dtypes.NONE.value)
        if MLParamKeys.CLASS_WEIGHTS.value in self.meta.keys():
            if self.meta[MLParamKeys.CLASS_WEIGHTS.value] not in Options.CLASS_WEIGHT_OPTIONS.value:
                self.meta[MLParamKeys.CLASS_WEIGHTS.value] = Dtypes.NONE.value
            self.class_weights_dropdown.setChoices(self.meta["class_weights"])
            if self.meta[MLParamKeys.CLASS_WEIGHTS.value] == "custom":
                self.__create_class_weight_table(value="custom")
                weights = ast.literal_eval(self.meta[MLParamKeys.CLASS_CUSTOM_WEIGHTS.value])

                self.weight_present.setChoices(weights[1])
                self.weight_absent.setChoices(weights[0])
        else:
            self.class_weights_dropdown.setChoices("None")
            self.__create_class_weight_table(value='None')

        print(f"Loaded parameters from config {config_name}...")

    def get_expected_meta_dict_entry_keys(self):
        self.expected_meta_dict_entries = [
            MLParamKeys.CLASSIFIER_NAME.value,
            MLParamKeys.RF_ESTIMATORS.value,
            MLParamKeys.RF_MAX_FEATURES.value,
            MLParamKeys.RF_CRITERION.value,
            MLParamKeys.TT_SIZE.value,
            MLParamKeys.MIN_LEAF.value,
            MLParamKeys.UNDERSAMPLE_RATIO.value,
            MLParamKeys.UNDERSAMPLE_SETTING.value,
            MLParamKeys.OVERSAMPLE_RATIO.value,
            MLParamKeys.OVERSAMPLE_SETTING.value,
            MLParamKeys.RF_METADATA.value,
            MLParamKeys.EX_DECISION_TREE.value,
            MLParamKeys.CLF_REPORT.value,
            MLParamKeys.IMPORTANCE_LOG.value,
            MLParamKeys.IMPORTANCE_BAR_CHART.value,
            MLParamKeys.N_FEATURE_IMPORTANCE_BARS.value,
            MLParamKeys.PERMUTATION_IMPORTANCE.value,
            MLParamKeys.LEARNING_CURVE.value,
            MLParamKeys.PRECISION_RECALL.value,
            MLParamKeys.LEARNING_CURVE_K_SPLITS.value,
            MLParamKeys.LEARNING_CURVE_DATA_SPLITS.value,
            MLParamKeys.SHAP_SCORES.value,
            MLParamKeys.SHAP_PRESENT.value,
            MLParamKeys.SHAP_ABSENT.value,
            MLParamKeys.SHAP_SAVE_ITERATION.value,
            MLParamKeys.SHAP_MULTIPROCESS.value,
            MLParamKeys.PARTIAL_DEPENDENCY.value,
            MLParamKeys.CLASS_WEIGHTS.value,
            MLParamKeys.CLASS_CUSTOM_WEIGHTS.value,
        ]


#_ = MachineModelSettingsPopUp(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini")

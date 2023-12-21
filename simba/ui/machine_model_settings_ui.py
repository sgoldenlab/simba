__author__ = "Simon Nilsson"

import ast
import os
import webbrowser
from tkinter import *

import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        Entry_Box, FileSelect, hxtScrollbar)
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_int)
from simba.utils.enums import Formats, Keys, Links, Options, ConfigKey, Dtypes, MachineLearningMetaKeys
from simba.utils.errors import InvalidHyperparametersFileError
from simba.utils.printing import stdout_success, stdout_trash, stdout_warning
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_fn_ext, read_config_entry)


class MachineModelSettingsPopUp(PopUpMixin, ConfigReader):
    """
    Launch GUI window for specifying ML model training parameters.
    """

    def __init__(self, config_path: str):
        ConfigReader.__init__(self, config_path=config_path)
        PopUpMixin.__init__(self, title="MACHINE MODEL SETTINGS", size=(450, 700))
        if not os.path.exists(self.configs_meta_dir):
            os.makedirs(self.configs_meta_dir)
        self.clf_options = Options.CLF_MODELS.value
        self.max_features_options = Options.CLF_MAX_FEATURES.value
        self.criterion_options = Options.CLF_CRITERION.value
        self.under_sample_options = Options.UNDERSAMPLE_OPTIONS.value
        self.over_sample_options = Options.OVERSAMPLE_OPTIONS.value
        self.class_weighing_options = Options.CLASS_WEIGHT_OPTIONS.value
        self.train_test_sizes_options = Options.CLF_TEST_SIZE_OPTIONS.value
        self.class_weights_options = list(range(1, 11, 1))
        load_meta_data_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="LOAD META-DATA",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.TRAIN_ML_MODEL.value,
        )
        self.select_config_file = FileSelect(load_meta_data_frm, "CONFIG PATH:")
        load_config_btn = Button(
            load_meta_data_frm,
            text="LOAD",
            fg="blue",
            command=lambda: self.load_config(),
        )
        label_link = Label(
            load_meta_data_frm, text="[MODEL SETTINGS TUTORIAL]", fg="blue"
        )
        label_link.bind(
            "<Button-1>",
            lambda e: webbrowser.open_new(
                "https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-7-train-machine-model"
            ),
        )

        machine_model_frm = LabelFrame(
            self.main_frm,
            text="MACHINE MODEL ALGORITHM",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
        )
        self.machine_model_dropdown = DropDownMenu(
            machine_model_frm, "ALGORITHM: ", self.clf_options, "25"
        )
        self.machine_model_dropdown.setChoices(self.clf_options[0])

        behavior_frm = LabelFrame(
            self.main_frm, text="BEHAVIOR", font=Formats.LABELFRAME_HEADER_FORMAT.value
        )
        self.behavior_name_dropdown = DropDownMenu(
            behavior_frm, "BEHAVIOR: ", self.clf_names, "25"
        )
        self.behavior_name_dropdown.setChoices(self.clf_names[0])

        self.hyperparameters_frm = LabelFrame(
            self.main_frm,
            text="HYPER-PARAMETERS",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
        )
        self.estimators_entrybox = Entry_Box(
            self.hyperparameters_frm,
            "Random forest estimators:",
            "25",
            validation="numeric",
        )
        n_estimators = read_config_entry(
            self.config,
            ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
            MachineLearningMetaKeys.RF_ESTIMATORS.value,
            data_type=Dtypes.INT.value,
        )

        self.estimators_entrybox.entry_set(val=n_estimators)
        max_features = read_config_entry(
            self.config,
            ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
            MachineLearningMetaKeys.RF_MAX_FEATURES.value,
            data_type=Dtypes.STR.value,
        )

        self.max_features_dropdown = DropDownMenu(
            self.hyperparameters_frm, "Max features: ", self.max_features_options, "25"
        )
        self.max_features_dropdown.setChoices(max_features)
        self.criterion_dropdown = DropDownMenu(
            self.hyperparameters_frm, "Criterion: ", self.criterion_options, "25"
        )
        self.criterion_dropdown.setChoices(self.criterion_options[0])
        train_test_size = read_config_entry(
            self.config,
            ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
            MachineLearningMetaKeys.TT_SIZE.value,
            data_type=Dtypes.STR.value,
        )
        self.train_test_size_dropdown = DropDownMenu(
            self.hyperparameters_frm, "Test Size: ", self.train_test_sizes_options, "25"
        )
        self.train_test_size_dropdown.setChoices(str(train_test_size))
        train_test_split_type = read_config_entry(
            self.config,
            ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
            MachineLearningMetaKeys.TRAIN_TEST_SPLIT_TYPE.value,
            data_type=Dtypes.STR.value,
        )
        self.train_test_type_dropdown = DropDownMenu(
            self.hyperparameters_frm,
            "Train-test Split Type: ",
            Options.TRAIN_TEST_SPLIT.value,
            "25",
        )
        self.train_test_type_dropdown.setChoices(str(train_test_split_type))
        max_depth = read_config_entry(
            self.config,
            ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
            MachineLearningMetaKeys.RF_MAX_DEPTH.value,
            data_type=Dtypes.STR.value,
        )
        self.max_depth_eb = Entry_Box(
            self.hyperparameters_frm, "Max Depth: ", "25", validation="numeric"
        )
        self.max_depth_eb.entry_set(val=max_depth)
        min_sample_leaf = read_config_entry(
            self.config,
            ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
            MachineLearningMetaKeys.MIN_LEAF.value,
            data_type=Dtypes.INT.value,
        )
        self.min_sample_leaf_eb = Entry_Box(
            self.hyperparameters_frm, "Minimum sample leaf", "25", validation="numeric"
        )
        self.min_sample_leaf_eb.entry_set(val=min_sample_leaf)
        undersample_settings = read_config_entry(
            self.config,
            ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
            MachineLearningMetaKeys.UNDERSAMPLE_SETTING.value,
            data_type=Dtypes.STR.value,
        )
        self.under_sample_ratio_entrybox = Entry_Box(
            self.hyperparameters_frm, "UNDER-sample ratio: ", "25", status=DISABLED
        )
        self.undersample_settings_dropdown = DropDownMenu(
            self.hyperparameters_frm,
            "UNDER-sample setting: ",
            self.under_sample_options,
            "25",
            com=lambda x: self.dropdown_switch_entry_box_state(
                self.under_sample_ratio_entrybox, self.undersample_settings_dropdown
            ),
        )
        self.undersample_settings_dropdown.setChoices(undersample_settings)
        self.over_sample_ratio_entrybox = Entry_Box(
            self.hyperparameters_frm, "OVER-sample ratio: ", "25", status=DISABLED
        )
        self.oversample_settings_dropdown = DropDownMenu(
            self.hyperparameters_frm,
            "OVER-sample setting: ",
            self.over_sample_options,
            "25",
            com=lambda x: self.dropdown_switch_entry_box_state(
                self.over_sample_ratio_entrybox, self.oversample_settings_dropdown
            ),
        )
        self.oversample_settings_dropdown.setChoices("None")
        self.class_weights_dropdown = DropDownMenu(
            self.hyperparameters_frm,
            "Class-weights setting: ",
            self.class_weighing_options,
            "25",
            com=lambda x: self.create_class_weight_table(),
        )
        self.class_weights_dropdown.setChoices("None")

        self.evaluations_frm = LabelFrame(
            self.main_frm,
            text="MODEL EVALUATION SETTINGS",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
        )
        self.create_meta_data_file_var = BooleanVar()
        self.create_example_decision_tree_graphviz_var = BooleanVar()
        self.create_example_decision_tree_dtreeviz_var = BooleanVar()
        self.create_clf_report_var = BooleanVar()
        self.create_clf_importance_bars_var = BooleanVar()
        self.feature_permutation_importance_var = BooleanVar()
        self.create_pr_curve_var = BooleanVar()
        self.partial_dependency_var = BooleanVar()
        self.calc_shap_scores_var = BooleanVar()
        self.learning_curve_var = BooleanVar()

        self.meta_data_file_cb = Checkbutton(
            self.evaluations_frm,
            text="Create model meta data file",
            variable=self.create_meta_data_file_var,
        )
        self.decision_tree_graphviz_cb = Checkbutton(
            self.evaluations_frm,
            text='Create Example Decision Tree (requires "graphviz")',
            variable=self.create_example_decision_tree_graphviz_var,
        )
        self.decision_tree_dtreeviz_cb = Checkbutton(
            self.evaluations_frm,
            text='Create Fancy Example Decision Tree (requires "dtreeviz")',
            variable=self.create_example_decision_tree_dtreeviz_var,
        )
        self.clf_report_cb = Checkbutton(
            self.evaluations_frm,
            text="Create Classification Report",
            variable=self.create_clf_report_var,
        )
        self.n_features_bars_entry_box = Entry_Box(
            self.evaluations_frm,
            "# Features: ",
            "25",
            status=DISABLED,
            validation="numeric",
        )
        self.bar_graph_cb = Checkbutton(
            self.evaluations_frm,
            text="Create Features Importance Bar Graph",
            variable=self.create_clf_importance_bars_var,
            command=lambda: self.enable_entrybox_from_checkbox(
                check_box_var=self.create_clf_importance_bars_var,
                entry_boxes=[self.n_features_bars_entry_box],
            ),
        )
        self.feature_permutation_cb = Checkbutton(
            self.evaluations_frm,
            text="Compute Feature Permutation Importances (Note: CPU intensive)",
            variable=self.feature_permutation_importance_var,
        )
        self.learning_curve_k_splits_entry_box = Entry_Box(
            self.evaluations_frm,
            "Learning Curve Shuffle K Splits",
            "25",
            status=DISABLED,
            validation="numeric",
        )
        self.learning_curve_data_splits_entry_box = Entry_Box(
            self.evaluations_frm,
            "Learning Curve Shuffle Data Splits",
            "25",
            status=DISABLED,
            validation="numeric",
        )
        self.learning_curve_cb = Checkbutton(
            self.evaluations_frm,
            text="Create Learning Curves (Note: CPU intensive)",
            variable=self.learning_curve_var,
            command=lambda: self.enable_entrybox_from_checkbox(
                check_box_var=self.learning_curve_var,
                entry_boxes=[
                    self.learning_curve_k_splits_entry_box,
                    self.learning_curve_data_splits_entry_box,
                ],
            ),
        )
        self.create_pr_curve_cb = Checkbutton(
            self.evaluations_frm,
            text="Create Precision Recall Curves",
            variable=self.create_pr_curve_var,
        )
        self.shap_present = Entry_Box(
            self.evaluations_frm,
            "# target present",
            "25",
            status=DISABLED,
            validation="numeric",
        )
        self.shap_absent = Entry_Box(
            self.evaluations_frm,
            "# target absent",
            "25",
            status=DISABLED,
            validation="numeric",
        )
        self.shap_save_it_dropdown = DropDownMenu(
            self.evaluations_frm,
            "SHAP save cadence: ",
            [1, 10, 100, 1000, "ALL FRAMES"],
            "25",
        )
        self.shap_save_it_dropdown.setChoices("ALL FRAMES")
        self.shap_save_it_dropdown.disable()
        self.partial_dependency_cb = Checkbutton(
            self.evaluations_frm,
            text="Calculate partial dependencies (Note: CPU intensive)",
            variable=self.partial_dependency_var,
        )
        self.calculate_shap_scores_cb = Checkbutton(
            self.evaluations_frm,
            text="Calculate SHAP scores",
            variable=self.calc_shap_scores_var,
            command=lambda: [
                self.enable_entrybox_from_checkbox(
                    check_box_var=self.calc_shap_scores_var,
                    entry_boxes=[self.shap_present, self.shap_absent],
                ),
                self.enable_dropdown_from_checkbox(
                    check_box_var=self.calc_shap_scores_var,
                    dropdown_menus=[self.shap_save_it_dropdown],
                ),
            ],
        )
        self.save_frame = LabelFrame(
            self.main_frm, text="SAVE", font=Formats.LABELFRAME_HEADER_FORMAT.value
        )
        save_global_btn = Button(
            self.save_frame,
            text="SAVE SETTINGS (GLOBAL ENVIRONMENT)",
            font=("Helvetica", 12, "bold"),
            fg="blue",
            command=lambda: self.save_global(),
        )
        save_meta_btn = Button(
            self.save_frame,
            text="SAVE SETTINGS (SPECIFIC MODEL)",
            font=("Helvetica", 12, "bold"),
            fg="green",
            command=lambda: self.save_config(),
        )
        clear_cache_btn = Button(
            self.save_frame,
            text="CLEAR CACHE",
            font=("Helvetica", 12, "bold"),
            fg="red",
            command=lambda: self.clear_cache(),
        )

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
        self.max_depth_eb.grid(row=1, column=0, sticky=NW)
        self.max_features_dropdown.grid(row=2, column=0, sticky=NW)
        self.criterion_dropdown.grid(row=3, column=0, sticky=NW)
        self.train_test_size_dropdown.grid(row=4, column=0, sticky=NW)
        self.train_test_type_dropdown.grid(row=5, column=0, sticky=NW)
        self.min_sample_leaf_eb.grid(row=6, column=0, sticky=NW)
        self.undersample_settings_dropdown.grid(row=7, column=0, sticky=NW)
        self.under_sample_ratio_entrybox.grid(row=8, column=0, sticky=NW)
        self.oversample_settings_dropdown.grid(row=9, column=0, sticky=NW)
        self.over_sample_ratio_entrybox.grid(row=10, column=0, sticky=NW)
        self.class_weights_dropdown.grid(row=11, column=0, sticky=NW)

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

        self.save_frame.grid(row=5, column=0, sticky=NW)
        save_global_btn.grid(row=0, column=0, sticky=NW)
        save_meta_btn.grid(row=1, column=0, sticky=NW)
        clear_cache_btn.grid(row=2, column=0, sticky=NW)

        self.main_frm.mainloop()

    def dropdown_switch_entry_box_state(self, box, dropdown):
        if dropdown.getChoices() != "None":
            box.set_state(NORMAL)
        else:
            box.set_state(DISABLED)

    def create_class_weight_table(self):
        if hasattr(self, "class_weight_frm"):
            self.weight_present.destroy()
            self.weight_absent.destroy()
            self.class_weight_frm.destroy()

        if self.class_weights_dropdown.getChoices() == "custom":
            self.class_weight_frm = LabelFrame(
                self.hyperparameters_frm,
                text="CLASS WEIGHTS",
                font=("Helvetica", 12, "bold"),
                pady=10,
            )
            self.weight_present = DropDownMenu(
                self.class_weight_frm,
                "{} PRESENT: ".format(self.behavior_name_dropdown.getChoices()),
                self.class_weights_options,
                "25",
            )
            self.weight_present.setChoices(2)
            self.weight_absent = DropDownMenu(
                self.class_weight_frm,
                "{} ABSENT: ".format(self.behavior_name_dropdown.getChoices()),
                self.class_weights_options,
                "25",
            )
            self.weight_absent.setChoices(1)

            self.class_weight_frm.grid(row=11, column=0, sticky=NW)
            self.weight_present.grid(row=0, column=0, sticky=NW)
            self.weight_absent.grid(row=1, column=0, sticky=NW)

    def __checks(self):
        check_int(
            name="Random forest estimators", value=self.estimators_entrybox.entry_get
        )
        check_int(name="Minimum sample leaf", value=self.min_sample_leaf_eb.entry_get)
        if self.undersample_settings_dropdown.getChoices() != "None":
            check_float(
                name="UNDER SAMPLE RATIO",
                value=self.under_sample_ratio_entrybox.entry_get,
            )
        if self.oversample_settings_dropdown.getChoices() != "None":
            check_float(
                name="OVER SAMPLE RATIO",
                value=self.over_sample_ratio_entrybox.entry_get,
            )
        if self.create_clf_importance_bars_var.get():
            check_int(
                name="# FEATURES",
                value=self.n_features_bars_entry_box.entry_get,
                min_value=1,
            )
        if self.learning_curve_var.get():
            check_int(
                name="LEARNING CURVE K SPLITS",
                value=self.learning_curve_k_splits_entry_box.entry_get,
            )
            check_int(
                name="LEARNING CURVE DATA SPLITS",
                value=self.learning_curve_data_splits_entry_box.entry_get,
            )
        if self.calc_shap_scores_var.get():
            check_int(
                name="SHAP TARGET PRESENT",
                value=self.shap_present.entry_get,
                min_value=1,
            )
            check_int(
                name="SHAP TARGET ABSENT", value=self.shap_absent.entry_get, min_value=1
            )

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
        if self.undersample_settings_dropdown.getChoices() != "None":
            self.under_sample_ratio = self.under_sample_ratio_entrybox.entry_get
        self.over_sample_setting = self.oversample_settings_dropdown.getChoices()
        self.over_sample_ratio = "NaN"
        if self.oversample_settings_dropdown.getChoices() != "None":
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
        self.shap_save_it = "ALL FRAMES"
        self.shap_scores = self.calc_shap_scores_var.get()
        if self.shap_scores:
            self.shap_scores_absent = self.shap_absent.entry_get
            self.shap_scores_present = self.shap_present.entry_get
            self.shap_save_it = self.shap_save_it_dropdown.getChoices()
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
        self.config.set("create ensemble settings", "model_to_run", self.algorithm)
        self.config.set(
            "create ensemble settings", "RF_n_estimators", str(self.n_estimators)
        )
        self.config.set(
            "create ensemble settings", "RF_max_features", str(self.max_features)
        )
        self.config.set("create ensemble settings", "RF_criterion", self.criterion)
        self.config.set(
            "create ensemble settings", "train_test_size", str(self.test_size)
        )
        self.config.set(
            "create ensemble settings",
            "train_test_split_type",
            str(self.train_test_type),
        )
        self.config.set(
            "create ensemble settings", "RF_min_sample_leaf", str(self.min_sample_leaf)
        )
        self.config.set(
            "create ensemble settings",
            "under_sample_ratio",
            str(self.under_sample_ratio),
        )
        self.config.set(
            "create ensemble settings",
            "under_sample_setting",
            str(self.under_sample_setting),
        )
        self.config.set(
            "create ensemble settings", "over_sample_ratio", str(self.over_sample_ratio)
        )
        self.config.set(
            "create ensemble settings",
            "over_sample_setting",
            str(self.over_sample_setting),
        )
        self.config.set("create ensemble settings", "classifier", self.behavior_name)
        self.config.set(
            "create ensemble settings", "RF_meta_data", str(self.meta_info_file)
        )
        self.config.set(
            "create ensemble settings",
            "generate_example_decision_tree",
            str(self.example_graphviz),
        )
        self.config.set(
            "create ensemble settings",
            "generate_classification_report",
            str(self.clf_report),
        )
        self.config.set(
            "create ensemble settings",
            "generate_features_importance_log",
            str(self.clf_importance_bars),
        )
        self.config.set(
            "create ensemble settings",
            "generate_features_importance_bar_graph",
            str(self.clf_importance_bars),
        )
        self.config.set(
            "create ensemble settings",
            "N_feature_importance_bars",
            str(self.clf_importance_bars_n),
        )
        self.config.set(
            "create ensemble settings",
            "compute_permutation_importance",
            str(self.permutation_importances),
        )
        self.config.set(
            "create ensemble settings",
            "generate_learning_curve",
            str(self.learning_curve),
        )
        self.config.set(
            "create ensemble settings",
            "generate_precision_recall_curve",
            str(self.pr_curve),
        )
        self.config.set(
            "create ensemble settings",
            "LearningCurve_shuffle_k_splits",
            str(self.learning_curve_k_split),
        )
        self.config.set(
            "create ensemble settings",
            "LearningCurve_shuffle_data_splits",
            str(self.learning_curve_data_split),
        )
        self.config.set(
            "create ensemble settings",
            "generate_example_decision_tree_fancy",
            str(self.example_dtreeviz),
        )
        self.config.set(
            "create ensemble settings", "generate_shap_scores", str(self.shap_scores)
        )
        self.config.set(
            "create ensemble settings",
            "shap_target_present_no",
            str(self.shap_scores_present),
        )
        self.config.set(
            "create ensemble settings",
            "shap_target_absent_no",
            str(self.shap_scores_absent),
        )
        self.config.set(
            "create ensemble settings", "shap_save_iteration", str(self.shap_save_it)
        )
        self.config.set(
            "create ensemble settings",
            "partial_dependency",
            str(self.partial_dependency),
        )
        self.config.set(
            "create ensemble settings", "class_weights", str(self.class_weight_method)
        )
        self.config.set(
            "create ensemble settings", "custom_weights", str(self.class_custom_weights)
        )

        with open(self.config_path, "w") as f:
            self.config.write(f)

        stdout_success(
            msg="Global model settings saved in the project_folder/project_config.ini"
        )

    def save_config(self):
        self.__checks()
        self.__get_variables()

        meta = {
            "RF_n_estimators": self.n_estimators,
            "RF_max_features": self.max_features,
            "RF_criterion": self.criterion,
            "train_test_size": self.test_size,
            "train_test_split_type": self.train_test_type,
            "RF_min_sample_leaf": self.min_sample_leaf,
            "under_sample_ratio": self.under_sample_ratio,
            "under_sample_setting": self.under_sample_setting,
            "over_sample_ratio": self.over_sample_ratio,
            "over_sample_setting": self.over_sample_setting,
            "generate_rf_model_meta_data_file": self.meta_info_file,
            "generate_example_decision_tree": self.example_graphviz,
            "generate_classification_report": self.clf_report,
            "generate_features_importance_log": self.clf_importance_bars,
            "generate_features_importance_bar_graph": self.clf_importance_bars,
            "n_feature_importance_bars": self.clf_importance_bars_n,
            "compute_feature_permutation_importance": self.permutation_importances,
            "generate_sklearn_learning_curves": self.learning_curve,
            "generate_precision_recall_curves": self.pr_curve,
            "learning_curve_k_splits": self.learning_curve_k_split,
            "learning_curve_data_splits": self.learning_curve_data_split,
            "generate_shap_scores": self.shap_scores,
            "shap_target_present_no": self.shap_scores_present,
            "shap_target_absent_no": self.shap_scores_absent,
            "shap_save_iteration": self.shap_save_it,
            "partial_dependency": self.partial_dependency,
            "class_weights": self.class_weight_method,
            "class_custom_weights": str(self.class_custom_weights),
        }

        meta_df = pd.DataFrame(meta, index=[0])
        meta_df.insert(0, "Classifier_name", self.behavior_name)
        self.find_meta_file_cnt()
        file_name = "{}_meta_{}.csv".format(self.behavior_name, str(self.meta_file_cnt))
        save_path = os.path.join(self.configs_meta_dir, file_name)
        meta_df.to_csv(save_path, index=FALSE)
        stdout_success(
            msg=f"Hyper-parameter config saved ({str(len(self.total_meta_files) + 1)} saved in project_folder/configs folder)."
        )

    def clear_cache(self):
        self.behavior_name = self.behavior_name_dropdown.getChoices()
        self.find_meta_file_cnt()
        for file_path in self.total_meta_files:
            os.remove(os.path.join(file_path))
            print("Deleted hyperparameters config {} ...".format(file_path))
        stdout_trash(msg=f"{str(len(self.total_meta_files))} config files deleted")

    def check_meta_data_integrity(self):
        self.meta = {k.lower(): v for k, v in self.meta.items()}
        for i in MachineLearningMetaKeys:
            if i not in self.meta.keys():
                stdout_warning(
                    msg=f"The file does not contain an expected entry for {i} parameter"
                )
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
            raise InvalidHyperparametersFileError(
                msg="SIMBA ERROR: {} is not a valid SimBA meta hyper-parameters file.".format(
                    config_name
                )
            )
        self.meta = {}
        for m in meta_df.columns:
            self.meta[m] = meta_df[m][0]
        self.check_meta_data_integrity()
        self.behavior_name_dropdown.setChoices(self.meta[MachineLearningMetaKeys.CLASSIFIER])
        self.estimators_entrybox.entry_set(val=self.meta[MachineLearningMetaKeys.RF_ESTIMATORS])
        self.max_features_dropdown.setChoices(self.meta[MachineLearningMetaKeys.RF_MAX_FEATURES])
        self.criterion_dropdown.setChoices(self.meta[MachineLearningMetaKeys.RF_CRITERION])
        self.train_test_size_dropdown.setChoices(self.meta[MachineLearningMetaKeys.TT_SIZE])
        self.min_sample_leaf_eb.entry_set(val=self.meta[MachineLearningMetaKeys.MIN_LEAF])
        self.max_depth_eb.entry_set(val=self.meta[MachineLearningMetaKeys.RF_MAX_DEPTH])
        self.undersample_settings_dropdown.setChoices(self.meta[MachineLearningMetaKeys.UNDER_SAMPLE_SETTING])
        if self.undersample_settings_dropdown.getChoices() != "None":
            self.under_sample_ratio_entrybox.entry_set(
                val=self.meta[MachineLearningMetaKeys.UNDER_SAMPLE_RATIO]
            )
            self.under_sample_ratio_entrybox.set_state(NORMAL)
        else:
            self.under_sample_ratio_entrybox.set_state(DISABLED)
        self.oversample_settings_dropdown.setChoices(self.meta[MachineLearningMetaKeys.OVER_SAMPLE_SETTING])
        if self.oversample_settings_dropdown.getChoices() != "None":
            self.over_sample_ratio_entrybox.entry_set(
                val=self.meta[MachineLearningMetaKeys.OVER_SAMPLE_RATIO]
            )
            self.over_sample_ratio_entrybox.set_state(NORMAL)
        else:
            self.over_sample_ratio_entrybox.set_state(DISABLED)

        if self.meta[MachineLearningMetaKeys.RF_METADATA]:
            self.create_meta_data_file_var.set(value=True)
        else:
            self.create_meta_data_file_var.set(value=False)
        if self.meta[MachineLearningMetaKeys.EX_DECISION_TREE]:
            self.create_example_decision_tree_graphviz_var.set(value=True)
        else:
            self.create_example_decision_tree_graphviz_var.set(value=False)
        if self.meta[MachineLearningMetaKeys.CLF_REPORT]:
            self.create_clf_report_var.set(value=True)
        else:
            self.create_clf_report_var.set(value=False)
        if (
                self.meta[MachineLearningMetaKeys.IMPORTANCE_LOG]
                or self.meta[MachineLearningMetaKeys.IMPORTANCE_BAR_CHART]
        ):
            self.create_clf_importance_bars_var.set(value=True)
            self.n_features_bars_entry_box.set_state(NORMAL)
            self.n_features_bars_entry_box.entry_set(
                val=self.meta[MachineLearningMetaKeys.N_FEATURE_IMPORTANCE_BARS]
            )
        else:
            self.create_clf_importance_bars_var.set(value=False)
            self.n_features_bars_entry_box.set_state(DISABLED)

        if self.meta[MachineLearningMetaKeys.PERMUTATION_IMPORTANCE]:
            self.feature_permutation_importance_var.set(value=True)

        if self.meta[MachineLearningMetaKeys.LEARNING_CURVE]:
            self.learning_curve_var.set(value=True)
            self.learning_curve_k_splits_entry_box.set_state(NORMAL)
            self.learning_curve_data_splits_entry_box.set_state(NORMAL)
            self.learning_curve_k_splits_entry_box.entry_set(
                val=self.meta[MachineLearningMetaKeys.LEARNING_CURVE_K_SPLITS]
            )
            self.learning_curve_data_splits_entry_box.entry_set(
                val=self.meta[MachineLearningMetaKeys.LEARNING_CURVE_DATA_SPLITS]
            )
        else:
            self.learning_curve_var.set(value=False)
            self.learning_curve_k_splits_entry_box.set_state(DISABLED)
            self.learning_curve_data_splits_entry_box.set_state(DISABLED)

        if self.meta[MachineLearningMetaKeys.SHAP_SCORES]:
            self.calc_shap_scores_var.set(value=True)
            self.shap_present.set_state(NORMAL)
            self.shap_absent.set_state(NORMAL)
            self.shap_absent.set_state(NORMAL)
            self.shap_save_it_dropdown.enable()
            self.shap_present.entry_set(val=self.meta[MachineLearningMetaKeys.SHAP_PRESENT])
            self.shap_absent.entry_set(val=self.meta[MachineLearningMetaKeys.SHAP_ABSENT])
            if "shap_save_iteration" in self.meta.keys():
                self.shap_save_it_dropdown.setChoices(self.meta[MachineLearningMetaKeys.SHAP_SAVE_ITERATION])
            else:
                self.shap_save_it_dropdown.setChoices("ALL FRAMES")
        else:
            self.calc_shap_scores_var.set(value=False)
            self.shap_present.set_state(DISABLED)
            self.shap_absent.set_state(DISABLED)
            self.shap_save_it_dropdown.enable()

        if MachineLearningMetaKeys.TRAIN_TEST_SPLIT_TYPE in self.meta.keys():
            self.train_test_type_dropdown.setChoices(self.meta[MachineLearningMetaKeys.TT_SIZE])
        else:
            self.train_test_type_dropdown.setChoices(Options.TRAIN_TEST_SPLIT.value[0])
        if MachineLearningMetaKeys.PARTIAL_DEPENDENCY in self.meta.keys():
            if self.meta[MachineLearningMetaKeys.PARTIAL_DEPENDENCY] in Options.RUN_OPTIONS_FLAGS.value:
                self.partial_dependency_var.set(value=True)
        else:
            self.shap_save_it_dropdown.setChoices("None")
        if MachineLearningMetaKeys.CLASS_WEIGHTS in self.meta.keys():
            if self.meta[MachineLearningMetaKeys.CLASS_WEIGHTS] not in Options.CLASS_WEIGHT_OPTIONS.value:
                self.meta[MachineLearningMetaKeys.CLASS_WEIGHTS] = "None"
            self.class_weights_dropdown.setChoices(self.meta[MachineLearningMetaKeys.CLASS_WEIGHTS])
            if self.meta[MachineLearningMetaKeys.CLASS_WEIGHTS] == "custom":
                self.create_class_weight_table()
                weights = ast.literal_eval(self.meta[MachineLearningMetaKeys.CLASS_CUSTOM_WEIGHTS])

                self.weight_present.setChoices(weights[1])
                self.weight_absent.setChoices(weights[0])
        else:
            self.class_weights_dropdown.setChoices("None")
            self.create_class_weight_table()

        print("Loaded parameters from config {}...".format(config_name))

    def get_expected_meta_dict_entry_keys(self):
        self.expected_meta_dict_entries = [
            "classifier_name",
            "rf_n_estimators",
            "rf_max_features",
            "rf_criterion",
            "train_test_size",
            "rf_min_sample_leaf",
            "under_sample_ratio",
            "under_sample_setting",
            "over_sample_ratio",
            "over_sample_setting",
            "generate_rf_model_meta_data_file",
            "generate_example_decision_tree",
            "generate_classification_report",
            "generate_features_importance_log",
            "generate_features_importance_bar_graph",
            "n_feature_importance_bars",
            "compute_feature_permutation_importance",
            "generate_sklearn_learning_curves",
            "generate_precision_recall_curves",
            "learning_curve_k_splits",
            "learning_curve_data_splits",
            "generate_example_decision_tree_fancy",
            "generate_shap_scores",
            "shap_target_present_no",
            "shap_target_absent_no",
            "shap_save_iteration",
            "partial_dependency",
            "class_weights",
            "class_custom_weights",
        ]

# _ = MachineModelSettingsPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')

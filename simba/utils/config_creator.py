__author__ = "Simon Nilsson"

import csv
import os
import platform
from configparser import ConfigParser
from typing import List

import simba
from simba.utils.enums import ConfigKey, DirNames, Dtypes, MLParamKeys, Paths
from simba.utils.errors import DirectoryExistError
from simba.utils.printing import SimbaTimer, stdout_success


class ProjectConfigCreator(object):
    """
    Create SimBA project directory tree and associated project_config.ini config file.

    :parameter str project_path: path to directory where to save the SimBA project directory tree
    :parameter str project_name: Name of the SimBA project
    :parameter List[str] target_list: Classifier names in the SimBA project
    :parameter str pose_estimation_bp_cnt: String representing the number of body-parts in the pose-estimation data used in the simba project.
                                           E.g., '4', '7', '8', '9', '14', '16' or 'user_defined', '3D_user_defined'.
    :parameter int body_part_config_idx: The index of the SimBA GUI dropdown pose-estimation selection. E.g., ``1``. I.e., the row representing
                                         your pose-estimated body-parts in `this file <https://github.com/sgoldenlab/simba/blob/master/simba/pose_configurations/bp_names/bp_names.csv>`_.
    :parameter int animal_cnt: Number of animals tracked in the input pose-estimation data.
    :parameter str file_type: The SimBA project file type. OPTIONS: ``csv`` or ``parquet``.

    .. note::
       `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#part-1-create-a-new-project-1>`__.

    Examples
    ----------
    >>> _ = ProjectConfigCreator(project_path = 'project/path', project_name='project_name', target_list=['Attack'], pose_estimation_bp_cnt='16', body_part_config_idx=9, animal_cnt=2, file_type='csv')

    """

    def __init__(self,
                  project_path: str,
                  project_name: str,
                  target_list: List[str],
                  pose_estimation_bp_cnt: str,
                  body_part_config_idx: int,
                  animal_cnt: int,
                  file_type: str = "csv"):


        self.simba_dir = os.path.dirname(simba.__file__)
        self.animal_cnt = animal_cnt
        self.os_platform = platform.system()
        self.project_path = project_path
        self.project_name = project_name
        self.target_list = target_list
        self.pose_estimation_bp_cnt = pose_estimation_bp_cnt
        self.body_part_config_idx = body_part_config_idx
        self.file_type = file_type
        self.timer = SimbaTimer(start=True)
        self.__create_directories()
        self.__create_configparser_config()

    def __create_directories(self):
        self.project_folder = os.path.join(
            self.project_path, self.project_name, DirNames.PROJECT.value
        )
        self.models_folder = os.path.join(
            self.project_path, self.project_name, DirNames.MODEL.value
        )
        self.config_folder = os.path.join(self.project_folder, DirNames.CONFIGS.value)
        self.csv_folder = os.path.join(self.project_folder, DirNames.CSV.value)
        self.frames_folder = os.path.join(self.project_folder, DirNames.FRAMES.value)
        self.logs_folder = os.path.join(self.project_folder, DirNames.LOGS.value)
        self.measures_folder = os.path.join(self.logs_folder, DirNames.MEASURES.value)
        self.pose_configs_folder = os.path.join(
            self.measures_folder, DirNames.POSE_CONFIGS.value
        )
        self.bp_names_folder = os.path.join(
            self.pose_configs_folder, DirNames.BP_NAMES.value
        )
        self.videos_folder = os.path.join(self.project_folder, DirNames.VIDEOS.value)
        self.features_extracted_folder = os.path.join(
            self.csv_folder, DirNames.FEATURES_EXTRACTED.value
        )
        self.input_csv_folder = os.path.join(self.csv_folder, DirNames.INPUT_CSV.value)
        self.machine_results_folder = os.path.join(
            self.csv_folder, DirNames.MACHINE_RESULTS.value
        )
        self.outlier_corrected_movement_folder = os.path.join(
            self.csv_folder, DirNames.OUTLIER_MOVEMENT.value
        )
        self.outlier_corrected_location_folder = os.path.join(
            self.csv_folder, DirNames.OUTLIER_MOVEMENT_LOCATION.value
        )
        self.targets_inserted_folder = os.path.join(
            self.csv_folder, DirNames.TARGETS_INSERTED.value
        )
        self.input_folder = os.path.join(self.frames_folder, DirNames.INPUT.value)
        self.output_folder = os.path.join(self.frames_folder, DirNames.OUTPUT.value)

        folder_lst = [
            self.project_folder,
            self.models_folder,
            self.config_folder,
            self.csv_folder,
            self.frames_folder,
            self.logs_folder,
            self.videos_folder,
            self.features_extracted_folder,
            self.input_csv_folder,
            self.machine_results_folder,
            self.outlier_corrected_movement_folder,
            self.outlier_corrected_location_folder,
            self.targets_inserted_folder,
            self.input_folder,
            self.output_folder,
            self.measures_folder,
            self.pose_configs_folder,
            self.bp_names_folder,
        ]

        for folder_path in folder_lst:
            if os.path.isdir(folder_path):
                raise DirectoryExistError(
                    msg=f"SimBA tried to create {folder_path}, but it already exists. Please create your SimBA project in a new path, or move/delete your previous SimBA project"
                )
            else:
                os.makedirs(folder_path)

    def __create_configparser_config(self):
        self.config = ConfigParser(allow_no_value=True)
        self.config.add_section(ConfigKey.GENERAL_SETTINGS.value)
        self.config[ConfigKey.GENERAL_SETTINGS.value][
            ConfigKey.PROJECT_PATH.value
        ] = self.project_folder
        self.config[ConfigKey.GENERAL_SETTINGS.value][
            ConfigKey.PROJECT_NAME.value
        ] = self.project_name
        self.config[ConfigKey.GENERAL_SETTINGS.value][
            ConfigKey.FILE_TYPE.value
        ] = self.file_type
        self.config[ConfigKey.GENERAL_SETTINGS.value][ConfigKey.ANIMAL_CNT.value] = str(
            self.animal_cnt
        )
        self.config[ConfigKey.GENERAL_SETTINGS.value][
            ConfigKey.OS.value
        ] = self.os_platform

        self.config.add_section(ConfigKey.SML_SETTINGS.value)
        self.config[ConfigKey.SML_SETTINGS.value][
            ConfigKey.MODEL_DIR.value
        ] = self.models_folder
        for clf_cnt in range(len(self.target_list)):
            self.config[ConfigKey.SML_SETTINGS.value][
                "model_path_{}".format(str(clf_cnt + 1))
            ] = os.path.join(
                self.models_folder, str(self.target_list[clf_cnt]) + ".sav"
            )

        self.config[ConfigKey.SML_SETTINGS.value][ConfigKey.TARGET_CNT.value] = str(
            len(self.target_list)
        )
        for clf_cnt in range(len(self.target_list)):
            self.config[ConfigKey.SML_SETTINGS.value][
                "target_name_{}".format(str(clf_cnt + 1))
            ] = str(self.target_list[clf_cnt])

        self.config.add_section(ConfigKey.THRESHOLD_SETTINGS.value)
        for clf_cnt in range(len(self.target_list)):
            self.config[ConfigKey.THRESHOLD_SETTINGS.value][
                "threshold_{}".format(str(clf_cnt + 1))
            ] = Dtypes.NONE.value
        self.config[ConfigKey.THRESHOLD_SETTINGS.value][
            ConfigKey.SKLEARN_BP_PROB_THRESH.value
        ] = str(0.00)

        self.config.add_section(ConfigKey.MIN_BOUT_LENGTH.value)
        for clf_cnt in range(len(self.target_list)):
            self.config[ConfigKey.MIN_BOUT_LENGTH.value][
                "min_bout_{}".format(str(clf_cnt + 1))
            ] = Dtypes.NONE.value

        self.config.add_section(ConfigKey.FRAME_SETTINGS.value)
        self.config[ConfigKey.FRAME_SETTINGS.value][ConfigKey.DISTANCE_MM.value] = 0.00
        self.config.add_section(ConfigKey.LINE_PLOT_SETTINGS.value)
        self.config.add_section(ConfigKey.PATH_PLOT_SETTINGS.value)
        self.config.add_section(ConfigKey.ROI_SETTINGS.value)
        self.config.add_section(ConfigKey.DIRECTIONALITY_SETTINGS.value)
        self.config.add_section(ConfigKey.PROCESS_MOVEMENT_SETTINGS.value)

        self.config.add_section(ConfigKey.CREATE_ENSEMBLE_SETTINGS.value)
        self.config[ConfigKey.CREATE_ENSEMBLE_SETTINGS.value][
            ConfigKey.POSE_SETTING.value
        ] = str(self.pose_estimation_bp_cnt)
        self.config[ConfigKey.CREATE_ENSEMBLE_SETTINGS.value][
            MLParamKeys.CLASSIFIER.value
        ] = Dtypes.NONE.value
        self.config[ConfigKey.CREATE_ENSEMBLE_SETTINGS.value][
            MLParamKeys.TT_SIZE.value
        ] = str(0.20)
        self.config[ConfigKey.CREATE_ENSEMBLE_SETTINGS.value][
            MLParamKeys.UNDERSAMPLE_SETTING.value
        ] = Dtypes.NONE.value
        self.config[ConfigKey.CREATE_ENSEMBLE_SETTINGS.value][
            MLParamKeys.UNDERSAMPLE_RATIO.value
        ] = Dtypes.NONE.value
        self.config[ConfigKey.CREATE_ENSEMBLE_SETTINGS.value][
            MLParamKeys.OVERSAMPLE_SETTING.value
        ] = Dtypes.NONE.value
        self.config[ConfigKey.CREATE_ENSEMBLE_SETTINGS.value][
            MLParamKeys.OVERSAMPLE_RATIO.value
        ] = Dtypes.NONE.value
        self.config[ConfigKey.CREATE_ENSEMBLE_SETTINGS.value][
            MLParamKeys.RF_ESTIMATORS.value
        ] = str(2000)
        self.config[ConfigKey.CREATE_ENSEMBLE_SETTINGS.value][
            MLParamKeys.MIN_LEAF.value
        ] = str(1)
        self.config[ConfigKey.CREATE_ENSEMBLE_SETTINGS.value][
            MLParamKeys.RF_MAX_FEATURES.value
        ] = Dtypes.SQRT.value
        self.config[ConfigKey.CREATE_ENSEMBLE_SETTINGS.value][
            ConfigKey.RF_JOBS.value
        ] = str(-1)
        self.config[ConfigKey.CREATE_ENSEMBLE_SETTINGS.value][
            MLParamKeys.RF_CRITERION.value
        ] = Dtypes.ENTROPY.value
        self.config[ConfigKey.CREATE_ENSEMBLE_SETTINGS.value][
            MLParamKeys.RF_METADATA.value
        ] = Dtypes.NONE.value
        self.config[ConfigKey.CREATE_ENSEMBLE_SETTINGS.value][
            MLParamKeys.EX_DECISION_TREE.value
        ] = Dtypes.NONE.value
        self.config[ConfigKey.CREATE_ENSEMBLE_SETTINGS.value][
            MLParamKeys.EX_DECISION_TREE_FANCY.value
        ] = Dtypes.NONE.value
        self.config[ConfigKey.CREATE_ENSEMBLE_SETTINGS.value][
            MLParamKeys.IMPORTANCE_LOG.value
        ] = Dtypes.NONE.value
        self.config[ConfigKey.CREATE_ENSEMBLE_SETTINGS.value][
            MLParamKeys.IMPORTANCE_BAR_CHART.value
        ] = Dtypes.NONE.value
        self.config[ConfigKey.CREATE_ENSEMBLE_SETTINGS.value][
            MLParamKeys.PERMUTATION_IMPORTANCE.value
        ] = Dtypes.NONE.value
        self.config[ConfigKey.CREATE_ENSEMBLE_SETTINGS.value][
            MLParamKeys.LEARNING_CURVE.value
        ] = Dtypes.NONE.value
        self.config[ConfigKey.CREATE_ENSEMBLE_SETTINGS.value][
            MLParamKeys.PRECISION_RECALL.value
        ] = Dtypes.NONE.value
        self.config[ConfigKey.CREATE_ENSEMBLE_SETTINGS.value][
            MLParamKeys.IMPORTANCE_BARS_N.value
        ] = Dtypes.NONE.value
        self.config[ConfigKey.CREATE_ENSEMBLE_SETTINGS.value][
            MLParamKeys.LEARNING_CURVE_K_SPLITS.value
        ] = Dtypes.NONE.value
        self.config[ConfigKey.CREATE_ENSEMBLE_SETTINGS.value][
            MLParamKeys.LEARNING_DATA_SPLITS.value
        ] = Dtypes.NONE.value

        self.config.add_section(ConfigKey.MULTI_ANIMAL_ID_SETTING.value)
        self.config[ConfigKey.MULTI_ANIMAL_ID_SETTING.value][
            ConfigKey.MULTI_ANIMAL_IDS.value
        ] = Dtypes.NONE.value

        self.config.add_section(ConfigKey.OUTLIER_SETTINGS.value)
        self.config[ConfigKey.OUTLIER_SETTINGS.value][
            ConfigKey.MOVEMENT_CRITERION.value
        ] = Dtypes.NONE.value
        self.config[ConfigKey.OUTLIER_SETTINGS.value][
            ConfigKey.LOCATION_CRITERION.value
        ] = Dtypes.NONE.value

        self.config_path = os.path.join(self.project_folder, "project_config.ini")
        with open(self.config_path, "w") as file:
            self.config.write(file)

        bp_dir_path = os.path.join(self.simba_dir, Paths.SIMBA_BP_CONFIG_PATH.value)
        with open(bp_dir_path, "r", encoding="utf8") as f:
            cr = csv.reader(f, delimiter=",")  # , is default
            rows = list(cr)  # create a list of rows for instance

        chosen_bps = rows[self.body_part_config_idx]
        chosen_bps = list(filter(None, chosen_bps))

        project_bp_file_path = os.path.join(
            self.bp_names_folder, "project_bp_names.csv"
        )
        f = open(project_bp_file_path, "w+")
        for i in chosen_bps:
            f.write(i + "\n")
        f.close()
        self.timer.stop_timer()

        stdout_success(
            msg=f"Project directory tree and project_config.ini created in {self.project_folder}",
            elapsed_time=self.timer.elapsed_time_str,
        )

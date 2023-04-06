from simba.read_config_unit_tests import (read_config_file,
                                          read_project_path_and_file_type,
                                          read_config_entry)
from simba.feature_extractors.unit_tests import read_video_info_csv
from simba.drop_bp_cords import createColorListofList
from simba.misc_tools import SimbaTimer, find_core_cnt, check_multi_animal_status
from simba.train_model_functions import get_all_clf_names
from simba.drop_bp_cords import getBpNames, create_body_part_dictionary, getBpHeaders
from simba.utils.errors import NoROIDataError
import os, glob
import pandas as pd
import itertools
import cv2
import numpy as np
from simba.enums import Paths, ReadConfig, Dtypes, Defaults, Keys
from datetime import datetime

class ConfigReader(object):
    def __init__(self,
                 config_path: str,
                 read_video_info: bool=True):

        self.timer = SimbaTimer()
        self.timer.start_timer()
        self.config = read_config_file(ini_path=config_path)
        self.config_path = config_path
        self.datetime = datetime.now().strftime('%Y%m%d%H%M%S')
        self.project_path, self.file_type = read_project_path_and_file_type(config=self.config)
        self.input_csv_dir = os.path.join(self.project_path, Paths.INPUT_CSV.value)
        self.features_dir = os.path.join(self.project_path, Paths.FEATURES_EXTRACTED_DIR.value)
        self.targets_folder = os.path.join(self.project_path, Paths.TARGETS_INSERTED_DIR.value)
        self.machine_results_dir = os.path.join(self.project_path, Paths.MACHINE_RESULTS_DIR.value)
        self.directionality_df_dir = os.path.join(self.project_path, Paths.DIRECTIONALITY_DF_DIR.value)
        self.outlier_corrected_dir = os.path.join(self.project_path, Paths.OUTLIER_CORRECTED.value)
        self.outlier_corrected_movement_dir = os.path.join(self.project_path, Paths.OUTLIER_CORRECTED_MOVEMENT.value)
        self.heatmap_clf_location_dir = os.path.join(self.project_path, Paths.HEATMAP_CLF_LOCATION_DIR.value)
        self.heatmap_location_dir = os.path.join(self.project_path, Paths.HEATMAP_LOCATION_DIR.value)
        self.line_plot_dir = os.path.join(self.project_path, Paths.LINE_PLOT_DIR.value)
        self.probability_plot_dir = os.path.join(self.project_path, Paths.PROBABILITY_PLOTS_DIR.value)
        self.gantt_plot_dir = os.path.join(self.project_path, Paths.GANTT_PLOT_DIR.value)
        self.path_plot_dir = os.path.join(self.project_path, Paths.PATH_PLOT_DIR.value)
        self.font = cv2.FONT_HERSHEY_COMPLEX
        self.roi_features_save_dir = os.path.join(self.project_path, Paths.ROI_FEATURES.value)
        self.sklearn_plot_dir = os.path.join(self.project_path, Paths.SKLEARN_RESULTS.value)
        self.detailed_roi_data_dir = os.path.join(self.project_path, Paths.DETAILED_ROI_DATA_DIR.value)
        self.directing_animals_video_output_path = os.path.join(self.project_path, Paths.DIRECTING_BETWEEN_ANIMALS_OUTPUT_PATH.value)
        self.animal_cnt = read_config_entry(config=self.config, section=ReadConfig.GENERAL_SETTINGS.value, option=ReadConfig.ANIMAL_CNT.value, data_type=Dtypes.INT.value)
        self.clf_cnt = read_config_entry(self.config, ReadConfig.SML_SETTINGS.value, ReadConfig.TARGET_CNT.value, Dtypes.INT.value)
        self.clf_names = get_all_clf_names(config=self.config, target_cnt=self.clf_cnt)
        self.feature_file_paths = glob.glob(self.features_dir + '/*.' + self.file_type)
        self.target_file_paths = glob.glob(self.targets_folder + '/*.' + self.file_type)
        self.outlier_corrected_paths = glob.glob(self.outlier_corrected_dir + '/*.' + self.file_type)
        self.cpu_cnt, self.cpu_to_use = find_core_cnt()
        self.machine_results_paths = glob.glob(self.machine_results_dir + '/*.' + self.file_type)
        self.logs_path = os.path.join(self.project_path, 'logs')
        self.body_parts_path = os.path.join(self.project_path, Paths.BP_NAMES.value)
        self.pose_setting = read_config_entry(config=self.config, section=ReadConfig.CREATE_ENSEMBLE_SETTINGS.value, option=ReadConfig.POSE_SETTING.value, data_type=Dtypes.STR.value)
        self.roi_coordinates_path = os.path.join(self.logs_path, Paths.ROI_DEFINITIONS.value)
        self.video_dir = os.path.join(self.project_path, 'videos')
        self.clf_validation_dir = os.path.join(self.project_path, Paths.CLF_VALIDATION_DIR.value)
        self.clf_data_validation_dir = os.path.join(self.project_path, 'csv', 'validation')
        self.annotated_frm_dir = os.path.join(self.project_path, Paths.ANNOTATED_FRAMES_DIR.value)
        self.single_validation_video_save_dir = os.path.join(self.project_path, Paths.SINGLE_CLF_VALIDATION.value)
        self.data_table_path = os.path.join(self.project_path, Paths.DATA_TABLE.value)
        self.multi_animal_status, self.multi_animal_id_list = check_multi_animal_status(self.config, self.animal_cnt)
        self.x_cols, self.y_cols, self.p_cols = getBpNames(config_path)
        self.column_headers = getBpHeaders(config_path)
        self.multiprocess_chunksize = Defaults.CHUNK_SIZE.value
        self.maxtasksperchild = Defaults.MAX_TASK_PER_CHILD.value
        self.clr_lst = createColorListofList(self.animal_cnt, int(len(self.x_cols)/self.animal_cnt) + 1)
        self.animal_bp_dict = create_body_part_dictionary(self.multi_animal_status, self.multi_animal_id_list, self.animal_cnt, self.x_cols, self.y_cols, self.p_cols, [])
        self.project_bps = list(set([x[:-2] for x in self.column_headers]))
        if read_video_info:
            self.video_info_df = read_video_info_csv(os.path.join(self.project_path, Paths.VIDEO_INFO.value))


    def read_roi_data(self):
        """
        Method to read in ROI definitions from SimBA project

        Returns
        -------
        roi_df: dict
        rectangles_df: pd.DataFrame
        circles_df: pd.DataFrame
        polygon_df: pd.DataFrame
        shape_names: list
        roi_dict: dict
        """
        if not os.path.isfile(self.roi_coordinates_path):
            raise NoROIDataError(msg='SIMBA ERROR: No ROI definitions were found in your SimBA project. Please draw some ROIs before analyzing your ROI data')
        else:
            self.rectangles_df = pd.read_hdf(self.roi_coordinates_path, key=Keys.ROI_RECTANGLES.value).dropna(how='any')
            self.circles_df = pd.read_hdf(self.roi_coordinates_path, key=Keys.ROI_CIRCLES.value).dropna(how='any')
            self.polygon_df = pd.read_hdf(self.roi_coordinates_path, key=Keys.ROI_POLYGONS.value).dropna(how='any')
            self.shape_names = list(itertools.chain(self.rectangles_df['Name'].unique(), self.circles_df['Name'].unique(), self.polygon_df['Name'].unique()))
            self.roi_dict = {Keys.ROI_RECTANGLES.value: self.rectangles_df,
                             Keys.ROI_CIRCLES.value: self.circles_df,
                             Keys.ROI_POLYGONS.value: self.polygon_df}
            for shape_type, shape_data in self.roi_dict.items():
                if shape_type == Keys.ROI_CIRCLES.value:
                    self.roi_dict[Keys.ROI_CIRCLES.value]['Center_X'] = self.roi_dict[Keys.ROI_CIRCLES.value]['centerX']
                    self.roi_dict[Keys.ROI_CIRCLES.value]['Center_Y'] = self.roi_dict[Keys.ROI_CIRCLES.value]['centerY']
                elif shape_type == Keys.ROI_RECTANGLES.value:
                    self.roi_dict[Keys.ROI_RECTANGLES.value]['Center_X'] = self.roi_dict[Keys.ROI_RECTANGLES.value]['Bottom_right_X'] - ((self.roi_dict[Keys.ROI_RECTANGLES.value]['Bottom_right_X'] - self.roi_dict[Keys.ROI_RECTANGLES.value]['width']) / 2)
                    self.roi_dict[Keys.ROI_RECTANGLES.value]['Center_Y'] = self.roi_dict[Keys.ROI_RECTANGLES.value]['Bottom_right_Y'] - ((self.roi_dict[Keys.ROI_RECTANGLES.value]['Bottom_right_Y'] - self.roi_dict[Keys.ROI_RECTANGLES.value]['height']) / 2)
                elif shape_type == Keys.ROI_POLYGONS.value:
                    self.roi_dict[Keys.ROI_POLYGONS.value]['Center_X'] = self.roi_dict[Keys.ROI_POLYGONS.value]['Center_X']
                    self.roi_dict[Keys.ROI_POLYGONS.value]['Center_Y'] = self.roi_dict[Keys.ROI_POLYGONS.value]['Center_Y']




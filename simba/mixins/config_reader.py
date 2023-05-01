__author__ = "Simon Nilsson"


from datetime import datetime
import shutil
import logging
from configparser import ConfigParser
import os, glob
import pandas as pd
import itertools
import cv2
from simba.utils.enums import (Paths,
                               ConfigKey,
                               Dtypes,
                               Defaults,
                               Keys)
from simba.utils.lookups import get_emojis, get_color_dict
from simba.utils.data import create_color_palettes
from simba.utils.errors import (NoROIDataError,
                                DataHeaderError,
                                NoFilesFoundError,
                                MissingProjectConfigEntryError,
                                NotDirectoryError,
                                InvalidInputError,
                                DuplicationError,
                                ParametersFileError)
from simba.utils.warnings import (NoFileFoundWarning,
                                  BodypartColumnNotFoundWarning,
                                  InvalidValueWarning)
from simba.utils.read_write import (read_project_path_and_file_type,
                                    get_fn_ext,
                                    SimbaTimer,
                                    read_config_file,
                                    find_core_cnt,
                                    get_all_clf_names)
from simba.utils.checks import check_file_exist_and_readable


class ConfigReader(object):
    def __init__(self,
                 config_path: str,
                 read_video_info: bool=True):

        """
        Methods for reading SimBA configparser.Configparser project config..

        :param configparser.Configparser config_path: path to SimBA project_config.ini
        :param bool read_video_info: if true, read the project_folder/logs/video_info.csv file.
        """

        self.timer = SimbaTimer(start=True)
        self.config_path = config_path
        self.config = read_config_file(config_path=config_path)
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
        self.shap_logs_path = os.path.join(self.project_path, Paths.SHAP_LOGS.value)
        self.video_info_path = os.path.join(self.project_path, Paths.VIDEO_INFO.value)
        self.font = cv2.FONT_HERSHEY_COMPLEX
        self.roi_features_save_dir = os.path.join(self.project_path, Paths.ROI_FEATURES.value)
        self.configs_meta_dir = os.path.join(self.project_path, 'configs')
        self.sklearn_plot_dir = os.path.join(self.project_path, Paths.SKLEARN_RESULTS.value)
        self.detailed_roi_data_dir = os.path.join(self.project_path, Paths.DETAILED_ROI_DATA_DIR.value)
        self.directing_animals_video_output_path = os.path.join(self.project_path, Paths.DIRECTING_BETWEEN_ANIMALS_OUTPUT_PATH.value)
        self.animal_cnt = self.read_config_entry(config=self.config, section=ConfigKey.GENERAL_SETTINGS.value, option=ConfigKey.ANIMAL_CNT.value, data_type=Dtypes.INT.value)
        self.clf_cnt = self.read_config_entry(self.config, ConfigKey.SML_SETTINGS.value, ConfigKey.TARGET_CNT.value, Dtypes.INT.value)
        self.clf_names = get_all_clf_names(config=self.config, target_cnt=self.clf_cnt)
        self.feature_file_paths = glob.glob(self.features_dir + '/*.' + self.file_type)
        self.target_file_paths = glob.glob(self.targets_folder + '/*.' + self.file_type)
        self.input_csv_paths = glob.glob(self.input_csv_dir + '/*.' + self.file_type)
        self.outlier_corrected_paths = glob.glob(self.outlier_corrected_dir + '/*.' + self.file_type)
        self.outlier_corrected_movement_paths = glob.glob(self.outlier_corrected_movement_dir + '/*.' + self.file_type)
        self.cpu_cnt, self.cpu_to_use = find_core_cnt()
        self.machine_results_paths = glob.glob(self.machine_results_dir + '/*.' + self.file_type)
        self.logs_path = os.path.join(self.project_path, 'logs')
        self.body_parts_path = os.path.join(self.project_path, Paths.BP_NAMES.value)
        check_file_exist_and_readable(file_path=self.body_parts_path)
        self.body_parts_lst = pd.read_csv(self.body_parts_path, header=None).iloc[:, 0].to_list()
        self.body_parts_lst = [x for x in self.body_parts_lst if str(x) != 'nan']
        self.get_body_part_names()
        self.get_bp_headers()
        self.bp_col_names = self.x_cols + self.y_cols + self.p_cols
        self.pose_setting = self.read_config_entry(config=self.config, section=ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, option=ConfigKey.POSE_SETTING.value, data_type=Dtypes.STR.value)
        self.roi_coordinates_path = os.path.join(self.logs_path, Paths.ROI_DEFINITIONS.value)
        self.video_dir = os.path.join(self.project_path, 'videos')
        self.clf_validation_dir = os.path.join(self.project_path, Paths.CLF_VALIDATION_DIR.value)
        self.clf_data_validation_dir = os.path.join(self.project_path, 'csv', 'validation')
        self.annotated_frm_dir = os.path.join(self.project_path, Paths.ANNOTATED_FRAMES_DIR.value)
        self.single_validation_video_save_dir = os.path.join(self.project_path, Paths.SINGLE_CLF_VALIDATION.value)
        self.data_table_path = os.path.join(self.project_path, Paths.DATA_TABLE.value)
        self.check_multi_animal_status()
        self.multiprocess_chunksize = Defaults.CHUNK_SIZE.value
        self.maxtasksperchild = Defaults.MAX_TASK_PER_CHILD.value
        self.clr_lst = create_color_palettes(self.animal_cnt, int(len(self.x_cols)/self.animal_cnt) + 1)
        self.animal_bp_dict = self.create_body_part_dictionary(self.multi_animal_status, self.multi_animal_id_list, self.animal_cnt, self.x_cols, self.y_cols, self.p_cols, self.clr_lst)
        self.project_bps = list(set([x[:-2] for x in self.bp_headers]))
        self.color_dict = get_color_dict()
        self.emojis = get_emojis()
        if read_video_info:
            self.video_info_df = self.read_video_info_csv(file_path=self.video_info_path)

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
            self.roi_types_names_lst = set()
            for idx, r in self.roi_dict[Keys.ROI_RECTANGLES.value].iterrows():
                self.roi_types_names_lst.add(f'Rectangle: {r["Name"]}')
            for idx, r in self.roi_dict[Keys.ROI_CIRCLES.value].iterrows():
                self.roi_types_names_lst.add(f'Circle: {r["Name"]}')
            for idx, r in self.roi_dict[Keys.ROI_POLYGONS.value].iterrows():
                self.roi_types_names_lst.add(f'Polygon: {r["Name"]}')
            self.roi_types_names_lst = list(self.roi_types_names_lst)
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

    def get_all_clf_names(self):
        model_names = []
        for i in range(self.clf_cnt):
            entry_name = 'target_name_{}'.format(str(i + 1))
            model_names.append(self.read_config_entry(self.config, ConfigKey.SML_SETTINGS.value, entry_name, data_type=Dtypes.STR.value))
        return model_names

    def insert_column_headers_for_outlier_correction(self,
                                                     data_df: pd.DataFrame,
                                                     new_headers: list,
                                                     filepath: str):
        """
        Helper to insert new column headers onto a dataframe.

        Parameters
        ----------
        data_df:  pd.DataFrame
            Dataframe where headers to to-bo replaced.
        new_headers: list
            Names of new headers.
        filepath: str
            Path to where ``data_df`` is stored on disk

        Returns
        -------
        data_df: pd.DataFrame
            Dataframe with new headers
        """

        if len(new_headers) != len(data_df.columns):
            difference = int(len(data_df.columns) - len(new_headers))
            bp_missing = int(abs(difference) / 3)
            if difference < 0:
                print(
                    'SIMBA ERROR: SimBA expects {} columns of data inside the files within project_folder/csv/input_csv directory. However, '
                    'within file {} file, SimBA found {} columns. Thus, there is {} missing data columns in the imported data, which may represent {} '
                    'bodyparts if each body-part has an x, y and p value. Either revise the SimBA project pose-configuration with {} less body-part, or '
                    'include {} more body-part in the imported data'.format(str(len(new_headers)), filepath,
                                                                            str(len(data_df.columns)),
                                                                            str(abs(difference)),
                                                                            str(int(bp_missing)), str(bp_missing),
                                                                            str(bp_missing)))
            else:
                print(
                    'SIMBA ERROR: SimBA expects {} columns of data inside the files within project_folder/csv/input_csv directory. However, '
                    'within file {} file, SimBA found {} columns. Thus, there is {} more data columns in the imported data than anticipated, which may represent {} '
                    'bodyparts if each body-part has an x, y and p value. Either revise the SimBA project pose-configuration with {} more body-part, or '
                    'include {} less body-part in the imported data'.format(str(len(new_headers)), filepath,
                                                                            str(len(data_df.columns)),
                                                                            str(abs(difference)),
                                                                            str(int(bp_missing)), str(bp_missing),
                                                                            str(bp_missing)))
            raise ValueError()
        else:
            data_df.columns = new_headers
            return data_df

    def get_number_of_header_columns_in_df(self,
                                           df: pd.DataFrame):
        for i in range(len(df)):
            try:
                temp = df.iloc[i:].apply(pd.to_numeric).reset_index(drop=True)
                return i
            except ValueError:
                pass
        raise DataHeaderError(msg='Could find the count of header columns in dataframe')


    def find_video_of_file(self,
                           video_dir: str,
                           filename: str):
        """
        Helper to find the video file that represents a data file.

        Parameters
        ----------
        video_dir: str
            Directory holding putative video file
        filename: str
            Data file name, e.g., ``Video_1``.

        Returns
        -------
        return_path: str

        """
        try:
            all_files_in_video_folder = [f for f in next(os.walk(video_dir))[2] if not f[0] == '.']
        except StopIteration:
            raise NoFilesFoundError(msg=f'No files found in the {video_dir} directory')
        all_files_in_video_folder = [os.path.join(video_dir, x) for x in all_files_in_video_folder]
        return_path = None
        for file_path in all_files_in_video_folder:
            _, video_filename, ext = get_fn_ext(file_path)
            if ((video_filename == filename) and ((ext.lower() == '.mp4') or (ext.lower() == '.avi'))):
                return_path = file_path

        if return_path is None:
            NoFileFoundWarning(
                f'SimBA could not find a video file representing {filename} in the project video directory')
        return return_path

    def add_missing_ROI_cols(self,
                             shape_df: pd.DataFrame) -> pd.DataFrame:
        """
        Helper to add missing ROI definitions in ROI info dataframes created by the first version of the SimBA ROI
        user-interface but analyzed using newer versions of SimBA.

        Parameters
        ----------
        shape_df: pd.DataFrame
            Dataframe holding ROI definitions.

        Returns
        -------
        pd.DataFrame
        """

        if not 'Color BGR' in shape_df.columns:
            shape_df['Color BGR'] = [(255, 255, 255)] * len(shape_df)
        if not 'Thickness' in shape_df.columns:
            shape_df['Thickness'] = [5] * len(shape_df)
        if not 'Color name' in shape_df.columns:
            shape_df['Color name'] = 'White'

        return shape_df

    def remove_a_folder(self,
                        folder_dir: str):
        """Helper to remove a directory"""
        shutil.rmtree(folder_dir, ignore_errors=True)

    def find_animal_name_from_body_part_name(self,
                                             bp_name: str,
                                             bp_dict: dict) -> str:

        """Given body-part name and animal body-part dict, return animal name"""

        for animal_name, animal_bps in bp_dict.items():
            if bp_name in [x[:-2] for x in animal_bps['X_bps']]:
                return animal_name

    def create_logger(self,
                      path: str):

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(filename=path, encoding='utf-8')  # or whatever
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(handler)

    def create_body_part_dictionary(self,
                                    multi_animal_status: bool,
                                    animal_id_lst: list,
                                    animal_cnt: int,
                                    x_cols: list,
                                    y_cols: list,
                                    p_cols: list or None=None,
                                    colors: list or None=None):
        """
        Helper to create dict of dict lookup of body-parts where the keys are animal names, and
        values are the body-part names.

        Parameters
        ----------
        multi_animal_status: bool
            If True, it is a multi-animal SimBA project
        multi_animal_id_lst: list
            Animal names. Eg., ['Animal_1, 'Animals_2']
        animal_cnt: int
            Number of animals in the SimBA project.
        x_cols: list
            column names for body-part coordinates on x-axis
        y_cols: list
             column names for body-part coordinates on y-axis
        p_cols: list
            Column names for body-part pose-estimation probability values
        colors: list or None
            bgr colors

        Returns
        -------
        dict
        """

        animal_bp_dict = {}
        if multi_animal_status:
            for animal in range(animal_cnt):
                animal_bp_dict[animal_id_lst[animal]] = {}
                animal_bp_dict[animal_id_lst[animal]]['X_bps'] = [i for i in x_cols if animal_id_lst[animal] in i]
                animal_bp_dict[animal_id_lst[animal]]['Y_bps'] = [i for i in y_cols if animal_id_lst[animal] in i]
                if colors:
                    animal_bp_dict[animal_id_lst[animal]]['colors'] = colors[animal]
                if p_cols:
                    animal_bp_dict[animal_id_lst[animal]]['P_bps'] = [i for i in p_cols if animal_id_lst[animal] in i]
                if not animal_bp_dict[animal_id_lst[animal]]['X_bps']:
                    multi_animal_status = False
                    break

        if not multi_animal_status:
            if animal_cnt > 1:
                for animal in range(animal_cnt):
                    currAnimalName = f'Animal_{str(animal + 1)}'
                    search_string_x = f'_{str(animal + 1)}_x'
                    search_string_y = f'_{str(animal + 1)}_y'
                    search_string_p = f'_{str(animal + 1)}_p'
                    animal_bp_dict[currAnimalName] = {}
                    animal_bp_dict[currAnimalName]['X_bps'] = [i for i in x_cols if i.endswith(search_string_x)]
                    animal_bp_dict[currAnimalName]['Y_bps'] = [i for i in y_cols if i.endswith(search_string_y)]
                    if colors:
                        animal_bp_dict[currAnimalName]['colors'] = colors[animal]
                    if p_cols:
                        animal_bp_dict[currAnimalName]['P_bps'] = [i for i in p_cols if i.endswith(search_string_p)]
                if animal_id_lst[0] != '':
                    for animal in range(len(animal_id_lst)):
                        currAnimalName = f'Animal_{str(animal + 1)}'
                        animal_bp_dict[animal_id_lst[animal]] = animal_bp_dict.pop(currAnimalName)

            else:
                animal_bp_dict['Animal_1'] = {}
                animal_bp_dict['Animal_1']['X_bps'] = [i for i in x_cols]
                animal_bp_dict['Animal_1']['Y_bps'] = [i for i in y_cols]
                if colors:
                    animal_bp_dict['Animal_1']['colors'] = colors[0]
                if p_cols:
                    animal_bp_dict['Animal_1']['P_bps'] = [i for i in p_cols]

        return animal_bp_dict

    def get_body_part_names(self):
        """
        Helper to extract pose-estimation data field names (x, y, p)
        """
        self.x_cols, self.y_cols, self.p_cols = [], [], []
        for bp in self.body_parts_lst:
            self.x_cols.append(f'{bp}_x'); self.y_cols.append(f'{bp}_y'); self.p_cols.append(f'{bp}_p')

    def drop_bp_cords(self,
                      df: pd.DataFrame) -> pd.DataFrame:

        """
        Helper to remove pose-estimation data from dataframe.

        Parameters
        ----------
        df: pd.DataFrame
            pandas dataframe containing pose-estimation data

        """


        missing_body_part_fields = list(set(self.bp_col_names) - set(list(df.columns)))
        if len(missing_body_part_fields) > 0:
            BodypartColumnNotFoundWarning(msg=f'SimBA could not drop body-part coordinates, some body-part names are missing in dataframe. SimBA expected the following body-parts, that could not be found inside the file: {missing_body_part_fields}')
        else:
            return df.drop(self.bp_col_names, axis=1)

    def get_bp_headers(self) -> list:
        """
        Helper to create ordered list of all column header fields for SimBA project dataframes.
        """
        self.bp_headers = []
        for bp in self.body_parts_lst:
            c1, c2, c3 = (f'{bp}_x', f'{bp}_y', f'{bp}_p')
            self.bp_headers.extend((c1, c2, c3))

    def read_config_entry(self,
                          config: ConfigParser,
                          section: str,
                          option: str,
                          data_type: str,
                          default_value=None,
                          options=None):

        """ Helper to read entry in a configparser.ConfigParser object"""
        try:
            if config.has_option(section, option):
                if data_type == Dtypes.FLOAT.value:
                    value = config.getfloat(section, option)
                elif data_type == Dtypes.INT.value:
                    value = config.getint(section, option)
                elif data_type == Dtypes.STR.value:
                    value = config.get(section, option).strip()
                elif data_type == Dtypes.FOLDER.value:
                    value = config.get(section, option).strip()
                    if not os.path.isdir(value):
                        raise NotDirectoryError(
                            msg=f'The SimBA config file includes paths to a folder ({value}) that does not exist.')
                if options != None:
                    if value not in options:
                        raise InvalidInputError(
                            msg=f'{option} is set to {str(value)} in SimBA, but this is not among the valid options: ({options})')
                    else:
                        return value
                return value
            elif default_value != None:
                return default_value
            else:
                raise MissingProjectConfigEntryError(
                    msg=f'SimBA could not find an entry for option {option} under section {section} in the project_config.ini. Please specify the settings in the settings menu.')
        except ValueError:
            if default_value != None:
                return default_value
            else:
                raise MissingProjectConfigEntryError(
                    msg=f'SimBA could not find an entry for option {option} under section {section} in the project_config.ini. Please specify the settings in the settings menu.')

    def read_video_info_csv(self,
                            file_path: str):
        """
        Helper to read the project_folder/logs/video_info.csv of the SimBA project in as a pd.DataFrame
        Parameters
        ----------
        file_path: str

        Returns
        -------
        pd.DataFrame
        """

        check_file_exist_and_readable(file_path=file_path)
        info_df = pd.read_csv(file_path)
        for c in ['Video', 'fps', 'Resolution_width', 'Resolution_height', 'Distance_in_mm', 'pixels/mm']:
            if c not in info_df.columns:
                raise ParametersFileError(
                    msg=f'The project "project_folder/logs/video_info.csv" does not not have an anticipated header ({c}). Please re-create the file and make sure each video has a {c} value')
        info_df['Video'] = info_df['Video'].astype(str)
        for c in ['fps', 'Resolution_width', 'Resolution_height', 'Distance_in_mm', 'pixels/mm']:
            try:
                info_df[c] = info_df[c].astype(float)
            except:
                raise ParametersFileError(msg=f'One or more values in the {c} column of the "project_folder/logs/video_info.csv" file could not be interpreted as a numeric value. Please re-create the file and make sure the entries in the {c} column are all numeric.')
        if info_df['fps'].min() <= 1:
            InvalidValueWarning(msg='Videos in your SimBA project have an FPS of 1 or less. Please use videos with more than one frame per second, or correct the inaccurate fps inside the `project_folder/logs/videos_info.csv` file')
        return info_df

    def read_video_info(self,
                        video_name: str):
        """
        Helper to read the meta-data (pixels per mm, resolution, fps etc) from the video_info.csv for a single input file

        Parameters
        ----------
        video_name: str
            The name of the video without extension to get the meta data for.

        Returns
        -------
        video_settings: pd.DataFrame
        px_per_mm: float
        fps: float
        """

        video_settings = self.video_info_df.loc[self.video_info_df['Video'] == video_name]
        if len(video_settings) > 1:
            raise DuplicationError(msg=f'SimBA found multiple rows in the project_folder/logs/video_info.csv named {str(video_name)}. Please make sure that each video name is represented ONCE in the video_info.csv')
        elif len(video_settings) < 1:
            raise ParametersFileError(msg=f'SimBA could not find {str(video_name)} in the video_info.csv file. Make sure all videos analyzed are represented in the project_folder/logs/video_info.csv file.')
        else:
            try:
                px_per_mm = float(video_settings['pixels/mm'])
                fps = float(video_settings['fps'])
                return video_settings, px_per_mm, fps
            except TypeError:
                raise ParametersFileError(msg=f'Make sure the videos that are going to be analyzed are represented with APPROPRIATE VALUES inside the project_folder/logs/video_info.csv file in your SimBA project. Could not interpret the fps, pixels per millimeter and/or fps as numerical values for video {video_name}')

    def check_multi_animal_status(self):
        """
        Helper to check if the project is a multi-animal SimBA project.

        Parameters
        ----------
        config: configparser.ConfigParser
            Parsed SimBA project_config.ini file.
        no_animals: int
            Number of animals in the SimBA project

        Returns
        -------
        multi_animal_status: bool
        multi_animal_id_lst: list
        """
        multi_animal_id_lst = []
        if not self.config.has_section('Multi animal IDs'):
            for animal in range(self.animal_cnt):
                multi_animal_id_lst.append('Animal_' + str(animal + 1))
            multi_animal_status = False

        else:
            multi_animal_id_str = self.read_config_entry(config=self.config, section='Multi animal IDs', option='id_list', data_type='str')
            multi_animal_id_lst = [x.lstrip() for x in multi_animal_id_str.split(",")]
            multi_animal_id_lst = [x for x in multi_animal_id_lst if x != 'None']
            if (self.animal_cnt > 1) and (len(multi_animal_id_lst) > 1):
                multi_animal_status = True
            else:
                for animal in range(self.animal_cnt):
                    multi_animal_id_lst.append('Animal_{}'.format(str(animal + 1)))
                multi_animal_status = False

        self.multi_animal_status = multi_animal_status
        self.multi_animal_id_list = multi_animal_id_lst[:self.animal_cnt]

#test = ConfigReader(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')


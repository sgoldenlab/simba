__author__ = "Simon Nilsson"

import os
from tkinter import *
from tkinter import ttk
from typing import Dict, Optional, Union

try:
    from typing import Literal
except:
    from typing_extensions import Literal

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.pose_importers.dlc_importer_csv import import_dlc_csv_data
from simba.pose_importers.facemap_h5_importer import FaceMapImporter
from simba.pose_importers.import_mars import MarsImporter
from simba.pose_importers.madlc_importer import MADLCImporterH5
from simba.pose_importers.read_DANNCE_mat import (import_DANNCE_file,
                                                  import_DANNCE_folder)
from simba.pose_importers.simba_blob_importer import SimBABlobImporter
from simba.pose_importers.sleap_csv_importer import SLEAPImporterCSV
from simba.pose_importers.sleap_h5_importer import SLEAPImporterH5
from simba.pose_importers.sleap_slp_importer import SLEAPImporterSLP
from simba.pose_importers.superanimal_import import SuperAnimalTopViewImporter
from simba.pose_importers.trk_importer import TRKImporter
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        Entry_Box, FileSelect, FolderSelect,
                                        SimBADropDown)
from simba.utils.checks import check_instance, check_int, check_str
from simba.utils.enums import ConfigKey, Dtypes, Formats, Options
from simba.utils.errors import InvalidInputError
from simba.utils.read_write import read_config_file

GAUSSIAN = 'Gaussian'
SAVITZKY_GOLAY = 'Savitzky Golay'
INTERPOLATION_MAP = {'Animal(s)': 'animals', 'Body-parts': 'body-parts'}
SMOOTHING_MAP = {'Savitzky Golay': 'savitzky-golay', 'Gaussian': 'gaussian'}

FRAME_DIR_IMPORT_TITLES = {'CSV (DLC/DeepPoseKit)': 'IMPORT DLC CSV DIRECTORY',  'MAT (DANNCE 3D)': 'IMPORT DANNCE MAT DIRECTORY', 'JSON (BENTO)': 'IMPORT MARS JSON DIRECTORY', 'CSV (SimBA BLOB)': 'IMPORT SIMBA BLOB CSV DIRECTORY', 'H5 (FaceMap)': 'IMPORT FaceMap H5 DIRECTORY'}
FRAME_FILE_IMPORT_TITLES = {'CSV (DLC/DeepPoseKit)': 'IMPORT DLC CSV FILE',  'MAT (DANNCE 3D)': 'IMPORT DANNCE MAT FILE', 'JSON (BENTO)': 'IMPORT MARS JSON FILE', 'CSV (SimBA BLOB)': 'IMPORT SIMBA BLOB CSV FILE', 'H5 (FaceMap)': 'IMPORT FaceMap H5 FILE'}
FILE_TYPES = {'CSV (DLC/DeepPoseKit)': '*.csv', 'MAT (DANNCE 3D)': '*.mat', 'JSON (BENTO)': '*.json', 'CSV (SimBA BLOB)': '*.csv', 'H5 (FaceMap)': '*.h5'}


class ImportPoseFrame(ConfigReader, PopUpMixin):

    """
    .. image:: _static/img/ImportPoseFrame.webp
       :width: 500
       :align: center

    :param Optional[Union[Frame, Canvas, LabelFrame, ttk.Frame]] parent_frm: Parent frame to insert the Import pose frame into. If None, one is created.
    :param Optional[Union[str, os.PathLike]] config_path:
    :param Optional[int] idx_row: The row in parent_frm to insert the Import pose frame into. Default: 0.
    :param Optional[int] idx_column: The column in parent_frm to insert the Import pose frame into. Default: 0.

    :example:
    >>> _ = ImportPoseFrame(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
    >>> _ = ImportPoseFrame(config_path=r"C:\troubleshooting\simba_blob_project\project_folder\project_config.ini")
    """

    def __init__(self,
                 parent_frm: Optional[Union[Frame, Canvas, LabelFrame, ttk.Frame]] = None,
                 config_path: Optional[Union[str, os.PathLike]] = None,
                 idx_row: Optional[int] = 0,
                 idx_column: Optional[int] = 0):

        if parent_frm is None and config_path is None:
            raise InvalidInputError(msg='If parent_frm is None, please pass config_path', source=self.__class__.__name__)

        elif parent_frm is None and config_path is not None:
            PopUpMixin.__init__(self, config_path=config_path, title='IMPORT POSE ESTIMATION')
            parent_frm = self.main_frm

        check_instance(source=f'{self.__class__.__name__} parent_frm', accepted_types=(Frame, Canvas, LabelFrame, ttk.Frame), instance=parent_frm)
        check_int(name=f'{self.__class__.__name__} idx_row', value=idx_row, min_value=0)
        check_int(name=f'{self.__class__.__name__} idx_column', value=idx_column, min_value=0)

        self.import_tracking_frm = CreateLabelFrameWithIcon(parent=parent_frm, header="IMPORT TRACKING DATA", relief='solid', icon_name='pose')
        self.import_tracking_frm.grid(row=0, column=0, sticky=NW)
        if config_path is None:
            Label(self.import_tracking_frm, text="Please CREATE PROJECT CONFIG before importing tracking data \n", font=Formats.FONT_REGULAR.value).grid(row=0, column=0, sticky=NW)
        else:
            ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
            self.data_type_dropdown = SimBADropDown(parent=self.import_tracking_frm, dropdown_options=Options.IMPORT_TYPE_OPTIONS.value, label="DATA TYPE: ", label_width=25, command=self.create_import_menu, dropdown_width=25, value=Options.IMPORT_TYPE_OPTIONS.value[0])
            self.data_type_dropdown.grid(row=0, column=0, sticky=NW)
            self.create_import_menu(data_type_choice=Options.IMPORT_TYPE_OPTIONS.value[0])
            self.import_tracking_frm.grid(row=idx_row, column=idx_column, sticky=NW)

        #parent_frm.mainloop()

    def __show_smoothing_entry_box_from_dropdown(self, choice: str):
        if (choice == GAUSSIAN) or (choice == SAVITZKY_GOLAY):
            self.smoothing_time_eb.grid(row=0, column=1, sticky=E)
        else:
            self.smoothing_time_eb.grid_forget()


    def __get_smooth_interpolation_settings(self,
                                            interpolation_settings: str,
                                            smoothing_setting: str,
                                            smoothing_time: Union[str, int]):
        METHODS_MAP = {'Gaussian': 'gaussian', 'Savitzky Golay': 'savitzky-golay'}
        if interpolation_settings not in [Dtypes.NONE.value, None]:
            interpolation_settings = interpolation_settings.split(':')
            interpolation_settings = {'type': INTERPOLATION_MAP[interpolation_settings[0]].lower().strip(), 'method': interpolation_settings[1].lower().strip()}
        else:
            interpolation_settings = None
        if smoothing_setting not in [Dtypes.NONE.value, None]:
            check_int(name='SMOOTHING TIME', value=smoothing_time, min_value=1)
            smoothing_setting = {'method': METHODS_MAP[smoothing_setting], 'time_window': int(smoothing_time)}
        else:
            smoothing_setting = None
        return smoothing_setting, interpolation_settings

    def __import_dlc_csv_data(self,
                              interpolation_settings: str,
                              smoothing_time: Union[str, int],
                              data_path: Union[str, os.PathLike],
                              smoothing_setting: Literal['Gaussian', 'None', 'Savitzky Golay']):

        if not os.path.isfile(data_path) and not os.path.isdir(data_path):
            raise InvalidInputError(msg=f'{data_path} is NOT a valid path', source=self.__class__.__name__)
        smoothing_settings, interpolation_settings = self.__get_smooth_interpolation_settings(interpolation_settings, smoothing_setting, smoothing_time)

        import_dlc_csv_data(config_path=self.config_path,
                            data_path=data_path,
                            interpolation_settings=interpolation_settings,
                            smoothing_settings=smoothing_settings)

    def __import_simba_blob_data(self,
                                interpolation_settings: str,
                                smoothing_time: Union[str, int],
                                data_path: Union[str, os.PathLike],
                                smoothing_setting: Literal['Gaussian', 'None', 'Savitzky Golay']):

            if not os.path.isfile(data_path) and not os.path.isdir(data_path):
                raise InvalidInputError(msg=f'{data_path} is NOT a valid path', source=self.__class__.__name__)
            smoothing_settings, interpolation_settings = self.__get_smooth_interpolation_settings(interpolation_settings, smoothing_setting, smoothing_time)

            blob_importer = SimBABlobImporter(config_path=self.config_path, data_path=data_path, interpolation_settings=interpolation_settings, smoothing_settings=smoothing_settings, verbose=True)
            blob_importer.run()

    def __import_facemap_data(self,
                              interpolation_settings: str,
                              smoothing_time: Union[str, int],
                              data_path: Union[str, os.PathLike],
                              smoothing_setting: Literal['Gaussian', 'None', 'Savitzky Golay']):

            if not os.path.isfile(data_path) and not os.path.isdir(data_path):
                raise InvalidInputError(msg=f'{data_path} is NOT a valid path', source=self.__class__.__name__)
            smoothing_settings, interpolation_settings = self.__get_smooth_interpolation_settings(interpolation_settings, smoothing_setting, smoothing_time)

            facemap_importer = FaceMapImporter(config_path=self.config_path, data_path=data_path, interpolation_settings=interpolation_settings, smoothing_settings=smoothing_settings, verbose=True)
            facemap_importer.run()

    def __multi_animal_run_call(self,
                                pose_estimation_tool: str,
                                interpolation_settings: str,
                                smoothing_settings: str,
                                smoothing_window: int,
                                animal_names: Dict[int, Entry_Box],
                                data_path: Union[str, os.PathLike],
                                tracking_data_type: Optional[str] = None):

        if not os.path.isfile(data_path) and not os.path.isdir(data_path):
            raise InvalidInputError(msg=f'{data_path} is NOT a valid path', source=self.__class__.__name__)
        smoothing_settings, interpolation_settings = self.__get_smooth_interpolation_settings(interpolation_settings, smoothing_settings, smoothing_window)
        animal_ids = []
        if len(list(animal_names.items())) == 1: animal_ids.append("Animal_1")
        else:
            for animal_cnt, animal_entry_box in animal_names.items():
                check_str(name=f"ANIMAL {str(animal_cnt)} NAME", value=animal_entry_box.entry_get, allow_blank=False)
                animal_ids.append(animal_entry_box.entry_get)

        config = read_config_file(config_path=self.config_path)
        config.set(ConfigKey.MULTI_ANIMAL_ID_SETTING.value, ConfigKey.MULTI_ANIMAL_IDS.value, ",".join(animal_ids))
        with open(self.config_path, "w") as f: config.write(f)

        if pose_estimation_tool == "H5 (multi-animal DLC)":
            data_importer = MADLCImporterH5(config_path=self.config_path,
                                            data_folder=data_path,
                                            file_type=tracking_data_type,
                                            id_lst=animal_ids,
                                            interpolation_settings=interpolation_settings,
                                            smoothing_settings=smoothing_settings)

        elif pose_estimation_tool == "SLP (SLEAP)":
            data_importer = SLEAPImporterSLP(project_path=self.config_path,
                                             data_folder=data_path,
                                             id_lst=animal_ids,
                                             interpolation_settings=interpolation_settings,
                                             smoothing_settings=smoothing_settings)

        elif pose_estimation_tool == "TRK (multi-animal APT)":
            data_importer = TRKImporter(config_path=self.config_path,
                                        data_path=data_path,
                                        animal_id_lst=animal_ids,
                                        interpolation_method=interpolation_settings,
                                        smoothing_settings=smoothing_settings)

        elif pose_estimation_tool == "CSV (SLEAP)":
            data_importer = SLEAPImporterCSV(config_path=self.config_path,
                                            data_folder=data_path,
                                            id_lst=animal_ids,
                                            interpolation_settings=interpolation_settings,
                                            smoothing_settings=smoothing_settings)

        elif pose_estimation_tool == "H5 (SLEAP)":
            data_importer = SLEAPImporterH5(config_path=self.config_path,
                                            data_folder=data_path,
                                            id_lst=animal_ids,
                                            interpolation_settings=interpolation_settings,
                                            smoothing_settings=smoothing_settings)


        elif pose_estimation_tool == "H5 (SuperAnimal-TopView)":
            data_importer = SuperAnimalTopViewImporter(config_path=self.config_path,
                                                        data_folder=data_path,
                                                        id_lst=animal_ids,
                                                        interpolation_settings=interpolation_settings,
                                                        smoothing_settings=smoothing_settings)


        else:
            raise InvalidInputError(msg=f'pose estimation tool {pose_estimation_tool} not recognized', source=self.__class__.__name__)
        data_importer.run()

    def __create_animal_names_entry_boxes(self,
                                          animal_cnt: str) -> None:
        check_int(name="NUMBER OF ANIMALS", value=animal_cnt, min_value=0)
        if hasattr(self, "animal_names_frm"):
            self.animal_names_frm.destroy()
        if not hasattr(self, "multi_animal_id_list"):
            self.multi_animal_id_list = []
            for i in range(int(animal_cnt)):
                self.multi_animal_id_list.append(f"Animal {i+1}")
        self.animal_names_frm = Frame(self.animal_settings_frm, pady=5, padx=5)
        self.animal_name_entry_boxes = {}
        for i in range(int(animal_cnt)):
            self.animal_name_entry_boxes[i + 1] = Entry_Box(self.animal_names_frm, f"Animal {str(i+1)} name: ", "25")
            if i <= len(self.multi_animal_id_list) - 1:
                self.animal_name_entry_boxes[i + 1].entry_set(self.multi_animal_id_list[i])
            self.animal_name_entry_boxes[i + 1].grid(row=i, column=0, sticky=NW)
        self.animal_names_frm.grid(row=1, column=0, sticky=NW)

    def create_import_menu(self, data_type_choice: Literal["CSV (DLC/DeepPoseKit)", "JSON (BENTO)", "H5 (multi-animal DLC)", "SLP (SLEAP)", "CSV (SLEAP)", "H5 (SLEAP)", "TRK (multi-animal APT)", "MAT (DANNCE 3D)", "CSV (SimBA BLOB)"]):
        if hasattr(self, "choice_frm"):
            self.choice_frm.destroy()

        self.choice_frm = Frame(self.import_tracking_frm)
        self.choice_frm.grid(row=1, column=0, sticky=NW)
        self.animal_name_entry_boxes = None

        self.interpolation_frm = CreateLabelFrameWithIcon(parent=self.choice_frm, header="INTERPOLATION METHOD", pady=5, padx=5,font=Formats.FONT_HEADER.value, icon_name='fill', relief='groove')
        self.interpolation_dropdown = SimBADropDown(parent=self.interpolation_frm, dropdown_options=Options.INTERPOLATION_OPTIONS_W_NONE.value, label='INTERPOLATION METHOD: ', label_width=25, dropdown_width=35, value=Options.INTERPOLATION_OPTIONS_W_NONE.value[0])
        self.interpolation_frm.grid(row=0, column=0, sticky=NW)
        self.interpolation_dropdown.grid(row=0, column=0, sticky=NW)

        self.smoothing_frm = CreateLabelFrameWithIcon(parent=self.choice_frm, header="SMOOTHING METHOD", pady=5, padx=5, font=Formats.FONT_HEADER.value, icon_name='smooth', relief='groove')
        self.smoothing_dropdown = SimBADropDown(parent=self.smoothing_frm, dropdown_options=Options.SMOOTHING_OPTIONS_W_NONE.value, label='SMOOTHING: ', label_width=25, dropdown_width=35, value=Options.SMOOTHING_OPTIONS_W_NONE.value[0], command=self.__show_smoothing_entry_box_from_dropdown)
        self.smoothing_time_eb = Entry_Box(self.smoothing_frm, "SMOOTHING PERIOD (MS):", labelwidth=30,  validation="numeric", entry_box_width=10)
        self.smoothing_frm.grid(row=1, column=0, sticky=NW)
        self.smoothing_dropdown.grid(row=0, column=0, sticky=NW)

        if data_type_choice in ["CSV (DLC/DeepPoseKit)", "MAT (DANNCE 3D)", "JSON (BENTO)", "CSV (SimBA BLOB)", 'H5 (FaceMap)']: # DATA TYPES WHERE NO TRACKS HAVE TO BE SPECIFIED
            self.import_directory_frm = LabelFrame(self.choice_frm, text=FRAME_DIR_IMPORT_TITLES[data_type_choice], pady=5, padx=5, font=Formats.FONT_HEADER.value,)
            self.import_directory_select = FolderSelect(self.import_directory_frm, "Input data DIRECTORY:", lblwidth=25, initialdir=self.project_path)
            self.import_single_frm = LabelFrame(self.choice_frm, text=FRAME_FILE_IMPORT_TITLES[data_type_choice], pady=5, padx=5, font=Formats.FONT_HEADER.value,)
            self.import_file_select = FileSelect(self.import_single_frm, "Input data FILE:", lblwidth=25, file_types=[("Pose data file", FILE_TYPES[data_type_choice])])

            if data_type_choice == "CSV (DLC/DeepPoseKit)":
                self.import_dir_btn = Button(self.import_directory_frm, fg="blue", font=Formats.FONT_REGULAR.value, text="Import DLC CSV DIRECTORY to SimBA project", command=lambda: self.__import_dlc_csv_data(interpolation_settings=self.interpolation_dropdown.getChoices(),
                                                                                                                                                                                smoothing_setting=self.smoothing_dropdown.getChoices(),
                                                                                                                                                                                smoothing_time=self.smoothing_time_eb.entry_get,
                                                                                                                                                                                data_path=self.import_directory_select.folder_path))
                self.import_file_btn = Button(self.import_single_frm, fg="blue", font=Formats.FONT_REGULAR.value, text="Import DLC CSV FILE to SimBA project", command=lambda: self.__import_dlc_csv_data(interpolation_settings=self.interpolation_dropdown.getChoices(),
                                                                                                                                                                         smoothing_setting=self.smoothing_dropdown.getChoices(),
                                                                                                                                                                         smoothing_time=self.smoothing_time_eb.entry_get,
                                                                                                                                                                         data_path=self.import_file_select.file_path))
            elif data_type_choice == "MAT (DANNCE 3D)":
                self.import_dir_btn = Button(self.import_directory_frm, fg="blue", font=Formats.FONT_REGULAR.value, text="Import DANNCE MAT DIRECTORY to SimBA project", command=lambda: import_DANNCE_folder(config_path=self.config_path,
                                                                                                                                                                  folder_path=self.import_directory_select.folder_path,
                                                                                                                                                                  interpolation_method=self.interpolation_dropdown.getChoices()))

                self.import_file_btn = Button(self.import_single_frm, fg="blue", font=Formats.FONT_REGULAR.value, text="Import DANNCE MAT FILE to SimBA project", command=lambda: import_DANNCE_file(config_path=self.config_path,
                                                                                                                                                         file_path=self.import_file_select.file_path,
                                                                                                                                                         interpolation_method=self.interpolation_dropdown.getChoices()))
            elif data_type_choice == "JSON (BENTO)":
                self.import_dir_btn = Button(self.import_directory_frm, fg="blue", font=Formats.FONT_REGULAR.value, text="Import BENTO JSON DIRECTORY to SimBA project", command=lambda: MarsImporter(config_path=self.config_path, data_path=self.import_directory_select.folder_path, interpolation_method=self.interpolation_dropdown.getChoices(), smoothing_method={"Method": self.smoothing_dropdown.getChoices(), "Parameters": {"Time_window": self.smoothing_time_eb.entry_get}}))
                self.import_file_btn = Button(self.import_single_frm, fg="blue", font=Formats.FONT_REGULAR.value, text="Import BENTO JSON FILE to SimBA project", command=lambda: MarsImporter(config_path=self.config_path, data_path=self.import_directory_select.folder_path, interpolation_method=self.interpolation_dropdown.getChoices(), smoothing_method={"Method": self.smoothing_dropdown.getChoices(), "Parameters": {"Time_window": self.smoothing_time_eb.entry_get}}))

            elif data_type_choice == "CSV (SimBA BLOB)":
                self.import_dir_btn = Button(self.import_directory_frm, fg="blue", font=Formats.FONT_REGULAR.value, text="Import SimBA BLOB CSV DIRECTORY to SimBA project", command=lambda: self.__import_simba_blob_data(interpolation_settings=self.interpolation_dropdown.getChoices(), smoothing_time=self.smoothing_time_eb.entry_get, data_path=self.import_directory_select.folder_path, smoothing_setting=self.smoothing_dropdown.getChoices()))
                self.import_file_btn = Button(self.import_single_frm, fg="blue", font=Formats.FONT_REGULAR.value, text="Import SimBA BLOB CSV FILE to SimBA project", command=lambda: self.__import_simba_blob_data(interpolation_settings=self.interpolation_dropdown.getChoices(), smoothing_time=self.smoothing_time_eb.entry_get, data_path=self.import_file_select.file_path, smoothing_setting=self.smoothing_dropdown.getChoices()))

            else:
                self.import_dir_btn = Button(self.import_directory_frm, fg="blue", font=Formats.FONT_REGULAR.value, text="Import SimBA FaceMap H5 DIRECTORY to SimBA project", command=lambda: self.__import_facemap_data(interpolation_settings=self.interpolation_dropdown.getChoices(), smoothing_time=self.smoothing_time_eb.entry_get, data_path=self.import_directory_select.folder_path, smoothing_setting=self.smoothing_dropdown.getChoices()))
                self.import_file_btn = Button(self.import_single_frm, fg="blue", font=Formats.FONT_REGULAR.value, text="Import SimBA FaceMap H5 FILE to SimBA project", command=lambda: self.__import_facemap_data(interpolation_settings=self.interpolation_dropdown.getChoices(), smoothing_time=self.smoothing_time_eb.entry_get, data_path=self.import_file_select.file_path, smoothing_setting=self.smoothing_dropdown.getChoices()))


            self.import_directory_frm.grid(row=2, column=0, sticky=NW)
            self.import_directory_select.grid(row=0, column=0, sticky=NW)
            self.import_dir_btn.grid(row=1, column=0, sticky=NW)

            self.import_single_frm.grid(row=3, column=0, sticky=NW)
            self.import_file_select.grid(row=0, column=0, sticky=NW)
            self.import_file_btn.grid(row=1, column=0, sticky=NW)

        else: # DATA TYPES WHERE TRACKS HAVE TO BE SPECIFIED
            self.animal_settings_frm = LabelFrame(self.choice_frm, text="ANIMAL SETTINGS", font=Formats.FONT_HEADER.value, pady=5, padx=5)
            animal_cnt_entry_box = Entry_Box(self.animal_settings_frm, "ANIMAL COUNT:", "25", validation="numeric")
            animal_cnt_entry_box.entry_set(val=self.animal_cnt)
            animal_cnt_confirm = Button(self.animal_settings_frm, text="CONFIRM", fg="blue", font=Formats.FONT_REGULAR.value, command=lambda: self.create_animal_names_entry_boxes( animal_cnt=animal_cnt_entry_box.entry_get))
            self.create_animal_names_entry_boxes(animal_cnt=animal_cnt_entry_box.entry_get)
            self.animal_settings_frm.grid(row=4, column=0, sticky=NW)
            animal_cnt_entry_box.grid(row=0, column=0, sticky=NW)
            animal_cnt_confirm.grid(row=0, column=1, sticky=NW)

            self.data_dir_frm = LabelFrame(self.choice_frm, text="DATA DIRECTORY", font=Formats.FONT_HEADER.value, pady=5, padx=5)
            self.import_frm = LabelFrame(self.choice_frm, text="IMPORT", font=Formats.FONT_HEADER.value, pady=5, padx=5)

            if data_type_choice == "H5 (multi-animal DLC)":
                self.tracking_type_frm = LabelFrame(self.choice_frm, text="TRACKING DATA TYPE", font=Formats.FONT_HEADER.value, pady=5, padx=5)
                self.dlc_data_type_option_dropdown = DropDownMenu(self.tracking_type_frm, "TRACKING TYPE", Options.MULTI_DLC_TYPE_IMPORT_OPTION.value, labelwidth=25)
                self.dlc_data_type_option_dropdown.setChoices(Options.MULTI_DLC_TYPE_IMPORT_OPTION.value[1])
                self.tracking_type_frm.grid(row=5, column=0, sticky=NW)
                self.dlc_data_type_option_dropdown.grid(row=0, column=0, sticky=NW)
                self.data_dir_select = FolderSelect(self.data_dir_frm, "H5 DLC DIRECTORY: ", lblwidth=25)
                self.instructions_lbl = Label(self.data_dir_frm, text="Please import videos BEFORE importing the \n multi animal DLC tracking data", font=Formats.FONT_REGULAR.value)
                self.run_btn = Button(self.import_frm, text="IMPORT DLC .H5", fg="blue", command=lambda: self.__multi_animal_run_call(pose_estimation_tool=data_type_choice,
                                                                                                                                      interpolation_settings=self.interpolation_dropdown.getChoices(),
                                                                                                                                      smoothing_settings=self.smoothing_dropdown.getChoices(),
                                                                                                                                      smoothing_window=self.smoothing_time_eb.entry_get,
                                                                                                                                      animal_names=self.animal_name_entry_boxes,
                                                                                                                                      data_path=self.data_dir_select.folder_path,
                                                                                                                                      tracking_data_type=self.dlc_data_type_option_dropdown.getChoices()))
            elif data_type_choice == "SLP (SLEAP)":
                self.data_dir_select = FolderSelect(self.data_dir_frm, "SLP SLEAP DIRECTORY: ", lblwidth=25)
                self.instructions_lbl = Label(self.data_dir_frm, font=Formats.FONT_REGULAR.value, text="Please import videos before importing the \n multi animal SLEAP tracking data if you are tracking more than ONE animal")
                self.run_btn = Button(self.import_frm, text="IMPORT SLEAP .SLP", fg="blue", font=Formats.FONT_REGULAR.value, command=lambda: self.__multi_animal_run_call(pose_estimation_tool=data_type_choice,
                                                                                                                                         interpolation_settings=self.interpolation_dropdown.getChoices(),
                                                                                                                                         smoothing_settings=self.smoothing_dropdown.getChoices(),
                                                                                                                                         smoothing_window=self.smoothing_time_eb.entry_get,
                                                                                                                                         animal_names=self.animal_name_entry_boxes,
                                                                                                                                         data_path=self.data_dir_select.folder_path))

            elif data_type_choice == "TRK (multi-animal APT)":
                self.data_dir_select = FolderSelect(self.data_dir_frm, "TRK APT DIRECTORY: ", lblwidth=25)
                self.instructions_lbl = Label(self.data_dir_frm, text="Please import videos before importing the \n multi animal TRK tracking data", font=Formats.FONT_REGULAR.value,)
                self.run_btn = Button(self.import_frm, text="IMPORT APT .TRK", font=Formats.FONT_REGULAR.value, fg="blue", command=lambda: self.__multi_animal_run_call(pose_estimation_tool=data_type_choice,
                                                                                                                                       interpolation_settings=self.interpolation_dropdown.getChoices(),
                                                                                                                                       smoothing_settings=self.smoothing_dropdown.getChoices(),
                                                                                                                                       smoothing_window=self.smoothing_time_eb.entry_get,
                                                                                                                                       animal_names=self.animal_name_entry_boxes,
                                                                                                                                       data_path=self.data_dir_select.folder_path))

            elif data_type_choice == "CSV (SLEAP)":
                self.data_dir_select = FolderSelect(self.data_dir_frm, "CSV SLEAP DIRECTORY:", lblwidth=25)
                self.instructions_lbl = Label(self.data_dir_frm, font=Formats.FONT_REGULAR.value, text="Please import videos before importing the SLEAP tracking data \n IF you are tracking more than ONE animal")
                self.run_btn = Button(self.import_frm, text="IMPORT SLEAP .CSV", fg="blue", font=Formats.FONT_REGULAR.value, command=lambda: self.__multi_animal_run_call(pose_estimation_tool=data_type_choice,
                                                                                                                                         interpolation_settings=self.interpolation_dropdown.getChoices(),
                                                                                                                                         smoothing_settings=self.smoothing_dropdown.getChoices(),
                                                                                                                                         smoothing_window=self.smoothing_time_eb.entry_get,
                                                                                                                                         animal_names=self.animal_name_entry_boxes,
                                                                                                                                         data_path=self.data_dir_select.folder_path))

            elif data_type_choice == "H5 (SLEAP)":
                self.data_dir_select = FolderSelect(self.data_dir_frm, "H5 SLEAP DIRECTORY", lblwidth=25)
                self.instructions_lbl = Label(self.data_dir_frm, font=Formats.FONT_REGULAR.value, text="Please import videos before importing the SLEAP H5 tracking data \n IF you are tracking more than ONE animal")
                self.run_btn = Button(self.import_frm, text="IMPORT SLEAP H5", font=Formats.FONT_REGULAR.value, fg="blue", command=lambda: self.__multi_animal_run_call(pose_estimation_tool=data_type_choice,
                                                                                                                                       interpolation_settings=self.interpolation_dropdown.getChoices(),
                                                                                                                                       smoothing_settings=self.smoothing_dropdown.getChoices(),
                                                                                                                                       smoothing_window=self.smoothing_time_eb.entry_get,
                                                                                                                                       animal_names=self.animal_name_entry_boxes,
                                                                                                                                       data_path=self.data_dir_select.folder_path))
            elif data_type_choice == "H5 (SuperAnimal-TopView)":
                self.data_dir_select = FolderSelect(self.data_dir_frm, "H5 SuperAnimal DIRECTORY:", lblwidth=25)
                self.instructions_lbl = Label(self.data_dir_frm, font=Formats.FONT_REGULAR.value, text="Please import videos before importing the H5 SuperAnimal tracking data \n IF you are tracking more than ONE animal")
                self.run_btn = Button(self.import_frm, text="IMPORT SuperAnimal-Mouse-TopView H5", fg="blue", font=Formats.FONT_REGULAR.value, command=lambda: self.__multi_animal_run_call(pose_estimation_tool=data_type_choice,
                                                                                                                                                                           interpolation_settings=self.interpolation_dropdown.getChoices(),
                                                                                                                                                                           smoothing_settings=self.smoothing_dropdown.getChoices(),
                                                                                                                                                                           smoothing_window=self.smoothing_time_eb.entry_get,
                                                                                                                                                                           animal_names=self.animal_name_entry_boxes,
                                                                                                                                                                           data_path=self.data_dir_select.folder_path))



            self.data_dir_frm.grid(row=self.frame_children(frame=self.choice_frm), column=0, sticky=NW)
            self.data_dir_select.grid(row=0, column=0, sticky=NW)
            self.instructions_lbl.grid(row=1, column=0, sticky=NW)
            self.import_frm.grid(row=self.frame_children(frame=self.choice_frm) + 1, column=0, sticky=NW)
            self.run_btn.grid(row=0, column=0, sticky=NW)
            self.choice_frm.grid(row=1, column=0, sticky=NW)

#_ = ImportPoseFrame(config_path=r"C:\troubleshooting\facemap_project\project_folder\project_config.ini")
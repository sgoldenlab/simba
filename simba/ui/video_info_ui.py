__author__ = "Simon Nilsson"

import os
from tkinter import *
from typing import Optional, Union

import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.px_to_mm_ui import GetPixelsPerMillimeterInterface
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, Entry_Box,
                                        SimbaButton, SimBALabel)
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists, check_int)
from simba.utils.enums import (ConfigKey, Dtypes, Formats, Keys, Links,
                               Options, TagNames)
from simba.utils.errors import InvalidInputError, PermissionError
from simba.utils.printing import log_event, stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_video_meta_data, read_config_entry,
                                    read_video_info_csv)

TABLE_HEADERS = ["INDEX", "VIDEO", "FPS", "RESOLUTION WIDTH", "RESOLUTION HEIGHT", "FIND DISTANCE", "DISTANCE IN MM", "PIXELS PER MM"]


class VideoInfoTable(ConfigReader, PopUpMixin):
    """
    Create GUI that allows users to modify resolutions, fps, and pixels-per-mm
    interactively of videos within the SimBA project. Data is stored within the project_folder/logs/video_info.csv
    file in the SimBA project.

    :param Union[str, os.PathLike] config_path: path to SimBA project config file in Configparser format
    :param Union[str, os.PathLike] video_dir: Optional path to directory with video files. If None, then read from the SimBA project as dictated by project config. Default None.
    :param Optional[Union[str, os.PathLike]] video_info_path: Optional path to vide_info.csv file. If None, then read from the SimBA project as dictated by project config. Default None.

    ..seealso::
       `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-3-set-video-parameters>`__.

    :example:
    >>> ui = VideoInfoTable(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini")
    >>> ui.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 video_dir: Optional[Union[str, os.PathLike]] = None,
                 video_info_path: Optional[Union[str, os.PathLike]] = None):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        log_event(logger_name=str(__class__.__name__), log_type=TagNames.CLASS_INIT.value, msg=self.create_log_msg_from_init_args(locals=locals()))
        if video_dir is not None:
            check_if_dir_exists(in_dir=video_dir, source=self.__class__.__name__)
            self.video_dir = video_dir
        if video_info_path is not None:
            check_file_exist_and_readable(file_path=video_info_path)
            self.video_info_path = video_info_path
        if os.path.isfile(self.video_info_path):
            self.video_info_df = read_video_info_csv(self.video_info_path).reset_index(drop=True)
            self.prior_videos = list(self.video_info_df['Video'])
        else:
            self.video_info_df, self.prior_videos = None, []
        self.distance_mm = read_config_entry(self.config, ConfigKey.FRAME_SETTINGS.value, ConfigKey.DISTANCE_MM.value, Dtypes.FLOAT.value,0.00)
        self.video_paths = find_files_of_filetypes_in_directory(directory=self.video_dir, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, raise_error=True, as_dict=True)
        self.max_char_vid_name = len(max(self.video_paths.keys(), key=len))
        PopUpMixin.__init__(self, title="VIDEO INFO", size=(1550, 800), icon='video')
        self.TABLE_COL_WIDTHS = ["5", self.max_char_vid_name + 5, "18", "18", "18", "18", "18", "18"]


    def _get_video_table_rows(self):
        self.videos = {}
        for cnt, (video_name, video_path) in enumerate(self.video_paths.items()):
            self.videos[video_name] = {}
            if video_name in self.prior_videos:
                try:
                    prior_data = self.read_video_info(video_name=video_name, raise_error=False)[0].reset_index(drop=True).iloc[0].to_dict()
                    fps, width, height, distance, pixels_per_mm = prior_data['fps'], prior_data['Resolution_width'], prior_data['Resolution_height'], prior_data['Distance_in_mm'], prior_data['pixels/mm']
                except:
                    fps, width, height = self.video_meta_data[video_name]['fps'], self.video_meta_data[video_name]['width'], self.video_meta_data[video_name]['height']
                    distance, pixels_per_mm = self.distance_mm, 'None'
            else:
                fps, width, height = self.video_meta_data[video_name]['fps'], self.video_meta_data[video_name]['width'], self.video_meta_data[video_name]['height']
                distance, pixels_per_mm = self.distance_mm, 'None'
            self.videos[video_name]["video_idx_lbl"] = Label(self.video_frm, text=str(cnt), font=Formats.FONT_REGULAR.value,  width=6)
            self.videos[video_name]["video_name_lbl"] = Label(self.video_frm, text=video_name, font=Formats.FONT_REGULAR.value,  width=self.max_char_vid_name)
            self.videos[video_name]["video_name_w_ext"] = os.path.basename(video_path)
            self.videos[video_name]["video_fps_eb"] = Entry_Box(parent=self.video_frm, fileDescription='', labelwidth=0, value=fps, entry_box_width=20, justify='center')
            self.videos[video_name]["video_width_eb"] = Entry_Box(parent=self.video_frm, fileDescription='', labelwidth=0, value=width, entry_box_width=20, justify='center', validation='numeric')
            self.videos[video_name]["video_height_eb"] = Entry_Box(parent=self.video_frm, fileDescription='', labelwidth=0, value=height, entry_box_width=20, justify='center', validation='numeric')
            self.videos[video_name]["video_known_distance_eb"] = Entry_Box(parent=self.video_frm, fileDescription='', labelwidth=0, value=distance, entry_box_width=20, justify='center')
            #self.videos[video_name]["find_dist_btn"] = Button(self.video_frm, text="CALCULATE DISTANCE", fg="black", command=lambda k=video_name: self._initate_find_distance(k))
            self.videos[video_name]["find_dist_btn"] = SimbaButton(parent=self.video_frm, txt="CALCULATE DISTANCE", txt_clr='black', img='calipher', font=Formats.FONT_HEADER.value, cmd=lambda k=video_name: self._initate_find_distance(k), hover_font=Formats.FONT_HEADER.value)
            self.videos[video_name]["video_px_per_mm"] = Entry_Box(parent=self.video_frm, fileDescription='', labelwidth=0, value=pixels_per_mm, entry_box_width=20, justify='center')
            self.videos[video_name]["video_idx_lbl"].grid(row=cnt+2, column=0, sticky=NW)
            self.videos[video_name]["video_name_lbl"].grid(row=cnt+2, column=1, sticky=NW)
            self.videos[video_name]["video_fps_eb"].grid(row=cnt+2, column=2, sticky=NW)
            self.videos[video_name]["video_width_eb"].grid(row=cnt+2, column=3, sticky=NW)
            self.videos[video_name]["video_height_eb"].grid(row=cnt+2, column=4, sticky=NW)
            self.videos[video_name]["find_dist_btn"].grid(row=cnt + 2, column=5, sticky=NW)
            self.videos[video_name]["video_known_distance_eb"].grid(row=cnt+2, column=6, sticky=NW)
            self.videos[video_name]["video_px_per_mm"].grid(row=cnt+2, column=7, sticky=NW)

    def _get_quick_set(self):
        self.quick_set_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="BATCH QUICK SET", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_PARAMETERS.value)
        self.quick_set_frm_known_distance_eb = Entry_Box(parent=self.quick_set_frm, fileDescription='KNOWN DISTANCE:', labelwidth=30, value='', entry_box_width=10)
        self.quick_set_frm_known_distance_btn = SimbaButton(parent=self.quick_set_frm, txt="APPLY", txt_clr='black', img='tick', font=Formats.FONT_HEADER.value, cmd=self._set_known_distance, cmd_kwargs={'distance': lambda: self.quick_set_frm_known_distance_eb.entry_get})
        self.quick_set_px_mm_eb = Entry_Box(parent=self.quick_set_frm, fileDescription='PIXEL PER MILLIMETER:', labelwidth=30, value='', entry_box_width=10)
        self.quick_set_px_mm_btn = SimbaButton(parent=self.quick_set_frm, txt="APPLY", txt_clr='black', img='tick', font=Formats.FONT_HEADER.value, cmd=self._set_px_per_mm, cmd_kwargs={'px_per_mm': lambda: self.quick_set_px_mm_eb.entry_get})
        self.quick_set_frm.grid(row=1, column=0, sticky=NW)
        self.quick_set_frm_known_distance_eb.grid(row=0, column=0, sticky=NW)
        self.quick_set_frm_known_distance_btn.grid(row=0, column=1, sticky=NW)
        self.quick_set_px_mm_eb.grid(row=1, column=0, sticky=NW)
        self.quick_set_px_mm_btn.grid(row=1, column=1, sticky=NW)


    def _set_known_distance(self, distance: str):
        check_float(name=f'{self.__class__.__name__} KNOWN DISTANCE', value=distance, min_value=10e-6, raise_error=True)
        for cnt, (video_name, video_path) in enumerate(self.video_paths.items()):
            self.videos[video_name]["video_known_distance_eb"].entry_set(val=distance)

    def _set_px_per_mm(self, px_per_mm: str):
        check_float(name=f'{self.__class__.__name__} PIXEL PER MILLIMETER', value=px_per_mm, min_value=10e-6, raise_error=True)
        for cnt, (video_name, video_path) in enumerate(self.video_paths.items()):
            self.videos[video_name]["video_px_per_mm"].entry_set(val=px_per_mm)

    def _get_video_meta_data(self):
        self.video_meta_data = {}
        for cnt, (video_name, video_path) in enumerate(self.video_paths.items()):
            self.video_meta_data[video_name] = get_video_meta_data(video_path=video_path)

    def _duplicate_idx_1_distance(self):
        first_key = list(self.video_paths.keys())[0]
        first_val = self.videos[first_key]["video_known_distance_eb"].entry_get
        if not check_float(name=self.__class__.__name__, value=first_val, min_value=10e-6, raise_error=False)[0]:
            raise InvalidInputError(msg=f'Distance for {first_key} cannot be duplicated. Not a valid distance value: {first_val}', source=self.__class__.__name__)
        for cnt, (video_name, video_path) in enumerate(self.video_paths.items()):
            self.videos[video_name]["video_known_distance_eb"].entry_set(val=first_val)

    def _duplicate_idx_1_px_mm(self):
        first_key = list(self.video_paths.keys())[0]
        first_val = self.videos[first_key]["video_px_per_mm"].entry_get
        if not check_float(name=self.__class__.__name__, value=first_val, min_value=10e-6, raise_error=False)[0]:
            raise InvalidInputError(msg=f'Pixel per millimeter for {first_key} cannot be duplicated. Not a valid distance value: {first_val}', source=self.__class__.__name__)
        for cnt, (video_name, video_path) in enumerate(self.video_paths.items()):
            self.videos[video_name]["video_px_per_mm"].entry_set(val=first_val)

    def _initate_find_distance(self, video_name: str):
        video_path = self.video_paths[video_name]
        known_distance = self.videos[video_name]['video_known_distance_eb'].entry_get
        check_float(name=f'{video_name} known distance', value=known_distance, min_value=10e-16, raise_error=True)
        px_per_mm_interface = GetPixelsPerMillimeterInterface(video_path=video_path, known_metric_mm=float(known_distance))
        px_per_mm_interface.run()
        self.videos[video_name]["video_px_per_mm"].entry_set(str(round(px_per_mm_interface.ppm, 5)))


    def _get_instructions_frm(self):
        self.instructions_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="INSTRUCTIONS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_PARAMETERS.value)
        self.instructions_label_1 = SimBALabel(parent=self.instructions_frm, txt='1. Enter the known distance (mm) in "DISTANCE IN MM." Use "autopopulate" or "batch quick set" for similar videos.', font=Formats.FONT_REGULAR.value)
        self.instructions_label_2 = SimBALabel(parent=self.instructions_frm, txt='2. Click on "CALCULTE DISTANCE" button(s) to calculate pixels/mm for each video.', font=Formats.FONT_REGULAR.value)
        self.instructions_label_3 = SimBALabel(parent=self.instructions_frm, txt="3. Click <SAVE PIXEL PER MILLIMETER DATA> when all the data are filled in.", font=Formats.FONT_REGULAR.value)
        self.instructions_frm.grid(row=0, column=0, sticky=W)
        self.instructions_label_1.grid(row=0, column=0, sticky=W)
        self.instructions_label_2.grid(row=1, column=0, sticky=W)
        self.instructions_label_3.grid(row=2, column=0, sticky=W)

    def _get_save_frm(self, index: int):
        self.save_frm = LabelFrame(self.main_frm, text="SAVE", font=Formats.FONT_HEADER.value)
        self.save_data_btn = SimbaButton(parent=self.save_frm, txt="SAVE PIXEL PER MILLIMETER DATA", txt_clr='black', img='save_large', font=Formats.FONT_HEADER.value, cmd=self._save)
        self.save_frm.grid(row=index, column=0, sticky=NW)
        self.save_data_btn.grid(row=0, column=0, sticky=NW)

    def _get_video_table(self):
        self.video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="PROJECT VIDEOS", icon_name='video_large', icon_link=Links.VIDEO_PARAMETERS.value, font=Formats.FONT_HEADER.value)
        for cnt, (col_name, col_width) in enumerate(zip(TABLE_HEADERS, self.TABLE_COL_WIDTHS)):
            col_header_label = Label( self.video_frm, text=col_name, width=col_width, font=Formats.FONT_HEADER.value)
            col_header_label.grid(row=0, column=cnt, sticky=W)
        self.video_frm.grid(row=3, column=0, sticky=NW)
        self._get_video_table_rows()

    def _get_duplicate_btns(self):
        self.duplicate_distance_btn = SimbaButton(parent=self.video_frm, txt="DUPLICATE INDEX 1", txt_clr='red', img='danger', font=Formats.FONT_REGULAR.value, cmd=self._duplicate_idx_1_distance)
        self.duplicate_distance_btn.grid(row=1, column=6, sticky=W, padx=5)
        self.duplicate_btn = SimbaButton(parent=self.video_frm, txt="DUPLICATE INDEX 1", txt_clr='red', img='danger', font=Formats.FONT_REGULAR.value, cmd=self._duplicate_idx_1_px_mm)
        self.duplicate_btn.grid(row=1, column=7, sticky=W, padx=5)


    def _save(self):
        self.video_info_df = pd.DataFrame(columns=Formats.EXPECTED_VIDEO_INFO_COLS.value)
        for video_name, video_path in self.videos.items():
            fps = self.videos[video_name]["video_fps_eb"].entry_get
            width = self.videos[video_name]["video_width_eb"].entry_get
            height = self.videos[video_name]["video_height_eb"].entry_get
            known_distance = self.videos[video_name]["video_known_distance_eb"].entry_get
            px_per_mm = self.videos[video_name]["video_px_per_mm"].entry_get
            check_float(name=f'{video_name} fps', value=fps, min_value=10e-16)
            check_int(name=f'{video_name} width', value=width, min_value=1, raise_error=True)
            check_int(name=f'{video_name} height', value=height, min_value=1, raise_error=True)
            check_float(name=f'{video_name} known_distance', value=known_distance, min_value=10e-16, raise_error=True)
            check_float(name=f'{video_name} px_per_mm', value=px_per_mm, min_value=10e-16, raise_error=True)
            self.video_info_df.loc[len(self.video_info_df)] = [video_name, fps, width, height, known_distance, px_per_mm]
        self.video_info_df.drop_duplicates(subset=["Video"], inplace=True)
        self.video_info_df = self.video_info_df.set_index("Video")
        try:
            self.video_info_df.to_csv(self.video_info_path)
        except PermissionError:
            raise PermissionError(msg=f"SimBA tried to write to {self.video_info_path}, but was not allowed. If this file is open in another program, try closing it.", source=self.__class__.__name__)
        stdout_success(msg=f"Video info saved at {self.video_info_path}", source=self.__class__.__name__)

    def run(self):
        self._get_video_meta_data()
        self._get_instructions_frm()
        self._get_save_frm(index=2)
        self._get_quick_set()
        self._get_video_table()
        self._get_duplicate_btns()
        self._get_save_frm(index=4)


# ui = VideoInfoTable(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini")
# ui.run()



# test.create_window()
#test.main_frm.mainloop()

# test = VideoInfoTable(config_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini')
# test.create_window()
# test.main_frm.mainloop()

#
# test = VideoInfoTable(config_path='/Users/simon/Desktop/envs/troubleshooting/sleap_5_animals_2/project_folder/project_config.ini')
# test.create_window()
# test.main_frm.mainloop()

# test = VideoInfoTable(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
# test.create_window()
# test.main_frm.mainloop()

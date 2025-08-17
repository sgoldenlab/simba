__author__ = "Simon Nilsson"

from copy import deepcopy
from tkinter import *

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.pose_plotter import PosePlotter
from simba.plotting.pose_plotter_mp import PosePlotterMultiProcess
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, FileSelect,
                                        FolderSelect, SimbaButton,
                                        SimBADropDown, SimBALabel)
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists, check_int)
from simba.utils.enums import Formats, Keys, Links, Options
from simba.utils.read_write import find_core_cnt, get_desktop_path, str_2_bool

ENTIRE_VIDEOS = "ENTIRE VIDEO(S)"
AUTO = 'AUTO'
KEYPOINT_SIZES = list(range(1, 101))
KEYPOINT_SIZES.insert(0, AUTO)
VIDEO_LENGTHS = list(range(10, 210, 10))
VIDEO_LENGTHS.insert(0, ENTIRE_VIDEOS)
COLOR_OPTIONS = deepcopy(Options.PALETTE_OPTIONS_CATEGORICAL.value)
COLOR_OPTIONS.insert(0, AUTO)


class VisualizePoseInFolderPopUp(PopUpMixin):

    """
    .. image:: _static/img/VisualizePoseInFolderPopUp.webp
       :width: 500
       :align: center

    :example:
    >>> VisualizePoseInFolderPopUp()
    """
    def __init__(self):

        PopUpMixin.__init__(self, title="VISUALIZE POSE ESTIMATION DATA", size=(800, 800), icon='pose')
        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.keypoint_size_dropdown = SimBADropDown(parent=self.settings_frm, label='KEY-POINT SIZES', label_width=30, dropdown_width=20, value=AUTO, command=None, dropdown_options=KEYPOINT_SIZES)
        self.video_slice_dropdown = SimBADropDown(parent=self.settings_frm, label="VIDEO SECTION (SECONDS):", label_width=30, dropdown_width=20, value=ENTIRE_VIDEOS, command=None, dropdown_options=VIDEO_LENGTHS)
        self.cpu_cnt_dropdown = SimBADropDown(parent=self.settings_frm, label="CPU COUNT:", label_width=30, dropdown_width=20, value=find_core_cnt()[1], command=None, dropdown_options=list(range(1, find_core_cnt()[0])))
        self.gpu_dropdown = SimBADropDown(parent=self.settings_frm, label="USE GPU:", label_width=30, dropdown_width=20, value='FALSE', command=None, dropdown_options=['TRUE', 'FALSE'])
        self.include_bbox_dropdown = SimBADropDown(parent=self.settings_frm, label="INCLUDE BOUNDING BOX:", label_width=30, dropdown_width=20, value='FALSE', command=None, dropdown_options=['TRUE', 'FALSE'])
        self.save_dir = FolderSelect(self.settings_frm, "SAVE DIRECTORY: ", title="Select a data folder", lblwidth=30, initialdir=get_desktop_path())
        self.number_of_animals_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=list(range(1, 17)), label="NUMBER OF ANIMALS:", label_width=30, dropdown_width=20, command=self.__create_table, value=1)
        self.__create_table(animal_cnt=1)
        self.settings_frm.grid(row=0, column=0, sticky=NW)
        self.keypoint_size_dropdown.grid(row=0, column=0, sticky=NW)
        self.video_slice_dropdown.grid(row=1, column=0, sticky=NW)
        self.cpu_cnt_dropdown.grid(row=2, column=0, sticky=NW)
        self.gpu_dropdown.grid(row=3, column=0, sticky=NW)
        self.include_bbox_dropdown.grid(row=4, column=0, sticky=NW)
        self.save_dir.grid(row=5, column=0, sticky=NW)
        self.number_of_animals_dropdown.grid(row=6, column=0, sticky=NW)

        self.main_frm.mainloop()

    def __create_table(self, animal_cnt):
        if hasattr(self, 'color_keypoint_frm'):
            self.color_keypoint_frm.destroy()
        animal_cnt = int(animal_cnt)
        dropdown_color_titles = [f'ANIMAL {x + 1} COLOR PALETTE:' for x in range(animal_cnt)]
        self.animal_clr_dropdowns = {}
        self.color_keypoint_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="ANIMAL COLORS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        for cnt, title in enumerate(dropdown_color_titles):
            self.animal_clr_dropdowns[title] = SimBADropDown(parent=self.color_keypoint_frm, dropdown_options=COLOR_OPTIONS, label=title, label_width=30, dropdown_width=20, value=COLOR_OPTIONS[0])
            self.animal_clr_dropdowns[title].grid(row=cnt, column=0, sticky=NW)
        self.color_keypoint_frm.grid(row=1, column=0, sticky=NW)

        self.single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="VISUALIZE SINGLE DATA FILE", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.single_instruction_lbl = SimBALabel(parent=self.single_video_frm, txt="SELECT CSV OR PARQUET FILE INSIDE 'PROJECT_FOLDER/CSV' SUB-DIRECTORIES", font=Formats.FONT_REGULAR_ITALICS.value)
        self.single_file_select = FileSelect(self.single_video_frm, "SELECT DATA FILE (CSV/PARQUET): ",  title="Select a data file", file_types=[("CSV or PARQUET", Options.WORKFLOW_FILE_TYPE_STR_OPTIONS.value)], lblwidth=40)
        self.single_file_run_btn = SimbaButton(parent=self.single_video_frm, txt='RUN', img='rocket', cmd=self.run, cmd_kwargs={'directory': lambda: False})
        self.single_video_frm.grid(row=2, column=0, sticky=NW)
        self.single_instruction_lbl.grid(row=0, column=0, sticky=NW)
        self.single_file_select.grid(row=1, column=0, sticky=NW)
        self.single_file_run_btn.grid(row=2, column=0, sticky=NW)

        self.video_dir_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="VISUALIZE DATA DIRECTORY", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.video_dir_lbl = SimBALabel(parent=self.video_dir_frm, txt="SELECT SUB-DIRECTORY INSIDE 'PROJECT_FOLDER/CSV' DIRECTORY", font=Formats.FONT_REGULAR_ITALICS.value)
        self.video_dir_select = FolderSelect(self.video_dir_frm, "SELECT DATA DIRECTORY (CSV/PARQUET): ",  title="Select a data folder", lblwidth=40)
        self.dir_run_btn = SimbaButton(parent=self.video_dir_frm, txt='RUN', img='rocket', cmd=self.run, cmd_kwargs={'directory': lambda: True})
        self.video_dir_frm.grid(row=3, column=0, sticky=NW)
        self.video_dir_lbl.grid(row=0, column=0, sticky=NW)
        self.video_dir_select.grid(row=1, column=0, sticky=NW)
        self.dir_run_btn.grid(row=2, column=0, sticky=NW)

    def run(self, directory: bool):
        if directory:
            data_path = self.video_dir_select.folder_path
            check_if_dir_exists(in_dir=data_path, source=self.__class__.__name__)
        else:
            data_path = self.single_file_select.file_path
            check_file_exist_and_readable(file_path=data_path)
        save_dir = self.save_dir.folder_path
        check_if_dir_exists(in_dir=save_dir, source=self.__class__.__name__)
        circle_size = self.keypoint_size_dropdown.getChoices()
        if not check_int(name='circle_size', value=circle_size, min_value=1, raise_error=False)[0]:
            circle_size = None
        else:
            circle_size = int(circle_size)
        gpu = str_2_bool(self.gpu_dropdown.get_value())
        bbox = str_2_bool(self.include_bbox_dropdown.get_value())
        cpu_cnt = int(self.cpu_cnt_dropdown.get_value())
        sample_time = self.video_slice_dropdown.getChoices()
        if sample_time == ENTIRE_VIDEOS: sample_time = None
        else: sample_time = int(sample_time)
        palettes = {}
        for cnt, (k, v) in enumerate(self.animal_clr_dropdowns.items()):
            if v.getChoices() == AUTO: palettes[cnt] = Options.PALETTE_OPTIONS_CATEGORICAL.value[cnt]
            else: palettes[cnt] = v.getChoices()
        if cpu_cnt == 1:
            plotter = PosePlotter(data_path=data_path,
                                  out_dir=save_dir,
                                  palettes=palettes,
                                  circle_size=circle_size,
                                  sample_time=sample_time)
        else:
            plotter = PosePlotterMultiProcess(data_path=data_path,
                                               out_dir=save_dir,
                                               palettes=palettes,
                                               gpu=gpu,
                                               circle_size=circle_size,
                                               sample_time=sample_time,
                                               bbox=bbox,
                                               core_cnt=cpu_cnt)
        plotter.run()

#VisualizePoseInFolderPopUp()
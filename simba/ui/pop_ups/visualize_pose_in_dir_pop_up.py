__author__ = "Simon Nilsson"

from copy import deepcopy
from tkinter import *
from tkinter import font

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.pose_plotter import PosePlotter
from simba.plotting.pose_plotter_mp import PosePlotterMultiProcess
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        FileSelect, FolderSelect)
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists)
from simba.utils.enums import Formats, Keys, Links, Options
from simba.utils.read_write import find_core_cnt

ENTIRE_VIDEOS = "ENTIRE VIDEO(S)"
AUTO = 'AUTO'


class VisualizePoseInFolderPopUp(PopUpMixin):

    """
    .. image:: _static/img/VisualizePoseInFolderPopUp.webp
       :width: 500
       :align: center

    :example:
    >>> VisualizePoseInFolderPopUp()
    """
    def __init__(self):
        PopUpMixin.__init__(self, title="VISUALIZE POSE_ESTIMATION DATA", size=(800, 800))
        self.keypoint_sizes = list(range(1, 101))
        self.keypoint_sizes.insert(0, AUTO)
        self.video_lengths = list(range(10, 210, 10))
        self.video_lengths.insert(0, ENTIRE_VIDEOS)
        self.color_options = deepcopy(Options.PALETTE_OPTIONS_CATEGORICAL.value)
        self.color_options.insert(0, AUTO)
        self.animal_cnt_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="NUMBER OF ANIMALS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.number_of_animals_dropdown = DropDownMenu(self.animal_cnt_frm, "NUMBER OF ANIMALS:", list(range(1, 17)), "20", com= lambda x: self.__populate_menu(x))
        self.number_of_animals_dropdown.setChoices(1)
        self.animal_cnt_frm.grid(row=0, column=0, sticky=NW)
        self.number_of_animals_dropdown.grid(row=0, column=0, sticky=NW)
        self.__populate_menu(1)
        #self.main_frm.mainloop()

    def __populate_menu(self, x):
        if hasattr(self, 'settings_menu'):
            self.settings_menu.destroy()
            self.single_video_frm.destroy()
            self.video_dir_frm.destroy()
        self.settings_menu = LabelFrame(self.main_frm, text="SETTINGS", font=Formats.FONT_HEADER.value, pady=5, padx=5, fg="black")
        self.settings_menu.grid(row=1, column=0, sticky=NW)
        self.save_dir = FolderSelect(self.settings_menu, "SAVE DIRECTORY: ",  title="Select a data folder", lblwidth=35)
        self.keypoint_size_dropdown = DropDownMenu(self.settings_menu, "KEYPOINT SIZES:", self.keypoint_sizes, "35")
        self.keypoint_size_dropdown.setChoices(self.keypoint_sizes[0])
        self.video_slice_dropdown = DropDownMenu(self.settings_menu, "VIDEO SECTION (SECONDS):", self.video_lengths, "35")
        self.video_slice_dropdown.setChoices(ENTIRE_VIDEOS)
        self.keypoint_size_dropdown.grid(row=0, column=0, sticky=NW)
        self.video_slice_dropdown.grid(row=1, column=0, sticky=NW)
        self.save_dir.grid(row=2, column=0, sticky=NW)
        dropdown_color_titles = [f'Animal_{x+1} KEYPOINT PALETTE:' for x in range(int(self.number_of_animals_dropdown.getChoices()))]
        self.dropdown_dict = {}
        for cnt, title in enumerate(dropdown_color_titles):
            self.dropdown_dict[title] = DropDownMenu(self.settings_menu, title, self.color_options, "35")
            self.dropdown_dict[title].setChoices(self.color_options[0])
            self.dropdown_dict[title].grid(row=cnt+3, column=0, sticky=NW)
        self.multiprocess_var = BooleanVar(value=False)
        multiprocess_cb = Checkbutton(self.settings_menu, text="Multi-process (faster)", font=Formats.FONT_REGULAR.value, variable=self.multiprocess_var, command=lambda: self.enable_dropdown_from_checkbox(check_box_var=self.multiprocess_var, dropdown_menus=[self.multiprocess_dropdown]))
        self.multiprocess_dropdown = DropDownMenu(self.settings_menu, "CPU cores:", list(range(2, find_core_cnt()[0])), "12")
        multiprocess_cb.grid(row=self.frame_children(frame=self.settings_menu), column=0, sticky=NW)
        self.multiprocess_dropdown.setChoices(find_core_cnt()[0])
        self.multiprocess_dropdown.disable()
        self.multiprocess_dropdown.grid(row=self.frame_children(frame=self.settings_menu), column=1, sticky=NW)
        self.single_video_frm = LabelFrame(self.main_frm, text="VISUALIZE DATA FILE", font=Formats.FONT_HEADER  .value, pady=5, padx=5, fg="black")
        self.file_select = FileSelect(self.single_video_frm, "SELECT DATA FILE (CSV/PARQUET): ",  title="Select a data file", file_types=[("CSV or PARQUET", Options.WORKFLOW_FILE_TYPE_STR_OPTIONS.value)], lblwidth=35)
        self.run_single_file_btn = Button(self.single_video_frm, text="RUN", command=lambda: self.run(dir=False), fg='blue')
        self.single_video_frm.grid(row=2, column=0, sticky=NW)
        Label(self.single_video_frm, text="SELECT CSV OR PARQUET FILE INSIDE 'PROJECT_FOLDER/CSV' SUB-DIRECTORIES \n", font=font.Font(family="Helvetica", size=12, slant="italic")).grid(row=0, column=0, sticky=NW)
        self.file_select.grid(row=1, column=0, sticky=NW)
        self.run_single_file_btn.grid(row=2, column=0, sticky=NW)
        self.video_dir_frm = LabelFrame(self.main_frm, text="VISUALIZE DATA DIRECTORY", font=Formats.FONT_HEADER.value, pady=5, padx=5, fg="black")
        self.folder_select = FolderSelect(self.video_dir_frm, "SELECT DATA DIRECTORY (CSV/PARQUET): ",  title="Select a data folder", lblwidth=35)
        self.run_folder_btn = Button(self.video_dir_frm, text="RUN", font=Formats.FONT_REGULAR.value, command=lambda: self.run(dir=True), fg='blue')
        self.video_dir_frm.grid(row=3, column=0, sticky=NW)
        Label(self.video_dir_frm, text="SELECT SUB-DIRECTORY INSIDE 'PROJECT_FOLDER/CSV' DIRECTORY \n", font=font.Font(family="Helvetica", size=12, slant="italic")).grid(row=0, column=0, sticky=NW)
        self.folder_select.grid(row=1, column=0, sticky=NW)
        self.run_folder_btn.grid(row=2, column=0, sticky=NW)
        self.main_frm.mainloop()

    def run(self, dir):
        if dir:
            data_path = self.folder_select.folder_path
            check_if_dir_exists(in_dir=data_path, source=self.__class__.__name__)
        else:
            data_path = self.file_select.file_path
            check_file_exist_and_readable(file_path=data_path)
        save_dir = self.save_dir.folder_path
        check_if_dir_exists(in_dir=save_dir, source=self.__class__.__name__)
        circle_size = self.keypoint_size_dropdown.getChoices()
        if circle_size == AUTO: circle_size = None
        else: circle_size = int(circle_size)
        palettes = {}
        for cnt, (k, v) in enumerate(self.dropdown_dict.items()):
            print(k, v, v.getChoices())
            if v.getChoices() == AUTO:
                palettes[cnt] = Options.PALETTE_OPTIONS_CATEGORICAL.value[cnt]
            else:
                palettes[cnt] = v.getChoices()
        sample_time = self.video_slice_dropdown.getChoices()
        if sample_time == ENTIRE_VIDEOS:
            sample_time = None
        else:
            sample_time = int(sample_time)
        if not self.multiprocess_var.get():
            plotter = PosePlotter(data_path=data_path,
                                  out_dir=save_dir,
                                  palettes=palettes,
                                  circle_size=circle_size,
                                  sample_time=sample_time)
        else:
            core_cnt = int(self.multiprocess_dropdown.getChoices())
            plotter = PosePlotterMultiProcess(data_path=data_path,
                                              out_dir=save_dir,
                                              palettes=palettes,
                                              circle_size=circle_size,
                                              sample_time=sample_time,
                                              core_cnt=core_cnt)

        plotter.run()

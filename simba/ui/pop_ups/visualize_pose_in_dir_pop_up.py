__author__ = "Simon Nilsson"

import threading
from tkinter import *

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.pose_plotter_mp import PosePlotter
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        FolderSelect)
from simba.utils.checks import check_if_dir_exists
from simba.utils.enums import Formats, Keys, Links
from simba.utils.lookups import get_color_dict
from simba.utils.read_write import find_core_cnt

ENTIRE_VIDEOS = "ENTIRE VIDEO(S)"


class VisualizePoseInFolderPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="Visualize pose-estimation", size=(500, 200))

        settings_frame = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="SETTINGS",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.VIDEO_TOOLS.value,
        )
        self.input_folder = FolderSelect(
            settings_frame,
            "Input directory (with csv/parquet files):",
            title="Select input folder",
            lblwidth=30,
        )
        self.output_folder = FolderSelect(
            settings_frame,
            "Output directory (where videos are saved):",
            title="Select output folder",
            lblwidth=30,
        )
        self.circle_size_dropdown = DropDownMenu(
            settings_frame, "Circle size:", list(range(1, 20)), "30"
        )
        self.circle_size_dropdown.setChoices(5)
        self.sample_size_options = list(range(10, 210, 10))
        self.sample_size_options.insert(0, ENTIRE_VIDEOS)
        self.sample_size_dropdown = DropDownMenu(
            settings_frame,
            "Video output sample sizes (s):",
            self.sample_size_options,
            "30",
        )
        self.sample_size_dropdown.setChoices(ENTIRE_VIDEOS)
        self.core_cnt_dropdown = DropDownMenu(
            settings_frame,
            "Core count (higher for faster runtime):",
            list(range(1, find_core_cnt()[0])),
            "30",
        )
        self.core_cnt_dropdown.setChoices(1)
        self.run_btn = Button(
            self.main_frm,
            text="VISUALIZE POSE",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="blue",
            command=lambda: self.run(),
        )
        self.advanced_settings_btn = Button(
            self.main_frm,
            text="OPEN ADVANCED SETTINGS",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="red",
            command=lambda: self.launch_adv_settings(),
        )
        settings_frame.grid(row=0, sticky=W)
        self.input_folder.grid(row=0, column=0, sticky=NW)
        self.output_folder.grid(row=1, column=0, sticky=NW)
        self.circle_size_dropdown.grid(row=2, column=0, sticky=NW)
        self.sample_size_dropdown.grid(row=3, column=0, sticky=NW)
        self.core_cnt_dropdown.grid(row=4, column=0, sticky=NW)
        self.run_btn.grid(row=5, column=0)
        self.advanced_settings_btn.grid(row=6, column=0)
        self.color_lookup = None

        self.main_frm.mainloop()

    def run(self):
        circle_size_int = int(self.circle_size_dropdown.getChoices())
        input_folder = self.input_folder.folder_path
        output_folder = self.output_folder.folder_path
        check_if_dir_exists(in_dir=input_folder)
        check_if_dir_exists(in_dir=output_folder)
        if self.color_lookup is not None:
            cleaned_color_lookup = {}
            for k, v in self.color_lookup.items():
                cleaned_color_lookup[k] = v.getChoices()
            self.color_lookup = cleaned_color_lookup
        if self.sample_size_dropdown.getChoices() == ENTIRE_VIDEOS:
            sample_time = None
        else:
            sample_time = int(self.sample_size_dropdown.getChoices())

        pose_plotter = PosePlotter(
            in_dir=input_folder,
            out_dir=output_folder,
            circle_size=circle_size_int,
            core_cnt=int(self.core_cnt_dropdown.getChoices()),
            color_settings=self.color_lookup,
            sample_time=sample_time,
        )

        threading.Thread(target=pose_plotter.run()).start()

    def launch_adv_settings(self):
        if self.advanced_settings_btn["text"] == "OPEN ADVANCED SETTINGS":
            self.advanced_settings_btn.configure(text="CLOSE ADVANCED SETTINGS")
            self.adv_settings_frm = LabelFrame(
                self.main_frm,
                text="ADVANCED SETTINGS",
                font=Formats.LABELFRAME_HEADER_FORMAT.value,
                pady=5,
                padx=5,
            )
            self.run_btn = Button(
                self.adv_settings_frm,
                text="VISUALIZE POSE",
                font=Formats.LABELFRAME_HEADER_FORMAT.value,
                fg="blue",
                command=lambda: self.run(),
            )
            self.confirm_btn = Button(
                self.adv_settings_frm,
                text="Confirm",
                command=lambda: self.launch_clr_menu(),
            )
            self.specify_animals_dropdown = DropDownMenu(
                self.adv_settings_frm, "ANIMAL COUNT: ", list(range(1, 11)), "20"
            )
            self.specify_animals_dropdown.setChoices(1)
            self.adv_settings_frm.grid(row=5, column=0, pady=10)
            self.specify_animals_dropdown.grid(row=0, column=0, sticky=NW)
            self.confirm_btn.grid(row=0, column=1)
            self.run_btn.grid(row=2, column=0)
        elif self.advanced_settings_btn["text"] == "CLOSE ADVANCED SETTINGS":
            if hasattr(self, "adv_settings_frm"):
                self.adv_settings_frm.destroy()
                self.color_lookup = None
            self.advanced_settings_btn.configure(text="OPEN ADVANCED SETTINGS")

    def launch_clr_menu(self):
        if hasattr(self, "color_table_frme"):
            self.color_table_frme.destroy()
        clr_dict = get_color_dict()
        self.color_table_frme = LabelFrame(
            self.adv_settings_frm,
            text="SELECT COLORS",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            pady=5,
            padx=5,
        )
        self.color_lookup = {}
        for animal_cnt in list(range(int(self.specify_animals_dropdown.getChoices()))):
            self.color_lookup["Animal_{}".format(str(animal_cnt + 1))] = DropDownMenu(
                self.color_table_frme,
                "Animal {} color:".format(str(animal_cnt + 1)),
                list(clr_dict.keys()),
                "20",
            )
            self.color_lookup["Animal_{}".format(str(animal_cnt + 1))].setChoices(
                list(clr_dict.keys())[animal_cnt]
            )
            self.color_lookup["Animal_{}".format(str(animal_cnt + 1))].grid(
                row=animal_cnt, column=0, sticky=NW
            )
        self.color_table_frme.grid(row=1, column=0, sticky=NW)


# test = VisualizePoseInFolderPopUp()

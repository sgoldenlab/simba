__author__ = "Simon Nilsson"

import os
import threading
from tkinter import *
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.ROI_plotter import ROIPlot
from simba.plotting.ROI_plotter_mp import ROIPlotMultiprocess
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        Entry_Box, FileSelect)
from simba.utils.checks import check_file_exist_and_readable, check_float
from simba.utils.enums import Formats, Keys, Links, Options
from simba.utils.read_write import find_all_videos_in_directory


class VisualizeROITrackingPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: Union[str, os.PathLike]):
        check_file_exist_and_readable(file_path=config_path)
        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        self.read_roi_data()
        self.video_file_paths = find_all_videos_in_directory(
            directory=self.video_dir, as_dict=True, raise_error=True
        )
        self.video_names = list(self.video_file_paths.keys())
        PopUpMixin.__init__(self, title="VISUALIZE ROI TRACKING", size=(800, 500))
        self.multiprocess_var = BooleanVar()
        self.show_pose_var = BooleanVar(value=True)
        self.animal_name_var = BooleanVar(value=True)
        self.settings_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="SETTINGS",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.ROI_DATA_PLOT.value,
        )
        self.threshold_entry_box = Entry_Box(
            self.settings_frm, "Body-part probability threshold", "30"
        )
        self.threshold_entry_box.entry_set(0.0)
        threshold_label = Label(
            self.settings_frm,
            text="Note: body-part locations detected with probabilities \n below this threshold is removed from the visualization(s).",
            font=("Helvetica", 10, "italic"),
        )
        self.show_pose_cb = Checkbutton(
            self.settings_frm,
            text="Show pose-estimated location",
            variable=self.show_pose_var,
        )
        self.show_animal_name_cb = Checkbutton(
            self.settings_frm, text="Show animal names", variable=self.animal_name_var
        )
        self.multiprocess_cb = Checkbutton(
            self.settings_frm,
            text="Multi-process (faster)",
            variable=self.multiprocess_var,
            command=lambda: self.enable_dropdown_from_checkbox(
                check_box_var=self.multiprocess_var,
                dropdown_menus=[self.multiprocess_dropdown],
            ),
        )
        self.multiprocess_dropdown = DropDownMenu(
            self.settings_frm, "CPU cores:", list(range(2, self.cpu_cnt)), "12"
        )
        self.multiprocess_dropdown.setChoices(2)
        self.multiprocess_dropdown.disable()
        self.body_parts_frm = LabelFrame(
            self.main_frm,
            text="SELECT BODY-PARTS",
            pady=10,
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        self.animal_cnt_dropdown = DropDownMenu(
            self.body_parts_frm,
            "NUMBER OF ANIMALS",
            list(range(1, self.animal_cnt + 1)),
            "20",
            com=lambda x: self.__populate_bp_dropdown(bp_cnt=x),
        )
        self.animal_cnt_dropdown.setChoices(1)
        self.__populate_bp_dropdown(bp_cnt=1)
        self.run_frm = LabelFrame(
            self.main_frm,
            text="RUN VISUALIZATION",
            pady=10,
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        self.single_video_frm = LabelFrame(
            self.run_frm,
            text="SINGLE video",
            pady=10,
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        self.single_video_dropdown = DropDownMenu(
            self.single_video_frm,
            "Select video",
            self.video_names,
            "15",
            com=lambda x: self.update_file_select_box_from_dropdown(
                filename=x, fileselectbox=self.select_video_file_select
            ),
        )
        self.single_video_dropdown.setChoices(self.video_names[0])
        self.select_video_file_select = FileSelect(
            self.single_video_frm,
            "",
            lblwidth="1",
            file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)],
            dropdown=self.single_video_dropdown,
        )
        self.select_video_file_select.filePath.set(self.video_names[0])
        self.single_video_btn = Button(
            self.single_video_frm,
            text="Create SINGLE ROI video",
            fg="blue",
            command=lambda: self.run(multiple=False),
        )
        self.all_videos_frm = LabelFrame(
            self.run_frm,
            text="ALL videos",
            pady=10,
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        self.all_videos_btn = Button(
            self.all_videos_frm,
            text=f"Create ALL ROI videos ({len(self.video_names)} videos found)",
            fg="red",
            command=lambda: self.run(multiple=True),
        )
        self.settings_frm.grid(row=0, column=0, sticky=NW)
        self.threshold_entry_box.grid(row=0, column=0, sticky=NW)
        threshold_label.grid(row=1, column=0, sticky=NW)
        self.show_pose_cb.grid(row=2, column=0, sticky=NW)
        self.show_animal_name_cb.grid(row=3, column=0, sticky=NW)
        self.multiprocess_cb.grid(row=4, column=0, sticky=NW)
        self.multiprocess_dropdown.grid(row=4, column=1, sticky=NW)

        self.body_parts_frm.grid(row=1, column=0, sticky=NW)
        self.animal_cnt_dropdown.grid(row=0, column=0, sticky=NW)

        self.run_frm.grid(row=2, column=0, sticky=NW)
        self.single_video_frm.grid(row=0, column=0, sticky=NW)
        self.single_video_dropdown.grid(row=0, column=0, sticky=NW)
        self.select_video_file_select.grid(row=0, column=1, sticky=NW)
        self.single_video_btn.grid(row=1, column=0, sticky=NW)
        self.all_videos_frm.grid(row=1, column=0, sticky=NW)
        self.all_videos_btn.grid(row=0, column=0, sticky=NW)
        self.main_frm.mainloop()

    def __populate_bp_dropdown(self, bp_cnt: int):
        if hasattr(self, "bp_dropdown_dict"):
            for k, v in self.bp_dropdown_dict.items():
                v.destroy()

        self.bp_dropdown_dict = {}
        for cnt in range(int(self.animal_cnt_dropdown.getChoices())):
            self.bp_dropdown_dict[cnt] = DropDownMenu(
                self.body_parts_frm,
                self.multi_animal_id_list[cnt],
                self.body_parts_lst,
                "20",
            )
            self.bp_dropdown_dict[cnt].setChoices(self.body_parts_lst[cnt])
            self.bp_dropdown_dict[cnt].grid(row=cnt + 1, column=0, sticky=NW)

    def run(self, multiple: bool):
        if multiple:
            videos = list(self.video_file_paths.values())
        else:
            videos = [self.video_file_paths[self.single_video_dropdown.getChoices()]]
        for video_path in videos:
            self.check_if_selected_video_path_exist_in_project(video_path=video_path)

        check_float(
            name="Body-part probability threshold",
            value=self.threshold_entry_box.entry_get,
            min_value=0.0,
            max_value=1.0,
        )
        style_attr = {
            "show_body_part": self.show_pose_var.get(),
            "show_animal_name": self.animal_name_var.get(),
        }
        body_parts = [v.getChoices() for k, v in self.bp_dropdown_dict.items()]
        for video_path in videos:
            if not self.multiprocess_var.get():
                roi_plotter = ROIPlot(
                    config_path=self.config_path,
                    video_path=video_path,
                    style_attr=style_attr,
                    threshold=float(self.threshold_entry_box.entry_get),
                    body_parts=body_parts,
                )
            else:
                core_cnt = self.multiprocess_dropdown.getChoices()
                roi_plotter = ROIPlotMultiprocess(
                    config_path=self.config_path,
                    video_path=video_path,
                    core_cnt=int(core_cnt),
                    style_attr=style_attr,
                    threshold=float(self.threshold_entry_box.entry_get),
                    body_parts=body_parts,
                )
            threading.Thread(target=roi_plotter.run()).start()


# _ = VisualizeROITrackingPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/project_config.ini')
# _ = VisualizeROITrackingPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Termites_5/project_folder/project_config.ini')
# _ = VisualizeROITrackingPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
# _ = VisualizeROITrackingPopUp(config_path='/Users/simon/Desktop/envs/simba_dev/tests/data/test_projects/mouse_open_field/project_folder/project_config.ini')
# _ = VisualizeROITrackingPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/dorian_2/project_folder/project_config.ini')

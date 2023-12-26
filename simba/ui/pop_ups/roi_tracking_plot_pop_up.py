__author__ = "Simon Nilsson"

import multiprocessing
import os
from tkinter import *

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.ROI_plotter import ROIPlot
from simba.plotting.ROI_plotter_mp import ROIPlotMultiprocess
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        Entry_Box, FileSelect)
from simba.utils.checks import check_float, check_if_filepath_list_is_empty
from simba.utils.enums import Formats, Keys, Links, Options
from simba.utils.printing import stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_fn_ext)


class VisualizeROITrackingPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: str):
        PopUpMixin.__init__(self, title="VISUALIZE ROI TRACKING", size=(800, 500))
        ConfigReader.__init__(self, config_path=config_path)
        self.video_list = []
        video_file_paths = find_files_of_filetypes_in_directory(
            directory=self.video_dir, extensions=[".mp4", ".avi"]
        )

        for file_path in video_file_paths:
            _, video_name, ext = get_fn_ext(filepath=file_path)
            self.video_list.append(video_name + ext)

        check_if_filepath_list_is_empty(
            filepaths=self.video_list,
            error_msg="No videos in SimBA project. Import videos into you SimBA project to visualize ROI tracking.",
        )
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
            text="Note: body-part locations detected with probabilities below this threshold is removed from visualization.",
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
            "Number of animals",
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
            self.video_list,
            "15",
            com=lambda x: self.update_file_select_box_from_dropdown(
                filename=x, fileselectbox=self.select_video_file_select
            ),
        )
        self.single_video_dropdown.setChoices(self.video_list[0])
        self.select_video_file_select = FileSelect(
            self.single_video_frm,
            "",
            lblwidth="1",
            file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)],
            dropdown=self.single_video_dropdown,
        )
        self.select_video_file_select.filePath.set(self.video_list[0])
        self.single_video_btn = Button(
            self.single_video_frm,
            text="Create SINGLE ROI video",
            fg="blue",
            command=lambda: self.run_visualize(multiple=False),
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
            text="Create ALL ROI videos ({} videos found)".format(
                str(len(self.video_list))
            ),
            fg="red",
            command=lambda: self.run_visualize(multiple=True),
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
            if len(self.bp_dropdown_dict.keys()) != bp_cnt:
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

    def run_visualize(self, multiple: bool):
        if multiple:
            videos = self.video_list
        else:
            videos = [self.single_video_dropdown.getChoices()]

        check_float(
            name="Body-part probability threshold",
            value=self.threshold_entry_box.entry_get,
            min_value=0.0,
            max_value=1.0,
        )
        style_attr = {}
        style_attr["Show_body_part"] = False
        style_attr["Show_animal_name"] = False
        if self.show_pose_var.get():
            style_attr["Show_body_part"] = True
        if self.animal_name_var.get():
            style_attr["Show_animal_name"] = True

        body_parts = {}
        for k, v in self.bp_dropdown_dict.items():
            body_parts[self.multi_animal_id_list[k]] = v.getChoices()

        if not self.multiprocess_var.get():
            selected_video_path = os.path.join(
                self.video_dir, self.single_video_dropdown.getChoices()
            )
            self.check_if_selected_video_path_exist_in_project(
                video_path=selected_video_path
            )
            self.config.set(
                "ROI settings",
                "probability_threshold",
                str(self.threshold_entry_box.entry_get),
            )
            with open(self.config_path, "w") as f:
                self.config.write(f)
            for video in videos:
                roi_plotter = ROIPlot(
                    ini_path=self.config_path,
                    video_path=video,
                    style_attr=style_attr,
                    threshold=float(self.threshold_entry_box.entry_get),
                    body_parts=body_parts,
                )
                roi_plotter.insert_data()
                roi_plotter_multiprocessor = multiprocessing.Process(
                    target=roi_plotter.run()
                )
                roi_plotter_multiprocessor.start()

        else:
            with open(self.config_path, "w") as f:
                self.config.write(f)
            core_cnt = self.multiprocess_dropdown.getChoices()
            for video in videos:
                roi_plotter = ROIPlotMultiprocess(
                    ini_path=self.config_path,
                    video_path=video,
                    core_cnt=int(core_cnt),
                    style_attr=style_attr,
                    threshold=float(self.threshold_entry_box.entry_get),
                    body_parts=body_parts,
                )
                roi_plotter.run()

        stdout_success(
            msg="All ROI videos created and saved in project_folder/frames/output/ROI_analysis directory",
            source=self.__class__.__name__,
        )


# _ = VisualizeROITrackingPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Termites_5/project_folder/project_config.ini')
# _ = VisualizeROITrackingPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
# _ = VisualizeROITrackingPopUp(config_path='/Users/simon/Desktop/envs/simba_dev/tests/data/test_projects/mouse_open_field/project_folder/project_config.ini')
# _ = VisualizeROITrackingPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/dorian_2/project_folder/project_config.ini')

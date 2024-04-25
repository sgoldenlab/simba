__author__ = "Simon Nilsson"


import threading
from tkinter import *

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.ROI_feature_visualizer import ROIfeatureVisualizer
from simba.plotting.ROI_feature_visualizer_mp import \
    ROIfeatureVisualizerMultiprocess
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        Entry_Box)
from simba.utils.checks import check_float
from simba.utils.enums import ConfigKey, Formats, Keys, Links
from simba.utils.errors import NoFilesFoundError
from simba.utils.read_write import (find_all_videos_in_directory,
                                    find_files_of_filetypes_in_directory,
                                    get_fn_ext)

ROI_CENTERS = "roi_centers"
ROI_EAR_TAGS = "roi_ear_tags"
DIRECTIONALITY = "directionality"
DIRECTIONALITY_STYLE = "directionality_style"
BORDER_COLOR = "border_color"
POSE = "pose_estimation"
ANIMAL_NAMES = "animal_names"


class VisualizeROIFeaturesPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: str):
        PopUpMixin.__init__(self, title="VISUALIZE ROI FEATURES", size=(400, 500))
        ConfigReader.__init__(self, config_path=config_path)
        self.video_file_paths = find_all_videos_in_directory(
            directory=self.video_dir, as_dict=True
        )
        self.video_list = [k for k in self.video_file_paths.keys()]

        if len(self.video_list) == 0:
            raise NoFilesFoundError(
                msg="SIMBA ERROR: No videos in SimBA project. Import videos into you SimBA project to visualize ROI features.",
                source=self.__class__.__name__,
            )

        self.settings_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="SETTINGS",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.ROI_FEATURES_PLOT.value,
        )
        self.threshold_entry_box = Entry_Box(
            self.settings_frm, "Probability threshold", "15"
        )
        self.threshold_entry_box.entry_set(0.0)
        threshold_label = Label(
            self.settings_frm,
            text="Note: body-part locations detected with probabilities below this threshold will be filtered out.",
            font=("Helvetica", 10, "italic"),
        )
        self.border_clr_dropdown = DropDownMenu(
            self.settings_frm, "Border color:", list(self.colors_dict.keys()), "12"
        )
        self.border_clr_dropdown.setChoices("Black")

        self.show_pose_var = BooleanVar(value=True)
        self.show_ROI_centers_var = BooleanVar(value=True)
        self.show_ROI_tags_var = BooleanVar(value=True)
        self.show_animal_names_var = BooleanVar(value=False)
        self.show_direction_var = BooleanVar(value=False)
        self.multiprocess_var = BooleanVar(value=False)
        show_pose_cb = Checkbutton(
            self.settings_frm, text="Show pose", variable=self.show_pose_var
        )
        show_roi_center_cb = Checkbutton(
            self.settings_frm,
            text="Show ROI centers",
            variable=self.show_ROI_centers_var,
        )
        show_roi_tags_cb = Checkbutton(
            self.settings_frm, text="Show ROI ear tags", variable=self.show_ROI_tags_var
        )

        show_animal_names_cb = Checkbutton(
            self.settings_frm,
            text="Show animal names",
            variable=self.show_animal_names_var,
        )

        show_roi_directionality_cb = Checkbutton(
            self.settings_frm,
            text="Show directionality",
            variable=self.show_direction_var,
            command=lambda: self.enable_dropdown_from_checkbox(
                check_box_var=self.show_direction_var,
                dropdown_menus=[self.directionality_type_dropdown],
            ),
        )

        multiprocess_cb = Checkbutton(
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

        self.directionality_type_dropdown = DropDownMenu(
            self.settings_frm, "Direction type:", ["funnel", "Lines"], "12"
        )
        self.directionality_type_dropdown.setChoices(choice="funnel")
        self.directionality_type_dropdown.disable()

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
        self.animal_cnt_dropdown.grid(row=0, column=0, sticky=NW)
        self.single_video_frm = LabelFrame(
            self.main_frm,
            text="Visualize ROI features on SINGLE video",
            pady=10,
            padx=10,
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        self.single_video_dropdown = DropDownMenu(
            self.single_video_frm, "Select video", self.video_list, "15"
        )
        self.single_video_dropdown.setChoices(self.video_list[0])
        self.single_video_btn = Button(
            self.single_video_frm,
            text="Visualize ROI features for SINGLE video",
            command=lambda: self.run(multiple=False),
        )

        self.all_videos_frm = LabelFrame(
            self.main_frm,
            text="Visualize ROI features on ALL videos",
            pady=10,
            padx=10,
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )

        self.all_videos_btn = Button(
            self.all_videos_frm,
            text="Generate ROI visualization on ALL videos",
            command=lambda: self.run(multiple=True),
        )
        self.settings_frm.grid(row=0, column=0, sticky=NW)
        self.threshold_entry_box.grid(row=0, sticky=NW)
        threshold_label.grid(row=1, sticky=NW)
        self.border_clr_dropdown.grid(row=2, sticky=NW)
        show_pose_cb.grid(row=3, sticky=NW)
        show_roi_center_cb.grid(row=4, sticky=NW)
        show_roi_tags_cb.grid(row=5, sticky=NW)
        show_animal_names_cb.grid(row=6, sticky=NW)
        show_roi_directionality_cb.grid(row=7, sticky=NW)
        self.directionality_type_dropdown.grid(row=7, column=1, sticky=NW)
        multiprocess_cb.grid(row=8, column=0, sticky=NW)
        self.multiprocess_dropdown.grid(row=8, column=1, sticky=NW)
        self.body_parts_frm.grid(row=1, column=0, sticky=NW)
        self.single_video_frm.grid(row=2, sticky=W)
        self.single_video_dropdown.grid(row=0, sticky=W)
        self.single_video_btn.grid(row=1, pady=12)
        self.all_videos_frm.grid(row=3, sticky=W, pady=10)
        self.all_videos_btn.grid(row=0, sticky=W)

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

    def run(self, multiple: bool):
        check_float(
            name="Body-part probability threshold",
            value=self.threshold_entry_box.entry_get,
            min_value=0.0,
            max_value=1.0,
        )
        style_attr = {
            ROI_CENTERS: self.show_ROI_centers_var.get(),
            ROI_EAR_TAGS: self.show_ROI_tags_var.get(),
            POSE: self.show_pose_var.get(),
            ANIMAL_NAMES: self.show_animal_names_var.get(),
            DIRECTIONALITY: self.show_direction_var.get(),
            BORDER_COLOR: self.colors_dict[self.border_clr_dropdown.getChoices()],
            DIRECTIONALITY_STYLE: self.directionality_type_dropdown.getChoices(),
        }
        if multiple:
            video_paths = [v for k, v in self.video_file_paths.items()]
        else:
            video_paths = [
                self.video_file_paths[self.single_video_dropdown.getChoices()]
            ]
        body_parts = [v.getChoices() for v in self.bp_dropdown_dict.values()]
        for video_path in video_paths:
            if not self.multiprocess_var.get():
                roi_feature_visualizer = ROIfeatureVisualizer(
                    config_path=self.config_path,
                    video_path=video_path,
                    body_parts=body_parts,
                    style_attr=style_attr,
                )
            else:
                core_cnt = int(self.multiprocess_dropdown.getChoices())
                roi_feature_visualizer = ROIfeatureVisualizerMultiprocess(
                    config_path=self.config_path,
                    video_path=video_path,
                    body_parts=body_parts,
                    style_attr=style_attr,
                    core_cnt=core_cnt,
                )
            threading.Thread(target=roi_feature_visualizer.run()).start()


# _ = VisualizeROIFeaturesPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/project_config.ini')


# _ = VisualizeROIFeaturesPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini')
# _ = VisualizeROIFeaturesPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Termites_5/project_folder/project_config.ini')
# ROIAnalysisPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Termites_5/project_folder/project_config.ini')

# _ = VisualizeROIFeaturesPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')

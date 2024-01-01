__author__ = "Simon Nilsson"

import os.path
from tkinter import *

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.plot_clf_results import PlotSklearnResultsSingleCore
from simba.plotting.plot_clf_results_mp import PlotSklearnResultsMultiProcess
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        Entry_Box, FileSelect)
from simba.utils.checks import check_float
from simba.utils.enums import Formats, Keys, Links, Options
from simba.utils.errors import NoFilesFoundError
from simba.utils.read_write import find_all_videos_in_directory


class SklearnVisualizationPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: str):
        PopUpMixin.__init__(self, title="VISUALIZE CLASSIFICATION (SKLEARN) RESULTS")
        ConfigReader.__init__(self, config_path=config_path)
        self.video_lst = find_all_videos_in_directory(directory=self.video_dir)
        self.use_default_font_settings_val = BooleanVar(value=True)
        self.create_videos_var = BooleanVar()
        self.create_frames_var = BooleanVar()
        self.include_timers_var = BooleanVar()
        self.rotate_img_var = BooleanVar()
        self.multiprocess_var = BooleanVar()

        bp_threshold_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="BODY-PART VISUALIZATION THRESHOLD",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.SKLEARN_PLOTS.value,
        )
        self.bp_threshold_lbl = Label(
            bp_threshold_frm,
            text="Body-parts detected below the set threshold won't be shown in the output videos.",
            font=("Helvetica", 11, "italic"),
        )
        self.bp_threshold_entry = Entry_Box(
            bp_threshold_frm, "Body-part probability threshold", "32"
        )
        self.get_bp_probability_threshold()

        self.style_settings_frm = LabelFrame(
            self.main_frm,
            text="STYLE SETTINGS",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            pady=5,
            padx=5,
            fg="black",
        )
        self.auto_compute_font_cb = Checkbutton(
            self.style_settings_frm,
            text="Auto-compute font/key-point settings",
            variable=self.use_default_font_settings_val,
            command=lambda: self.enable_entrybox_from_checkbox(
                check_box_var=self.use_default_font_settings_val,
                reverse=True,
                entry_boxes=[
                    self.sklearn_text_size_entry_box,
                    self.sklearn_text_spacing_entry_box,
                    self.sklearn_text_thickness_entry_box,
                    self.sklearn_circle_size_entry_box,
                ],
            ),
        )
        self.sklearn_text_size_entry_box = Entry_Box(
            self.style_settings_frm, "Text size: ", "12"
        )
        self.sklearn_text_spacing_entry_box = Entry_Box(
            self.style_settings_frm, "Text spacing: ", "12"
        )
        self.sklearn_text_thickness_entry_box = Entry_Box(
            self.style_settings_frm, "Text thickness: ", "12"
        )
        self.sklearn_circle_size_entry_box = Entry_Box(
            self.style_settings_frm, "Circle size: ", "12"
        )
        self.sklearn_text_size_entry_box.set_state("disable")
        self.sklearn_text_spacing_entry_box.set_state("disable")
        self.sklearn_text_thickness_entry_box.set_state("disable")
        self.sklearn_circle_size_entry_box.set_state("disable")

        self.settings_frm = LabelFrame(
            self.main_frm,
            text="VISUALIZATION SETTINGS",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            pady=5,
            padx=5,
            fg="black",
        )
        self.create_videos_cb = Checkbutton(
            self.settings_frm, text="Create video", variable=self.create_videos_var
        )
        self.create_frames_cb = Checkbutton(
            self.settings_frm, text="Create frames", variable=self.create_frames_var
        )
        self.timers_cb = Checkbutton(
            self.settings_frm,
            text="Include timers overlay",
            variable=self.include_timers_var,
        )
        self.rotate_cb = Checkbutton(
            self.settings_frm, text="Rotate video 90°", variable=self.rotate_img_var
        )
        self.multiprocess_cb = Checkbutton(
            self.settings_frm,
            text="Multiprocess videos (faster)",
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

        self.run_frm = LabelFrame(
            self.main_frm,
            text="RUN",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            pady=5,
            padx=5,
            fg="black",
        )
        self.run_single_video_frm = LabelFrame(
            self.run_frm,
            text="SINGLE VIDEO",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            pady=5,
            padx=5,
            fg="black",
        )
        self.run_single_video_btn = Button(
            self.run_single_video_frm,
            text="Create single video",
            fg="blue",
            command=lambda: self.__initiate_video_creation(multiple_videos=False),
        )
        self.single_video_dropdown = DropDownMenu(
            self.run_single_video_frm,
            "Video:",
            self.video_lst,
            "12",
            com=lambda x: self.__update_single_video_file_path(filename=x),
        )
        self.select_video_file_select = FileSelect(
            self.run_single_video_frm,
            "",
            lblwidth="1",
            file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)],
            dropdown=self.single_video_dropdown,
        )
        self.single_video_dropdown.setChoices(self.video_lst[0])
        self.select_video_file_select.filePath.set(self.video_lst[0])

        self.run_multiple_videos = LabelFrame(
            self.run_frm,
            text="MULTIPLE VIDEO",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            pady=5,
            padx=5,
            fg="black",
        )
        self.run_multiple_video_btn = Button(
            self.run_multiple_videos,
            text=f"Create multiple videos ({len(self.machine_results_paths)} video(s) found)",
            fg="blue",
            command=lambda: self.__initiate_video_creation(multiple_videos=True),
        )

        bp_threshold_frm.grid(row=0, sticky=NW)
        self.bp_threshold_lbl.grid(row=0, sticky=NW)
        self.bp_threshold_entry.grid(row=1, sticky=NW)

        self.style_settings_frm.grid(row=1, sticky=NW)
        self.auto_compute_font_cb.grid(row=0, sticky=NW)
        self.sklearn_text_size_entry_box.grid(row=1, sticky=NW)
        self.sklearn_text_spacing_entry_box.grid(row=2, sticky=NW)
        self.sklearn_text_thickness_entry_box.grid(row=3, sticky=NW)
        self.sklearn_circle_size_entry_box.grid(row=4, sticky=NW)

        self.settings_frm.grid(row=2, sticky=NW)
        self.create_videos_cb.grid(row=0, sticky=NW)
        self.create_frames_cb.grid(row=1, sticky=NW)
        self.timers_cb.grid(row=2, sticky=NW)
        self.rotate_cb.grid(row=3, sticky=NW)
        self.multiprocess_cb.grid(row=4, sticky=NW)
        self.multiprocess_dropdown.grid(row=4, column=1, sticky=NW)
        self.multiprocess_dropdown.disable()

        self.run_frm.grid(row=3, sticky=NW)

        self.run_single_video_frm.grid(row=0, sticky=NW)
        self.run_single_video_btn.grid(row=1, sticky=NW)
        self.single_video_dropdown.grid(row=1, column=1, sticky=NW)
        self.select_video_file_select.grid(row=1, column=2, sticky=NW)
        self.run_multiple_videos.grid(row=1, sticky=NW)
        self.run_multiple_video_btn.grid(row=0, sticky=NW)

        self.main_frm.mainloop()

    def __update_single_video_file_path(self, filename: str):
        self.select_video_file_select.filePath.set(filename)

    def get_bp_probability_threshold(self):
        try:
            self.bp_threshold_entry.entry_set(
                self.config.getfloat("threshold_settings", "bp_threshold_sklearn")
            )
        except:
            self.bp_threshold_entry.entry_set(0.0)

    def __initiate_video_creation(self, multiple_videos: bool = False):
        check_float(
            name="BODY-PART PROBABILITY THRESHOLD",
            value=self.bp_threshold_entry.entry_get,
            min_value=0.0,
            max_value=1.0,
        )
        self.config.set(
            "threshold_settings",
            "bp_threshold_sklearn",
            self.bp_threshold_entry.entry_get,
        )
        with open(self.config_path, "w") as f:
            self.config.write(f)

        if not self.use_default_font_settings_val.get():
            print_settings = {
                "font_size": self.sklearn_text_size_entry_box.entry_get,
                "circle_size": self.sklearn_circle_size_entry_box.entry_get,
                "space_size": self.sklearn_text_spacing_entry_box.entry_get,
                "text_thickness": self.sklearn_text_thickness_entry_box.entry_get,
            }
            for k, v in print_settings.items():
                check_float(name=v, value=v)
        else:
            print_settings = False

        if not multiple_videos:
            video_file_path = self.single_video_dropdown.getChoices()
            if not os.path.isfile(os.path.join(self.video_dir, video_file_path)):
                raise NoFilesFoundError(
                    msg=f"Selected video {video_file_path} is not a video in the SimBA project videos directory."
                )

        else:
            video_file_path = None

        if self.multiprocess_var.get():
            simba_plotter = PlotSklearnResultsMultiProcess(
                config_path=self.config_path,
                video_setting=self.create_videos_var.get(),
                rotate=self.rotate_img_var.get(),
                video_file_path=video_file_path,
                frame_setting=self.create_frames_var.get(),
                print_timers=self.include_timers_var.get(),
                cores=int(self.multiprocess_dropdown.getChoices()),
                text_settings=print_settings,
            )

        else:
            simba_plotter = PlotSklearnResultsSingleCore(
                config_path=self.config_path,
                video_setting=self.create_videos_var.get(),
                rotate=self.rotate_img_var.get(),
                video_file_path=video_file_path,
                frame_setting=self.create_frames_var.get(),
                print_timers=self.include_timers_var.get(),
                text_settings=print_settings,
            )
        simba_plotter.run()


# _ = SklearnVisualizationPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')

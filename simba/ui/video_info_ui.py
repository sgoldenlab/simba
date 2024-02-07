__author__ = "Simon Nilsson"

import collections
import os
from tkinter import *

import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import CreateLabelFrameWithIcon, hxtScrollbar
from simba.utils.enums import (ConfigKey, Dtypes, Formats, Keys, Links, Paths,
                               TagNames)
from simba.utils.errors import (InvalidInputError, NoFilesFoundError,
                                ParametersFileError, PermissionError)
from simba.utils.printing import log_event, stdout_success
from simba.utils.read_write import (find_all_videos_in_project, get_fn_ext,
                                    get_video_meta_data, read_config_entry,
                                    read_video_info_csv)
from simba.utils.warnings import DuplicateNamesWarning, InvalidValueWarning
from simba.video_processors.px_to_mm import get_coordinates_nilsson


class VideoInfoTable(ConfigReader, PopUpMixin):
    """
    Create GUI that allows users to modify resolutions, fps, and pixels-per-mm
    interactively of videos within the SimBA project. Data is stored within the project_folder/logs/video_info.csv
    file in the SimBA project.

    :param str config_file_path: path to SimBA project config file in Configparser format

    ..seealso::
       `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-3-set-video-parameters>`__.

    :example:
    >>> video_info_gui = VideoInfoTable(config_path='MyProjectConfig')
    >>> video_info_gui.create_window()
    >>> video_info_gui.main_frm.mainloop()
    """

    def __init__(self, config_path: str):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        log_event(
            logger_name=str(__class__.__name__),
            log_type=TagNames.CLASS_INIT.value,
            msg=self.create_log_msg_from_init_args(locals=locals()),
        )
        if os.path.isfile(self.video_info_path):
            self.video_info_df = read_video_info_csv(self.video_info_path).reset_index(
                drop=True
            )
        else:
            self.video_info_df = None
        self.distance_mm = read_config_entry(
            self.config,
            ConfigKey.FRAME_SETTINGS.value,
            ConfigKey.DISTANCE_MM.value,
            Dtypes.FLOAT.value,
            0.00,
        )
        self.video_paths = find_all_videos_in_project(videos_dir=self.video_dir)
        self.video_names = []
        for i in self.video_paths:
            _, video_name, _ = get_fn_ext(filepath=i)
            self.video_names.append(video_name)
        self.video_basename_lst = []
        self.max_char_vid_name = len(max(self.video_names, key=len))

    def __check_that_value_is_numeric(
        self, value=None, value_name=None, video_name=None
    ):
        if str(value).isdigit():
            return int(value)
        elif str(value).replace(".", "", 1).isdigit() and str(value).count(".") < 2:
            return float(value)
        else:
            raise ParametersFileError(
                msg=f"The {value_name} setting for video {video_name} is set to {value} in the project_folder/logs/video_info. Please set it to a numeric value.",
                source=self.__class__.__name__,
            )

    def __append_videos_from_video_folder(self):
        for cnt, name in enumerate(self.video_paths):
            _, video_basename, _ = get_fn_ext(name)
            self.video_basename_lst.append(video_basename)
            self.videos[name] = {}
            self.videos[name]["video_idx_lbl"] = Label(
                self.video_frm, text=str(cnt), width=6
            )
            self.videos[name]["video_name_lbl"] = Label(
                self.video_frm, text=video_basename, width=self.max_char_vid_name
            )
            self.videos[name]["video_name_w_ext"] = name
            video_meta = get_video_meta_data(os.path.join(self.video_dir, name))
            self.videos[name]["fps_var"] = IntVar()
            self.videos[name]["fps_var"].set(video_meta["fps"])
            self.videos[name]["fps_entry"] = Entry(
                self.video_frm, width=20, textvariable=self.videos[name]["fps_var"]
            )
            self.videos[name]["width_var"] = IntVar()
            self.videos[name]["width_var"].set(video_meta["width"])
            self.videos[name]["width_entry"] = Entry(
                self.video_frm, width=20, textvariable=self.videos[name]["width_var"]
            )
            self.videos[name]["height_var"] = IntVar()
            self.videos[name]["height_var"].set(video_meta["height"])
            self.videos[name]["height_entry"] = Entry(
                self.video_frm, width=20, textvariable=self.videos[name]["height_var"]
            )
            self.videos[name]["distance"] = StringVar()
            self.videos[name]["distance"].set(self.distance_mm)
            self.videos[name]["distance_entry"] = Entry(
                self.video_frm, width=20, textvariable=self.videos[name]["distance"]
            )
            self.videos[name]["find_dist_btn"] = Button(
                self.video_frm,
                text="Calculate distance",
                fg="black",
                command=lambda k=(
                    self.videos[name]["video_name_w_ext"],
                    self.videos[name]["distance"],
                ): self.__initiate_find_distance(k),
            )
            self.videos[name]["px_mm"] = StringVar()
            self.videos[name]["px_mm"].set(0)
            self.videos[name]["px_mm_entry"] = Entry(
                self.video_frm, width=20, textvariable=self.videos[name]["px_mm"]
            )
            if isinstance(self.video_info_df, pd.DataFrame):
                prior_data = None
                try:
                    prior_data = self.read_video_info(
                        video_name=video_basename, raise_error=False
                    )[0]
                except ParametersFileError:
                    pass
                if prior_data is not None:
                    for value_name, set_name in zip(
                        [
                            "fps",
                            "Resolution_width",
                            "Resolution_height",
                            "Distance_in_mm",
                            "pixels/mm",
                        ],
                        ["fps_var", "width_var", "height_var", "distance", "px_mm"],
                    ):
                        float_val = self.__check_that_value_is_numeric(
                            value=prior_data[value_name].values[0],
                            value_name=value_name,
                            video_name=prior_data["Video"].values[0],
                        )
                        self.videos[name][set_name].set(float_val)

    def __check_for_duplicate_names(self):
        duplicate_video_names = [
            item
            for item, count in collections.Counter(self.video_basename_lst).items()
            if count > 1
        ]
        if len(duplicate_video_names) > 0:
            for video_name in duplicate_video_names:
                print(video_name)
            DuplicateNamesWarning(
                msg=f"SimBA found {str(len(duplicate_video_names))} duplicate video name(s) in your SimBA project. The video(s) with duplicated names are printed above. This can happen if you have imported a video called MyVideo.mp4, and MyVideo.avi. Please avoid non-unique video names with different file extensions / video formats.",
                source=self.__class__.__name__,
            )

    def __check_no_zero_px_per_mm_values(self):
        for index, row in self.video_df.iterrows():
            try:
                px_per_mm_float = float(row["pixels/mm"])
            except:
                raise ParametersFileError(
                    msg=f'The pixels/mm for video {row["Video"]} is not a numeric value. Please set pixels per mm using the "Calculate distance" button before proceding.',
                    source=self.__class__.__name__,
                )
            if px_per_mm_float <= 0:
                InvalidValueWarning(
                    msg=f'The pixels/mm for video {row["Video"]} is set to zero. Please calculate pixels per mm using the "Calculate distance" button before proceding.',
                    source=self.__class__.__name__,
                )

    def create_window(self):
        PopUpMixin.__init__(self, title="VIDEO INFO", size=(1550, 800))
        self.instructions_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="INSTRUCTIONS",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.VIDEO_PARAMETERS.value,
        )
        self.intructions_label_1 = Label(
            self.instructions_frm,
            text='1. Enter the known distance (millimeters) in the "DISTANCE IN MM" column. Consider using the "autopopulate" entry box in the main project window you have a lot of videos.',
            font=("Helvetica", 15),
        )
        self.intructions_label_2 = Label(
            self.instructions_frm,
            text='2. Click on "Calculate distance" button(s) to calculate pixels/mm for each video.',
            font=("Helvetica", 15),
        )
        self.intructions_label_3 = Label(
            self.instructions_frm,
            text="3. Click <SAVE DATA> when all the data are filled in.",
            font=("Helvetica", 15),
        )
        self.instructions_frm.grid(row=0, column=0, sticky=W)
        self.intructions_label_1.grid(row=0, column=0, sticky=W)
        self.intructions_label_2.grid(row=1, column=0, sticky=W)
        self.intructions_label_3.grid(row=2, column=0, sticky=W)
        self.execute_frm = LabelFrame(
            self.main_frm, text="EXECUTE", font=Formats.LABELFRAME_HEADER_FORMAT.value
        )
        self.save_data_btn = Button(
            self.execute_frm,
            text="SAVE DATA",
            fg="green",
            command=lambda: self.__save_data(),
        )
        self.execute_frm.grid(row=1, column=0, sticky=W)
        self.save_data_btn.grid(row=0, column=0, sticky=W)
        self.video_frm = LabelFrame(
            self.main_frm,
            text="PROJECT VIDEOS",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
        )
        self.column_names = [
            "INDEX",
            "VIDEO",
            "FPS",
            "RESOLUTION WIDTH",
            "RESOLUTION HEIGHT",
            "DISTANCE IN MM",
            "FIND DISTANCE",
            "PIXELS PER MM",
        ]
        self.col_widths = [
            "5",
            self.max_char_vid_name + 5,
            "18",
            "18",
            "18",
            "18",
            "18",
            "18",
        ]
        self.video_frm.grid(row=6, column=0)
        for cnt, (col_name, col_width) in enumerate(
            zip(self.column_names, self.col_widths)
        ):
            col_header_label = Label(
                self.video_frm,
                text=col_name,
                width=col_width,
                font=Formats.LABELFRAME_HEADER_FORMAT.value,
            )
            col_header_label.grid(row=0, column=cnt, sticky=W)
        self.videos = {}
        self.__append_videos_from_video_folder()
        self.__check_for_duplicate_names()
        self.duplicate_distance_btn = Button(
            self.video_frm,
            text="Duplicate index 1 distance/mm (CAUTION!)",
            fg="red",
            command=lambda: self.__duplicate_idx_1_distance_mm(),
        )
        self.duplicate_distance_btn.grid(row=1, column=5, sticky=W, padx=5)
        self.duplicate_btn = Button(
            self.video_frm,
            text="Duplicate index 1 pixels/mm (CAUTION!)",
            fg="red",
            command=lambda: self.__duplicate_idx_1_px_mm(),
        )
        self.duplicate_btn.grid(row=1, column=7, sticky=W, padx=5)

        for vid_cnt, name in enumerate(self.videos.keys()):
            vid_cnt += 2
            self.videos[name]["video_idx_lbl"].grid(
                row=vid_cnt, column=0, sticky=W, padx=5
            )
            self.videos[name]["video_name_lbl"].grid(
                row=vid_cnt, column=1, sticky=W, padx=5
            )
            self.videos[name]["fps_entry"].grid(row=vid_cnt, column=2, sticky=W)
            self.videos[name]["width_entry"].grid(
                row=vid_cnt, column=3, sticky=W, padx=5
            )
            self.videos[name]["height_entry"].grid(
                row=vid_cnt, column=4, sticky=W, padx=5
            )
            self.videos[name]["distance_entry"].grid(
                row=vid_cnt, column=5, sticky=W, padx=5
            )
            self.videos[name]["find_dist_btn"].grid(row=vid_cnt, column=6, padx=5)
            self.videos[name]["px_mm_entry"].grid(row=vid_cnt, column=7, padx=5)
        Label(
            self.execute_frm,
            text=f"THE PROJECT CONTAINS {len(self.videos.keys())} VIDEOS",
            fg="blue",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            padx=20,
        ).grid(row=0, column=1, sticky=NW)

    def __initiate_find_distance(self, k):
        video_name, distance = k[0], k[1].get()
        try:
            distance = float(distance)
        except:
            raise InvalidInputError(
                msg=f"The *DISTANCE IN MM* for video {video_name} is not an integer or float value. The *DISTANCE IN MM* has to be a numerical value.",
                source=self.__class__.__name__,
            )
        if distance <= 0:
            raise InvalidInputError(
                msg=f"The *DISTANCE IN MM* for video {video_name} is <=0. The *DISTANCE IN MM* has to be a value above 0.",
                source=self.__class__.__name__,
            )
        video_pixels_per_mm = get_coordinates_nilsson(
            os.path.join(self.video_dir, video_name), distance
        )
        self.videos[video_name]["px_mm"].set(str(round(video_pixels_per_mm, 5)))

    def __duplicate_idx_1_px_mm(self):
        px_value = self.videos[list(self.videos.keys())[0]]["px_mm_entry"].get()
        for cnt, name in enumerate(self.video_paths):
            self.videos[name]["px_mm"].set(px_value)

    def __duplicate_idx_1_distance_mm(self):
        px_value = self.videos[list(self.videos.keys())[0]]["distance_entry"].get()
        for cnt, name in enumerate(self.video_paths):
            self.videos[name]["distance"].set(px_value)

    def __save_data(self):
        self.video_df = pd.DataFrame(
            columns=[
                "Video",
                "fps",
                "Resolution_width",
                "Resolution_height",
                "Distance_in_mm",
                "pixels/mm",
            ]
        )
        for name, data in self.videos.items():
            _, name, _ = get_fn_ext(name)
            lst = [
                name,
                data["fps_var"].get(),
                data["width_var"].get(),
                data["height_var"].get(),
                data["distance"].get(),
                data["px_mm"].get(),
            ]
            self.video_df.loc[len(self.video_df)] = lst
        self.__check_no_zero_px_per_mm_values()
        self.video_df.drop_duplicates(subset=["Video"], inplace=True)
        self.video_df = self.video_df.set_index("Video")
        try:
            self.video_df.to_csv(
                os.path.join(self.project_path, Paths.VIDEO_INFO.value)
            )
        except PermissionError:
            raise PermissionError(
                msg="SimBA tried to write to project_folder/logs/video_info.csv, but was not allowed. If this file is open in another program, try closing it.",
                source=self.__class__.__name__,
            )
        stdout_success(
            msg="Video info saved at project_folder/logs/video_info.csv",
            source=self.__class__.__name__,
        )


# test = VideoInfoTable(config_path='/Users/simon/Desktop/envs/troubleshooting/locomotion/project_folder/project_config.ini')
# test.create_window()
# test.main_frm.mainloop()

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

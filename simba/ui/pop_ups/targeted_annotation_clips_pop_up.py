import os
from tkinter import *
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, Entry_Box,
                                        FileSelect)
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_string_value_is_valid_video_timestamp,
                                check_int,
                                check_that_hhmmss_start_is_before_end)
from simba.utils.data import find_frame_numbers_from_time_stamp
from simba.utils.enums import Formats, Keys, Links, Options
from simba.utils.read_write import (
    check_if_hhmmss_timestamp_is_valid_part_of_video, find_video_of_file,
    get_fn_ext, read_df, read_video_info, remove_a_folder, write_df)
from simba.video_processors.video_processing import multi_split_video


class TargetedAnnotationsWClipsPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: Union[str, os.PathLike]):
        PopUpMixin.__init__(self, title="TAGETED ANNOTATIONS WITH CLIPS")
        ConfigReader.__init__(self, config_path=config_path)
        settings_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="Split videos into different parts",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.VIDEO_TOOLS.value,
        )
        self.selected_video = FileSelect(
            settings_frm,
            "Data path",
            title="Select a data file",
            lblwidth=10,
            file_types=[("VIDEO FILE", Options.WORKFLOW_FILE_TYPE_OPTIONS.value)],
        )
        self.clip_cnt = Entry_Box(
            settings_frm, "# of clips", "10", validation="numeric"
        )
        confirm_settings_btn = Button(
            settings_frm, text="Confirm", command=lambda: self.show_start_stop()
        )
        settings_frm.grid(row=0, column=0, sticky=NW)
        self.selected_video.grid(row=0, column=0, sticky=NW)
        self.clip_cnt.grid(row=1, column=0, sticky=NW)
        confirm_settings_btn.grid(row=2, column=0, sticky=NW)
        self.main_frm.mainloop()

    def show_start_stop(self):
        check_int(name="Number of clips", value=self.clip_cnt.entry_get)
        if hasattr(self, "table"):
            self.table.destroy()
        self.table = LabelFrame(self.main_frm)
        self.table.grid(row=2, column=0, sticky=NW)
        Label(self.table, text="Clip #").grid(row=0, column=0)
        Label(self.table, text="Start Time").grid(row=0, column=1, sticky=NW)
        Label(self.table, text="Stop Time").grid(row=0, column=2, sticky=NW)
        self.clip_names, self.start_times, self.end_times = [], [], []
        for i in range(int(self.clip_cnt.entry_get)):
            Label(self.table, text="Clip " + str(i + 1)).grid(row=i + 2, sticky=W)
            self.start_times.append(Entry(self.table))
            self.start_times[i].insert(0, "00:00:00")
            self.start_times[i].grid(row=i + 2, column=1, sticky=NW)
            self.end_times.append(Entry(self.table))
            self.end_times[i].insert(0, "00:00:00")
            self.end_times[i].grid(row=i + 2, column=2, sticky=NW)
        run_button = Button(
            self.table,
            text="RUN",
            command=lambda: self.run(),
            fg="navy",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
        )
        run_button.grid(row=int(self.clip_cnt.entry_get) + 2, column=2, sticky=W)

    def run(self):
        check_file_exist_and_readable(self.selected_video.file_path)
        _, file_name, _ = get_fn_ext(filepath=self.selected_video.file_path)
        machine_results_path = os.path.join(
            self.machine_results_dir, f"{file_name}.{self.file_type}"
        )
        check_file_exist_and_readable(file_path=machine_results_path)
        _, _, fps = read_video_info(
            vid_info_df=self.video_info_df, video_name=file_name
        )
        video_path = find_video_of_file(
            video_dir=self.video_dir, filename=file_name, raise_error=True
        )
        start_times, end_times = [], []
        check_file_exist_and_readable(self.selected_video.file_path)
        for cnt, (start_time, end_time) in enumerate(
            zip(self.start_times, self.end_times)
        ):
            start_times.append(start_time.get())
            end_times.append(end_time.get())
            check_if_string_value_is_valid_video_timestamp(
                name=f"Clip {cnt+1} start", value=start_time.get()
            )
            check_if_string_value_is_valid_video_timestamp(
                name=f"Clip {cnt+1} end", value=end_time.get()
            )
            check_that_hhmmss_start_is_before_end(
                start_time=start_time.get(),
                end_time=end_time.get(),
                name=f"Clip {cnt+1}",
            )
            check_if_hhmmss_timestamp_is_valid_part_of_video(
                timestamp=start_time.get(), video_path=video_path
            )
            check_if_hhmmss_timestamp_is_valid_part_of_video(
                timestamp=end_time.get(), video_path=video_path
            )

        print("Preparing video and data clips...")
        temp_dir = os.path.join(
            self.input_frames_dir, "advanced_clip_annotator", file_name
        )
        if os.path.isdir(temp_dir):
            remove_a_folder(folder_dir=temp_dir)
        if not os.path.isdir(temp_dir):
            os.makedirs(temp_dir)
        multi_split_video(
            file_path=video_path,
            start_times=start_times,
            end_times=end_times,
            out_dir=temp_dir,
            include_clip_time_in_filename=True,
        )

        df = read_df(file_path=machine_results_path, file_type=self.file_type)
        for cnt, (start_time, end_time) in enumerate(zip(start_times, end_times)):
            frm_numbers = find_frame_numbers_from_time_stamp(
                start_time=start_time, end_time=end_time, fps=fps
            )
            sliced = df.iloc[frm_numbers, :]
            new_start_time = start_time.replace(":", "-")
            new_end_time = start_time.replace(":", "-")
            save_path = os.path.join(
                temp_dir,
                f"{file_name}_{new_start_time}_{new_end_time}.{self.file_type}",
            )
            write_df(df=sliced, file_type=self.file_type, save_path=save_path)
            print(f"Data clip {cnt+1} saved...")
        self.root.destroy()
        self.main_frm.destroy()


# TargetedAnnotationsWClipsPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')

import os
from typing import Union
from tkinter import *

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.utils.checks import check_if_dir_exists, check_int, check_float, check_that_hhmmss_start_is_before_end
from simba.utils.read_write import find_all_videos_in_directory, get_video_meta_data, seconds_to_timestamp, check_if_hhmmss_timestamp_is_valid_part_of_video
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, Entry_Box, FolderSelect)
from simba.utils.enums import Keys, Links
from simba.utils.errors import FrameRangeError
from simba.video_processors.video_processing import clip_videos_by_frame_ids, clip_video_in_range
from simba.utils.printing import stdout_success, SimbaTimer


class ClipMultipleVideosByFrameNumbers(PopUpMixin):

    def __init__(self,
                 data_dir: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike]):

        check_if_dir_exists(in_dir=data_dir, source=self.__class__.__name__, create_if_not_exist=False)
        check_if_dir_exists(in_dir=save_dir, source=self.__class__.__name__, create_if_not_exist=True)
        self.video_paths = find_all_videos_in_directory(directory=data_dir, as_dict=True, raise_error=True)
        self.video_meta_data = [get_video_meta_data(video_path=x)['frame_count'] for x in list(self.video_paths.values())]
        max_video_name_len = len(max(list(self.video_paths.keys())))
        super().__init__(title="CLIP MULTIPLE VIDEOS BY FRAME NUMBERS")
        self.save_dir = save_dir
        data_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="VIDEO SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        data_frm.grid(row=0, column=0, sticky=NW)
        Label(data_frm, text="VIDEO NAME", width=max_video_name_len).grid(row=0, column=0, sticky=NW)
        Label(data_frm, text="TOTAL FRAMES", width=10).grid(row=0, column=1)
        Label(data_frm, text="START FRAME", width=10).grid(row=0, column=2)
        Label(data_frm, text="END FRAME", width=10).grid(row=0, column=3)
        self.entry_boxes = {}
        for cnt, video_name in enumerate(self.video_paths.keys()):
            self.entry_boxes[video_name] = {}
            Label(data_frm, text=video_name, width=max_video_name_len).grid(row=cnt+1, column=0, sticky=NW)
            Label(data_frm, text=self.video_meta_data[cnt], width=max_video_name_len).grid(row=cnt + 1, column=1, sticky=NW)
            self.entry_boxes[video_name]['start'] = Entry_Box(data_frm, "", 5, validation="numeric")
            self.entry_boxes[video_name]['end'] = Entry_Box(data_frm, "", 5, validation="numeric")
            self.entry_boxes[video_name]['start'].grid(row=cnt+1, column=2, sticky=NW)
            self.entry_boxes[video_name]['end'].grid(row=cnt+1, column=3, sticky=NW)
        self.create_run_frm(run_function=self.run, btn_txt_clr='blue')
        self.main_frm.mainloop()


    def run(self):
        video_paths, frame_ids = [], []
        for cnt, (video_name, v) in enumerate(self.entry_boxes.items()):
            video_paths.append(self.video_paths[video_name])
            video_frm_cnt = self.video_meta_data[cnt]
            check_int(name=f'START {video_name}', value=v['start'].entry_get, min_value=0)
            check_int(name=f'START {video_name}', value=v['end'].entry_get, min_value=1)
            start, end = int(v['start'].entry_get), int(v['end'].entry_get)
            if start >= end: raise FrameRangeError(msg=f'For video {video_name}, the start frame ({start}) is after or the same as the end frame ({end})', source=__class__.__name__)
            if (start < 0) or (end < 1):
                raise FrameRangeError(msg=f'For video {video_name}, start frame has to be at least 0 and end frame has to be at least 1', source=__class__.__name__)
            if start > video_frm_cnt:
                raise FrameRangeError(
                    msg=f'The video {video_name} has {video_frm_cnt} frames, which is less than the start frame: {start}', source=__class__.__name__)
            if end > video_frm_cnt:
                raise FrameRangeError(msg=f'The video {video_name} has {video_frm_cnt} frames, which is less than the end frame: {end}', source=__class__.__name__)
            frame_ids.append([start, end])

        _ = clip_videos_by_frame_ids(file_paths=video_paths, frm_ids=frame_ids, save_dir=self.save_dir)

class InitiateClipMultipleVideosByFrameNumbersPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title="CLIP MULTIPLE VIDEOS BY FRAME NUMBERS")
        data_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SELECT DATA DIRECTORIES", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.input_folder = FolderSelect(data_frm, "Video directory:", title="Select Folder with videos", lblwidth=20)
        self.output_folder = FolderSelect(data_frm, "Output directory:", title="Select a folder for your output videos", lblwidth=20)
        data_frm.grid(row=0, column=0, sticky=NW)
        self.input_folder.grid(row=0, column=0, sticky=NW)
        self.output_folder.grid(row=1, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        check_if_dir_exists(in_dir=self.input_folder.folder_path, source=self.__class__.__name__, create_if_not_exist=False)
        check_if_dir_exists(in_dir=self.output_folder.folder_path, source=self.__class__.__name__, create_if_not_exist=True)
        self.video_paths = find_all_videos_in_directory(directory=self.input_folder.folder_path, as_dict=True, raise_error=True)
        self.root.destroy()
        _ = ClipMultipleVideosByFrameNumbers(data_dir=self.input_folder.folder_path, save_dir=self.output_folder.folder_path)


class ClipMultipleVideosByTimestamps(PopUpMixin):

    def __init__(self,
                 data_dir: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike]):

        check_if_dir_exists(in_dir=data_dir, source=self.__class__.__name__, create_if_not_exist=False)
        check_if_dir_exists(in_dir=save_dir, source=self.__class__.__name__, create_if_not_exist=True)
        self.video_paths = find_all_videos_in_directory(directory=data_dir, as_dict=True, raise_error=True)
        self.video_meta_data = [get_video_meta_data(video_path=x) for x in list(self.video_paths.values())]
        max_video_name_len = len(max(list(self.video_paths.keys())))
        super().__init__(title="CLIP MULTIPLE VIDEOS BY TIME-STAMPS")
        self.save_dir = save_dir
        data_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="VIDEO SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        data_frm.grid(row=0, column=0, sticky=NW)
        self.save_dir = save_dir
        data_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="VIDEO SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        data_frm.grid(row=0, column=0, sticky=NW)
        Label(data_frm, text="VIDEO NAME", width=max_video_name_len).grid(row=0, column=0, sticky=NW)
        Label(data_frm, text="VIDEO LENGTH", width=10).grid(row=0, column=1)
        Label(data_frm, text="START TIME (HH:MM:SS)", width=18).grid(row=0, column=2)
        Label(data_frm, text="END TIME (HH:MM:SS)", width=18).grid(row=0, column=3)

        self.entry_boxes = {}
        for cnt, video_name in enumerate(self.video_paths.keys()):
            self.entry_boxes[video_name] = {}
            Label(data_frm, text=video_name, width=max_video_name_len).grid(row=cnt+1, column=0, sticky=NW)
            video_length = self.video_meta_data[cnt]['video_length_s']
            video_length_hhmmss = seconds_to_timestamp(seconds=video_length)
            Label(data_frm, text=video_length_hhmmss, width=max_video_name_len).grid(row=cnt + 1, column=1, sticky=NW)
            self.entry_boxes[video_name]['start'] = Entry_Box(data_frm, "", 5)
            self.entry_boxes[video_name]['end'] = Entry_Box(data_frm, "", 5)
            self.entry_boxes[video_name]['start'].grid(row=cnt+1, column=2, sticky=NW)
            self.entry_boxes[video_name]['end'].grid(row=cnt+1, column=3, sticky=NW)
        self.create_run_frm(run_function=self.run, btn_txt_clr='blue')
        self.main_frm.mainloop()

    def run(self):
        timer = SimbaTimer(start=True)
        for cnt, (video_name, v) in enumerate(self.entry_boxes.items()):
            start, end = v['start'].entry_get, v['end'].entry_get
            check_that_hhmmss_start_is_before_end(start_time=start, end_time=end, name=video_name)
            check_if_hhmmss_timestamp_is_valid_part_of_video(timestamp=start, video_path=self.video_paths[video_name])
            check_if_hhmmss_timestamp_is_valid_part_of_video(timestamp=end, video_path=self.video_paths[video_name])
            clip_video_in_range(file_path=self.video_paths[video_name], start_time=start, end_time=end, out_dir=self.save_dir, overwrite=True, include_clip_time_in_filename=False)
        timer.stop_timer()
        stdout_success(msg=f'{len(self.entry_boxes)} videos clipped by time-stamps and saved in {self.save_dir}', elapsed_time=timer.elapsed_time_str)


class InitiateClipMultipleVideosByTimestampsPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title="CLIP MULTIPLE VIDEOS BY TIME-STAMPS")
        data_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SELECT DATA DIRECTORIES", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.input_folder = FolderSelect(data_frm, "Video directory:", title="Select Folder with videos", lblwidth=20)
        self.output_folder = FolderSelect(data_frm, "Output directory:", title="Select a folder for your output videos", lblwidth=20)
        data_frm.grid(row=0, column=0, sticky=NW)
        self.input_folder.grid(row=0, column=0, sticky=NW)
        self.output_folder.grid(row=1, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        check_if_dir_exists(in_dir=self.input_folder.folder_path, source=self.__class__.__name__, create_if_not_exist=False)
        check_if_dir_exists(in_dir=self.output_folder.folder_path, source=self.__class__.__name__, create_if_not_exist=True)
        self.video_paths = find_all_videos_in_directory(directory=self.input_folder.folder_path, as_dict=True, raise_error=True)
        self.root.destroy()
        _ = ClipMultipleVideosByTimestamps(data_dir=self.input_folder.folder_path, save_dir=self.output_folder.folder_path)

#InitiateClipMultipleVideosByTimestampsPopUp()

#ClipMultipleVideosByTimestamps(data_dir='/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/videos/test', save_dir='/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/videos/clipped')


#InitiateClipMultipleVideosByFrameNumbersPopUp()
#ClipMultipleVideosByFrameNumbers(data_dir='/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/videos/test', save_dir='/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/videos/clipped')

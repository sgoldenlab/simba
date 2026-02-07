from typing import Tuple, Union

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import glob
import json
import os
import shutil

from simba.utils.checks import (check_ffmpeg_available,
                                check_file_exist_and_readable, check_str)
from simba.utils.errors import CropError, FFMPEGNotFoundError, PermissionError
from simba.utils.lookups import (get_ffmpeg_encoders, gpu_quality_to_cpu_quality_lk)
from simba.utils.printing import stdout_information
from simba.utils.warnings import CropWarning
from simba.video_processors.video_processing import (change_single_video_fps,
                                                     clahe_enhance_video,
                                                     clip_video_in_range,
                                                     crop_video,
                                                     downsample_video,
                                                     superimpose_frame_count,
                                                     video_to_greyscale)


class FFMPEGCommandCreator(object):
    """
    Execute FFmpeg commands from instructions stored in json format.

    Parameters
    ----------
    json_path: str
        path to json file storing FFmpeg instructions as created by ``simba.batch_process_vides.BatchProcessFrame``.

    Notes
    ----------
    `Batch pre-process tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial_process_videos.md>`__.
    `Example expected JSON file <https://github.com/sgoldenlab/simba/blob/master/misc/batch_process_log.json>`__.

    Examples
    ----------
    >>> ffmpeg_executor = FFMPEGCommandCreator(json_path='MyJsonFilePath')
    >>> ffmpeg_executor.crop_videos()
    >>> ffmpeg_executor.clip_videos()
    >>> ffmpeg_executor.downsample_videos()
    >>> ffmpeg_executor.apply_fps()
    >>> ffmpeg_executor.apply_grayscale()
    >>> ffmpeg_executor.apply_frame_count()
    >>> ffmpeg_executor.apply_clahe()
    >>> ffmpeg_executor.move_all_processed_files_to_output_folder()
    """

    def __init__(self,
                 json_path: Union[str, os.PathLike],
                 codec: str = 'libx264'):

        if not check_ffmpeg_available(raise_error=False):
            raise FFMPEGNotFoundError(msg='Cannot perform batch video processing: FFMPEG not found', source=self.__class__.__name__)

        check_file_exist_and_readable(json_path)
        with open(json_path, "r") as fp:
            self.video_dict = json.load(fp)
        self.input_dir = self.video_dict["meta_data"]["in_dir"]
        self.out_dir = self.video_dict["meta_data"]["out_dir"]
        if "gpu" in list(self.video_dict["meta_data"].keys()):
            self.gpu = self.video_dict["meta_data"]["gpu"]
        else:
            self.gpu = False
        self.quality = 23
        self.temp_dir = os.path.join(self.out_dir, "temp")
        self.gpu_to_cpu_quality_lk = gpu_quality_to_cpu_quality_lk()
        self.time_format = "%H:%M:%S"
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        os.makedirs(self.temp_dir)
        self.copy_videos_to_temp_dir()
        check_str(name=f'{self.__class__.__name__} codec', value=codec, allow_blank=False)
        self.batch_codec = codec if codec in get_ffmpeg_encoders() else 'mpeg4'

    def replace_files_in_temp(self):
        processed_files = glob.glob(self.process_dir + "/*")
        for file_path in processed_files:
            file_path_basename = os.path.basename(file_path)
            shutil.move(file_path, os.path.join(self.temp_dir, file_path_basename))

    def create_process_dir(self):
        self.process_dir = os.path.join(self.temp_dir, "process_dir")
        if os.path.exists(self.process_dir):
            try:
                shutil.rmtree(self.process_dir)
            except Exception as e:
                print(e.args)
                raise PermissionError(msg=f'SimBA is not allowed to delete/manipulate {self.process_dir}. Are you doing batch processing on a VirtualDrive or Cloud storage? Try performing the batch processing on a local drive.', source=self.__class__.__name__)
        os.makedirs(self.process_dir)

    def find_relevant_videos(self, variable=None):
        videos = {}
        for video, video_data in self.video_dict["video_data"].items():
            if video_data[variable]:
                video_basename = (
                    video_data["video_info"]["video_name"]
                    + video_data["video_info"]["extension"]
                )
                videos[video_data["video_info"]["video_name"]] = {}
                videos[video_data["video_info"]["video_name"]]["path"] = os.path.join(
                    self.temp_dir, video_basename
                )
                videos[video_data["video_info"]["video_name"]]["settings"] = video_data[
                    variable + "_settings"
                ]
                videos[video_data["video_info"]["video_name"]]["last_operation"] = (
                    video_data["last_operation"]
                )
                videos[video_data["video_info"]["video_name"]]["output_quality"] = (
                    video_data["output_quality"]
                )
        return videos

    def downsample_videos(self):
        self.videos_to_downsample = self.find_relevant_videos(variable="downsample")
        self.create_process_dir()
        for cnt, (video, video_data) in enumerate(self.videos_to_downsample.items()):
            msg = f"Down-sampling {video} to {video_data['settings']['width']}x{video_data['settings']['height']} ({cnt+1}/{len(list(self.videos_to_downsample.keys()))}) ... "
            stdout_information(msg=msg)
            if video_data["last_operation"] == "downsample":
                self.quality = video_data["output_quality"]
            in_path, out_path = video_data["path"], os.path.join(self.process_dir, os.path.basename(video_data["path"]))
            downsample_video(file_path=in_path, save_path=out_path, gpu=self.gpu, quality=self.quality, verbose=False, codec=self.batch_codec, video_width=int(video_data["settings"]["width"]), video_height=int(video_data["settings"]["height"]))
        self.replace_files_in_temp()
        stdout_information(msg="Down-sampling complete...")

    def clip_videos(self):
        self.videos_to_clip = self.find_relevant_videos(variable="clip")
        self.create_process_dir()
        for cnt, (video, video_data) in enumerate(self.videos_to_clip.items()):
            if video_data["last_operation"] == "clip":
                self.quality = video_data["output_quality"]
            in_path, out_path = video_data["path"], os.path.join(self.process_dir, os.path.basename(video_data["path"]))
            start_time, end_time = str(video_data["settings"]["start"]).replace(" ", ""), str(video_data["settings"]["stop"]).replace(" ", "")
            msg = f"Clipping {video} between {start_time} and {end_time} ({cnt+1}/{len(list(self.videos_to_clip.keys()))})..."
            stdout_information(msg=msg)
            clip_video_in_range(file_path=in_path, start_time=start_time, end_time=end_time, out_dir=None, overwrite=True, include_clip_time_in_filename=False, gpu=self.gpu, save_path=out_path, verbose=False)
        self.replace_files_in_temp()
        stdout_information(msg="Clipping complete...")

    def apply_fps(self):
        self.videos_to_change_fps = self.find_relevant_videos(variable="fps")
        self.create_process_dir()
        for cnt, (video, video_data) in enumerate(self.videos_to_change_fps.items()):
            msg = f"Changing FPS of {video} to {video_data['settings']['fps']} ({cnt+1}/{len(list(self.videos_to_change_fps.keys()))}) ..."
            stdout_information(msg=msg)
            if video_data["last_operation"] == "fps":
                self.quality = video_data["output_quality"]
            in_path, out_path = video_data["path"], os.path.join(self.process_dir, os.path.basename(video_data["path"]))
            change_single_video_fps(file_path=in_path, fps=video_data["settings"]["fps"], gpu=self.gpu, quality=self.quality, verbose=False, save_path=out_path)
        self.replace_files_in_temp()
        stdout_information(msg="FPS conversion complete...")

    def apply_grayscale(self):
        self.videos_to_greyscale = self.find_relevant_videos(variable="grayscale")
        self.create_process_dir()
        for cnt, (video, video_data) in enumerate(self.videos_to_greyscale.items()):
            msg = f"Applying grayscale {video} ({cnt+1}/{len(list(self.videos_to_greyscale.keys()))})..."
            stdout_information(msg=msg)
            if video_data["last_operation"] == "grayscale":
                self.quality = video_data["output_quality"]
            in_path, out_path = video_data["path"], os.path.join(self.process_dir, os.path.basename(video_data["path"]))
            video_to_greyscale(file_path=in_path, gpu=self.gpu, codec=self.batch_codec, verbose=False, quality=self.quality, save_path=out_path)
        self.replace_files_in_temp()
        stdout_information(msg=f"Grayscale complete...")

    def apply_frame_count(self, font_size: int = 25, font: str = 'Arial', loc: Literal['top_left', 'top_middle', 'top_right', 'bottom_left', 'bottom_middle', 'bottom_right'] = 'bottom_middle'):
        self.videos_to_frm_cnt = self.find_relevant_videos(variable="frame_cnt")
        self.create_process_dir()
        for cnt, (video, video_data) in enumerate(self.videos_to_frm_cnt.items()):
            msg = f"Applying frame count print {video} ({cnt+1}/{len(list(self.videos_to_frm_cnt.keys()))})..."
            stdout_information(msg=msg)
            if video_data["last_operation"] == "frame_cnt":
                self.quality = video_data["output_quality"]
            in_path, out_path = video_data["path"], os.path.join(self.process_dir, os.path.basename(video_data["path"]))
            superimpose_frame_count(file_path=in_path, gpu=self.gpu, font=font, save_path=out_path, loc=loc, fontsize=font_size, codec=self.batch_codec, quality=self.quality, verbose=False)
        self.replace_files_in_temp()
        stdout_information(msg="Applying frame count complete...")

    def apply_clahe(self, tile_size: Tuple[int, int] = (16, 16), clip_limit: int = 2):
        self.videos_to_frm_cnt = self.find_relevant_videos(variable="clahe")
        self.create_process_dir()
        for cnt, (video, video_data) in enumerate(self.videos_to_frm_cnt.items()):
            msg = f"Applying CLAHE {video} ({cnt+1}/{len(list(self.videos_to_frm_cnt.keys()))})... (note: process can be slow for long videos)"
            stdout_information(msg=msg)
            in_path, out_path = video_data["path"], os.path.join(self.process_dir, os.path.basename(video_data["path"]))
            clahe_enhance_video(file_path=in_path, clip_limit=clip_limit, tile_grid_size=tile_size, out_path=out_path, verbose=False)
        self.replace_files_in_temp()
        stdout_information(msg=f"Applying CLAHE complete...")

    def crop_videos(self):
        self.videos_to_crop = self.find_relevant_videos(variable="crop")
        self.create_process_dir()
        for cnt, (video, video_data) in enumerate(self.videos_to_crop.items()):
            msg = f"Applying crop {video} ({cnt+1}/{len(list(self.videos_to_crop.keys()))}) ... "
            stdout_information(msg=msg)
            if video_data["last_operation"] == "crop":
                self.quality = video_data["output_quality"]
            in_path, out_path = video_data["path"], os.path.join(self.process_dir, os.path.basename(video_data["path"]))
            crop_settings = self.video_dict["video_data"][video]["crop_settings"]
            width, height = int(crop_settings["width"]), int(crop_settings["height"])
            top_left_x, top_left_y = int(crop_settings["top_left_x"]), int(crop_settings["top_left_y"])
            try:
                crop_video(video_path=in_path, save_path=out_path, size=(width, height), top_left=(top_left_x, top_left_y), gpu=self.gpu, verbose=False, quality=self.quality, codec=self.batch_codec)
            except Exception as e:
                if self.gpu:
                    CropWarning(msg=f'GPU crop for video {video} failed, reverting to CPU crop.', source=self.__class__.__name__)
                    crop_video(video_path=in_path, save_path=out_path, size=(width, height), top_left=(top_left_x, top_left_y), gpu=False, verbose=False, quality=self.quality, codec=self.batch_codec)
                else:
                    raise CropError(msg=f'Could not crop video {video} at width: {width}, height: {height} top_left_x: {top_left_x}, top_left_y: {top_left_y}', source=self.__class__.__name__)
        self.replace_files_in_temp()
        stdout_information(msg="Applying crop complete...")
    def copy_videos_to_temp_dir(self):
        for video, video_data in self.video_dict["video_data"].items():
            source = video_data["video_info"]["file_path"]
            msg = f"Making a copy of {os.path.basename(source)} ..."
            stdout_information(msg=msg)
            destination = os.path.join(self.temp_dir, os.path.basename(source))
            shutil.copyfile(source, destination)

    def move_all_processed_files_to_output_folder(self):
        final_videos_path_lst = [f for f in glob.glob(self.temp_dir + "/*") if os.path.isfile(f)]
        for file_path in final_videos_path_lst:
            shutil.copy(file_path, os.path.join(self.out_dir, os.path.basename(file_path)))
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        if os.path.exists(self.process_dir):
            shutil.rmtree(self.process_dir)


# test = FFMPEGCommandCreator(json_path='/Users/simon/Desktop/train_model_project/project_folder/videos_2/batch_process.json_log')
# test.crop_videos()
# test.clip_videos()
# test.downsample_videos()
# test.apply_fps()
# test.apply_grayscale()
# test.apply_frame_count()
# test.apply_clahe()
# test.move_all_processed_files_to_output_folder()

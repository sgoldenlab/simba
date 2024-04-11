import datetime
import glob
import json
import os
import pathlib
import shutil
import subprocess

import cv2

import simba
from simba.utils.checks import check_file_exist_and_readable
from simba.utils.enums import Formats
from simba.utils.read_write import get_video_meta_data


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

    def __init__(self, json_path: str):

        check_file_exist_and_readable(json_path)
        with open(json_path, "r") as fp:
            self.video_dict = json.load(fp)
        self.input_dir = self.video_dict["meta_data"]["in_dir"]
        self.out_dir = self.video_dict["meta_data"]["out_dir"]
        if "gpu" in list(self.video_dict["meta_data"].keys()):
            self.gpu = self.video_dict["meta_data"]["gpu"]
            self.quality = "medium"
        else:
            self.gpu = False
            self.quality = 23
        self.temp_dir = os.path.join(self.out_dir, "temp")
        self.time_format = "%H:%M:%S"
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir)
        self.copy_videos_to_temp_dir()

    def replace_files_in_temp(self):
        processed_files = glob.glob(self.process_dir + "/*")
        for file_path in processed_files:
            file_path_basename = os.path.basename(file_path)
            shutil.move(file_path, os.path.join(self.temp_dir, file_path_basename))

    def create_process_dir(self):
        self.process_dir = os.path.join(self.temp_dir, "process_dir")
        if os.path.exists(self.process_dir):
            shutil.rmtree(self.process_dir)
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
        for video, video_data in self.videos_to_downsample.items():
            print(f"Down-sampling {video}...")
            if video_data["last_operation"] == "downsample":
                self.quality = video_data["output_quality"]
            in_path, out_path = video_data["path"], os.path.join(
                self.process_dir, os.path.basename(video_data["path"])
            )
            width, height = str(video_data["settings"]["width"]), str(
                video_data["settings"]["height"]
            )
            if self.gpu:
                command = f'ffmpeg -y -hwaccel auto -c:v h264_cuvid -i "{in_path}" -vf "scale=w={width}:h={height}" -c:v h264_nvenc -preset {self.quality} -hide_banner -loglevel error "{out_path}"'
            else:
                command = f'ffmpeg -i "{in_path}" -vf scale={width}:{height} "{out_path}" -c:v {Formats.BATCH_CODEC.value} -crf {self.quality} -hide_banner -loglevel error'
            subprocess.call(command, shell=True, stdout=subprocess.PIPE)
        self.replace_files_in_temp()
        print("Downsampling complete...")

    def clip_videos(self):
        self.videos_to_clip = self.find_relevant_videos(variable="clip")
        self.create_process_dir()

        for video, video_data in self.videos_to_clip.items():
            print("Clipping {}...".format(video))
            if video_data["last_operation"] == "clip":
                self.quality = video_data["output_quality"]
            in_path, out_path = video_data["path"], os.path.join(
                self.process_dir, os.path.basename(video_data["path"])
            )
            start_time, end_time = str(video_data["settings"]["start"]).replace(
                " ", ""
            ), str(video_data["settings"]["stop"]).replace(" ", "")
            start_time_shift, end_time = datetime.datetime.strptime(
                start_time, self.time_format
            ), datetime.datetime.strptime(end_time, self.time_format)
            time_difference = str(end_time - start_time_shift)
            if self.gpu:
                command = f'ffmpeg -hwaccel auto -i "{in_path}" -ss {start_time} -t {time_difference} -c:v h264_nvenc -async 1 "{out_path}" -preset {self.quality} -hide_banner -loglevel error'
            else:
                command = f'ffmpeg -i "{in_path}" -ss {start_time} -t {time_difference} -c:v {Formats.BATCH_CODEC.value} -async 1 -crf {self.quality} -c:a copy "{out_path}" -hide_banner -loglevel error'
            subprocess.call(command, shell=True, stdout=subprocess.PIPE)
        self.replace_files_in_temp()
        print("Clipping complete...")

    def apply_fps(self):
        self.videos_to_change_fps = self.find_relevant_videos(variable="fps")
        self.create_process_dir()
        for video, video_data in self.videos_to_change_fps.items():
            print("Changing FPS {}...".format(video))
            if video_data["last_operation"] == "fps":
                self.quality = video_data["output_quality"]
            in_path, out_path = video_data["path"], os.path.join(
                self.process_dir, os.path.basename(video_data["path"])
            )
            fps = str(video_data["settings"]["fps"])
            if self.gpu:
                command = f'ffmpeg -hwaccel auto -c:v h264_cuvid -i "{in_path}" -vf "fps={fps}" -c:v h264_nvenc -preset {self.quality} -c:a copy "{out_path}" -hide_banner -loglevel error'
            else:
                command = f'ffmpeg -i "{in_path}" -c:v {Formats.BATCH_CODEC.value} -crf {self.quality} -filter:v fps=fps={fps} "{out_path}"'
            subprocess.call(command, shell=True, stdout=subprocess.PIPE)
        self.replace_files_in_temp()
        print("FPS conversion complete...")

    def apply_grayscale(self):
        self.videos_to_greyscale = self.find_relevant_videos(variable="grayscale")
        self.create_process_dir()
        for video, video_data in self.videos_to_greyscale.items():
            print(f"Applying grayscale {video}...")
            if video_data["last_operation"] == "grayscale":
                self.quality = video_data["output_quality"]
            in_path, out_path = video_data["path"], os.path.join(
                self.process_dir, os.path.basename(video_data["path"])
            )
            if self.gpu:
                command = f'ffmpeg -hwaccel auto -c:v h264_cuvid -i "{in_path}" -vf "hwupload_cuda,hwdownload,format=nv12,format=gray" -c:v h264_nvenc -preset {self.quality} -c:a copy "{out_path}" -hide_banner -loglevel error'
            else:
                command = f'ffmpeg -i "{in_path}" -c:v {Formats.BATCH_CODEC.value} -crf {self.quality} -vf hue=s=0 "{out_path}" -hide_banner -loglevel error'
            subprocess.call(command, shell=True, stdout=subprocess.PIPE)
        self.replace_files_in_temp()
        print("Grayscale complete...")

    def apply_frame_count(self):
        self.videos_to_frm_cnt = self.find_relevant_videos(variable="frame_cnt")
        self.create_process_dir()
        for video, video_data in self.videos_to_frm_cnt.items():
            print(f"Applying frame count print {video}...")
            if video_data["last_operation"] == "frame_cnt":
                self.quality = video_data["output_quality"]
            in_path, out_path = video_data["path"], os.path.join(
                self.process_dir, os.path.basename(video_data["path"])
            )
            try:
                if self.gpu:
                    command = f'ffmpeg -hwaccel auto -c:v h264_cuvid -i "{in_path}" -vf "drawtext==fontfile=Arial.ttf:text=%{{n}}:x=(w-tw)/2:y=h-th-10:fontcolor=white:box=1:boxcolor=white@0.5" -c:v h264_nvenc -preset {self.quality} -c:a copy "{out_path}" -y -hide_banner -loglevel error'
                else:
                    command = f'ffmpeg -i "{in_path}" -c:v {Formats.BATCH_CODEC.value} -crf {self.quality} -vf "drawtext=fontfile=Arial.ttf:text=\'%{{frame_num}}\':start_number=0:x=(w-tw)/2:y=h-(2*lh):fontcolor=black:fontsize=20:box=1:boxcolor=white:boxborderw=5" -c:a copy -y "{out_path}" -hide_banner -loglevel error'
                subprocess.check_output(command, shell=True)
                subprocess.call(command, shell=True, stdout=subprocess.PIPE)
            except:
                simba_cw = os.path.dirname(simba.__file__)
                simba_font_path = pathlib.Path(
                    simba_cw, "assets", "UbuntuMono-Regular.ttf"
                )
                if self.gpu:
                    command = f'ffmpeg -hwaccel auto -c:v h264_cuvid -i "{in_path}" -vf "drawtext=fontsize=24:fontfile={simba_font_path}:text=%{{n}}:x=(w-tw)/2:y=h-th-10:fontcolor=white:box=1:boxcolor=white@0.5" -c:v h264_nvenc -preset {self.quality} -c:a copy "{out_path}" -y -hide_banner -loglevel error'
                else:
                    command = f'ffmpeg -i "{in_path}" -vf "drawtext=fontfile={simba_font_path}:text=\'%{{frame_num}}\':start_number=1:x=(w-tw)/2:y=h-(2*lh):fontcolor=black:fontsize=20:box=1:boxcolor=white:boxborderw=5" -c:v {Formats.BATCH_CODEC.value} -crf {self.quality} -c:a copy -y "{out_path}" -hide_banner -loglevel error'
                subprocess.call(command, shell=True, stdout=subprocess.PIPE)
        self.replace_files_in_temp()
        print("Applying frame count complete...")

    def apply_clahe(self):
        self.videos_to_frm_cnt = self.find_relevant_videos(variable="clahe")
        self.create_process_dir()
        for video, video_data in self.videos_to_frm_cnt.items():
            clahe_filter = cv2.createCLAHE(clipLimit=2, tileGridSize=(16, 16))
            print(
                f"Applying CLAHE {video}... (note: process can be slow for long videos)"
            )
            in_path, out_path = video_data["path"], os.path.join(
                self.process_dir, os.path.basename(video_data["path"])
            )
            video_info = get_video_meta_data(in_path)
            cap = cv2.VideoCapture(in_path)
            fps, width, height = (
                video_info["fps"],
                video_info["width"],
                video_info["height"],
            )
            writer = cv2.VideoWriter(
                out_path, cv2.VideoWriter_fourcc(*"MP4V"), fps, (width, height), 0
            )
            while True:
                ret, image = cap.read()
                if ret == True:
                    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    clahe_img = clahe_filter.apply(img)
                    writer.write(clahe_img)
                else:
                    break
            cap.release()
            writer.release()
        self.replace_files_in_temp()
        print("Applying CLAHE complete...")

    def crop_videos(self):
        self.videos_to_crop = self.find_relevant_videos(variable="crop")
        self.create_process_dir()
        for video, video_data in self.videos_to_crop.items():
            print(f"Applying crop {video}...")
            if video_data["last_operation"] == "crop":
                self.quality = video_data["output_quality"]
            in_path, out_path = video_data["path"], os.path.join(
                self.process_dir, os.path.basename(video_data["path"])
            )
            crop_settings = self.video_dict["video_data"][video]["crop_settings"]
            width, height = str(crop_settings["width"]), str(crop_settings["height"])
            top_left_x, top_left_y = str(crop_settings["top_left_x"]), str(
                crop_settings["top_left_y"]
            )
            if self.gpu:
                command = f'ffmpeg -hwaccel auto -c:v h264_cuvid -i "{in_path}" -vf "crop={width}:{height}:{top_left_x}:{top_left_y}" -c:v h264_nvenc -preset {self.quality} -c:a copy "{out_path}" -hide_banner -loglevel error'
            else:
                command = f'ffmpeg -i "{in_path}" -vf "crop={width}:{height}:{top_left_x}:{top_left_y}" -c:v {Formats.BATCH_CODEC.value} -crf {self.quality} -c:a copy "{out_path}" -hide_banner -loglevel error'
            subprocess.call(command, shell=True, stdout=subprocess.PIPE)
        self.replace_files_in_temp()
        print("Applying crop complete...")

    def copy_videos_to_temp_dir(self):
        for video, video_data in self.video_dict["video_data"].items():
            source = video_data["video_info"]["file_path"]
            print("Making a copy of {} ...".format(os.path.basename(source)))
            destination = os.path.join(self.temp_dir, os.path.basename(source))
            shutil.copyfile(source, destination)

    def move_all_processed_files_to_output_folder(self):
        final_videos_path_lst = [
            f for f in glob.glob(self.temp_dir + "/*") if os.path.isfile(f)
        ]
        for file_path in final_videos_path_lst:
            shutil.copy(
                file_path, os.path.join(self.out_dir, os.path.basename(file_path))
            )
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

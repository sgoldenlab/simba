__author__ = "Simon Nilsson"

import pandas as pd
import os, glob
import cv2
import subprocess
from typing import Union
try:
    from typing import Literal
except:
    from typing_extensions import Literal

from copy import deepcopy
from simba.utils.printing import stdout_success, SimbaTimer
from simba.utils.checks import  (check_int, check_str, check_if_filepath_list_is_empty)
from simba.utils.read_write import get_fn_ext
from simba.utils.enums import Formats
from simba.utils.errors import CountError, InvalidVideoFileError

class MultiCropper(object):
    """
    Crop single video into multiple videos

    Parameters
    ----------
    file_type: str
        File type of input video files (e.g., 'mp4', 'avi')
    input_folder: str
        Folder path holding videos to be cropped.
    output_folder: str
        Folder where to store the results.
    crop_cnt: int
        Integer representing the number of videos to produce from every input video.

    Notes
    ----------
    `Multi-crop tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#multi-crop-videos>`__.


    Examples
    ----------
    >>> _ = MultiCropper(file_type='mp4', input_folder='InputDirectory', output_folder='OutputDirectory', crop_cnt=2)

    """

    def __init__(self,
                 file_type: Literal['mp4', 'avi'],
                 input_folder: Union[str, os.PathLike],
                 output_folder: Union[str, os.PathLike],
                 crop_cnt: int):

        self.file_type, self.crop_cnt = file_type, crop_cnt
        self.input_folder, self.output_folder = input_folder, output_folder
        self.crop_df = pd.DataFrame(columns=['Video', "height", "width", "topLeftX", "topLeftY"])
        check_int(name='CROP COUNT', value=self.crop_cnt, min_value=2)
        self.crop_cnt = int(crop_cnt)
        if self.crop_cnt == 0:
            raise CountError(msg='The number of cropped output videos is set to ZERO. The number of crops has the be a value above 0.')
        check_str(name='FILE TYPE',value=self.file_type)
        if self.file_type.__contains__('.'): self.file_type.replace(".", "")
        self.files_found = glob.glob(self.input_folder + '/*.{}'.format(self.file_type))
        check_if_filepath_list_is_empty(filepaths=self.files_found,
                                        error_msg='SIMBA CROP ERROR: The input direct {} contains ZERO videos in {} format'.format(self.input_folder, self.file_type))
        if not os.path.exists(self.output_folder): os.makedirs(self.output_folder)
        self.font_scale, self.space_scale, self.font = 0.02, 1.1, Formats.FONT.value
        self.add_spacer = 2
        self.__crop_videos()

    def __test_crop_locations(self):
        for idx, row in self.crop_df.iterrows():
            lst = [row['height'], row['width'], row['topLeftX'], row['topLeftY']]
            video_name = row['Video']
            if all(v == 0 for v in lst):
                raise CountError(msg='SIMBA ERROR: A crop for video {} has all crop coordinates set to zero. Did you click ESC, space or enter before defining the rectangle crop coordinates?!'.format(video_name))
            else:
                pass


    def __crop_videos(self):
        for file_cnt, file_path in enumerate(self.files_found):
            video_name = str(os.path.basename(file_path))

            cap = cv2.VideoCapture(file_path)
            cap.set(1, 0)
            ret, frame = cap.read()
            original_frame = deepcopy(frame)
            if not ret:
                raise InvalidVideoFileError(msg='The first frame of video {} could not be read'.format(video_name))
            (height, width) = frame.shape[:2]
            font_size = min(width, height) / (25 / self.font_scale)
            space_size = int(min(width, height) / (25 / self.space_scale))
            cv2.namedWindow('VIDEO IMAGE', cv2.WINDOW_NORMAL)
            cv2.putText(frame, str(video_name), (10, ((height - height) + space_size)), cv2.FONT_HERSHEY_TRIPLEX, font_size, (180, 105, 255), 2)
            cv2.putText(frame, 'Define the rectangle bounderies of {} cropped videos'.format(str(self.crop_cnt)), (10, ((height - height) + space_size * self.add_spacer)), cv2.FONT_HERSHEY_TRIPLEX, font_size, (180, 105, 255), 2)
            cv2.putText(frame, 'Press ESC to continue' ,(10, ((height - height) + space_size * (self.add_spacer+2))), cv2.FONT_HERSHEY_TRIPLEX, font_size, (180, 105, 255), 2)

            while (1):
                cv2.imshow('VIDEO IMAGE', frame)
                k = cv2.waitKey(33)
                if k == 27:
                    cv2.destroyAllWindows()
                    cv2.waitKey(3)
                    break

            for box_cnt in range(self.crop_cnt):
                cv2.namedWindow('VIDEO IMAGE', cv2.WINDOW_NORMAL)
                frame = deepcopy(original_frame)
                cv2.putText(frame, str(video_name), (10, ((height - height) + space_size)), self.font, font_size, (180, 105, 255), 2)
                cv2.putText(frame, str('Define crop #{} coordinate bounderies and press enter'.format(str(box_cnt+1))), (10, ((height - height) + space_size * self.add_spacer)), cv2.FONT_HERSHEY_TRIPLEX, font_size, (180, 105, 255), 2)
                ROI = cv2.selectROI('VIDEO IMAGE', frame)
                width, height = (abs(ROI[0] - (ROI[2] + ROI[0]))), (abs(ROI[2] - (ROI[3] + ROI[2])))
                topLeftX, topLeftY = ROI[0], ROI[1]
                k = cv2.waitKey(20) & 0xFF
                cv2.destroyAllWindows()
                self.crop_df.loc[len(self.crop_df)] = [video_name, height, width, topLeftX, topLeftY]

        self.__test_crop_locations()
        cv2.destroyAllWindows()
        cv2.waitKey(50)

        self.timer = SimbaTimer()
        self.timer.start_timer()
        for video_name in self.crop_df['Video'].unique():
            video_crops = self.crop_df[self.crop_df['Video'] == video_name].reset_index(drop=True)
            _, name, ext = get_fn_ext(video_name)
            in_video_path = os.path.join(self.input_folder, video_name)
            for cnt, (idx, row) in enumerate(video_crops.iterrows()):
                height, width = row['height'], row['width']
                topLeftX, topLeftY = row['topLeftX'], row['topLeftY']
                out_file_fn = os.path.join(self.output_folder, name + '_{}.'.format(str(cnt+1)) + self.file_type)
                command = str('ffmpeg -y -i ') + str(in_video_path) + str(' -vf ') + str('"crop=') + str(width) + ':' + str(
                    height) + ':' + str(topLeftX) + ':' + str(topLeftY) + '" ' + str('-c:v libx264 -c:a copy ') + str(
                    out_file_fn + ' -hide_banner -loglevel error')
                subprocess.call(command, shell=True)
                print('Video {} crop {} complete...'.format(name, str(cnt+1)))

        self.timer.stop_timer()
        stdout_success(msg=f'{str(len(self.crop_df))} new cropped videos created from {str(len(self.files_found))} input videos. Cropped videos are saved in the {self.output_folder} directory', elapsed_time=self.timer.elapsed_time_str)

# test = MultiCropper(file_type='avi', input_folder='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/test',
#                     output_folder='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/test_2', crop_cnt=2)






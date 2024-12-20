from typing import Union, Tuple
import os

from datetime import datetime
import subprocess

from tkinter import Button
from simba.utils.enums import Keys, Links, Options
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import CreateLabelFrameWithIcon, FileSelect, FolderSelect
from simba.utils.checks import check_file_exist_and_readable, check_if_dir_exists
from simba.video_processors.brightness_contrast_ui import brightness_contrast_ui
from simba.utils.read_write import find_files_of_filetypes_in_directory, get_fn_ext
from simba.utils.printing import stdout_success, SimbaTimer


class BrightnessContrastPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title="CHANGE BRIGHTNESS / CONTRAST")
        self.datetime = datetime.now().strftime("%Y%m%d%H%M%S")
        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="CHANGE BRIGHTNESS / CONTRAST SINGLE VIDEO", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video = FileSelect(single_video_frm, "VIDEO PATH:", title="Select a video file", file_types=[("VIDEO", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)], lblwidth=25)
        run_video_btn = Button(single_video_frm, text="RUN SINGLE VIDEO", command=lambda: self.run_video(), fg="blue")

        single_video_frm.grid(row=0, column=0, sticky="NW")
        self.selected_video.grid(row=0, column=0, sticky="NW")
        run_video_btn.grid(row=1, column=0, sticky="NW")

        video_dir_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="CHANGE BRIGHTNESS / CONTRAST MULTIPLE VIDEOS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_dir = FolderSelect(video_dir_frm, "VIDEO DIRECTORY PATH:", lblwidth=25)
        run_dir_btn = Button(video_dir_frm, text="RUN VIDEO DIRECTORY", command=lambda: self.run_directory(), fg="blue")

        video_dir_frm.grid(row=1, column=0, sticky="NW")
        self.selected_dir.grid(row=0, column=0, sticky="NW")
        run_dir_btn.grid(row=1, column=0, sticky="NW")
        self.main_frm.mainloop()

    def run_video(self):
        video_path = self.selected_video.file_path
        check_file_exist_and_readable(file_path=video_path)
        self.brightness, self.contrast = brightness_contrast_ui(video_path=video_path)
        self.video_paths = [video_path]
        print(self.video_paths)
        self.apply()

    def run_directory(self):
        video_dir = self.selected_dir.folder_path
        check_if_dir_exists(in_dir=video_dir, source=self.__class__.__name__)
        self.video_paths = find_files_of_filetypes_in_directory(directory=video_dir, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, raise_error=True)
        self.brightness, self.contrast = brightness_contrast_ui(video_path=self.video_paths[0])
        self.apply()

    def apply(self):
        timer = SimbaTimer(start=True)
        for file_cnt, file_path in enumerate(self.video_paths):
            video_timer = SimbaTimer(start=True)
            dir, video_name, ext = get_fn_ext(filepath=file_path)
            print(f'Creating copy of {video_name}...')
            out_path = os.path.join(dir, f'{video_name}_eq_{self.datetime}{ext}')
            cmd = f'ffmpeg -i "{file_path}" -vf "eq=brightness={self.brightness}:contrast={self.contrast}" -loglevel error -stats "{out_path}" -y'
            subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
            video_timer.stop_timer()
            stdout_success(msg=f'Video {out_path} complete!', elapsed_time=video_timer.elapsed_time_str)
        timer.stop_timer()
        stdout_success(f'{len(self.video_paths)} video(s) converted.', elapsed_time=timer.stop_timer())











#
# def change_contrast(video_path: Union[str, os.PathLike]) -> Tuple[float, float]:
#     """
#     Create a user interface using OpenCV to explore and change the brightness and contrast of a video.
#
#     :param Union[str, os.PathLike] video_path: Path to the video file.
#     :return Tuple: The scaled brightness and scaled contrast values on scale -1 to +1 suitable for FFmpeg conversion
#
#     :example:
#     >>> change_contrast(video_path='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/frames/output/ROI_features/2022-06-20_NOB_DOT_4.mp4')
#     """
#     def _get_trackbar_values(v):
#         brightness = cv2.getTrackbarPos('Brightness', 'Contrast / Brightness')
#         contrast = cv2.getTrackbarPos('Contrast', 'Contrast / Brightness')
#         brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
#         contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))
#         if brightness != 0:
#             if brightness > 0:
#                 shadow, max = brightness, 255
#             else:
#                 shadow, max = 0, 255 + brightness
#             cal = cv2.addWeighted(original_img, (max - shadow) / 255, original_img, 0, shadow)
#         else:
#             cal = original_img
#         if contrast != 0:
#             Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
#             Gamma = 127 * (1 - Alpha)
#             cal = cv2.addWeighted(cal, Alpha, cal, 0, Gamma)
#         img = np.copy(cal)
#         cv2.imshow('Contrast / Brightness', img)
#
#     _ = get_video_meta_data(video_path=video_path)
#     original_img = read_frm_of_video(video_path=video_path, frame_index=0)
#     img = np.copy(original_img)
#     cv2.namedWindow('Contrast / Brightness', cv2.WINDOW_NORMAL)
#     cv2.imshow('Contrast / Brightness', img)
#     cv2.createTrackbar('Brightness', 'Contrast / Brightness', 255, 2 * 255, _get_trackbar_values)
#     cv2.createTrackbar('Contrast', 'Contrast / Brightness',  127, 2 * 127,  _get_trackbar_values)
#     while True:
#         k = cv2.waitKey(1) & 0xFF
#         if k == 27:
#             brightness = cv2.getTrackbarPos('Brightness', 'Change contrast')
#             contrast = cv2.getTrackbarPos('Contrast', 'Change contrast')
#             scaled_brightness = ((brightness - 0) / (510 - 0)) * (1 - -1) + -1
#             scaled_contrast= ((contrast - 0) / (254 - 0)) * (1 - -1) + -1
#             if scaled_contrast == 0.0 and scaled_brightness == 0.0:
#                 InValidUserInputWarning(msg=f'Both the selected brightness and contrast are the same as in the input video. Select different values.')
#             else:
#                 cv2.destroyAllWindows()
#                 return scaled_brightness, scaled_contrast


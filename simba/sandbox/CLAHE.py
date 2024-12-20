import os
from datetime import datetime
import threading
from tkinter import *
from tkinter import Button
from simba.utils.enums import Keys, Links, Options
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import CreateLabelFrameWithIcon, FileSelect, FolderSelect
from simba.utils.checks import check_file_exist_and_readable, check_if_dir_exists
from simba.video_processors.clahe_ui import interactive_clahe_ui
from simba.video_processors.video_processing import clahe_enhance_video
from simba.utils.read_write import find_files_of_filetypes_in_directory, get_fn_ext
from simba.utils.printing import stdout_success, SimbaTimer

class CLAHEPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title="CLAHE VIDEO CONVERSION")
        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO - Contrast Limited Adaptive Histogram Equalization", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video = FileSelect(single_video_frm, "VIDEO PATH:", title="Select a video file", file_types=[("VIDEO", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        run_single_video_btn = Button(single_video_frm, text="Apply CLAHE", command=lambda: self.run_single_video())

        multiple_videos_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="MULTIPLE VIDEOs - Contrast Limited Adaptive Histogram Equalization", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_dir = FolderSelect(multiple_videos_frm, "VIDEO DIRECTORY PATH:", lblwidth=25)
        run_multiple_btn = Button(multiple_videos_frm, text="RUN VIDEO DIRECTORY", command=lambda: self.run_directory(), fg="blue")

        single_video_frm.grid(row=0, column=0, sticky=NW)
        self.selected_video.grid(row=0, column=0, sticky=NW)
        run_single_video_btn.grid(row=1, column=0, sticky=NW)

        multiple_videos_frm.grid(row=1, column=0, sticky=NW)
        self.selected_dir.grid(row=0, column=0, sticky=NW)
        run_multiple_btn.grid(row=1, column=0, sticky=NW)

    def run_single_video(self):
        selected_video = self.selected_video.file_path
        check_file_exist_and_readable(file_path=selected_video)
        threading.Thread(target=clahe_enhance_video(file_path=selected_video)).start()

    def run_directory(self):
        timer = SimbaTimer(start=True)
        video_dir = self.selected_dir.folder_path
        check_if_dir_exists(in_dir=video_dir, source=self.__class__.__name__)
        self.video_paths = find_files_of_filetypes_in_directory(directory=video_dir, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, raise_error=True)
        for file_path in self.video_paths:
            threading.Thread(target=clahe_enhance_video(file_path=file_path)).start()
        timer.stop_timer()
        stdout_success(msg=f'CLAHE enhanced {len(self.video_paths)} video(s)', elapsed_time=timer.elapsed_time_str)











# # Function to update CLAHE
# def update_clahe(x):
#     global img, clahe
#     clip_limit = cv2.getTrackbarPos('Clip Limit', 'CLAHE') / 10.0  # Scale the trackbar value
#     tile_size = cv2.getTrackbarPos('Tile Size', 'CLAHE')
#     if tile_size % 2 == 0:
#         tile_size += 1  # Ensure tile size is odd
#     clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
#     img_clahe = clahe.apply(img)
#     cv2.imshow('CLAHE', img_clahe)
#
# # Load an image
# img = cv2.imread('/Users/simon/Downloads/PXL_20240429_222923838.jpg', cv2.IMREAD_GRAYSCALE)
#
# # Create a window
# cv2.namedWindow('CLAHE', cv2.WINDOW_NORMAL)
#
# # Initialize the clip limit trackbar
# cv2.createTrackbar('Clip Limit', 'CLAHE', 10, 300, update_clahe)
#
# # Initialize the tile size trackbar
# cv2.createTrackbar('Tile Size', 'CLAHE', 8, 64, update_clahe)
#
# # Apply CLAHE with initial parameters
# clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
# img_clahe = clahe.apply(img)
# cv2.imshow('Original', img)
# cv2.imshow('CLAHE', img_clahe)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import os
import time
from tkinter import *

import cv2
import numpy as np
from PIL import Image, ImageTk

from simba.enums import Formats
from simba.misc_tools import (find_all_videos_in_directory, get_fn_ext,
                              get_video_meta_data)
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.tkinter_functions import create_scalebar

MAX_FRM_SIZE = 1080, 650


class VideoColorChanger(PopUpMixin):
    def __init__(self, input_path: str, output_dir: str):
        super().__init__(title="CHANGE COLOR", size=(850, 400))
        self.save_dir = output_dir
        if os.path.isfile(input_path):
            self.video_paths = [input_path]
        else:
            self.video_paths = find_all_videos_in_directory(directory=input_path)

    def __insert_img(self, img: np.array):
        current_frm_pil = Image.fromarray(img)
        current_frm_pil.thumbnail(MAX_FRM_SIZE, Image.ANTIALIAS)
        current_frm_pil = ImageTk.PhotoImage(
            master=self.main_frm, image=current_frm_pil
        )
        self.video_frame = Label(self.main_frm, image=current_frm_pil)
        self.video_frame.image = current_frm_pil
        self.video_frame.grid(row=0, column=0)

    def titrate_brightness(self, x):
        value = int(self.brightness_scale.get())
        adjusted = cv2.convertScaleAbs(self.img, alpha=1.0, beta=value)
        adjusted = cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB)
        self.__insert_img(img=adjusted)

    def __run_interface(self, file_path: str):
        self.dif_angle = 0
        self.video_meta_data = get_video_meta_data(video_path=file_path)
        self.file_path = file_path
        _, self.video_name, _ = get_fn_ext(filepath=file_path)
        self.video_frm = Frame(self.main_frm)
        self.video_frm.grid(row=0, column=0, sticky=NW)
        self.cap = cv2.VideoCapture(file_path)
        _, self.img = self.cap.read()
        self._orig_img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
        self.dashboard = LabelFrame(
            self.main_frm, text="DASHBOARD", font=Formats.LABELFRAME_HEADER_FORMAT.value
        )
        self.dashboard.grid(row=0, column=1, sticky=NW)
        self.brightness_scale = create_scalebar(
            parent=self.dashboard,
            name="BRIGHTNESS",
            cmd=self.titrate_brightness,
            min=-100,
            max=100,
        )
        self.brightness_scale.grid(row=0, column=0, sticky=NW)
        self.__insert_img(img=self._orig_img)

    def run(self):
        self.results = {}
        self.__run_interface(self.video_paths[0])
        self.main_frm.mainloop()


# rotator = VideoColorChanger(
#     input_path="/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/videos/Together_1_downsampled.mp4",
#     output_dir="/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/rotated",
# )
# rotator.run()

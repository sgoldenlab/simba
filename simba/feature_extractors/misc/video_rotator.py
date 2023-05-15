import os
from simba.misc_tools import (find_all_videos_in_directory,
                              get_video_meta_data,
                              get_fn_ext)
from simba.mixins.config_reader import ConfigReader
from simba.enums import Formats
import cv2
import numpy as np
from tkinter import *
import time
from PIL import Image, ImageTk
from simba.utils.printing import stdout_success

MAX_FRM_SIZE = 1080, 650

class VideoRotator(ConfigReader):
    def __init__(self,
                 config_path: str,
                 input_path: str,
                 output_dir: str):

        super().__init__(config_path=config_path)
        self.save_dir = output_dir
        if os.path.isfile(input_path):
            self.video_paths = [input_path]
        else:
            self.video_paths = find_all_videos_in_directory(directory=input_path)

    def __insert_img(self, img: np.array):
        current_frm_pil = Image.fromarray(img)
        current_frm_pil.thumbnail(MAX_FRM_SIZE, Image.ANTIALIAS)
        current_frm_pil = ImageTk.PhotoImage(master=self.main_frm, image=current_frm_pil)
        self.video_frame = Label(self.main_frm, image=current_frm_pil)
        self.video_frame.image = current_frm_pil
        self.video_frame.grid(row=0, column=0)

    def __rotate(self, value: int, img: np.array):
        self.dif_angle += value
        rotation_matrix = cv2.getRotationMatrix2D((self.video_meta_data['width'] / 2, self.video_meta_data['height'] / 2), self.dif_angle, 1)
        img = cv2.warpAffine(img, rotation_matrix, (self.video_meta_data['width'], self.video_meta_data['height']))
        self.__insert_img(img=img)

    def __run_rotation(self):
        self.main_frm.destroy()
        fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        start = time.time()
        for video_cnt, (video_path, rotation) in enumerate(self.results.items()):
            cap = cv2.VideoCapture(video_path)
            rotation_matrix = cv2.getRotationMatrix2D((self.video_meta_data['width'] / 2, self.video_meta_data['height'] / 2), rotation, 1)
            save_path = os.path.join(self.save_dir, os.path.basename(video_path))
            video_meta = get_video_meta_data(video_path=video_path)
            writer = cv2.VideoWriter(save_path, fourcc, video_meta['fps'], (video_meta['width'], video_meta['height']))
            img_cnt = 0
            while True:
                ret, img = cap.read()
                if not ret:
                    break
                img = cv2.warpAffine(img, rotation_matrix, (self.video_meta_data['width'], self.video_meta_data['height']))
                writer.write(img)
                img_cnt+=1
                print(f'Rotating frame {img_cnt}/{video_meta["frame_count"]} (Video {video_cnt+1}/{len(self.results.keys())}) ')
            cap.release()
            writer.release()
        stdout_success(msg=f'All videos rotated and saved in {self.save_dir}', elapsed_time=str(round((time.time() - start), 2)))

    def __save(self):
        self.results[self.file_path] = self.dif_angle
        if len(self.results.keys()) == len(self.video_paths):
            self.__run_rotation()
        else:
            self.__run_interface(file_path=self.video_paths[len(self.results.keys())-1])

    def __bind_keys(self):
        self.main_frm.bind('<Left>', lambda x: self.__rotate(value = 1, img=self._orig_img))
        self.main_frm.bind('<Right>', lambda x: self.__rotate(value = -1, img=self._orig_img))
        self.main_frm.bind('<Escape>', lambda x: self.__save())

    def __run_interface(self, file_path: str):
        self.dif_angle = 0
        self.video_meta_data = get_video_meta_data(video_path=file_path)
        self.file_path = file_path
        _, self.video_name, _ = get_fn_ext(filepath=file_path)
        self.main_frm = Toplevel()
        self.main_frm.title(f'ROTATE VIDEO {self.video_name}')
        self.video_frm = Frame(self.main_frm)
        self.video_frm.grid(row=0, column=0)
        self.instruction_frm = Frame(self.main_frm, width=100, height=100)
        self.instruction_frm.grid(row=0, column=2, sticky=NW)
        self.key_lbls = Label(self.instruction_frm,
                                    text='\n\n Navigation: '
                                         '\n Left arrow = 1° left' 
                                         '\n Right arrow = 1° right'
                                         '\n Esc = Save')

        self.key_lbls.grid(sticky=NW)
        self.cap = cv2.VideoCapture(file_path)
        _, self.img = self.cap.read()
        self._orig_img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
        self.__insert_img(img=self._orig_img)
        self.__bind_keys()

    def run(self):
        self.results = {}
        self.__run_interface(self.video_paths[0])
        #self.main_frm.mainloop()





rotator = VideoRotator(input_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/videos/Together_1_downsampled.mp4',
                       config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
                       output_dir='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/rotated')
rotator.run()

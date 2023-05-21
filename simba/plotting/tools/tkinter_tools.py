__author__ = "Simon Nilsson"

from tkinter import *

import cv2
import numpy as np
from PIL import Image as Img
from PIL import ImageTk

from simba.utils.errors import FrameRangeError
from simba.utils.read_write import get_video_meta_data

PADDING = 5
MAX_SIZE = (1080, 650)


class InteractiveVideoPlotterWindow(object):
    def __init__(self, video_path: str, p_arr: np.array):
        self.main_frm = Toplevel()
        self.current_frm_number, self.jump_size = 0, 0
        self.img_frm = Frame(self.main_frm)
        self.img_frm.grid(row=0, column=1, sticky=NW)
        self.button_frame = Frame(self.main_frm, bd=2, width=700, height=300)
        self.button_frame.grid(row=1, column=0)
        self.video_meta_data = get_video_meta_data(video_path=video_path)
        self.cap = cv2.VideoCapture(video_path)
        self.cap.set(1, self.current_frm_number)
        self.max_frm = np.argmax(p_arr)
        self.frame_id_lbl = Label(self.button_frame, text="FRAME NUMBER")
        self.frame_id_lbl.grid(row=0, column=1, sticky=NW, padx=PADDING)
        self.forward_next_frm_btn = Button(
            self.button_frame,
            text=">",
            command=lambda: self.load_new_frame(frm_cnt=self.current_frm_number + 1),
        )
        self.forward_next_frm_btn.grid(row=1, column=3, sticky=E, padx=PADDING)

        self.forward_last_frm_btn = Button(
            self.button_frame,
            text=">>",
            command=lambda: self.load_new_frame(
                frm_cnt=self.video_meta_data["frame_count"] - 1
            ),
        )
        self.forward_last_frm_btn.grid(row=1, column=4, sticky=E, padx=PADDING)
        self.back_one_frm_btn = Button(
            self.button_frame,
            text="<",
            command=lambda: self.load_new_frame(frm_cnt=self.current_frm_number - 1),
        )
        self.back_one_frm_btn.grid(row=1, column=1, sticky=W, padx=PADDING)
        self.back_first_frm = Button(
            self.button_frame, text="<<", command=lambda: self.load_new_frame(frm_cnt=0)
        )
        self.back_first_frm.grid(row=1, column=0, sticky=W)
        self.frame_entry_var = IntVar(self.main_frm, value=self.current_frm_number)
        self.frame_entry_box = Entry(
            self.button_frame, width=7, textvariable=self.frame_entry_var
        )
        self.frame_entry_box.grid(row=1, column=1)
        self.select_frm_btn = Button(
            self.button_frame,
            text="Jump to selected frame",
            command=lambda: self.load_new_frame(frm_cnt=self.frame_entry_var.get()),
        )
        self.select_frm_btn.grid(row=2, column=1, sticky=N)

        self.jump_frm = Frame(self.main_frm)
        self.jump_frm.grid(row=2, column=0)
        self.jump_lbl = Label(self.jump_frm, text="Jump Size:")
        self.jump_lbl.grid(row=0, column=0, sticky=NW)
        self.jump_size_scale = Scale(
            self.jump_frm, from_=0, to=100, orient=HORIZONTAL, length=200
        )
        self.jump_size_scale.set(self.jump_size)
        self.jump_size_scale.grid(row=0, column=1, sticky=NW)
        self.jump_back_btn = Button(
            self.jump_frm,
            text="<<",
            command=lambda: self.load_new_frame(
                frm_cnt=self.current_frm_number - self.jump_size_scale.get()
            ),
        )
        self.jump_back_btn.grid(row=0, column=2, sticky=E)
        self.jump_forward_btn = Button(
            self.jump_frm,
            text=">>",
            command=lambda: self.load_new_frame(
                frm_cnt=self.current_frm_number + self.jump_size_scale.get()
            ),
        )
        self.jump_forward_btn.grid(row=0, column=3, sticky=W)

        self.load_new_frame(frm_cnt=self.current_frm_number)

        instructions_frm = Frame(self.main_frm, width=100, height=100)
        instructions_frm.grid(row=0, column=2, sticky=N)
        key_presses = Label(
            instructions_frm,
            text="\n\n Keyboard shortcuts for frame navigation: \n Right Arrow = +1 frame"
            "\n Left Arrow = -1 frame"
            "\n Ctrl + l = Last frame"
            "\n Ctrl + o = First frame",
        )

        move_to_highest_p_btn = Button(
            instructions_frm,
            text="SHOW HIGHEST \n PROBABILITY FRAME",
            command=lambda: self.load_new_frame(frm_cnt=self.max_frm),
        )
        key_presses.grid(row=0, column=0, sticky=S)
        move_to_highest_p_btn.grid(row=1, column=0, sticky=S)
        self.bind_keys()

    def bind_keys(self):
        self.main_frm.bind(
            "<Right>",
            lambda x: self.load_new_frame(frm_cnt=self.current_frm_number + 1),
        )
        self.main_frm.bind(
            "<Left>", lambda x: self.load_new_frame(frm_cnt=self.current_frm_number - 1)
        )
        self.main_frm.bind(
            "<Control-l>",
            lambda x: self.load_new_frame(
                frm_cnt=self.video_meta_data["frame_count"] - 1
            ),
        )
        self.main_frm.bind("<Control-o>", lambda x: self.load_new_frame(frm_cnt=0))

    def load_new_frame(self, frm_cnt: int):
        if (frm_cnt > self.video_meta_data["frame_count"] - 1) or (frm_cnt < 0):
            raise FrameRangeError(
                msg=f'Frame {str(frm_cnt)} is outside of the video frame range: (0-{self.video_meta_data["frame_count"]-1}).'
            )
        self.cap.set(1, int(frm_cnt))
        _, self.new_frm = self.cap.read()
        self.new_frm = cv2.cvtColor(self.new_frm, cv2.COLOR_RGB2BGR)
        self.new_frm = Img.fromarray(self.new_frm)
        self.new_frm.thumbnail(MAX_SIZE, Img.ANTIALIAS)
        self.new_frm = ImageTk.PhotoImage(master=self.main_frm, image=self.new_frm)
        self.img_frm = Label(self.main_frm, image=self.new_frm)
        self.img_frm.image = self.new_frm
        self.img_frm.grid(row=0, column=0)
        self.current_frm_number = frm_cnt
        self.frame_entry_var.set(value=self.current_frm_number)

import os
from tkinter import *

import simba
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.utils.enums import Paths


class AboutSimBAPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, size=(896, 507), title="ABOUT SIMBA")
        canvas = Canvas(self.main_frm, width=896, height=507, bg="black")
        canvas.pack()
        img_path = os.path.join(os.path.dirname(simba.__file__), Paths.ABOUT_ME.value)
        img = PhotoImage(file=os.path.join(img_path))
        canvas.create_image(0, 0, image=img, anchor=NW)
        canvas.image = img

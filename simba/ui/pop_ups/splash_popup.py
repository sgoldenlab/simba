from typing import Tuple, Optional, Union
import os
import cv2
from simba.utils.enums import Paths, Defaults, FontPaths
from simba.utils.read_write import get_video_meta_data
from simba.utils.checks import check_if_valid_rgb_tuple, check_file_exist_and_readable,check_float
from tkinter import *
from PIL import ImageTk, ImageDraw, ImageFont
import PIL.Image
import simba

class SplashMovie:
    def __init__(self,
                 outline_color: Tuple[int, int, int] = (255,192,203),
                 fill_color: Tuple[int, int, int] = (255, 255, 255),
                 font_path: Optional[Union[str, os.PathLike]] = FontPaths.POPPINS_REGULAR.value,
                 font_size_height: float = 0.05):

        self.parent, self.img_cnt = Tk(), 0
        self.parent.overrideredirect(True)
        self.parent.configure(bg="white")
        splash_path = os.path.join(os.path.dirname(simba.__file__), Paths.SPLASH_PATH_MOVIE.value)
        self.meta_ = get_video_meta_data(splash_path)
        self.cap = cv2.VideoCapture(splash_path)
        width, height = self.meta_["width"], self.meta_["height"]
        half_width = int((self.parent.winfo_screenwidth() - width) // 2)
        half_height = int((self.parent.winfo_screenheight() - height) // 2)
        self.parent.geometry("%ix%i+%i+%i" % (width, height, half_width, half_height))
        self.parent.attributes('-topmost', True)
        self.img_lbl = Label(self.parent, bg="white", image="")
        self.img_lbl.pack()
        check_if_valid_rgb_tuple(data=fill_color, source=f'{self.__class__.__name__} fill_color', raise_error=True)
        check_if_valid_rgb_tuple(data=outline_color, source=f'{self.__class__.__name__} outline_color', raise_error=True)
        self.fill_color, self.outline_color = fill_color, outline_color
        check_float(value=font_size_height, name=f'{self.__class__.__name__} font_size_height', max_value=1.0, min_value=0.01, raise_error=True)
        self.font_size = int(font_size_height * height)
        if font_path is not None:
            self.font_path = os.path.join(os.path.dirname(simba.__file__), font_path)
            check_file_exist_and_readable(file_path=self.font_path, raise_error=True)
        else:
            self.font_path = None
        if self.font_path is not None:
            try:
                self.font = ImageFont.truetype(self.font_path, self.font_size)
            except:
                self.font = ImageFont.load_default()
        else:
            self.font = ImageFont.load_default()

        self.show_animation()

    def add_welcome_text(self, pil_image):
        draw = ImageDraw.Draw(pil_image)
        width, height = pil_image.size
        welcome_text = Defaults.WELCOME_MSG.value.replace('\n', ' ').strip()
        bbox = draw.textbbox((0, 0), welcome_text, font=self.font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x, y = (width - text_width) // 2, height - text_height - 20
        for adj in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            draw.text((x + adj[0], y + adj[1]), welcome_text, font=self.font, fill=self.outline_color)

        draw.text((x, y), welcome_text, font=self.font, fill=self.fill_color)
        
        return pil_image

    def show_animation(self):
        for frm_cnt in range(self.meta_["frame_count"] - 1):
            self.cap.set(1, frm_cnt)
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            pil_image = PIL.Image.fromarray(frame)
            pil_image = self.add_welcome_text(pil_image)
            frame = ImageTk.PhotoImage(image=pil_image)
            self.img_lbl.configure(image=frame)
            self.img_lbl.imgtk = frame
            self.parent.update()
            cv2.waitKey(max(50, int(self.meta_["fps"] / 1000)))
        self.parent.destroy()



#SplashMovie()
import os
from datetime import datetime
from tkinter import *
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageTk
from shapely.geometry import MultiPolygon, Polygon

from simba.mixins.geometry_mixin import GeometryMixin
from simba.mixins.image_mixin import ImageMixin
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, Entry_Box,
                                        SimbaButton)
from simba.utils.checks import (check_if_valid_img, check_instance, check_int,
                                check_str, check_valid_tuple)
from simba.utils.enums import Formats
from simba.utils.errors import FrameRangeError, InvalidVideoFileError
from simba.utils.printing import SimbaTimer
from simba.utils.read_write import get_video_meta_data, read_frm_of_video
from simba.video_processors.video_processing import create_average_frm

FRAME_NAME = 'QUICK CHECK'

class BlobQuickChecker():
    """
     Interactive tool for visual comparisons using threshold-based difference detection with support for inclusion zones and interactive frame navigation.

    :param Union[str, os.PathLike] video_path: Path to the video file being analyzed.
    :param Union[str, os.PathLike] bg_video_path: Path to the background reference video.
    :param str method: Method for computing the difference image.  Options: 'absolute', 'light', 'dark'. Default is 'absolute'.
    :param int threshold: Threshold value for difference computation (1-255). Default is 70.
    :param Optional[Union[Polygon, MultiPolygon]] inclusion_zones: Optional geometric regions of interest to highlight in the processed frames.

    :example:
    >>> _ = BlobQuickChecker(video_path=r"C:\troubleshooting\mitra\test\501_MA142_Gi_Saline_0515.mp4", bg_video_path=r"C:\troubleshooting\mitra\test\background_dir\501_MA142_Gi_Saline_0515.mp4")
    """

    def __init__(self,
                 video_path: Union[str, os.PathLike],
                 bg_video_path: Union[str, os.PathLike],
                 method: str = 'absolute',
                 threshold: int = 70,
                 inclusion_zones: Optional[Union[Polygon, MultiPolygon]] = None,
                 status_label: Optional[Label] = None,
                 close_kernel_size: Optional[Tuple[int, int]] = None,
                 close_kernel_iterations: int = 3,
                 open_kernel_size: Optional[Tuple[int, int]] = None,
                 open_kernel_iterations: int = 3):

        self.video_meta, bg_meta = get_video_meta_data(video_path=video_path), get_video_meta_data(video_path=bg_video_path)
        if self.video_meta['resolution_str'] != bg_meta['resolution_str']:
            msg = f'The video, and background reference video, for {self.video_meta["video_name"]} have different resolutions: {self.video_meta["resolution_str"]} vs {bg_meta["resolution_str"]}'
            self._set_status_bar_panel(text=msg, fg='red')
            raise InvalidVideoFileError(msg=msg, source=self.__class__.__name__)
        if inclusion_zones is not None:
            check_instance(source=f'{self.__class__.__name__} inclusion_zones', instance=inclusion_zones, accepted_types=(Polygon, MultiPolygon,), raise_error=True)
        if status_label is not None:
            check_instance(source=f'{self.__class__.__name__} status_label', instance=status_label, accepted_types=(Label,), raise_error=True)
        self.status_label = status_label
        check_str(name='method', value=method, options=['absolute', 'light', 'dark'], raise_error=True)
        check_int(name='threshold', value=threshold, min_value=1, max_value=255)
        if close_kernel_size is not None:
            check_valid_tuple(x=close_kernel_size, source=f'{self.__class__.__name__} close kernel', accepted_lengths=(2,), valid_dtypes=(int,), min_integer=1)
            check_int(name=f'{self.__class__.__name__} close iterations', value=close_kernel_iterations, min_value=1, raise_error=True)
            self.closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_kernel_size)
        else:
            self.closing_kernel = None
        if open_kernel_size is not None:
            check_valid_tuple(x=open_kernel_size, source=f'{self.__class__.__name__} open_kernel_size', accepted_lengths=(2,), valid_dtypes=(int,), min_integer=1)
            check_int(name=f'{self.__class__.__name__} openiterations', value=open_kernel_iterations, min_value=1, raise_error=True)
            self.opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, open_kernel_size)
        else:
            self.opening_kernel = None
        self.bg_video_path, self.video_path, self.close_its, self.open_its = bg_video_path, video_path, close_kernel_iterations, open_kernel_iterations
        self.threshold, self.method = threshold, method
        self.img_idx = 0
        self.img_window = Toplevel()
        self.img_window.title(FRAME_NAME)
        self.img_lbl = Label(self.img_window, name='img_lbl')
        self.img_lbl.grid(row=0, column=0, sticky=NW)
        self.inclusion_zones = inclusion_zones
        self.interact_panel = CreateLabelFrameWithIcon(parent=self.img_window, header="CHANGE IMAGE", font=Formats.FONT_HEADER.value, padx=5, pady=5, icon_name='frames')
        self._set_status_bar_panel(text=f'WAIT: COMPUTING BACKGROUND FOR VIDEO {self.video_meta["video_name"]} ({datetime.now().strftime("%H:%M:%S")})...')
        bg_timer = SimbaTimer(start=True)
        self.avg_frm = create_average_frm(video_path=self.bg_video_path, verbose=False)

        bg_timer.stop_timer()
        self._set_status_bar_panel(text=f'BACKGROUND COMPLETE FOR VIDEO {self.video_meta["video_name"]} (elapsed time: {bg_timer.elapsed_time_str}s...')
        self.create_img(idx=self.img_idx)
        self.forward_1s_btn = SimbaButton(parent=self.interact_panel, txt="+1s", img='plus_green_2', font=Formats.FONT_REGULAR.value, txt_clr='darkgreen', cmd=self._change_img, cmd_kwargs={'stride': int(self.video_meta['fps'])})
        self.backwards_1s_btn = SimbaButton(parent=self.interact_panel, txt="-1s", img='minus_blue_2', font=Formats.FONT_REGULAR.value, txt_clr='darkblue', cmd=self._change_img, cmd_kwargs={'stride': -int(self.video_meta['fps'])})
        self.custom_seconds_entry = Entry_Box(parent=self.interact_panel, fileDescription='CUSTOM SECONDS:', labelwidth=18, validation='numeric', entry_box_width=4, value=10)
        self.custom_fwd_btn = SimbaButton(parent=self.interact_panel, txt="FORWARD", img='fastforward_green_2', font=Formats.FONT_REGULAR.value, txt_clr='darkgreen', cmd=self._change_img, cmd_kwargs={'stride': 'custom_forward'})
        self.custom_backwards_btn = SimbaButton(parent=self.interact_panel, txt="REVERSE", img='rewind_blue_2', font=Formats.FONT_REGULAR.value, txt_clr='darkblue', cmd=self._change_img,cmd_kwargs={'stride': 'custom_backward'})
        self.first_frm_btn = SimbaButton(parent=self.interact_panel, txt="FIRST FRAME", img='first_frame_blue', font=Formats.FONT_REGULAR.value, txt_clr='darkblue', cmd=self._change_img, cmd_kwargs={'stride': 'first'})
        self.last_frm_btn = SimbaButton(parent=self.interact_panel, txt="LAST FRAME", img='last_frame_blue', font=Formats.FONT_REGULAR.value, txt_clr='darkblue', cmd=self._change_img, cmd_kwargs={'stride': 'last'})
        self.interact_panel.grid(row=1, column=0, sticky=NW)
        self.forward_1s_btn.grid(row=0, column=0, sticky=NW, pady=10)
        self.backwards_1s_btn.grid(row=0, column=1, sticky=NW, pady=10, padx=10)
        self.custom_seconds_entry.grid(row=0, column=2, sticky=NW, pady=10)
        self.custom_fwd_btn.grid(row=0, column=3, sticky=NW, pady=10)
        self.custom_backwards_btn.grid(row=0, column=4, sticky=NW, pady=10)
        self.first_frm_btn.grid(row=0, column=5, sticky=NW, pady=10)
        self.last_frm_btn.grid(row=0, column=6, sticky=NW, pady=10)
        self.img_window.protocol("WM_DELETE_WINDOW", self._close)

        self.img_window.mainloop()

    def draw_img(self, img: np.ndarray):
        self.pil_image = Image.fromarray(img)
        self.tk_image = ImageTk.PhotoImage(self.pil_image)
        self.img_lbl.configure(image=self.tk_image)
        self.img_lbl.image = self.tk_image

    def create_img(self, idx: int):
        img = read_frm_of_video(video_path=self.video_path, frame_index=idx)
        diff_img = ImageMixin.img_diff(x=img, y=self.avg_frm, threshold=self.threshold, method=self.method)
        diff_img = cv2.cvtColor(diff_img, cv2.COLOR_GRAY2RGB)
        if self.opening_kernel is not None:
            diff_img = cv2.morphologyEx(diff_img, cv2.MORPH_OPEN, self.opening_kernel, iterations=self.open_its)
        if self.closing_kernel is not None:
            diff_img = cv2.morphologyEx(diff_img, cv2.MORPH_CLOSE, self.closing_kernel, iterations=self.close_its)
        if self.inclusion_zones is not None:
            diff_img = GeometryMixin().view_shapes(shapes=[self.inclusion_zones], bg_img=diff_img, pixel_buffer=1)
        self.draw_img(img=diff_img)


    def _set_status_bar_panel(self, text: str, fg: str = 'blue'):
        if self.status_label is not None:
            self.status_label.configure(text=text, fg=fg)
            self.status_label.update_idletasks()

    def _change_img(self, stride: Union[int, str]):
        custom_s, new_frm_idx = self.custom_seconds_entry.entry_get.strip(), None
        if isinstance(stride, int):
            new_frm_idx = self.img_idx + stride
        elif stride == 'custom_forward':
            check_int(name='CUSTOM SECONDS', value=custom_s, min_value=1)
            custom_s = int(custom_s)
            new_frm_idx = int(self.img_idx + (custom_s * self.video_meta['fps']))
        elif stride == 'custom_backward':
            check_int(name='CUSTOM SECONDS', value=custom_s, min_value=1)
            custom_s = int(custom_s)
            new_frm_idx = int(self.img_idx - (custom_s * self.video_meta['fps']))
        elif stride == 'first':
            new_frm_idx = 0
        elif stride == 'last':
            new_frm_idx = self.video_meta['frame_count']-1
        if (0 > new_frm_idx) or (new_frm_idx > self.video_meta['frame_count']-1):
            msg = f'Cannot change frame. The new frame index {new_frm_idx} is outside the video {self.video_meta["video_name"]} frame range (video has {self.video_meta["frame_count"]} frames).'
            self._set_status_bar_panel(text=msg, fg='red')
            raise FrameRangeError(msg=msg, source=self.__class__.__name__)
        else:
            self.create_img(idx=new_frm_idx)
            self.img_idx = new_frm_idx
            self._set_status_bar_panel(text=f'Showing frame {self.img_idx} (video: {self.video_meta["video_name"]})...')


    def _close(self):
        self._set_status_bar_panel(text=f'Closing check for video {self.video_meta["video_name"]}...', fg='blue')
        try:
            self.img_window.destroy()
            self.img_window.quit()
        except:
            pass


# quick_checker = BlobQuickChecker(video_path=r"D:\EPM\sample_2\2025-02-24 08-25-56.mp4",
#                                  bg_video_path=r"D:\EPM\sample_2\bg\2025-02-24 08-25-56.mp4",
#                                  close_kernel_size=(7, 7),
#                                  close_kernel_iterations=3)
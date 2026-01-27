import os
from tkinter import *
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image, ImageTk

from simba.mixins.image_mixin import ImageMixin
from simba.ui.tkinter_functions import SimbaButton, SimBALabel, SimBAScaleBar
from simba.utils.checks import (check_file_exist_and_readable, check_int,
                                check_valid_boolean)
from simba.utils.enums import Formats, TkBinds
from simba.utils.lookups import get_icons_paths, get_monitor_info
from simba.utils.read_write import (get_video_meta_data, read_frm_of_video,
                                    seconds_to_timestamp)


class TimelapseSlider():

    """
    Interactive timelapse viewer with segment selection sliders.

    Creates a Tkinter GUI window displaying a timelapse composite image generated from evenly-spaced frames
    across a video. Includes interactive sliders to select start and end times for video segments, with
    visual highlighting of the selected segment and frame previews.

    .. image:: _static/img/TimelapseSlider.png
       :width: 600
       :align: center

    :param Union[str, os.PathLike] video_path: Path to video file to create timelapse from.
    :param int frame_cnt: Number of frames to include in timelapse composite. Default 25.
    :param Optional[int] ruler_width: Width per frame in pixels. If None, calculated to match video width. Default None.
    :param Optional[int] crop_ratio: Percentage of frame width to keep (0-100). Default 50.
    :param int padding: Padding in pixels added to timelapse when ruler is shown. Default 60.
    :param int ruler_divisions: Number of major divisions on time ruler. Default 6.
    :param bool show_ruler: If True, display time ruler below timelapse. Default True.
    :param int ruler_height: Height of ruler in pixels. Default 60.

    :example:
    >>> slider = TimelapseSlider(video_path='path/to/video.mp4', frame_cnt=25, crop_ratio=75)
    >>> slider.run()
    >>> # Use sliders to select segment, then access selected times and frames:
    >>> start_time = slider.get_start_time()  # seconds (float)
    >>> end_time = slider.get_end_time()  # seconds (float)
    >>> start_time_str = slider.get_start_time_str()  # "HH:MM:SS" string
    >>> end_time_str = slider.get_end_time_str()  # "HH:MM:SS" string
    >>> start_frame = slider.get_start_frame()  # frame number (int)
    >>> end_frame = slider.get_end_frame()  # frame number (int)
    >>> slider.close()
    """

    def __init__(self,
                 video_path: Union[str, os.PathLike],
                 frame_cnt: int = 25,
                 crop_ratio: Optional[int] = 50,
                 padding: int = 60,
                 ruler_divisions: int = 6,
                 show_ruler: bool = True,
                 ruler_height: Optional[int] = None,
                 ruler_width: Optional[int] = None,
                 img_width: Optional[int] = None,
                 img_height: Optional[int] = None):

        check_file_exist_and_readable(file_path=video_path)
        check_int(name='frame_cnt', value=frame_cnt, min_value=1, raise_error=True)
        _, (self.monitor_width, self.monitor_height) = get_monitor_info()
        if ruler_width is not None: check_int(name='size', value=ruler_width, min_value=1, raise_error=True)
        else: ruler_width = int(self.monitor_width * 0.5)
        if ruler_height is not None: check_int(name='ruler_height', value=ruler_height, min_value=1, raise_error=True)
        else: ruler_height = int(self.monitor_height * 0.05)
        if img_width is not None:  check_int(name='img_width', value=img_width, min_value=1, raise_error=True)
        else: img_width = int(self.monitor_width * 0.5)
        if img_height is not None:  check_int(name='img_height', value=img_height, min_value=1, raise_error=True)
        else: img_height = int(self.monitor_height * 0.5)


        check_int(name='padding', value=padding, min_value=1, raise_error=True)
        check_valid_boolean(value=show_ruler, source=f'{self.__class__.__name__} show_ruler', raise_error=True)
        self.video_meta = get_video_meta_data(video_path=video_path, raise_error=True)
        if show_ruler: check_int(name='ruler_divisions', value=ruler_divisions, min_value=1, raise_error=True)
        self.size, self.padding, self.crop_ratio, self.frame_cnt = ruler_width, padding, crop_ratio, frame_cnt
        self.ruler_height, self.video_path, self.show_ruler, self.ruler_divisions = ruler_height, video_path, show_ruler, ruler_divisions
        self.img_width, self.img_height = img_width, img_height
        self.frm_name = f'{self.video_meta["video_name"]} - TIMELAPSE VIEWER - hit "X" or ESC t close'
        self.video_capture = None
        self._pending_frame_update = None
        self._frame_debounce_ms = 50

    def _draw_img(self, img: np.ndarray, lbl: SimBALabel):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.pil_image = Image.fromarray(img_rgb)
        self.tk_image = ImageTk.PhotoImage(self.pil_image)
        lbl.configure(image=self.tk_image)
        lbl.image = self.tk_image

    def _update_selection(self, slider_type: str):
        start_sec = int(self.start_scale.get_value())
        end_sec = int(self.end_scale.get_value())
        max_sec = int(self.video_meta['video_length_s'])
        if slider_type == 'start':
            if start_sec >= end_sec:
                end_sec = min(start_sec + 1, max_sec)
                self.end_scale.set_value(end_sec)
        else:
            if end_sec <= start_sec:
                start_sec = max(end_sec - 1, 0)
                self.start_scale.set_value(start_sec)

        self.selected_start[0] = start_sec
        self.selected_end[0] = end_sec
        
        start_frame = int(start_sec * self.video_meta['fps'])
        end_frame = int(end_sec * self.video_meta['fps'])
        if start_frame >= self.video_meta['frame_count']: start_frame = self.video_meta['frame_count'] - 1
        if end_frame >= self.video_meta['frame_count']: end_frame = self.video_meta['frame_count'] - 1
        if start_frame < 0: start_frame = 0
        if end_frame < 0: end_frame = 0
        self.selected_start_frame[0] = start_frame
        self.selected_end_frame[0] = end_frame

        self.start_time_label.config(text=seconds_to_timestamp(start_sec), fg='green')
        self.end_time_label.config(text=seconds_to_timestamp(end_sec), fg='red')

        if self.video_meta['video_length_s'] > 0:
            self._highlight_segment(start_sec, end_sec)
            self._schedule_frame_update(slider_type=slider_type)

    def _move_start_frame(self, direction: int):
        current_seconds = self.selected_start[0]
        new_seconds = current_seconds + direction
        new_seconds = max(0, min(new_seconds, int(self.video_meta['video_length_s'])))
        self.start_scale.set_value(int(new_seconds))
        self._update_selection(slider_type='start')
        if self._pending_frame_update is not None:
            if hasattr(self, 'img_window') and self.img_window.winfo_exists():
                self.img_window.after_cancel(self._pending_frame_update)
        self._update_frame_display(slider_type='start')

    def _move_end_frame(self, direction: int):
        current_seconds = self.selected_end[0]
        new_seconds = current_seconds + direction
        new_seconds = max(0, min(new_seconds, int(self.video_meta['video_length_s'])))
        self.end_scale.set_value(int(new_seconds))
        self._update_selection(slider_type='end')
        if self._pending_frame_update is not None:
            if hasattr(self, 'img_window') and self.img_window.winfo_exists():
                self.img_window.after_cancel(self._pending_frame_update)
        self._update_frame_display(slider_type='end')

    def _schedule_frame_update(self, slider_type: str):
        """Schedule frame preview update with debouncing.
        
        Cancels any pending frame update and schedules a new one. If the slider
        moves again before the delay expires, the update is cancelled and rescheduled.
        This prevents expensive frame reads during fast slider dragging.
        """
        if not hasattr(self, 'img_window') or not self.img_window.winfo_exists():
            return

        if self._pending_frame_update is not None: self.img_window.after_cancel(self._pending_frame_update)

        self._pending_frame_update = self.img_window.after(self._frame_debounce_ms, lambda: self._update_frame_display(slider_type=slider_type))

    def _update_frame_display(self, slider_type: str):
        if slider_type == 'start':
            seconds = self.selected_start[0]
            self.frame_label.config(text=f"Start Frame Preview ({seconds_to_timestamp(seconds)})", font=Formats.FONT_LARGE_BOLD.value, fg='green')
        else:
            seconds = self.selected_end[0]
            self.frame_label.config(text=f"End Frame Preview ({seconds_to_timestamp(seconds)})", font=Formats.FONT_LARGE_BOLD.value, fg='red')

        frame_index = int(seconds * self.video_meta['fps'])
        if frame_index >= self.video_meta['frame_count']: frame_index = self.video_meta['frame_count'] - 1
        if frame_index < 0: frame_index = 0

        if self.video_capture is not None and self.video_capture.isOpened():
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = self.video_capture.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                target_w, target_h = self.img_width, self.img_height
                scale = min(target_w / w, target_h / h)
                new_w, new_h = int(w * scale), int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                self._draw_img(img=frame, lbl=self.frame_display_lbl)

    def _highlight_segment(self, start_sec: int, end_sec: int):
        timelapse_width = self.original_timelapse.shape[1]
        start_x = int((start_sec / self.video_meta['video_length_s']) * timelapse_width)
        end_x = int((end_sec / self.video_meta['video_length_s']) * timelapse_width)
        highlighted = self.original_timelapse.copy()
        mask = np.ones(highlighted.shape[:2], dtype=np.uint8) * 128
        mask[:, start_x:end_x] = 255
        mask = cv2.merge([mask, mask, mask])
        highlighted = cv2.multiply(highlighted, mask.astype(np.uint8), scale=1/255.0)
        cv2.line(highlighted, (start_x, 0), (start_x, highlighted.shape[0]), (0, 255, 0), 2)
        cv2.line(highlighted, (end_x, 0), (end_x, highlighted.shape[0]), (0, 255, 0), 2)
        self._draw_img(img=highlighted, lbl=self.img_lbl)

    def run(self):
        self.video_capture = cv2.VideoCapture(self.video_path)
        if not self.video_capture.isOpened():
            raise ValueError(f"Failed to open video file: {self.video_path}")
        
        self.timelapse_img = ImageMixin.get_timelapse_img(video_path=self.video_path, frame_cnt=self.frame_cnt, size=self.size, crop_ratio=self.crop_ratio)
        if self.show_ruler:
            timelapse_height, timelapse_width = self.timelapse_img.shape[0], self.timelapse_img.shape[1]
            padded_timelapse = np.zeros((timelapse_height, timelapse_width + (2 * self.padding), 3), dtype=np.uint8)
            padded_timelapse[:, self.padding:self.padding + timelapse_width] = self.timelapse_img
            ruler = ImageMixin.create_time_ruler(video_path=self.video_path, width=timelapse_width, height=self.ruler_height, num_divisions=self.ruler_divisions)
            self.timelapse_img = cv2.vconcat([padded_timelapse, ruler])

        self.original_timelapse = self.timelapse_img.copy()
        self.img_window = Toplevel()
        self.img_window.resizable(True, True)
        self.img_window.title(self.frm_name)
        self.img_window.protocol("WM_DELETE_WINDOW", self.close)
        # Bind Escape key to close window
        self.img_window.bind(TkBinds.ESCAPE.value, lambda event: self.close())


        self.img_lbl = SimBALabel(parent=self.img_window, txt='')
        self.img_lbl.pack()
        self._draw_img(img=self.timelapse_img, lbl=self.img_lbl)
        self.frame_display_frame = Frame(self.img_window)
        self.frame_display_frame.pack(pady=10, padx=10, fill=BOTH, expand=True)
        self.frame_label = SimBALabel(parent=self.frame_display_frame, txt="Frame Preview", font=Formats.FONT_REGULAR_BOLD.value)
        self.frame_label.pack()
        self.frame_display_lbl = SimBALabel(parent=self.frame_display_frame, txt='', bg_clr='black')
        self.frame_display_lbl.pack(pady=5)
        self.slider_frame = Frame(self.img_window)
        self.slider_frame.pack(pady=10, padx=10, fill=X)
        self.slider_frame.columnconfigure(index=0, weight=1)
        self.slider_frame.columnconfigure(index=1, weight=0)
        self.slider_frame.columnconfigure(index=2, weight=0)
        self.slider_frame.columnconfigure(index=3, weight=0)
        self.slider_frame.columnconfigure(index=4, weight=0)
        self.slider_frame.columnconfigure(index=5, weight=1)
    
        self.start_scale = SimBAScaleBar(parent=self.slider_frame, label="START TIME:", from_=0, to=self.video_meta['video_length_s'], orient=HORIZONTAL, length=400, resolution=1, value=0, showvalue=False, label_width=15, sliderrelief='raised', troughcolor='white', activebackground='green', lbl_font=Formats.FONT_LARGE_BOLD.value)
        self.start_scale.grid(row=0, column=1, padx=5)
        self.start_scale.scale.config(command=lambda x: self._update_selection(slider_type='start'))
    
        self.start_time_label = SimBALabel(parent=self.slider_frame, txt="00:00:00", font=Formats.FONT_LARGE_BOLD.value, width=10, txt_clr='green')
        self.start_time_label.grid(row=0, column=2, padx=5)

        self.start_frame_left_btn = SimbaButton(parent=self.slider_frame, txt="-1s", tooltip_txt="Previous second", cmd=self._move_start_frame, cmd_kwargs={'direction': -1}, font=Formats.FONT_REGULAR_BOLD.value, img='left_arrow_green')
        self.start_frame_left_btn.grid(row=0, column=3, padx=2)
        self.start_frame_right_btn = SimbaButton(parent=self.slider_frame, txt="+1s", tooltip_txt="Next second", cmd=self._move_start_frame, cmd_kwargs={'direction': 1}, font=Formats.FONT_REGULAR_BOLD.value, img='right_arrow_green')
        self.start_frame_right_btn.grid(row=0, column=4, padx=2)

        self.end_scale = SimBAScaleBar(parent=self.slider_frame, label="END TIME:", from_=0, to=int(self.video_meta['video_length_s']), orient=HORIZONTAL, length=400, resolution=1, value=int(self.video_meta['video_length_s']), showvalue=False, label_width=15, sliderrelief='raised', troughcolor='white', activebackground='red',  lbl_font=Formats.FONT_LARGE_BOLD.value)
        self.end_scale.grid(row=1, column=1, padx=5)
        self.end_scale.scale.config(command=lambda x: self._update_selection(slider_type='end'))

        self.end_time_label = SimBALabel(parent=self.slider_frame, txt=seconds_to_timestamp(int(self.video_meta['video_length_s'])), font=Formats.FONT_LARGE_BOLD.value, width=10, txt_clr='red')
        self.end_time_label.grid(row=1, column=2, padx=5)

        self.end_frame_left_btn = SimbaButton(parent=self.slider_frame, txt="-1s", tooltip_txt="Previous second", cmd=self._move_end_frame, cmd_kwargs={'direction': -1}, font=Formats.FONT_REGULAR_BOLD.value, img='left_arrow_red')
        self.end_frame_left_btn.grid(row=1, column=3, padx=2)
        self.end_frame_right_btn = SimbaButton(parent=self.slider_frame, txt="+1s", tooltip_txt="Next second", cmd=self._move_end_frame, cmd_kwargs={'direction': 1}, font=Formats.FONT_REGULAR_BOLD.value, img='right_arrow_red')
        self.end_frame_right_btn.grid(row=1, column=4, padx=2)

        self.selected_start = [0]
        self.selected_end = [int(self.video_meta['video_length_s'])]
        self.selected_start_frame = [0]
        end_frame = int(self.video_meta['frame_count']) - 1
        if end_frame < 0: end_frame = 0
        self.selected_end_frame = [end_frame]
        
        self.img_window.update_idletasks()
        self.img_window.update()
        
        req_width, req_height = self.img_window.winfo_reqwidth(), self.img_window.winfo_reqheight()
        min_width = max(self.timelapse_img.shape[1] + 60, req_width + 20)
        timelapse_height = self.timelapse_img.shape[0]
        frame_preview_height = self.img_height
        slider_height, padding_total = 150, 50
        calculated_min_height = timelapse_height + frame_preview_height + slider_height + padding_total
        min_height = max(calculated_min_height, req_height + 50, timelapse_height + 400)
        max_height = int(self.monitor_height * 0.95)
        if min_height > max_height: min_height = max_height
        
        self.img_window.minsize(min_width, min_height)
        self.img_window.geometry(f"{min_width}x{min_height}")
        self._update_frame_display(slider_type='start')

    def get_start_time(self) -> float:
        return self.selected_start[0]
    
    def get_end_time(self) -> float:
        return self.selected_end[0]
    
    def get_start_time_str(self) -> str:
        return seconds_to_timestamp(self.selected_start[0])
    
    def get_end_time_str(self) -> str:
        return seconds_to_timestamp(self.selected_end[0])
    
    def get_start_frame(self) -> int:
        return self.selected_start_frame[0]
    
    def get_end_frame(self) -> int:
        return self.selected_end_frame[0]

    def close(self):
        if self._pending_frame_update is not None:
            if hasattr(self, 'img_window') and self.img_window.winfo_exists():
                self.img_window.after_cancel(self._pending_frame_update)
            self._pending_frame_update = None

        # Unbind Escape key if window still exists
        if hasattr(self, 'img_window') and self.img_window.winfo_exists():
            try:
                self.img_window.unbind(TkBinds.ESCAPE.value)
            except:
                pass

        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None

        if hasattr(self, 'img_window') and self.img_window.winfo_exists():
            self.img_window.destroy()



#
# x = TimelapseSlider(video_path=r"E:\troubleshooting\mitra_emergence\project_folder\clip_test\Box1_180mISOcontrol_Females_clipped_progress_bar.mp4",
#         frame_cnt=25,
#         crop_ratio=75)
# x.run()
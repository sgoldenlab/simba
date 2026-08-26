__author__ = "Simon Nilsson; sronilsson@gmail.com"

import os
from tkinter import *
from typing import List, Union, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image as Img
from PIL import ImageTk
from copy import deepcopy

from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import check_file_exist_and_readable, check_valid_array, check_valid_boolean, check_valid_dataframe, check_valid_tuple
from simba.utils.enums import Formats
from simba.utils.errors import ColumnNotFoundError, InvalidInputError
from simba.utils.lookups import get_icons_paths, get_simba_font_name_and_path, get_color_dict
from simba.utils.enums import TextOptions
from simba.utils.read_write import get_video_meta_data, read_frm_of_video, str_2_bool
from simba.utils.warnings import FrameRangeWarning
from simba.ui.tkinter_functions import SimBALabel, SimBAScaleBar, SimbaButton, Entry_Box, CreateLabelFrameWithIcon, SimBADropDown

PADDING = 5
MAX_SIZE = (1080, 650)


class InteractiveVideoPlotterWindow(ConfigReader, PlottingMixin):
    def __init__(self,
                 video_path: Union[str, os.PathLike],
                 p_arr: np.array,
                 config_path: Optional[Union[str, os.PathLike]] = None,
                 data_df: Optional[pd.DataFrame] = None,
                 data_path: Optional[Union[str, os.PathLike]] = None,
                 show_names: bool = False,
                 show_pose: bool = False,
                 max_img_size: Optional[Tuple[int, int]] = None):

        check_file_exist_and_readable(file_path=video_path, raise_error=True)
        check_valid_array(data=p_arr, source=f'{self.__class__.__name__} p_arr', accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_valid_boolean(value=show_names, source=f'{self.__class__.__name__} show_names', raise_error=True)
        check_valid_boolean(value=show_pose, source=f'{self.__class__.__name__} show_pose', raise_error=True)
        if data_path is not None:
            check_file_exist_and_readable(file_path=data_path)
        self.data_path = data_path
        if max_img_size is not None:
            check_valid_tuple(x=max_img_size, source=f'{self.__class__.__name__} max_img_size', accepted_lengths=(2,), valid_dtypes=(int,), min_integer=1)
        self.max_img_size = MAX_SIZE if max_img_size is None else tuple(max_img_size)
        if data_df is not None:
            check_valid_dataframe(df=data_df, source=f'{self.__class__.__name__} data_df', valid_dtypes=Formats.NUMERIC_DTYPES.value)
        self.animal_bp_cols, self.animal_clrs = {}, {}
        if config_path is not None:
            check_file_exist_and_readable(file_path=config_path, raise_error=True)
            ConfigReader.__init__(self, config_path=config_path, read_video_info=False, create_logger=False)
            for animal_name, animal_data in self.animal_bp_dict.items():
                if len(animal_data['X_bps']) != len(animal_data['Y_bps']):
                    raise InvalidInputError(msg=f'The animal {animal_name} has {len(animal_data["X_bps"])} X and {len(animal_data["Y_bps"])} Y body-part columns in the SimBA project {config_path}.', source=self.__class__.__name__)
                self.animal_bp_cols[animal_name] = list(zip(animal_data['X_bps'], animal_data['Y_bps']))
                self.animal_clrs[animal_name] = [tuple(int(c) for c in clr) for clr in animal_data['colors']]
        if (show_names or show_pose) and (not isinstance(data_df, pd.DataFrame) or config_path is None):
            raise InvalidInputError(msg='If showing names and/or pose, please pass config_path and data_df')
        self.video_meta_data = get_video_meta_data(video_path=video_path)
        if config_path is not None:   # NOTE: the text and circle sizes are computed for any project - not just when names/pose are displayed at startup - as they can be switched on later through the preferences menu.
            _, self.font_path = get_simba_font_name_and_path(font=TextOptions.DEFAULT_FONT.value)
            self.font_size, _, _ = PlottingMixin().get_optimal_font_size_ttf(text=list(self.animal_bp_dict.keys()), font_path=self.font_path, accepted_px_width=int(self.video_meta_data['width']/10), accepted_px_height=int(self.video_meta_data['height']/10))
            self.circle_size = PlottingMixin().get_optimal_circle_size(frame_size=(self.video_meta_data['width'], self.video_meta_data['height']), circle_frame_ratio=100)
        else:
            self.font_path, self.font_size, self.circle_size = None, None, None
        self.show_names, self.show_pose, self.data_df, self.config_path = show_names, show_pose, data_df, config_path
        missing_cols = self._get_missing_bp_cols(show_names=self.show_names, show_pose=self.show_pose)
        if len(missing_cols) > 0:
            raise ColumnNotFoundError(column_name=', '.join(missing_cols), file_name=self.data_path if self.data_path is not None else 'the passed data frame', source=self.__class__.__name__)
        self.main_frm = Toplevel()
        self.btn_icons = get_icons_paths()
        for k in self.btn_icons.keys():
            self.btn_icons[k]["img"] = ImageTk.PhotoImage(image=Img.open(os.path.join(os.path.dirname("__file__"), self.btn_icons[k]["icon_path"])))
        self.main_frm.iconphoto(False, self.btn_icons['frames']['img'])
        self.current_frm_number, self.jump_size = 0, 0
        self.img_frm = Frame(self.main_frm)
        self.clr_dict = get_color_dict()
        self.text_bg_clr, self.name_bg_opacity = 'Black', 0.4
        self.img_frm.grid(row=0, column=1, sticky=NW)
        self.button_frame = Frame(self.main_frm, bd=2, width=700, height=300)
        self.button_frame.grid(row=1, column=0)
        self.main_frm.wm_title(self.video_meta_data['video_name'])
        self.cap = cv2.VideoCapture(video_path)
        self.max_frm = np.argmax(p_arr)
        self.frame_id_lbl = SimBALabel(parent=self.button_frame, txt="FRAME NUMBER", justify='center')
        self.frame_id_lbl.grid(row=0, column=0, pady=(0, PADDING))

        self.nav_frm = Frame(self.button_frame)
        self.nav_frm.grid(row=1, column=0)
        self.back_first_frm = SimbaButton(parent=self.nav_frm, txt="<<", cmd=self.load_new_frame, cmd_kwargs={'frm_cnt': lambda: 0})
        self.back_first_frm.grid(row=0, column=0)
        self.back_one_frm_btn = SimbaButton(parent=self.nav_frm, txt="<", cmd=self.load_new_frame, cmd_kwargs={'frm_cnt': lambda: self.current_frm_number - 1})
        self.back_one_frm_btn.grid(row=0, column=1, padx=(0, PADDING))
        self.frame_entry_box = Entry_Box(parent=self.nav_frm, value=self.current_frm_number, allow_blank=False, justify='center', padx=PADDING, validation='numeric')
        self.frame_entry_box.grid(row=0, column=2)
        self.forward_next_frm_btn = SimbaButton(parent=self.nav_frm, txt=">", cmd=self.load_new_frame, cmd_kwargs={'frm_cnt': lambda: self.current_frm_number + 1})
        self.forward_next_frm_btn.grid(row=0, column=3, padx=(PADDING, 0))
        self.forward_last_frm_btn = SimbaButton(parent=self.nav_frm, txt=">>", cmd=self.load_new_frame, cmd_kwargs={'frm_cnt': lambda: self.video_meta_data["frame_count"] - 1})
        self.forward_last_frm_btn.grid(row=0, column=4)
        self.select_frm_btn = SimbaButton(parent=self.button_frame, txt="Jump to selected frame", cmd=self.load_new_frame, cmd_kwargs={'frm_cnt': lambda: int(self.frame_entry_box.entry_get)}, img='jump')
        self.select_frm_btn.grid(row=2, column=0, pady=(PADDING, 0))

        self.jump_frm = Frame(self.button_frame)
        self.jump_frm.grid(row=3, column=0, pady=(PADDING, 0))

        self.jump_size_scale = SimBAScaleBar(parent=self.jump_frm, label="Jump Size:", from_=0, to=100, length=200, value=0)
        self.jump_size_scale.grid(row=0, column=0)
        self.jump_back_btn = SimbaButton(parent=self.jump_frm, txt="<<", cmd=self.load_new_frame, cmd_kwargs={'frm_cnt': lambda: self.current_frm_number - self.jump_size_scale.get()})
        self.jump_back_btn.grid(row=0, column=1, padx=(PADDING, 0))
        self.jump_forward_btn = SimbaButton(parent=self.jump_frm, txt=">>", cmd=self.load_new_frame, cmd_kwargs={'frm_cnt': lambda: self.current_frm_number + self.jump_size_scale.get()})
        self.jump_forward_btn.grid(row=0, column=2)

        self.load_new_frame(frm_cnt=self.current_frm_number)

        instructions_frm = Frame(self.main_frm, width=100, height=100)
        instructions_frm.grid(row=0, column=2, sticky=N)

        key_presses = SimBALabel(parent=instructions_frm, txt="\n\n Keyboard shortcuts for frame navigation: \n Right Arrow = +1 frame"
                                                "\n Left Arrow = -1 frame"
                                                "\n Ctrl + l = Last frame"
                                                "\n Ctrl + o = First frame", font=Formats.FONT_REGULAR_BOLD.value)
        key_presses.grid(row=0, column=0, sticky=S)
        self.move_to_highest_p_btn = SimbaButton(parent=instructions_frm, txt="SHOW HIGHEST \n PROBABILITY FRAME", cmd=self.load_new_frame, cmd_kwargs={'frm_cnt': lambda: self.max_frm}, img='pct')
        self.move_to_highest_p_btn.grid(row=1, column=0, sticky=N, pady=(PADDING, 0))

        self.menu = Menu(self.main_frm, tearoff=0)
        self.file_menu = Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="File...", menu=self.file_menu)
        self.file_menu.add_command(label="Preferences", compound="left", image=self.btn_icons["settings"]["img"], command=lambda: self.preferences_pop_up())
        self.main_frm.config(menu=self.menu)

        self._bind_keys()

    def _get_missing_bp_cols(self, show_names: bool, show_pose: bool) -> List[str]:
        """Return the body-part columns required for drawing animal names and/or pose that are missing from ``self.data_df``."""
        if not (show_names or show_pose) or not isinstance(self.data_df, pd.DataFrame):
            return []
        required_cols = []
        for bp_cols in self.animal_bp_cols.values():
            required_cols.extend([c for bp in (bp_cols if show_pose else bp_cols[:1]) for c in bp])
        df_cols = set(self.data_df.columns)
        return [c for c in required_cols if c not in df_cols]

    def _show_names(self, img: np.ndarray, frm_cnt: int) -> np.ndarray:
        for animal_name, bp_cols in self.animal_bp_cols.items():
            name_loc = self.data_df.loc[frm_cnt, list(bp_cols[0])].values.astype(np.float32)
            if np.isnan(name_loc).any():
                continue
            img = PlottingMixin().put_text(img=img, text=animal_name, pos=(int(name_loc[0]), int(name_loc[1])), font_size=self.font_size, font_path=self.font_path, text_color=self.animal_clrs[animal_name][0], text_bg_alpha=self.name_bg_opacity, text_color_bg=self.clr_dict[self.text_bg_clr])
        return img

    def _show_pose(self, img: np.ndarray, frm_cnt: int) -> np.ndarray:
        for animal_name, bp_cols in self.animal_bp_cols.items():
            animal_pose_data = self.data_df.loc[frm_cnt, [c for bp in bp_cols for c in bp]].values.astype(np.float32).reshape(-1, 2)
            for bp_idx in range(animal_pose_data.shape[0]):
                if np.isnan(animal_pose_data[bp_idx]).any():
                    continue
                img = cv2.circle(img=img, center=(int(animal_pose_data[bp_idx][0]), int(animal_pose_data[bp_idx][1])), radius=self.circle_size, color=self.animal_clrs[animal_name][bp_idx], thickness=-1)
        return img

    def preferences_pop_up(self):
        if hasattr(self, 'preferences_frm'):
            self.preferences_frm.destroy()
        self.preferences_frm = Toplevel()
        self.preferences_frm.minsize(400, 300)
        self.preferences_frm.wm_title("PREFERENCES")
        self.preferences_frm.iconphoto(False, self.btn_icons['settings']["img"])

        pref_frm_panel = CreateLabelFrameWithIcon(parent=self.preferences_frm, header="PREFERENCES", icon_name='settings', padx=5, pady=5)
        self.show_names_dd = SimBADropDown(parent=pref_frm_panel, dropdown_options=['TRUE', 'FALSE'], label="SHOW ANIMAL NAMES: ", label_width=35, dropdown_width=35, value='TRUE' if self.show_names else 'FALSE', tooltip_key='ROI_POLYGON_TOLERANCE', img='label')
        self.show_pose_dd = SimBADropDown(parent=pref_frm_panel, dropdown_options=['TRUE', 'FALSE'], label="SHOW ANIMAL POSE: ", label_width=35, dropdown_width=35, value='TRUE' if self.show_pose else 'FALSE', tooltip_key='ROI_KEYBOARD_SENSITIVITY', img='pose')
        self.name_bg_clr_dd = SimBADropDown(parent=pref_frm_panel, dropdown_options=list(self.clr_dict.keys()), label="NAME BACKGROUND COLOR: ", label_width=35, dropdown_width=35, value=self.text_bg_clr, tooltip_key='ROI_KEYBOARD_SENSITIVITY', img='paint')
        self.name_bg_opacity_dd = SimBADropDown(parent=pref_frm_panel, dropdown_options=[f"{x / 10:.1f}" for x in range(0, 11)], label="NAME BACKGROUND OPACITY: ", label_width=35, dropdown_width=35, value=self.name_bg_opacity, tooltip_key='ROI_KEYBOARD_SENSITIVITY', img='opacity')

        pref_save_btn = SimbaButton(parent=pref_frm_panel, txt="SAVE", img='save_large', font=Formats.FONT_REGULAR.value, cmd=self._set_preferences)
        pref_frm_panel.grid(row=0, column=0)
        self.show_names_dd.grid(row=0, column=0, sticky=NW, pady=5)
        self.show_pose_dd.grid(row=1, column=0, sticky=NW, pady=5)
        self.name_bg_clr_dd.grid(row=2, column=0, sticky=NW, pady=5)
        self.name_bg_opacity_dd.grid(row=3, column=0, sticky=NW, pady=5)
        pref_save_btn.grid(row=4, column=0, sticky=NW, pady=5)
        self.status_bar = SimBALabel(parent=self.preferences_frm, txt='', txt_clr='black', bg_clr=None, font=Formats.FONT_REGULAR.value, relief='sunken')
        self.status_bar.grid(row=1, column=0, sticky='we')
        self.preferences_frm.grid_rowconfigure(1, weight=0)

    def _set_preferences(self):
        show_pose_selection = str_2_bool(self.show_pose_dd.get_value())
        show_name_selection = str_2_bool(self.show_names_dd.get_value())
        missing_cols = self._get_missing_bp_cols(show_names=show_name_selection, show_pose=show_pose_selection)
        if (show_pose_selection or show_name_selection) and (not isinstance(self.data_df, pd.DataFrame) or self.config_path is None):
            self.show_names_dd.set_value(value='FALSE')
            self.show_pose_dd.set_value(value='FALSE')
            self.status_bar.configure(text='If showing names and/or pose, please pass config_path and data_df', fg='red')
            self.status_bar.update_idletasks()
            raise InvalidInputError(msg='If showing names and/or pose, please pass config_path and data_df')
        elif len(missing_cols) > 0:
            self.show_names_dd.set_value(value='TRUE' if self.show_names else 'FALSE')
            self.show_pose_dd.set_value(value='TRUE' if self.show_pose else 'FALSE')
            self.status_bar.configure(text=f'The column(s) {missing_cols} could not be found in the data: names and pose cannot be displayed', fg='red')
            self.status_bar.update_idletasks()
        else:
            self.show_pose = deepcopy(show_pose_selection)
            self.show_names = deepcopy(show_name_selection)
            self.text_bg_clr = self.name_bg_clr_dd.get_value()
            self.name_bg_opacity = float(self.name_bg_opacity_dd.get_value())
            self.load_new_frame(frm_cnt=self.current_frm_number)
            self.status_bar.configure(text='Updated interactive probability grapher visualization settings', fg='blue')
            self.status_bar.update_idletasks()



    def _bind_keys(self):
        self.main_frm.bind("<Right>", lambda x: self.load_new_frame(frm_cnt=self.current_frm_number + 1))
        self.main_frm.bind("<Left>", lambda x: self.load_new_frame(frm_cnt=self.current_frm_number - 1))
        self.main_frm.bind("<Control-l>", lambda x: self.load_new_frame(frm_cnt=self.video_meta_data["frame_count"] - 1))
        self.main_frm.bind("<Control-o>", lambda x: self.load_new_frame(frm_cnt=0))

    def load_new_frame(self, frm_cnt: int):
        if (frm_cnt > self.video_meta_data["frame_count"] - 1) or (frm_cnt < 0):
            FrameRangeWarning(msg=f'Frame {str(frm_cnt)} is outside of the video frame range: (0-{self.video_meta_data["frame_count"]-1}).')
        else:
            self.new_frm = read_frm_of_video(video_path=self.cap, frame_index=frm_cnt)
            if self.show_pose or self.show_names:
                if frm_cnt in self.data_df.index:
                    if self.show_pose:
                        self.new_frm = self._show_pose(img=self.new_frm, frm_cnt=frm_cnt)
                    if self.show_names:
                        self.new_frm = self._show_names(img=self.new_frm, frm_cnt=frm_cnt)
                else:
                    FrameRangeWarning(msg=f'Frame {str(frm_cnt)} is outside of the data frame range: (0-{len(self.data_df)-1}). Pose and animal names cannot be displayed.', source=self.__class__.__name__)
            self.new_frm = cv2.cvtColor(self.new_frm, cv2.COLOR_RGB2BGR)
            self.new_frm = Img.fromarray(self.new_frm)
            self.new_frm.thumbnail(self.max_img_size, Img.LANCZOS)
            self.new_frm = ImageTk.PhotoImage(master=self.main_frm, image=self.new_frm)
            self.img_frm = Label(self.main_frm, image=self.new_frm)
            self.img_frm.image = self.new_frm
            self.img_frm.grid(row=0, column=0)
            self.current_frm_number = frm_cnt
            self.frame_entry_box.entry_set(val=self.current_frm_number)

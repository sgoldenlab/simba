__author__ = "Simon Nilsson"

import math
import os
from copy import deepcopy
from subprocess import PIPE, Popen
from tkinter import *
from tkinter import filedialog
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib

matplotlib.use('agg')
import cv2
import pandas as pd
from PIL import Image, ImageTk
from tabulate import tabulate

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import simba
from simba.mixins.config_reader import ConfigReader
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, Entry_Box,
                                        SimbaButton, SimbaCheckbox, SimBALabel)
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_int, check_str, check_that_column_exist,
                                check_valid_boolean, check_valid_dataframe,
                                check_valid_dict)
from simba.utils.enums import ConfigKey, Dtypes, Formats, Options, TagNames
from simba.utils.errors import FrameRangeError, NoDataError, NoFilesFoundError
from simba.utils.lookups import (get_current_time, get_display_resolution,
                                 get_labelling_img_kbd_bindings,
                                 get_labelling_video_kbd_bindings)
from simba.utils.printing import log_event, stdout_success
from simba.utils.read_write import (get_fn_ext, get_video_meta_data,
                                    read_config_entry, read_config_file,
                                    read_df, read_frm_of_video, write_df)

PLAY_VIDEO_SCRIPT_PATH = os.path.join(os.path.dirname(simba.__file__), "labelling/play_annotation_video.py")
PADDING = 5

DISPLAY_RESOLUTION = get_display_resolution()
MAX_FRAME_SIZE = (int(DISPLAY_RESOLUTION[0] * 0.5), int(DISPLAY_RESOLUTION[1] * 0.05))


class LabellingInterface(ConfigReader):
    """
    Launch ``standard`` or ``pseudo``-labelling (annotation) GUI interface in SimBA.

    .. note::
       Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/label_behavior.md>`__.

    .. image:: _static/img/annotator.png
       :width: 500
       :align: center


    :param Union[str, os.PathLike] config_path: path to SimBA project config file in Configparser format
    :param Union[str, os.PathLike] file_path: Path to video that is to be annotated.
    :param Literal["from_scratch", "pseudo"] setting: String representing annotation method. OPTIONS: ``from_scratch`` or ``pseudo``
    :param Optional[Dict[str, float]] threshold_dict: If setting ``pseudo``, threshold_dict dict contains the machine probability thresholds, with the classifier names as keys and the classification probabilities as values, e.g. {'Attack': 0.40, 'Sniffing': 0.7).
    :param bool continuing: If True, continouing previously started annotation session.

    Examples
    ----------
    >>> select_labelling_video(config_path='MyConfigPath', threshold_dict={'Attack': 0.4}, file_path='MyVideoFilePath', setting='pseudo', continuing=False)
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 file_path: Union[str, os.PathLike],
                 threshold_dict: Optional[Dict[str, float]] = None,
                 setting: Literal["from_scratch", "pseudo"] = "from_scratch",
                 continuing: bool = False):

        ConfigReader.__init__(self, config_path=config_path)
        log_event(logger_name=str(self.__class__.__name__), log_type=TagNames.CLASS_INIT.value, msg=self.create_log_msg_from_init_args(locals=locals()))
        check_file_exist_and_readable(file_path=file_path)
        self.video_meta_data = get_video_meta_data(video_path=file_path)
        check_str(name='setting', value=setting, options=('pseudo', "from_scratch",))
        check_valid_boolean(value=continuing, source=f'{self.__class__.__name__} continuing')
        if threshold_dict is not None:
            check_valid_dict(x=threshold_dict, valid_key_dtypes=(str,), valid_values_dtypes=(float,), min_value=0.0, max_value=1.0)
        self.img_idx, self.threshold_dict = 0, threshold_dict
        self.setting, self.video_path = setting, file_path
        self.video_name = self.video_meta_data['video_name']
        self.features_extracted_file_path = os.path.join(self.features_dir, f"{self.video_name}.{self.file_type}")
        self.targets_inserted_file_path = os.path.join(self.targets_folder, f"{self.video_name}.{self.file_type}")
        self.machine_results_file_path = os.path.join(self.machine_results_dir, f"{self.video_name}.{self.file_type}")
        self.cap = cv2.VideoCapture(self.video_path)
        self.img_kbd_bindings = get_labelling_img_kbd_bindings()
        self.video_kbd_bindings = get_labelling_video_kbd_bindings()
        self.key_presses_lbl = self.get_keyboard_shortcuts_lbl(self.img_kbd_bindings, header="FRAME NAVIGATION SHORTCUTS: ")
        self.video_presses_lbl = self.get_keyboard_shortcuts_lbl(self.video_kbd_bindings, header="VIDEO NAVIGATION SHORTCUTS: ")
        self.max_frm_id = self.video_meta_data["frame_count"] - 1
        self.temp_file_path = os.path.join(os.path.dirname(self.config_path), "subprocess.txt")
        if len(self.clf_names) == 0:
            raise NoDataError(msg='To annotate behaviors, your SimBA project needs at least one defined classifier. Found 0 classifiers defined in SimBA project', source=self.__class__.__name__)
        self.main_window = Toplevel()

        if continuing:
            if not os.path.isfile(self.targets_inserted_file_path):
                raise NoFilesFoundError(msg=f'When continuing annotations, SimBA expects a file at {self.targets_inserted_file_path}. SimBA could not find this file.', source=self.__class__.__name__)
            self.data_df = read_df(self.targets_inserted_file_path, self.file_type)
            self.data_df_features = read_df(self.features_extracted_file_path, self.file_type)
            missing_idx = self.data_df_features.index.difference(self.data_df.index)
            if len(missing_idx) > 0:
                self.data_df_features = self.data_df_features.iloc[self.data_df_features.index.difference(self.data_df.index)]
                self.data_df = pd.concat([self.data_df.astype(int), self.data_df_features], axis=0).sort_index()
            self.main_window.title("SIMBA ANNOTATION INTERFACE (CONTINUING ANNOTATIONS) - {}".format( self.video_name))
            self.img_idx = read_config_entry(self.config, "Last saved frames", self.video_name, data_type="int", default_value=0)
        else:
            if setting == "from_scratch":
                if not os.path.isfile(self.features_extracted_file_path):
                    raise NoFilesFoundError(msg=f'When annotating data, SimBA expects a file at {self.features_extracted_file_path}. SimBA could not find this file. Extract features for video {self.video_name} before annotating data.', source=self.__class__.__name__)
                self.data_df = read_df(self.features_extracted_file_path, self.file_type)
                self.main_window.title("SIMBA ANNOTATION INTERFACE (ANNOTATING FROM SCRATCH) - {}".format(self.video_name))
                for target in self.clf_names:
                    self.data_df[target] = 0
            elif setting == "pseudo":
                if not os.path.isfile(self.machine_results_file_path):
                    raise NoFilesFoundError(msg=f'When doing pseudo-annotations, SimBA expects a file at {self.machine_results_file_path}. SimBA could not find this file.', source=self.__class__.__name__)
                self.data_df = read_df(self.machine_results_file_path, self.file_type)
                check_valid_dataframe(df=self.data_df, source=self.__class__.__name__, required_fields=self.clf_names)
                for target in self.clf_names:
                    self.data_df.loc[self.data_df[f"Probability_{target}"] > self.threshold_dict[target], target] = 1
                    self.data_df.loc[self.data_df[f"Probability_{target}"] <= self.threshold_dict[target], target] = 0
                self.main_window.title("SIMBA ANNOTATION INTERFACE (PSEUDO-LABELLING) - {}".format(self.video_name))

        self.data_df_targets = self.data_df.copy()
        check_valid_dataframe(df=self.data_df_targets, source=f'{self.__class__.__name__} file_path', valid_dtypes=Formats.NUMERIC_DTYPES.value, required_fields=self.clf_names)
        self.data_df_targets = self.data_df_targets[self.clf_names]

        self.img_frm = CreateLabelFrameWithIcon(parent=self.main_window, header='VIDEO FRAME', icon_name='video', relief='solid')
        self.img_lbl = Label(self.img_frm, name='img_lbl')
        self.img_frm.grid(row=0, column=0, sticky=NW, padx=10, pady=10, columnspan=3, rowspan=2)
        self.img_lbl.grid(row=0, column=0, sticky=NW)
        self.set_img(img_lbl=self.img_lbl, video_path=self.video_path, img_idx=self.img_idx)


        self.navigation_frm = CreateLabelFrameWithIcon(parent=self.main_window, header='FRAME NAVIGATION',  icon_name='navigation', relief='solid')
        self.current_frm_eb = Entry_Box(parent=self.navigation_frm, fileDescription='CURRENT FRAME: ', labelwidth=30, validation='numeric', entry_box_width=20, value=self.img_idx, justify='center', label_font=Formats.FONT_HEADER.value, entry_font=Formats.FONT_HEADER.value)
        self.got_to_frm_btn = SimbaButton(parent=self.navigation_frm, txt="GO TO FRAME", img='walk', font=Formats.FONT_HEADER.value, cmd=self.__advance_frame, cmd_kwargs={'frm_id': lambda: int(self.current_frm_eb.entry_get)}, hover_font=Formats.FONT_HEADER.value)
        self.first_frm_btn = SimbaButton(parent=self.navigation_frm, txt="", tooltip_txt='GO TO FIRST FRAME', img='rewind_large', cmd=self.__advance_frame, cmd_kwargs={'frm_id': 0})
        self.reverse_btn = SimbaButton(parent=self.navigation_frm, txt="", img='reverse_large', tooltip_txt="-1 FRAME", cmd=self.__advance_frame, cmd_kwargs={'frm_id': lambda: self.img_idx - 1})
        self.forward_btn = SimbaButton(parent=self.navigation_frm, txt="", img='forward_large', tooltip_txt="+1 FRAME", cmd=self.__advance_frame, cmd_kwargs={'frm_id': lambda: self.img_idx + 1})
        self.last_frm_btn = SimbaButton(parent=self.navigation_frm, txt="", img='fast_forward_large_2', tooltip_txt="GO TO LAST FRAME", cmd=self.__advance_frame, cmd_kwargs={'frm_id': self.max_frm_id})

        self.navigation_frm.grid(row=2, column=0, sticky=NSEW, pady=10, padx=10, columnspan=1)
        self.current_frm_eb.grid(row=0, column=0, sticky=EW, columnspan=4, padx=5, pady=32)
        self.got_to_frm_btn.grid(row=0, column=5, sticky=EW, columnspan=2, padx=5, pady=32)
        self.first_frm_btn.grid(row=1, column=0, sticky=E, padx=5, pady=5)
        self.reverse_btn.grid(row=1, column=1, sticky=E, padx=5, pady=5)
        self.forward_btn.grid(row=1, column=2, sticky=E, padx=5, pady=5)
        self.last_frm_btn.grid(row=1, column=3, sticky=E, padx=5, pady=5)


        self.jump_frm = CreateLabelFrameWithIcon(parent=self.main_window, header='JUMP TO FRAME',  icon_name='jump', relief='solid')
        self.jump_scale = Scale(self.jump_frm, from_=0, to=100, orient=HORIZONTAL, length=200, label="FRAME JUMP SIZE", tickinterval=25)
        self.jump_scale.set(0)
        self.jump_back = SimbaButton(parent=self.jump_frm, txt="", tooltip_txt="JUMP BACKWARDS", img='pointing_left_large', cmd=self.__advance_frame, cmd_kwargs={'frm_id': lambda: self.img_idx - int(self.jump_scale.get())})
        self.jump_forwards = SimbaButton(parent=self.jump_frm, txt="", img='point_right_large', tooltip_txt='JUMP FORWARDS', cmd=self.__advance_frame, cmd_kwargs={'frm_id': lambda: self.img_idx + int(self.jump_scale.get())})

        self.jump_frm.grid(row=2, column=1, sticky=NSEW, pady=10, padx=10, columnspan=2)
        self.jump_scale.grid(row=0, column=0, sticky=W, padx=5, pady=5, columnspan=2)
        self.jump_back.grid(row=1, column=0, sticky=E, padx=5, pady=5, columnspan=1)
        self.jump_forwards.grid(row=1, column=1, sticky=W, padx=5, pady=5, columnspan=1)

        self.range_frm = CreateLabelFrameWithIcon(parent=self.main_window, header='ANNOTATE FRAME RANGE', icon_name='range_large', relief='solid')
        self.range_start_eb = Entry_Box(parent=self.range_frm, fileDescription='START FRAME: ', labelwidth=30, validation='numeric', entry_box_width=20, justify='center', label_font=Formats.FONT_HEADER.value, entry_font=Formats.FONT_HEADER.value)
        self.range_end_eb = Entry_Box(parent=self.range_frm, fileDescription='RANGE END FRAME: ', labelwidth=30, validation='numeric', entry_box_width=20, justify='center', label_font=Formats.FONT_HEADER.value, entry_font=Formats.FONT_HEADER.value)
        self.save_range_btn = SimbaButton(parent=self.range_frm, txt="SAVE RANGE", img='save_small', cmd=self.__save_behavior_in_range, cmd_kwargs={'start_frm': lambda: self.range_start_eb.entry_get, 'end_frm': lambda: self.range_end_eb.entry_get})
        self.range_frm.grid(row=3, column=0, sticky=NSEW, padx=10, pady=10, columnspan=1)
        self.range_start_eb.grid(row=0, column=0, sticky=NW)
        self.range_end_eb.grid(row=1, column=0, sticky=NW)
        self.save_range_btn.grid(row=2, column=0, sticky=NW)

        self.save_frm = CreateLabelFrameWithIcon(parent=self.main_window, header='SAVE ANNOTATIONS', icon_name='save_large', relief='solid')
        self.save_btn = SimbaButton(parent=self.save_frm, txt="ANNOTATIONS SAVE", tooltip_txt="SAVE ANNOTATIONS TO DISK", img='rocket', cmd=self.__advance_frame, cmd_kwargs={'frm_id': self.max_frm_id}, font=Formats.FONT_HEADER.value)
        self.save_frm.grid(row=3, column=1, sticky=NSEW, padx=10, pady=10, columnspan=2)
        self.save_btn.grid(row=0, column=0, sticky=NW, padx=5,  pady=20)


        self.clf_cb_frm = CreateLabelFrameWithIcon(parent=self.main_window, header='CHECK BEHAVIOR', icon_name='forest', relief='solid')
        self.clf_cbs = self.get_clf_cbs(parent=self.clf_cb_frm, clfs=self.clf_names)
        self.clf_cb_frm.grid(row=0, column=4, sticky=NSEW, padx=10, pady=10)

        self.video_player_frm = CreateLabelFrameWithIcon(parent=self.main_window, header='PLAY VIDEO', icon_name='tv_large', relief='solid')
        self.play_video_btn = SimbaButton(parent=self.video_player_frm, txt="OPEN VIDEO", img='video_file_large', cmd=self.__play_video)
        self.update_img_from_video = SimbaButton(parent=self.video_player_frm, txt="SHOW CURRENT VIDEO FRAME", img='display_large', cmd=self.__update_frame_from_video)
        self.video_player_frm.grid(row=1, column=4, sticky=NSEW, padx=10, pady=10)
        self.play_video_btn.grid(row=0, column=0, sticky=NSEW, padx=3, pady=3)
        self.update_img_from_video.grid(row=0, column=1, sticky=NSEW,  padx=3, pady=3)

        SimBALabel(parent=self.video_player_frm, txt=self.video_presses_lbl, font=Formats.FONT_HEADER.value).grid(row=2, column=0, sticky=NSEW, padx=5, columnspan=2)
        SimBALabel(parent=self.video_player_frm, txt=self.key_presses_lbl, font=Formats.FONT_HEADER.value, justify='center').grid(row=3, column=0, sticky=NSEW, padx=5, columnspan=2)
        self.__bind_frm_shortcut_keys()

        self.main_window.mainloop()

    def __bind_frm_shortcut_keys(self):
        self.main_window.bind(self.img_kbd_bindings['save']['kbd'], lambda x: self.__save_results())
        self.main_window.bind(self.img_kbd_bindings['frame+1_keep_choices']['kbd'], lambda x: self.__advance_frame(frm_id=self.img_idx+1, save_and_keep_checks=True))
        self.main_window.bind(self.img_kbd_bindings['print_annotation_statistic']['kbd'], lambda x: self.print_annotation_statistics(video_meta_data=self.video_meta_data, clf_names=self.clf_names, data_df=self.data_df_targets))
        self.main_window.bind(self.img_kbd_bindings['frame+1']['kbd'], lambda x: self.__advance_frame(frm_id=self.img_idx+1))
        self.main_window.bind(self.img_kbd_bindings['frame-1']['kbd'], lambda x: self.__advance_frame(frm_id=self.img_idx-1))
        self.main_window.bind(self.img_kbd_bindings['last_frame']['kbd'], lambda x: self.__advance_frame(frm_id=self.max_frm_id))
        self.main_window.bind(self.img_kbd_bindings['first_frame']['kbd'], lambda x: self.__advance_frame(0))


    def __play_video(self):
        p = Popen(f"python {PLAY_VIDEO_SCRIPT_PATH}",stdin=PIPE, stdout=PIPE, shell=True)
        p.stdin.write(bytes(self.video_path, "utf-8")); p.stdin.close()
        with open(self.temp_file_path, "w") as text_file: text_file.write(str(p.pid))

    def __update_frame_from_video(self):
        f = open(os.path.join(os.path.dirname(self.config_path), "labelling_info.txt"), "r+")
        os.fsync(f.fileno())
        vid_frame_no = int(f.readline())
        self.__advance_frame(frm_id=vid_frame_no)
        f.close()


    def get_keyboard_shortcuts_lbl(self,
                                   kbd_bindings: Dict[str, Dict[str, str]],
                                   header: str = "FRAME NAVIGATION SHORTCUTS: "):

        labels = [v['label'] for v in kbd_bindings.values()]
        num_items = len(labels)
        mid = math.ceil(num_items / 2)
        col1 = labels[:mid]
        col2 = labels[mid:]
        if len(col2) < len(col1):
            col2 += [''] * (len(col1) - len(col2))
        key_presses_lbl = f"\n\n {header}\n"
        for left, right in zip(col1, col2):
            key_presses_lbl += f"{left:<40} {right}\n"

        return key_presses_lbl



    def set_img(self,
                img_lbl: Label,
                video_path: Union[str, os.PathLike],
                img_idx: int,
                display_size: Optional[Tuple[int, int]] = None):

        self.img = read_frm_of_video(video_path=video_path, frame_index=img_idx, size=display_size)
        img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.pil_image = Image.fromarray(img_rgb)
        self.tk_image = ImageTk.PhotoImage(self.pil_image)
        img_lbl.configure(image=self.tk_image)
        img_lbl.image = self.tk_image

    def get_clf_cbs(self,
                    parent: Toplevel,
                    clfs: List[str]) -> Dict[str, BooleanVar]:

        clf_cb_vars = {}
        for clf_cnt, clf_name in enumerate(clfs):
            cb, clf_cb_vars[clf_name] = SimbaCheckbox(parent=parent, txt=clf_name, font=Formats.FONT_HEADER.value)
            cb.grid(row=clf_cnt, column=0, sticky=NW, padx=5, pady=5)
        return clf_cb_vars


    def print_annotation_statistics(self,
                                    video_meta_data: Dict[str, Any],
                                    clf_names: List[str],
                                    data_df: pd.DataFrame):

        table_view = [["Video name", video_meta_data['video_name']],
                      ["Video frames", video_meta_data["frame_count"]],
                      ['Current time:', get_current_time()]]
        for clf in clf_names:
            present = len(data_df[data_df[clf] == 1])
            absent = len(data_df[data_df[clf] == 0])
            table_view.append([clf + " present labels", present])
            table_view.append([clf + " absent labels", absent])
            table_view.append([clf + " % present", present / video_meta_data["frame_count"]])
            table_view.append([clf + " % absent", absent / video_meta_data["frame_count"]])
        headers = ["VARIABLE", "VALUE"]
        print(tabulate(table_view, headers, tablefmt="github"))

    def __advance_frame(self,
                        frm_id: int,
                        save_and_keep_checks: bool = False):

        if frm_id > self.max_frm_id:
            raise FrameRangeError(msg=f"TRYING TO SHOW FRAME {frm_id} WHICH CANNOT BE SHOWN - THE VIDEO {self.video_name} HAS {self.video_meta_data['frame_count']} FRAMES")
        elif frm_id < 0:
            raise FrameRangeError(msg=f"TRYING TO SHOW FRAME {frm_id} WHICH CANNOT BE SHOWN. FRAME INDEX HAS TO BE 0 OR ABOVE")
        else:
            prior_frm_idx = int(self.current_frm_eb.entry_get)
            self.__create_print_statements(frame=prior_frm_idx)
            self.current_frm_eb.entry_set(val=frm_id)
            if not save_and_keep_checks:
                for target in self.clf_names:
                    self.clf_cbs[target].set(bool(self.data_df_targets[target].loc[frm_id]))
            else:
                for target in self.clf_names:
                    self.save_behavior_in_frm(clf_name=target, frame_idx=prior_frm_idx)
                    self.clf_cbs[target].set(bool(self.data_df_targets[target].loc[prior_frm_idx]))

            self.img_idx = deepcopy(frm_id)
            self.set_img(img_lbl=self.img_lbl, video_path=self.video_path, img_idx=frm_id)

    def __save_behavior_in_range(self,
                                 start_frm: int,
                                 end_frm: int):

        check_int(name="RANGE START FRAME", value=start_frm, min_value=0, max_value=self.max_frm_id)
        check_int(name="RANGE END FRAME", value=end_frm, min_value=0, max_value=self.max_frm_id)
        start_frm, end_frm = int(start_frm), int(end_frm)
        if start_frm >= end_frm:
            raise FrameRangeError(msg=f'The RANGE START FRAME ({start_frm}) has to be smaller than the RANGE END FRAME ({end_frm})', source=self.__class__.__name__)
        for frm_no in range(int(start_frm), int(end_frm) + 1):
            for target in self.clf_names:
                self.data_df_targets[target].loc[frm_no] = self.clf_cbs[target].get()
        self.set_img(img_lbl=self.img_lbl, video_path=self.video_path, img_idx=end_frm)
        self.current_frm_eb.entry_set(end_frm)
        self.img_idx = deepcopy(end_frm)
        self.__create_print_statements(frame=(start_frm, end_frm))

    def save_behavior_in_frm(self,
                             frame_idx: int,
                             clf_name: str):

        self.data_df_targets[clf_name].loc[frame_idx] = int(self.clf_cbs[clf_name].get())

    def __save_results(self):
        self.features_df = read_df(self.features_extracted_file_path, self.file_type)
        self.save_df = pd.concat([self.features_df, self.data_df_targets], axis=1)
        try:
            write_df(self.save_df, self.file_type, self.targets_inserted_file_path)
        except Exception as e:
            print(e, f"SIMBA ERROR: File for video {get_fn_ext(self.features_extracted_file_path)[1]} could not be saved.")
            raise FileExistsError
        stdout_success(msg=f"SAVED: Annotation file for video {self.video_name} saved within the {self.targets_folder} directory.", source=self.__class__.__name__)
        if not self.config.has_section("Last saved frames"):
            self.config.add_section("Last saved frames")
        self.config.set("Last saved frames", str(self.video_name), str(self.img_idx))
        with open(self.config_path, "w") as configfile:
            self.config.write(configfile)

    def __create_print_statements(self,
                                  frame: Union[int, Tuple[int, int]]):

        for target in self.clf_names:
            target_present_choice = self.clf_cbs[target].get()
            if target_present_choice:
                if isinstance(frame, int):
                    print(f"{target} PRESENT in frame {frame}")
                else:
                    print(f"{target} PRESENT in frames {frame[0]} to {frame[1]}")
            else:
                if isinstance(frame, int):
                    print(f"{target} ABSENT in frame {frame}")
                else:
                    print(f"{target} ABSENT in frames {frame[0]} to {frame[1]}")

def select_labelling_video(config_path: Union[str, os.PathLike],
                           threshold_dict: Optional[Dict[str, Entry_Box]] = None,
                           setting: str = None,
                           continuing: bool = None,
                           video_file_path: Union[str, os.PathLike] = None):

    check_file_exist_and_readable(file_path=config_path)
    if threshold_dict is not None:
        check_valid_dict(x=threshold_dict, valid_key_dtypes=(str,), valid_values_dtypes=(float,))
    if setting is not None:
        check_str(name='setting', value=setting, options=('pseudo', "from_scratch",))
    check_valid_boolean(value=[continuing], source=select_labelling_video.__name__)

    config = read_config_file(config_path=config_path)
    project_path = read_config_entry(config , ConfigKey.GENERAL_SETTINGS.value, ConfigKey.PROJECT_PATH.value,data_type=Dtypes.STR.value)
    video_dir = os.path.join(project_path, 'videos')
    video_dir = None if not os.path.isdir(video_dir) else video_dir

    if setting is not "pseudo":
        video_file_path = filedialog.askopenfilename(filetypes=[("Video files", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)], initialdir=video_dir)
    else:
        threshold_dict_values = {}
        for k, v in threshold_dict.items():
            check_float(name=k, value=float(v.entry_get), min_value=0.0, max_value=1.0)
            threshold_dict_values[k] = float(v.entry_get)
        threshold_dict = threshold_dict_values

    check_file_exist_and_readable(file_path=video_file_path)
    video_meta = get_video_meta_data(video_file_path)
    _, video_name, _ = get_fn_ext(video_file_path)



    print(f"ANNOTATING VIDEO {video_name} \n VIDEO INFO: {video_meta}")
    _ = LabellingInterface(config_path=config_path,
                           file_path=video_file_path,
                           threshold_dict=threshold_dict,
                           setting=setting,
                           continuing=continuing)


# test = select_labelling_video(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini",
#                               threshold_dict=None, #threshold_dict={'Attack': 0.4}
#                               setting='from_scratch',
#                               continuing=False)

# _ = LabellingInterface(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini",
#                        file_path=r"C:\troubleshooting\mitra\project_folder\videos\501_MA142_Gi_CNO_0521.mp4",
#                        threshold_dict=None,
#                        setting='from_scratch',
#                        continuing=False)





# #
# test = select_labelling_video(config_path=r"/Users/simon/Desktop/envs/simba/troubleshooting/mitra/project_folder/project_config.ini",
#                               threshold_dict={'Attack': 0.4},
#                               setting='from_scratch',
#                               continuing=False)

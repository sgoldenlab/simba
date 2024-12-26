__author__ = "Simon Nilsson"

import os
from subprocess import PIPE, Popen
from tkinter import *
from tkinter import filedialog
from typing import Dict, Optional, Union

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
from simba.ui.tkinter_functions import Entry_Box
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_int, check_str, check_that_column_exist,
                                check_valid_boolean, check_valid_dataframe,
                                check_valid_dict)
from simba.utils.enums import Options, TagNames
from simba.utils.errors import FrameRangeError, NoDataError, NoFilesFoundError
from simba.utils.lookups import (get_labelling_img_kbd_bindings,
                                 get_labelling_video_kbd_bindings)
from simba.utils.printing import log_event, stdout_success
from simba.utils.read_write import (get_all_clf_names, get_fn_ext,
                                    get_video_meta_data, read_config_entry,
                                    read_df, read_frm_of_video, write_df)
from simba.utils.warnings import FrameRangeWarning

PLAY_VIDEO_SCRIPT_PATH = os.path.join(os.path.dirname(simba.__file__), "labelling/play_annotation_video.py")
PADDING = 5

class LabellingInterface(ConfigReader):
    """
    Launch ``standard`` or ``pseudo``-labelling (annotation) GUI interface in SimBA.

    .. note::
       Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/label_behavior.md>`__.

    .. image:: _static/img/annotator.png
       :width: 500
       :align: center


    :param Union[str, os.PathLike] config_path: path to SimBA project config file in Configparser format
    :param Union[str, os.PathLike] video_path: Path to video that is to be annotated.
    :param Literal["from_scratch", "pseudo"] setting: String representing annotation method. OPTIONS: ``from_scratch`` or ``pseudo``
    :param Optional[Dict[str, float]] thresholds: If setting ``pseudo``, threshold_dict dict contains the machine probability thresholds, with the classifier names as keys and the classification probabilities as values, e.g. {'Attack': 0.40, 'Sniffing': 0.7).
    :param bool continuing: If True, continouing previously started annotation session.

    :example:
    >>> select_labelling_video(config_path='MyConfigPath', threshold_dict={'Attack': 0.4}, file_path='MyVideoFilePath', setting='pseudo', continuing=False)
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 video_path: Union[str, os.PathLike],
                 thresholds: Optional[Dict[str, float]] = None,
                 setting: Literal["from_scratch", "pseudo"] = "pseudo",
                 continuing: Optional[bool] = False):

        ConfigReader.__init__(self, config_path=config_path)
        if len(self.clf_names) == 0:
            raise NoDataError(msg='To annotate behaviors, your SimBA project needs at least one defined classifier. Found 0 classifiers defined in SimBA project', source=self.__class__.__name__)
        log_event(logger_name=str(self.__class__.__name__), log_type=TagNames.CLASS_INIT.value, msg=self.create_log_msg_from_init_args(locals=locals()))
        self.video_meta_data = get_video_meta_data(video_path=video_path)
        check_str(name='setting', value=setting, options=('pseudo', "from_scratch"))
        check_valid_boolean(value=[continuing], source=select_labelling_video.__name__)
        if thresholds is not None:
            check_valid_dict(x=thresholds, valid_key_dtypes=(str,), valid_values_dtypes=(float,), min_value=0, max_value=1.0)
        self.frm_no, self.thresholds, self.file_path, self.setting, self.video_path = 0, thresholds, video_path, setting, video_path
        _, self.video_name, _ = get_fn_ext(filepath=video_path)
        self.features_extracted_file_path = os.path.join(self.features_dir, f"{self.video_name}.{self.file_type}")
        self.targets_inserted_file_path = os.path.join(self.targets_folder, f"{self.video_name}.{self.file_type}")
        self.machine_results_file_path = os.path.join(self.machine_results_dir, f"{self.video_name}.{self.file_type}")
        self.cap = cv2.VideoCapture(self.video_path)
        self.img_kbd_bindings = get_labelling_img_kbd_bindings()
        self.video_kbd_bindings = get_labelling_video_kbd_bindings()
        self.__create_frm_key_presses_lbl()
        self.__create_video_key_presses_lbl()
        self.frame_lst = list(range(0, self.video_meta_data["frame_count"]))
        self.max_frm_no = max(self.frame_lst)
        self.max_frm_size = 1080, 650
        self.main_window = Toplevel()
        if continuing:
            if not os.path.isfile(self.targets_inserted_file_path):
                raise NoFilesFoundError(msg=f'When continuing annotations, SimBA expects a file at {self.targets_inserted_file_path}. SimBA could not find this file.', source=self.__class__.__name__)
            if not os.path.isfile(self.features_extracted_file_path):
                raise NoFilesFoundError(msg=f'When continuing annotations, SimBA expects a file at {self.features_extracted_file_path}. SimBA could not find this file.', source=self.__class__.__name__)
            self.data_df = read_df(self.targets_inserted_file_path, self.file_type)
            self.data_df_features = read_df(self.features_extracted_file_path, self.file_type)
            missing_idx = self.data_df_features.index.difference(self.data_df.index)
            if len(missing_idx) > 0:
                self.data_df_features = self.data_df_features.iloc[self.data_df_features.index.difference(self.data_df.index)]
                self.data_df = pd.concat([self.data_df.astype(int), self.data_df_features], axis=0).sort_index()
            self.main_window.title("SIMBA ANNOTATION INTERFACE (CONTINUING ANNOTATIONS) - {}".format( self.video_name))
            self.frm_no = read_config_entry(self.config, "Last saved frames", self.video_name, data_type="int", default_value=0)
            if self.frm_no not in self.frame_lst:
                FrameRangeWarning(msg=f'SimBA attempted to open the last saved frame of video {self.video_name} as denoted in the section [Last saved frames] in the project_config.ini. However, this frame does not exist in the video. The video {self.video_name} has {self.max_frm_no} frames. SimBA will begin with the first frame instead.')
                self.frm_no = 0

        else:
            if setting == "from_scratch":
                check_file_exist_and_readable(file_path=self.features_extracted_file_path)
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
                    self.data_df.loc[self.data_df[f"Probability_{target}"] > self.thresholds[target], target] = 1
                    self.data_df.loc[self.data_df[f"Probability_{target}"] <= self.thresholds[target], target] = 0
                self.main_window.title("SIMBA ANNOTATION INTERFACE (PSEUDO-LABELLING) - {}".format(self.video_name))

        self.data_df_targets = self.data_df
        for target in self.clf_names:
            check_that_column_exist(df=self.data_df_targets, column_name=target, file_name=video_path)
        self.data_df_targets = self.data_df_targets[self.clf_names]
        self.folder = Frame(self.main_window)
        self.buttons_frm = Frame(self.main_window, bd=2, width=700, height=300)
        self.current_frm_n = IntVar(self.main_window, value=self.frm_no)
        self.change_frm_box = Entry(self.buttons_frm, width=7, textvariable=self.current_frm_n)
        self.frame_number_lbl = Label(self.buttons_frm, text="Frame number")
        self.forward_btn = Button(self.buttons_frm, text=">", command=lambda: self.__advance_frame(new_frm_number=int(self.current_frm_n.get() + 1)))
        self.backward_btn = Button(self.buttons_frm, text="<", command=lambda: self.__advance_frame(new_frm_number=int(self.current_frm_n.get() - 1)))
        self.forward_max_btn = Button(self.buttons_frm, text=">>", command=lambda: self.__advance_frame(len(self.frame_lst) - 1))
        self.backward_max_btn = Button(self.buttons_frm, text="<<", command=lambda: self.__advance_frame(0))
        self.select_frm_btn = Button(self.buttons_frm, text="Jump to selected frame", command=lambda: self.__advance_frame(new_frm_number=int(self.change_frm_box.get())))
        self.jump_frame = Frame(self.main_window)
        self.jump = Label(self.jump_frame, text="Jump Size:")
        self.jump_size = Scale(self.jump_frame, from_=0, to=100, orient=HORIZONTAL, length=200)
        self.jump_size.set(0)
        self.jump_back = Button(self.jump_frame, text="<<", command=lambda: self.__advance_frame(int(self.change_frm_box.get()) - self.jump_size.get()))
        self.jump_forward = Button(self.jump_frame, text=">>", command=lambda: self.__advance_frame(int(self.change_frm_box.get()) + self.jump_size.get()))

        self.folder.grid(row=0, column=1, sticky=N)
        self.buttons_frm.grid(row=1, column=0)
        self.change_frm_box.grid(row=0, column=1)
        self.forward_btn.grid(row=1, column=3, sticky=E, padx=PADDING)
        self.backward_btn.grid(row=1, column=1, sticky=W, padx=PADDING)
        self.change_frm_box.grid(row=1, column=1)
        self.forward_max_btn.grid(row=1, column=4, sticky=W, padx=PADDING)
        self.backward_max_btn.grid(row=1, column=0, sticky=W, padx=PADDING)
        self.select_frm_btn.grid(row=2, column=1, sticky=N)
        self.jump_frame.grid(row=2, column=0)
        self.jump.grid(row=0, column=0, sticky=W)
        self.jump_size.grid(row=0, column=1, sticky=W)
        self.jump_back.grid(row=0, column=2, sticky=E)
        self.jump_forward.grid(row=0, column=3, sticky=W)

        self.check_frame = Frame(self.main_window, bd=2, width=300, height=500)
        self.check_frame.grid(row=0, column=1)
        self.check_behavior_lbl = Label(self.check_frame, text="Check Behavior:")
        self.check_behavior_lbl.config(font=("Calibri", 16))
        self.check_behavior_lbl.grid(sticky=N)

        self.checkboxes = {}
        for target_cnt, target in enumerate(self.clf_names):
            self.checkboxes[target] = {}
            self.checkboxes[target]["name"] = target
            if self.current_frm_n.get() not in list(self.data_df_targets.index):
                raise FrameRangeError(msg=f'Frame pose-estimation data for frame {self.current_frm_n.get()} could not be found for video {self.video_name}. This suggests that the video ({self.video_path}) has a different frame number count than the rows in the data files (e.g., {self.targets_inserted_file_path}, {self.features_extracted_file_path}, {self.machine_results_file_path}, where they exist).  Alternatively, modify the [Last saved frames] section in the {self.config_path} if needed.', source=self.__class__.__name__)
            self.checkboxes[target]["var"] = IntVar(value=self.data_df_targets[target].iloc[self.current_frm_n.get()])
            self.checkboxes[target]["cb"] = Checkbutton(self.check_frame,
                text=target,
                variable=self.checkboxes[target]["var"],
                command=lambda k=self.checkboxes[target][
                    "name"
                ]: self.save_behavior_in_frm(
                    frame_number=int(self.current_frm_n.get()), target=k
                ),
            )
            self.checkboxes[target]["cb"].grid(row=target_cnt + 1, sticky=W)

        self.range_on = IntVar(value=0)
        self.range_frames = Frame(self.main_window)
        self.range_frames.grid(row=1, column=1, sticky=S)
        self.select_range = Checkbutton(self.range_frames, text="Frame range", variable=self.range_on)
        self.select_range.grid(row=0, column=0, sticky=W)
        self.first_frame = Entry(self.range_frames, width=7)
        self.first_frame.grid(row=0, column=1, sticky=E)
        self.to_label = Label(self.range_frames, text=" to ")
        self.to_label.grid(row=0, column=2, sticky=E)
        self.last_frame = Entry(self.range_frames, width=7)
        self.last_frame.grid(row=0, column=3, sticky=E)

        save = Button(self.main_window, text="Save Range", command=lambda: self.__save_behavior_in_range(self.first_frame.get(), self.last_frame.get()))
        save.grid(row=2, column=1, sticky=N)

        self.generate = Button(self.main_window, text="Save Annotations", command=lambda: self.__save_results(), fg="blue")
        self.generate.config(font=("Calibri", 16))
        self.generate.grid(row=10, column=1, sticky=N)

        self.video_player_frm = Frame(self.main_window, width=100, height=100)
        self.video_player_frm.grid(row=0, column=2, sticky=N)
        self.play_video_btn = Button(self.video_player_frm, text="Open Video", command=self.__play_video)
        self.play_video_btn.grid(sticky=N, pady=10)
        Label(self.video_player_frm, text=self.video_presses_lbl).grid(sticky=W)
        self.update_img_from_video = Button(self.video_player_frm, text="Show current video frame", command=self.__update_frame_from_video)
        self.update_img_from_video.grid(sticky=N)
        self.__bind_frm_shortcut_keys()


        Label(self.video_player_frm, text=self.key_presses_lbl).grid(sticky=S)
        self.__advance_frame(new_frm_number=self.frm_no)
        self.main_window.mainloop()

    def __create_frm_key_presses_lbl(self):
        self.key_presses_lbl = "\n\n Keyboard shortcuts for frame navigation: "
        for k, v in self.img_kbd_bindings.items():
            self.key_presses_lbl += '\n'
            self.key_presses_lbl += v['label']

    def __create_video_key_presses_lbl(self):
        self.video_presses_lbl = "\n\n Keyboard shortcuts for video navigation: "
        for k, v in self.video_kbd_bindings.items():
            self.video_presses_lbl += '\n'
            self.video_presses_lbl += v['label']


    def __bind_frm_shortcut_keys(self):
        self.main_window.bind(self.img_kbd_bindings['save']['kbd'], lambda x: self.__save_results())
        self.main_window.bind(self.img_kbd_bindings['frame+1_keep_choices']['kbd'], lambda x: self.__advance_frame(new_frm_number=int(self.current_frm_n.get() + 1), save_and_keep_checks=True))
        self.main_window.bind(self.img_kbd_bindings['print_annotation_statistic']['kbd'], lambda x: self.print_annotation_statistics())
        self.main_window.bind(self.img_kbd_bindings['frame+1']['kbd'], lambda x: self.__advance_frame(new_frm_number=int(self.current_frm_n.get() + 1)))
        self.main_window.bind(self.img_kbd_bindings['frame-1']['kbd'], lambda x: self.__advance_frame(new_frm_number=int(self.current_frm_n.get() - 1)))
        self.main_window.bind(self.img_kbd_bindings['last_frame']['kbd'], lambda x: self.__advance_frame(new_frm_number=self.max_frm_no))
        self.main_window.bind(self.img_kbd_bindings['first_frame']['kbd'], lambda x: self.__advance_frame(0))




    def print_annotation_statistics(self):
        table_view = [["Video name", self.video_name], ["Video frames", self.video_meta_data["frame_count"]]]
        for target in self.clf_names:
            present = len(self.data_df_targets[self.data_df_targets[target] == 1])
            absent = len(self.data_df_targets[self.data_df_targets[target] == 0])
            table_view.append([target + " present labels", present])
            table_view.append([target + " absent labels", absent])
            table_view.append([target + " % present", present / self.video_meta_data["frame_count"]])
            table_view.append([target + " % absent", absent / self.video_meta_data["frame_count"]])
        headers = ["VARIABLE", "VALUE"]
        print(tabulate(table_view, headers, tablefmt="github"))

    def __play_video(self):
        p = Popen(f"python {PLAY_VIDEO_SCRIPT_PATH}",stdin=PIPE, stdout=PIPE, shell=True,)
        main_project_dir = os.path.dirname(self.config_path)
        p.stdin.write(bytes(self.video_path, "utf-8"))
        p.stdin.close()
        temp_file = os.path.join(main_project_dir, "subprocess.txt")
        with open(temp_file, "w") as text_file:
            text_file.write(str(p.pid))

    def __update_frame_from_video(self):
        f = open(os.path.join(os.path.dirname(self.config_path), "labelling_info.txt"), "r+")
        os.fsync(f.fileno())
        vid_frame_no = int(f.readline())
        self.__advance_frame(new_frm_number=vid_frame_no)
        f.close()

    def __read_frm(self, frm_number: int):
        check_int(name=f'{self.video_name} {frm_number}', value=frm_number, min_value=0)
        self.current_frm_npy = read_frm_of_video(video_path=self.cap, frame_index=frm_number)
        self.current_frm_npy = cv2.cvtColor(self.current_frm_npy, cv2.COLOR_RGB2BGR)
        self.current_frm_pil = Image.fromarray(self.current_frm_npy)
        self.current_frm_pil.thumbnail(self.max_frm_size, Image.LANCZOS)
        self.current_frm_pil = ImageTk.PhotoImage(master=self.main_window, image=self.current_frm_pil)
        self.video_frame = Label(self.main_window, image=self.current_frm_pil)
        self.video_frame.image = self.current_frm_pil
        self.video_frame.grid(row=0, column=0)

    def __advance_frame(self, new_frm_number: int, save_and_keep_checks=False):
        if new_frm_number > self.max_frm_no:
            print(f"FRAME {new_frm_number} CANNOT BE SHOWN - YOU ARE VIEWING THE FINAL FRAME OF THE VIDEO (FRAME NUMBER {self.max_frm_no})")
            self.current_frm_n = IntVar(value=self.max_frm_no)
            self.change_frm_box.delete(0, END)
            self.change_frm_box.insert(0, str(self.current_frm_n.get()))
        elif new_frm_number < 0:
            print(f"FRAME {new_frm_number} CANNOT BE SHOWN - YOU ARE VIEWING THE FIRST FRAME OF THE VIDEO (FRAME NUMBER 0)")
            self.current_frm_n = IntVar(value=0)
            self.change_frm_box.delete(0, END)
            self.change_frm_box.insert(0, str(self.current_frm_n.get()))
        else:
            self.__create_print_statements()
            self.current_frm_n = IntVar(value=new_frm_number)
            self.change_frm_box.delete(0, END)
            self.change_frm_box.insert(0, str(self.current_frm_n.get()))
            if not save_and_keep_checks:
                for target in self.clf_names:
                    self.checkboxes[target]["var"].set(bool(self.data_df_targets[target].loc[int(self.current_frm_n.get())]))
            else:
                for target in self.clf_names:
                    self.checkboxes[target]["var"].set(
                        self.data_df_targets[target].loc[int(self.current_frm_n.get() - 1)])
                    self.save_behavior_in_frm(target=target)
        self.__read_frm(frm_number=int(self.current_frm_n.get()))

    def __save_behavior_in_range(self, start_frm=None, end_frm=None):
        if not self.range_on.get():
            raise FrameRangeError("SAVE RANGE ERROR: TO SAVE RANGE OF FRAMES, TICK THE `Frame range` checkbox before clicking `Save Range`", source=self.__class__.__name__)
        else:
            check_int("START FRAME", int(start_frm), max_value=self.max_frm_no, min_value=0)
            check_int("END FRAME", int(end_frm), max_value=self.max_frm_no, min_value=0)
            for frm_no in range(int(start_frm), int(end_frm) + 1):
                for target in self.clf_names:
                    self.data_df_targets[target].loc[frm_no] = self.checkboxes[target]["var"].get()
            self.__read_frm(frm_number=int(end_frm))
            self.change_frm_box.delete(0, END)
            self.change_frm_box.insert(0, end_frm)
        self.__create_print_statements(frame_range=True, start_frame=start_frm, end_frame=end_frm)

    def save_behavior_in_frm(self, frame_number=None, target=None):
        self.data_df_targets[target].loc[int(self.current_frm_n.get())] = int(self.checkboxes[target]["var"].get())

    def __save_results(self):
        self.save_df = read_df(self.features_extracted_file_path, self.file_type)
        self.save_df = pd.concat([self.save_df, self.data_df_targets], axis=1)
        try:
            write_df(self.save_df, self.file_type, self.targets_inserted_file_path)
        except Exception as e:
            print(e, f"SIMBA ERROR: File for video {get_fn_ext(self.features_extracted_file_path)[1]} could not be saved.")
            raise FileExistsError
        stdout_success(msg=f"SAVED: Annotation file for video {self.video_name} saved within the project_folder/csv/targets_inserted directory.", source=self.__class__.__name__)
        if not self.config.has_section("Last saved frames"):
            self.config.add_section("Last saved frames")
        self.config.set("Last saved frames", str(self.video_name), str(self.current_frm_n.get()))
        with open(self.config_path, "w") as configfile:
            self.config.write(configfile)

    def __create_print_statements(self, frame_range: bool = None, start_frame: int = None, end_frame: int = None):
        print("USER FRAME SELECTION(S):")
        if not frame_range:
            for target in self.clf_names:
                target_present_choice = self.checkboxes[target]["var"].get()
                if target_present_choice == 0:
                    print("{} ABSENT IN FRAME {}".format(target, self.current_frm_n.get()))
                if target_present_choice == 1:
                    print("{} PRESENT IN FRAME {}".format(target, self.current_frm_n.get()))

        if frame_range:
            for target in self.clf_names:
                target_present_choice = self.checkboxes[target]["var"].get()
                if target_present_choice == 1:
                    print("{} PRESENT IN FRAMES {} to {}".format(target, str(start_frame), str(end_frame)))
                elif target_present_choice == 0:
                    print("{} ABSENT IN FRAMES {} to {}".format(target, str(start_frame), str(end_frame)))


def select_labelling_video(config_path: Union[str, os.PathLike],
                           threshold_dict: Optional[Dict[str, Entry_Box]] = None,
                           setting: Literal['pseudo', 'from_scratch'] = 'from_scratch',
                           continuing: bool = None,
                           video_file_path: Union[str, os.PathLike] = None):

    check_file_exist_and_readable(file_path=config_path)
    if threshold_dict is not None:
        check_valid_dict(x=threshold_dict, valid_key_dtypes=(str,), valid_values_dtypes=(float,))
    check_str(name='setting', value=setting, options=('pseudo', "from_scratch"))
    check_valid_boolean(value=[continuing], source=select_labelling_video.__name__)
    if setting is not "pseudo":
        video_file_path = filedialog.askopenfilename(filetypes=[("Video files", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
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
                           video_path=video_file_path,
                           thresholds=threshold_dict,
                           setting=setting,
                           continuing=continuing)

#
# test = select_labelling_video(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini",
#                               threshold_dict={'Attack': 0.4},
#                               setting='from_scratch',
#                               continuing=True)

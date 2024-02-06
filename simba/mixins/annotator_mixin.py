import os
import re
from copy import deepcopy
from subprocess import PIPE, Popen
from tkinter import *
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import pandas as pd
from PIL import Image, ImageTk

try:
    from typing import Literal
except:
    from typing_extensions import Literal

from simba.mixins.config_reader import ConfigReader
from simba.ui.tkinter_functions import Entry_Box
from simba.utils.checks import check_int
from simba.utils.enums import Formats, Labelling
from simba.utils.errors import FrameRangeError, InvalidInputError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (find_time_stamp_from_frame_numbers,
                                    get_all_clf_names, get_fn_ext,
                                    get_video_meta_data, read_df, write_df)
from simba.utils.warnings import FrameRangeWarning
from simba.video_processors.video_processing import clip_video_in_range


class AnnotatorMixin(ConfigReader):
    """
    Methods for creating tkinter GUI frames and functions associated with annotating videos.

    Currently under development (starting 01/24). As the number of different annotation methods and interfaces increases,
    this class will contain common methods for all annotation interfaces to decrease code duplication.

    :param Union[str, os.PathLike] config_path: Path to SimBA project config ini file.
    :param Union[str, os.PathLike] video_path: Path to video file to-be annotated.
    :param Union[str, os.PathLike] data_path: Path to featurized pose-estimation data associated with the video.
    :param Optional[Tuple[int]] frame_size: The size of the subframe displaying the video frame in the GUI.

    """

    def __init__(
        self,
        config_path: Union[str, os.PathLike],
        video_path: Union[str, os.PathLike],
        data_path: Union[str, os.PathLike],
        frame_size: Optional[Tuple[int]] = Labelling.MAX_FRM_SIZE.value,
        title: Optional[str] = None,
    ):

        ConfigReader.__init__(self, config_path=config_path)
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        self.video_meta_data = get_video_meta_data(video_path=self.video_path)
        self.frame_lst = list(range(0, self.video_meta_data["frame_count"]))
        _, self.video_name, _ = get_fn_ext(filepath=video_path)
        _, self.pixels_per_mm, self.fps = self.read_video_info(
            video_name=self.video_name
        )
        self.target_lst = get_all_clf_names(config=self.config, target_cnt=self.clf_cnt)
        self.original_data = read_df(file_path=data_path, file_type=self.file_type)
        self.save_path = os.path.join(
            self.targets_folder, f"{self.video_name}.{self.file_type}"
        )
        self.max_frm_no, self.min_frm_no = max(self.original_data.index), min(
            self.original_data.index
        )
        self.main_frm = Toplevel()
        self.max_size = frame_size
        self.main_frm.title(title)

    def video_frm_label(
        self,
        frm_number: int,
        max_size: Optional[Tuple[int]] = None,
        loc: Tuple[int] = (0, 0),
    ) -> None:
        """
        Inserts a video frame as a tkinter label at a specified maximum size at specified grid location.

        .. image:: _static/img/video_frm_label.png
           :width: 500
           :align: center

        :param int frm_number: The frame number if the video that should be displayed as a tkinter label.
        :param Optional[Tuple[int, int]] max_size: The maximum size of the image when displayed. If None, then ``frame_size`` defined at instance init.
        :param Tuple[int, int] loc: The grid location (row, column) within the main frame at which the video frame should be displayed.
        """

        if max_size is None:
            max_size = self.max_size
        if frm_number > self.video_meta_data["frame_count"]:
            raise FrameRangeError(
                msg=f'Cannot display frame {frm_number}. Video contains {self.video_meta_data["frame_count"]} frames.'
            )
        self.cap.set(1, frm_number)
        _, frm = self.cap.read()
        frm = cv2.cvtColor(frm, cv2.COLOR_RGB2BGR)
        frm = Image.fromarray(frm)
        frm.thumbnail(max_size, Image.ANTIALIAS)
        frm = ImageTk.PhotoImage(master=self.main_frm, image=frm)
        video_frame = Label(self.main_frm, image=frm)
        video_frame.image = frm
        video_frame.grid(row=loc[0], column=loc[1], sticky=NW)

    def h_nav_bar(
        self,
        parent: Frame,
        update_funcs: Optional[List[Callable[[int], None]]] = None,
        store_funcs: Optional[List[Callable[[], None]]] = None,
        size: Optional[Tuple[int, int]] = (300, 700),
        loc: Optional[Tuple[int, int]] = (1, 0),
        previous_next_clf: Optional[bool] = False,
    ) -> None:
        """
        Creates a horizontal frame navigation bar where the buttons are tied to callbacks for changing and displaying video frames.

        .. image:: _static/img/h_nav_bar.png
           :width: 500
           :align: center

        :param Frame parent: The tkinter Frame to place navigation bar within.
        :param Optional[List[Callable[[int], None]]] update_funcs: Optional list of callables that accepts a single integer inputs.  Can be methods that updates part of the interface as the frame number changes.
        :param Optional[List[Callable[[], None]]] store_funcs: Optional list of callables without arguments. Can be methods that stores the selections in memory as users proceeds through the frames.
        :param Optional[Tuple[int, int]] size: The size of the navigation bar in h x w. Default 300 x 700.
        :param Optional[Tuple[int, int]] loc: The grid location (row, column) within the parent frame at which the navigation bar should be displayed. Defualt: (1, 0).
        :param Optional[bool] previous_next_clf: If True, then include four buttons allowing users to navigate to the most proximal preceding or proceeding frame where behaviors are annotated as present or absent.
        """

        out_frm = Frame(parent, bd=2, width=size[1], height=size[0])
        nav_frm = Frame(out_frm)
        jump_frame = Frame(out_frm, bd=2)
        jump_lbl = Label(jump_frame, text="JUMP SIZE:")
        jump_size = Scale(jump_frame, from_=0, to=100, orient=HORIZONTAL, length=200)
        jump_size.set(0)
        self.frm_box = Entry_Box(nav_frm, "FRAME NUMBER", "15", validation="numeric")
        self.frm_box.entry_set(val=self.min_frm_no)
        forward_btn = Button(
            nav_frm,
            text=">",
            command=lambda: self.change_frame(
                new_frm_id=int(self.frm_box.entry_get) + 1,
                update_funcs=update_funcs,
                store_funcs=store_funcs,
            ),
        )
        backward_btn = Button(
            nav_frm,
            text="<",
            command=lambda: self.change_frame(
                new_frm_id=int(self.frm_box.entry_get) - 1,
                update_funcs=update_funcs,
                store_funcs=store_funcs,
            ),
        )
        forward_max_btn = Button(
            nav_frm,
            text=">>",
            command=lambda: self.change_frame(
                new_frm_id=self.max_frm_no - 1,
                update_funcs=update_funcs,
                store_funcs=store_funcs,
            ),
        )
        backward_max_btn = Button(
            nav_frm,
            text="<<",
            command=lambda: self.change_frame(
                new_frm_id=self.min_frm_no,
                update_funcs=update_funcs,
                store_funcs=store_funcs,
            ),
        )
        jump_back = Button(
            jump_frame,
            text="<<",
            command=lambda: self.change_frame(
                new_frm_id=int(self.frm_box.entry_get) - jump_size.get(),
                update_funcs=update_funcs,
                store_funcs=store_funcs,
            ),
        )
        jump_forward = Button(
            jump_frame,
            text=">>",
            command=lambda: self.change_frame(
                new_frm_id=int(self.frm_box.entry_get) + jump_size.get(),
                update_funcs=update_funcs,
                store_funcs=store_funcs,
            ),
        )
        select_frm_btn = Button(
            nav_frm,
            text="VIEW SELECTED FRAME",
            command=lambda: self.change_frame(
                new_frm_id=int(self.frm_box.entry_get),
                update_funcs=update_funcs,
                store_funcs=store_funcs,
            ),
        )
        nav_frm.grid(row=0, column=0)
        self.frm_box.grid(row=0, column=0)

        backward_max_btn.grid(row=0, column=31, sticky=NE, padx=Labelling.PADDING.value)
        backward_btn.grid(row=0, column=2, sticky=NE, padx=Labelling.PADDING.value)
        forward_btn.grid(row=0, column=3, sticky=NE, padx=Labelling.PADDING.value)
        forward_max_btn.grid(row=0, column=4, sticky=NE, padx=Labelling.PADDING.value)
        select_frm_btn.grid(row=1, column=0, sticky=NE, padx=Labelling.PADDING.value)
        jump_frame.grid(row=1, column=0)
        jump_lbl.grid(row=0, column=0, sticky=NE)
        jump_size.grid(row=0, column=1, sticky=SE)
        jump_back.grid(row=0, column=2, sticky=SE)
        jump_forward.grid(row=0, column=3, sticky=SE)

        if previous_next_clf:
            next_clf_btn_frm = Frame(out_frm, bd=2)
            next_absent_btn = Button(
                next_clf_btn_frm,
                text="NEXT ABSENT",
                fg="red",
                command=lambda: self.find_proximal_annotated_frm(
                    forwards=True, present=False
                ),
            )
            next_present_btn = Button(
                next_clf_btn_frm,
                text="NEXT PRESENT",
                fg="blue",
                command=lambda: self.find_proximal_annotated_frm(
                    forwards=True, present=True
                ),
            )
            last_absent_btn = Button(
                next_clf_btn_frm,
                text="PREVIOUS ABSENT",
                fg="red",
                command=lambda: self.find_proximal_annotated_frm(
                    forwards=False, present=False
                ),
            )
            last_present_btn = Button(
                next_clf_btn_frm,
                text="PREVIOUS PRESENT",
                fg="blue",
                command=lambda: self.find_proximal_annotated_frm(
                    forwards=False, present=True
                ),
            )
            next_clf_btn_frm.grid(row=2, column=0)
            last_present_btn.grid(row=0, column=0, sticky=NE)
            last_absent_btn.grid(row=0, column=1, sticky=NE)
            next_absent_btn.grid(row=0, column=2, sticky=NE)
            next_present_btn.grid(row=0, column=3, sticky=NE)

        out_frm.grid(row=loc[0], column=loc[1], sticky=NW)

    def v_navigation_pane_targeted(
        self,
        parent: Frame,
        save_func: Callable[[], None],
        update_funcs: Optional[List[Callable[[int], None]]] = None,
        store_funcs: Optional[List[Callable[[], None]]] = None,
        loc: Optional[Tuple[int, int]] = (0, 2),
    ) -> None:
        """
        Create a vertical navigation pane for playing a video and displaying and activating keyboard shortcuts when annotating bouts.

        .. image:: _static/img/h_nav_bar.png
           :width: 500
           :align: center

        :param Frame parent: The tkinter Frame to place the vertical navigation bar within.
        :param Callable[[], None] save_func: The save-data-to-disk function that should be called when using the save data shortcut.
        :param Optional[List[Callable[[int], None]]] update_funcs: Optional list of callables that accepts a single integer inputs.  Can be methods that updates part of the interface as the frame number changes.
        :param Optional[List[Callable[[], None]]] store_funcs: Optional list of callables without arguments. Can be methods that stores the selections in memory as users proceeds through the frames.
        :param Optional[Tuple[int, int]] loc: The grid location (row, column) within the parent frame at which the navigation bar should be displayed. Default: (1, 0).

        """

        def bind_shortcut_keys(parent):
            parent.bind("<Control-s>", lambda x: save_func())
            parent.bind(
                "<Right>",
                lambda x: self.change_frame(
                    new_frm_id=int(self.frm_box.entry_get) + 1,
                    update_funcs=update_funcs,
                    store_funcs=store_funcs,
                ),
            )
            parent.bind(
                "<Left>",
                lambda x: self.change_frame(
                    new_frm_id=int(self.frm_box.entry_get) - 1,
                    update_funcs=update_funcs,
                    store_funcs=store_funcs,
                ),
            )
            parent.bind(
                "<Control-l>",
                lambda x: self.change_frame(
                    new_frm_id=self.max_frm_no - 1,
                    update_funcs=update_funcs,
                    store_funcs=store_funcs,
                ),
            )
            parent.bind(
                "<Control-o>",
                lambda x: self.change_frame(
                    new_frm_id=self.min_frm_no,
                    update_funcs=update_funcs,
                    store_funcs=store_funcs,
                ),
            )
            parent.bind(
                "<Control-a>",
                lambda x: self.change_frame(
                    new_frm_id=int(self.frm_box.entry_get) + 1,
                    keep_radio_btn_choices=True,
                ),
            )

        video_player_frm = Frame(parent)
        play_video_btn = Button(
            video_player_frm, text="OPEN VIDEO", command=lambda: self.__play_video()
        )
        play_video_btn.grid(sticky=N, pady=10)
        Label(
            video_player_frm,
            text="\n\n  Keyboard shortcuts for video navigation: \n p = Pause/Play"
            "\n\n After pressing pause:"
            "\n o = +2 frames \n e = +10 frames \n w = +1 second"
            "\n\n t = -2 frames \n s = -10 frames \n x = -1 second"
            "\n\n q = Close video window \n\n",
        ).grid(sticky=W)
        update_img_from_video = Button(
            video_player_frm, text="Show current video frame", command=None
        )
        update_img_from_video.grid(sticky=N)

        key_presses_lbl = Label(
            video_player_frm,
            text="\n\n Keyboard shortcuts for frame navigation: \n Right Arrow = +1 frame"
            "\n Left Arrow = -1 frame"
            "\n Ctrl + a = +1 frame and keep choices"
            "\n Ctrl + s = Save annotations file"
            "\n Ctrl + l = Last frame"
            "\n Ctrl + o = First frame",
        )
        key_presses_lbl.grid(sticky=S)
        bind_shortcut_keys(parent)
        video_player_frm.grid(row=loc[0], column=loc[1], sticky=NW)

    def targeted_bouts_pane(self, parent: Frame) -> Frame:
        """Create a pane for choosing bouts start and end and a radiobutton truth table for targeted bouts annotations. Used by simba.labelling.targeted_annotations_bouts.TargetedAnnotatorBouts"""

        def set_selection():
            selection_txt.config(
                text=f"CURRENT SELECTION START: {start_frm_bx.entry_get}, END: {end_frm_bx.entry_get}"
            )
            save_button.config(
                text=f"SAVE ANNOTATION CLIP \n START FRAME: {start_frm_bx.entry_get}, \n END FRAME: {end_frm_bx.entry_get}"
            )
            self.set_start, self.set_end = start_frm_bx.entry_get, end_frm_bx.entry_get

        pane, self.set_start, self.set_end = Frame(parent, bd=2), 0, 1
        selection_txt = Label(
            pane,
            text=f"CURRENT SELECTION START: {0}, END: {1}",
            font=("Helvetica", 14, "bold"),
        )
        start_frm_bx = Entry_Box(
            pane, "SELECT START FRAME:", "20", validation="numeric", entry_box_width=10
        )
        start_frm_bx.entry_set("0")
        end_frm_bx = Entry_Box(
            pane, "SELECT END FRAME:", "20", validation="numeric", entry_box_width=10
        )
        end_frm_bx.entry_set("1")
        confirm_btn = Button(
            pane, text="SET SELECTION", command=lambda: set_selection()
        )
        selection_txt.grid(row=0, column=0, pady=30, sticky=N)
        start_frm_bx.grid(row=1, column=0, pady=30, sticky=S)
        end_frm_bx.grid(row=2, column=0, pady=10, sticky=S)
        confirm_btn.grid(row=3, column=0, sticky=S)

        Label(pane, text=f"PRESENT", font=("Helvetica", 12, "bold")).grid(
            row=4, column=1, pady=20, sticky=NW
        )
        Label(pane, text=f"ABSENT", font=("Helvetica", 12, "bold")).grid(
            row=4, column=2, pady=20, sticky=NW
        )
        self.clf_radio_btns = {}
        for target_cnt, target in enumerate(self.target_lst):
            self.clf_radio_btns[target] = StringVar(value=0)
            Label(pane, text=target, font=("Helvetica", 12, "bold")).grid(
                row=target_cnt + 5, column=0, sticky=NW
            )
            present = Radiobutton(
                pane,
                text=None,
                variable=self.clf_radio_btns[target],
                value=1,
                command=None,
            )
            absent = Radiobutton(
                pane,
                text=None,
                variable=self.clf_radio_btns[target],
                value=0,
                command=None,
            )
            present.grid(row=target_cnt + 5, column=1, pady=5, sticky=NW)
            absent.grid(row=target_cnt + 5, column=2, pady=5, sticky=NW)

        save_button = Button(
            pane,
            text=f"SAVE ANNOTATION CLIP \n START FRAME: {start_frm_bx.entry_get}, \n END FRAME: {end_frm_bx.entry_get}",
            fg="green",
            font=("Helvetica", 14, "bold"),
            command=lambda: self.__targeted_clips_save(),
        )
        save_button.grid(row=5 + len(self.target_lst), column=0, pady=40, sticky=S)

        return pane

    def update_current_selected_frm_lbl(self, new_frm: Union[int, str]):
        """Helper to update label showing current frame text shown when annotating bouts frame-wise."""
        self.frame_selection_txt.config(text=f"ANNOTATING FRAME: {int(new_frm)}")
        integers = [
            int(x) for x in re.findall(r"\b\d+\b", self.bout_selection_txt.cget("text"))
        ]
        if (int(new_frm) < integers[0]) or (int(new_frm) > integers[1]):
            FrameRangeWarning(
                msg=f"You are viewing a frame ({new_frm}) that is OUTSIDE your selected bout (frame {integers[0]} to frame {integers[1]}). Any annotation for this frame will not be saved."
            )

    def __update_bout_selection_txt(
        self, new_start_frm: Union[int, str], new_end_frm: Union[int, str]
    ):
        """Helper to update label showing current bout text shown when annotating bouts."""
        self.bout_selection_txt.config(
            text=f"BOUT SELECTION START: {new_start_frm}, END: {new_end_frm}"
        )
        self.save_button.config(
            text=f"SAVE ANNOTATION CLIP \n START FRAME: {new_start_frm}, \n END FRAME: {new_end_frm}"
        )

    def targeted_frames_selection_pane(
        self, parent: Frame, loc: Optional[Tuple[int, int]] = (0, 1)
    ) -> None:
        """
        Creates a vertical pane that includes tkinter frames for selecting bouts and annotating behaviours in those bouts frame-wise.

        .. image:: _static/img/targeted_frames_selection_pane.png
           :width: 500
           :align: center

        :param Frame parent: The tkinter Frame to place the vertical annotation frame within.
        :param Optional[Tuple[int, int]] loc: The grid location (row, column) within the parent frame at which the pane should be displayed. Default: (0, 1).

        """

        def set_selection():
            if int(self.start_frm_bx.entry_get) >= int(self.end_frm_bx.entry_get):
                raise FrameRangeError(
                    msg=f"Start frame ({self.start_frm_bx.entry_get}) cannot be the same or larger than the end frame ({self.end_frm_bx.entry_get})"
                )
            self.__update_bout_selection_txt(
                new_start_frm=self.start_frm_bx.entry_get,
                new_end_frm=self.end_frm_bx.entry_get,
            )
            self.change_frame(
                new_frm_id=self.start_frm_bx.entry_get,
                update_funcs=[self.update_current_selected_frm_lbl],
            )

        self.results = {}
        self.session_annotated_frames = []
        pane, self.set_start, self.set_end = Frame(parent, bd=2), 0, 1
        self.bout_selection_txt = Label(
            pane,
            text=f"BOUT SELECTED START: {0}, END: {1}",
            font=("Helvetica", 14, "bold"),
        )
        self.frame_selection_txt = Label(
            pane, text=f"ANNOTATING FRAME: {0}", font=("Helvetica", 14, "bold")
        )
        self.start_frm_bx = Entry_Box(
            pane, "BOUT START FRAME:", "20", validation="numeric", entry_box_width=10
        )
        self.start_frm_bx.entry_set("0")
        self.end_frm_bx = Entry_Box(
            pane, "BOUT END FRAME:", "20", validation="numeric", entry_box_width=10
        )
        self.end_frm_bx.entry_set("1")
        confirm_btn = Button(
            pane, text="SET SELECTION", command=lambda: set_selection()
        )
        self.bout_selection_txt.grid(row=0, column=0, sticky=N)
        self.frame_selection_txt.grid(row=1, column=0, sticky=N)
        self.start_frm_bx.grid(row=2, column=0, sticky=S)
        self.end_frm_bx.grid(row=3, column=0, sticky=S)
        confirm_btn.grid(row=4, column=0, sticky=S)

        Label(pane, text=f"PRESENT", font=("Helvetica", 12, "bold")).grid(
            row=5, column=1, pady=20, sticky=NW
        )
        Label(pane, text=f"ABSENT", font=("Helvetica", 12, "bold")).grid(
            row=5, column=2, pady=20, sticky=NW
        )
        self.clf_radio_btns = {}
        for target_cnt, target in enumerate(self.target_lst):
            self.clf_radio_btns[target] = StringVar(value=0)
            Label(pane, text=target, font=Formats.LABELFRAME_HEADER_FORMAT.value).grid(
                row=target_cnt + 6, column=0, sticky=NW
            )
            present = Radiobutton(
                pane,
                text=None,
                variable=self.clf_radio_btns[target],
                value=1,
                command=None,
            )
            absent = Radiobutton(
                pane,
                text=None,
                variable=self.clf_radio_btns[target],
                value=0,
                command=None,
            )
            present.grid(row=target_cnt + 6, column=1, pady=5, sticky=NW)
            absent.grid(row=target_cnt + 6, column=2, pady=5, sticky=NW)

        self.save_button = Button(
            pane,
            text=f"SAVE ANNOTATION CLIP \n START FRAME: {self.start_frm_bx.entry_get}, \n END FRAME: {self.end_frm_bx.entry_get}",
            fg="green",
            font=("Helvetica", 14, "bold"),
            command=lambda: self.targeted_annotations_frames_save(),
        )
        self.save_button.grid(row=6 + len(self.target_lst), column=0, pady=40, sticky=S)
        pane.grid(row=loc[0], column=loc[1], sticky=NW)

    def advanced_labelling_pane(
        self,
        parent: Frame,
        data: Optional[Dict[str, List[int]]] = None,
        loc: Optional[Tuple[int]] = (0, 1),
    ) -> None:

        pane = Frame(parent, bd=2)
        Label(
            pane, text="SELECT BEHAVIOR:", font=Formats.LABELFRAME_HEADER_FORMAT.value
        ).grid(row=0, column=0, sticky=N)
        Label(pane, text=f"PRESENT", font=Formats.LABELFRAME_HEADER_FORMAT.value).grid(
            row=1, column=1, pady=20, sticky=NW
        )
        Label(pane, text=f"ABSENT", font=Formats.LABELFRAME_HEADER_FORMAT.value).grid(
            row=1, column=2, pady=20, sticky=NW
        )
        Label(pane, text=f"NONE", font=Formats.LABELFRAME_HEADER_FORMAT.value).grid(
            row=1, column=3, pady=20, sticky=NW
        )
        self.results = deepcopy(data)
        self.clf_radio_btns = {}
        for target_cnt, target in enumerate(self.target_lst):
            self.clf_radio_btns[target] = IntVar(
                value=self.get_annotation_of_frame(
                    frm_num=int(self.frm_box.entry_get),
                    clf=target,
                    allowed_vals=Labelling.VALID_ANNOTATIONS_ADVANCED.value,
                )
            )
            Label(pane, text=target, font=Formats.LABELFRAME_HEADER_FORMAT.value).grid(
                row=target_cnt + 6, column=0, sticky=NW
            )
            present = Radiobutton(
                pane,
                text=None,
                variable=self.clf_radio_btns[target],
                value=1,
                command=lambda: self.store_annotation(),
            )
            absent = Radiobutton(
                pane,
                text=None,
                variable=self.clf_radio_btns[target],
                value=0,
                command=lambda: self.store_annotation(),
            )
            none = Radiobutton(
                pane,
                text=None,
                variable=self.clf_radio_btns[target],
                value=2,
                command=lambda: self.store_annotation(),
            )
            present.grid(row=target_cnt + 6, column=1, pady=5, sticky=NW)
            absent.grid(row=target_cnt + 6, column=2, pady=5, sticky=NW)
            none.grid(row=target_cnt + 6, column=3, pady=5, sticky=NW)
        self.save_button = Button(
            pane,
            text=f"SAVE DATA",
            fg="green",
            font=("Helvetica", 18, "bold"),
            anchor="center",
            command=lambda: self.save_data(),
        )
        self.save_button.grid(
            row=6 + len(self.target_lst), column=0, pady=100, sticky=S
        )
        pane.grid(row=loc[0], column=loc[1], sticky=NW)

    def get_annotation_of_frame(
        self,
        frm_num: int,
        clf: str,
        allowed_vals: Optional[List[Union[int, None]]] = None,
    ) -> Union[int, None]:
        """Helper to retrieve the stored annotation of a specific classifier at a specific frame index"""
        if int(frm_num) not in self.results[clf]:
            raise FrameRangeError(
                msg=f"You are annotating frame {frm_num} with is not a frame in the dataset (row count: {self.max_frm_no}",
                source=self.__class__.__name__,
            )
        if allowed_vals is not None:
            if self.results[clf][frm_num] not in allowed_vals:
                raise InvalidInputError(
                    msg=f"Found labelling value {self.results[clf][frm_num]}. Allowed: {allowed_vals}"
                )
        else:
            return self.results[clf][frm_num]

    def update_clf_radiobtns(self, frm_num: int):
        """Update helper to set the radio button to annotated values"""
        for clf in self.clf_names:
            value = self.get_annotation_of_frame(frm_num=frm_num, clf=clf)
            if value is not None:
                value = int(value)
            self.clf_radio_btns[clf].set(value=value)

    def __play_video(self):
        """Helper to play video when the ``play video`` button is pressed"""
        p = Popen(
            "python {}".format(Labelling.PLAY_VIDEO_SCRIPT_PATH.value),
            stdin=PIPE,
            stdout=PIPE,
            shell=True,
        )
        p.stdin.write(bytes(self.video_path, "utf-8"))
        p.stdin.close()
        temp_file = os.path.join(os.path.dirname(self.config_path), "subprocess.txt")
        with open(temp_file, "w") as text_file:
            text_file.write(str(p.pid))

    def __targeted_clips_save(self):
        """Helper method called by targeted_clips_pane to save video and dataframe sliced during targeted clips annotations"""
        timer = SimbaTimer(start=True)
        print("Saving video clip...")
        start_idx, end_idx = int(self.set_start), int(self.set_end)
        if start_idx >= end_idx:
            raise FrameRangeError(
                msg=f"Start frame ({start_idx}) has to be before end frame ({end_idx}).",
                source=self.__class__.__name__,
            )
        if (
            (start_idx > self.max_frm_no)
            or (start_idx < self.min_frm_no)
            or (end_idx > self.max_frm_no)
            or (end_idx < self.min_frm_no)
        ):
            raise FrameRangeError(
                msg=f"Start frame ({start_idx}) or end frame ({end_idx}) cannot be smaller or larger than the minimum ({self.min_frm_no}) and maximum ({self.max_frm_no}) frame of the video",
                source=self.__class__.__name__,
            )
        timestamps = find_time_stamp_from_frame_numbers(
            start_frame=start_idx, end_frame=end_idx, fps=self.video_meta_data["fps"]
        )
        clip_video_in_range(
            file_path=self.video_path,
            start_time=timestamps[0],
            end_time=timestamps[1],
            out_dir=self.video_dir,
            include_clip_time_in_filename=True,
            overwrite=True,
        )
        df_filename = os.path.join(
            self.video_name
            + f'_{timestamps[0].replace(":", "-")}_{timestamps[1].replace(":", "-")}.{self.file_type}'
        )
        video_name = os.path.join(
            self.video_name
            + f'_{timestamps[0].replace(":", "-")}_{timestamps[1].replace(":", "-")}'
        )
        print(f"Saving targets data {df_filename}...")
        df = self.original_data.iloc[start_idx:end_idx, :]
        for clf, btn in self.clf_radio_btns.items():
            df[clf] = btn.get()
        write_df(
            df=df,
            file_type=self.file_type,
            save_path=os.path.join(self.targets_folder, df_filename),
        )
        self.add_video_to_video_info(
            video_name=video_name,
            fps=self.fps,
            width=self.video_meta_data["width"],
            height=self.video_meta_data["height"],
            pixels_per_mm=self.pixels_per_mm,
        )
        timer.stop_timer()
        stdout_success(
            msg=f"Annotated clip {df_filename} saved into SimBA project",
            elapsed_time=timer.elapsed_time_str,
        )

    def change_frame(
        self,
        new_frm_id: int,
        min_frm: Optional[int] = None,
        max_frm: Optional[int] = None,
        update_funcs: Optional[List[Callable[[int], None]]] = None,
        store_funcs: Optional[List[Callable[[], None]]] = None,
        keep_radio_btn_choices: Optional[bool] = False,
    ) -> None:
        """
        Change the frame displayed in annotator GUI.

        .. note::
           store_funcs will be executed before update_funcs.

        :parameter int new_frm_id: The frame number of the new frame.
        :parameter Optional[int] min_frm: If the minimum frame number is not the first frame of the video, pass the minimum frame number here.
        :parameter Optional[int] max_frm: If the maximum frame number is not the last frame of the video, pass the max frame number here.
        :parameter Optional[int] max_frm: If the maximum frame number is not the last frame of the video, pass the max frame number here.
        :parameter Optional[List[Callable[[int], None]]] update_funcs: Optional functions that takes accepts the new frame numers that should be called. E.g., if updating the frame number should cause the display of the new frame numbers in any other Frame.
        :parameter Optional[List[Callable[[], None]]] store_funcs: Optional functions that saves user frame selections in memory.
        :parameter Optional[bool] keep_radio_btn_choices: If True, then any update_funcs that causes the update of radio button choices in the newly displayed frame will be supressed. Thus, the choices in the prior frame is maintained.
        """

        check_int(name="FRAME NUMBER", value=new_frm_id)
        if min_frm != None:
            check_int(
                name="MIN FRAME NUMBER",
                value=min_frm,
                min_value=self.min_frm_no,
                max_value=self.max_frm_no,
            )
        if max_frm != None:
            check_int(
                name="MIN FRAME NUMBER",
                value=max_frm,
                min_value=self.min_frm_no,
                max_value=self.max_frm_no,
            )
        if int(new_frm_id) > (self.max_frm_no - 1):
            self.frm_box.entry_set(val=self.max_frm_no)
            raise FrameRangeError(
                msg=f"FRAME {new_frm_id} CANNOT BE SHOWN - YOU ARE VIEWING THE FINAL FRAME OF THE VIDEO (FRAME NUMBER {self.max_frm_no})",
                source=self.__class__.__name__,
            )
        elif int(new_frm_id) < 0:
            self.frm_box.entry_set(val=self.min_frm_no)
            raise FrameRangeError(
                msg=f"FRAME {new_frm_id} CANNOT BE SHOWN - YOU ARE VIEWING THE FIRST FRAME OF THE VIDEO (FRAME NUMBER {self.min_frm_no})",
                source=self.__class__.__name__,
            )
        else:
            self.old_frm_val = self.frm_box.entry_get
            self.frm_box.entry_set(val=new_frm_id)
            self.video_frm_label(frm_number=int(new_frm_id))
        if (keep_radio_btn_choices is True) and (update_funcs is not None):
            update_funcs = [x for x in update_funcs if x != self.update_clf_radiobtns]
        if update_funcs is not None:
            for f in update_funcs:
                f(new_frm_id)
        if store_funcs is not None:
            for f in store_funcs:
                f()

    def store_targeted_annotations_frames(self):
        """Method to store annotations in memory while annotating targeted bouts frame-wise"""
        integers = [
            int(x) for x in re.findall(r"\b\d+\b", self.bout_selection_txt.cget("text"))
        ]
        if (int(self.old_frm_val) < integers[0]) or (
            int(self.old_frm_val) > integers[1]
        ):
            FrameRangeWarning(
                msg=f"ANNOTATION NOT SAVED: Frame {self.old_frm_val} is outside the selected bout ({integers[0]}-{integers[1]})"
            )
        else:
            self.results[int(self.old_frm_val)] = {}
            for clf, btn in self.clf_radio_btns.items():
                self.results[int(self.old_frm_val)][clf] = int(btn.get())

    def store_annotation(self) -> None:
        """Method to store annotations in memory"""
        frm_num = self.frm_box.entry_get
        for clf, btn in self.clf_radio_btns.items():
            if int(frm_num) not in self.results[clf]:
                raise FrameRangeError(
                    msg=f"You are annotating frame {frm_num} with is not a frame in the dataset (row count: {self.max_frm_no}",
                    source=self.__class__.__name__,
                )
            else:
                self.results[clf][int(frm_num)] = btn.get()

    def targeted_annotations_frames_save(self):
        """Method to save annotations to disk when using targeted bout frame-wise annotations"""
        save_timer = SimbaTimer(start=True)
        annotated_frms = []
        integers = [
            int(x) for x in re.findall(r"\b\d+\b", self.bout_selection_txt.cget("text"))
        ]
        for frm, annotation_data in self.results.items():
            annotated_frms.append(int(frm))
        missing_frms = list(
            set(list(range(integers[0], integers[1] + 1))) - set(annotated_frms)
        )
        additional_frms = list(
            set(annotated_frms) - set(list(range(integers[0], integers[1] + 1)))
        )
        if len(missing_frms) > 0:
            raise FrameRangeError(
                msg=f"MISSING ANNOTATIONS FOR FRAMES IN SELECTED BOUT: {missing_frms}. ANNOTATE THESE FRAMES BEFORE SAVING SELECTED BOUT DATA."
            )
        if len(additional_frms) > 0:
            FrameRangeWarning(
                msg=f"Annotations found for frames that are not within the selected bout {additional_frms}. These annotations will not be saved."
            )
        annotated_frms = [x for x in annotated_frms if x not in additional_frms]
        duplication_annotations = [
            x for x in annotated_frms if x in self.session_annotated_frames
        ]
        if len(duplication_annotations) > 0:
            raise FrameRangeError(
                msg=f"Frames {duplication_annotations} of video {self.video_name} have already been annotated as part of a different bout. You can not annotate these frame again in the same annotation session."
            )
        annotated_rows = self.original_data.loc[annotated_frms]
        for frm, annotation_data in self.results.items():
            for clf, btn_val in annotation_data.items():
                annotated_rows.loc[frm, clf] = int(btn_val)
        save_path = os.path.join(
            self.targets_folder,
            f"{self.video_name}_{integers[0]}_{integers[1]}.{self.file_type}",
        )
        write_df(df=annotated_rows, file_type=self.file_type, save_path=save_path)
        timestamps = find_time_stamp_from_frame_numbers(
            start_frame=min(annotated_frms),
            end_frame=max(annotated_frms),
            fps=self.video_meta_data["fps"],
        )
        clip_video_in_range(
            file_path=self.video_path,
            start_time=timestamps[0],
            end_time=timestamps[1],
            out_dir=self.video_dir,
            include_frame_numbers_in_filename=True,
            overwrite=True,
        )
        self.add_video_to_video_info(
            video_name=f"{self.video_name}_{integers[0]}_{integers[1]}",
            fps=self.fps,
            width=self.video_meta_data["width"],
            height=self.video_meta_data["height"],
            pixels_per_mm=self.pixels_per_mm,
        )
        save_timer.stop_timer()
        stdout_success(
            msg=f"Annotated clip {save_path} saved into SimBA project",
            elapsed_time=save_timer.elapsed_time_str,
        )
        self.session_annotated_frames.extend(annotated_frms)

    def find_proximal_annotated_frm(self, forwards: bool, present: bool):
        """Helper to find the most proximal preceding or proceeding frame where any one behavior are annotated as present or absent."""
        target_frames = set()
        if present:
            for clf in self.clf_names:
                target_frames = target_frames.union(
                    set([k for k, v in self.results[clf].items() if v == 1])
                )
        else:
            for clf in self.clf_names:
                target_frames = target_frames.union(
                    set([k for k, v in self.results[clf].items() if v == 0])
                )
        if forwards:
            target_frames = [
                x for x in list(target_frames) if x > int(self.frm_box.entry_get)
            ]
            if len(target_frames) == 0:
                if present:
                    raise FrameRangeError(
                        msg="No LATER frames with behavior annotated as PRESENT found",
                        source=self.__class__.__name__,
                    )
                else:
                    raise FrameRangeError(
                        msg="No LATER frames with behavior annotated as ABSENT found",
                        source=self.__class__.__name__,
                    )

        else:
            target_frames = [
                x for x in list(target_frames) if x < int(self.frm_box.entry_get)
            ]
            if len(target_frames) == 0:
                if present:
                    raise FrameRangeError(
                        msg="No PRIOR frames with behavior annotated as PRESENT found",
                        source=self.__class__.__name__,
                    )
                else:
                    raise FrameRangeError(
                        msg="No PRIOR frames with behavior annotated as ABSENT found",
                        source=self.__class__.__name__,
                    )

        frm_num = min(target_frames, key=lambda x: abs(x - int(self.frm_box.entry_get)))
        self.frm_box.entry_set(val=frm_num)
        self.update_clf_radiobtns(frm_num=frm_num)
        self.video_frm_label(frm_number=frm_num)

    def save_data(self):
        results = pd.DataFrame(self.results)
        print(self.original_data)
        for target_col in results.columns:
            self.original_data = self.original_data.drop(
                [target_col], axis=1, errors="ignore"
            )
            self.original_data[target_col] = results[target_col]
        print(self.original_data)

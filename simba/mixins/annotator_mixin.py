import os
from tkinter import *
import cv2
from PIL import Image, ImageTk
from subprocess import Popen, PIPE
from typing import Union, Optional, Tuple, Callable
try:
    from typing import Literal
except:
    from typing_extensions import Literal
from simba.mixins.config_reader import ConfigReader
from simba.utils.read_write import (get_fn_ext, read_df, write_df, get_video_meta_data, get_all_clf_names, find_time_stamp_from_frame_numbers)
from simba.utils.enums import Labelling
from simba.utils.checks import check_int
from simba.utils.errors import FrameRangeError
from simba.ui.tkinter_functions import Entry_Box
from simba.video_processors.video_processing import clip_video_in_range
from simba.utils.printing import SimbaTimer, stdout_success


MAX_FRM_SIZE = 1280, 650

class AnnotatorMixin(ConfigReader):

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 video_path: Union[str, os.PathLike],
                 data_path: Union[str, os.PathLike],
                 frame_size: Optional[Tuple[int]] = MAX_FRM_SIZE):

        ConfigReader.__init__(self, config_path=config_path)
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        self.video_meta_data = get_video_meta_data(video_path=self.video_path)
        self.frame_lst = list(range(0, self.video_meta_data['frame_count']))
        _, self.video_name, _ = get_fn_ext(filepath=video_path)
        _, self.pixels_per_mm, self.fps = self.read_video_info(video_name=self.video_name)
        self.target_lst = get_all_clf_names(config=self.config, target_cnt=self.clf_cnt)
        self.data_df = read_df(file_path=data_path, file_type=self.file_type)
        self.max_frm_no, self.min_frm_no  = max(self.data_df.index), min(self.data_df.index)
        self.main_frm = Toplevel()
        self.max_size = frame_size
        self.main_frm.title(f'SIMBA CLIP ANNOTATOR - {self.video_name}')

    def video_frm_label(self,
                        frm_number: int,
                        max_size: Optional[Tuple[int, int]] = None,
                        loc: Tuple[int] = (0, 0)) -> None:

        """
        Inserts a video frame as a tkinter label fo specified maximum size at specified grid location

        .. image:: _static/img/video_frm_label.png
           :width: 500
           :align: center
        """

        if max_size is None: max_size = self.max_size
        self.cap.set(1, frm_number)
        _, frm = self.cap.read()
        frm = cv2.cvtColor(frm, cv2.COLOR_RGB2BGR)
        frm = Image.fromarray(frm)
        frm.thumbnail(max_size, Image.ANTIALIAS)
        frm = ImageTk.PhotoImage(master=self.main_frm, image=frm)
        video_frame = Label(self.main_frm, image=frm)
        video_frame.image = frm
        video_frame.grid(row=loc[0], column=loc[1], sticky=NW)


    def h_nav_bar(self, parent: Frame,
                  change_frm_func: Callable[[int], None],
                  width: int = 700,
                  height: int = 300) -> Frame:

        """
        Create a horizontal frame navigation bar where the buttons callbacks are tied to specified ``change_frm_func`` callback.

        .. image:: _static/img/h_nav_bar.png
           :width: 500
           :align: center
        """
        out_frm = Frame(parent, bd=2, width=width, height=height)
        nav_frm = Frame(out_frm)
        jump_frame = Frame(out_frm, bd=2)
        jump_lbl = Label(jump_frame, text="JUMP SIZE:")
        jump_size = Scale(jump_frame, from_=0, to=100, orient=HORIZONTAL, length=200)
        jump_size.set(0)
        self.frm_box = Entry_Box(nav_frm, 'FRAME NUMBER', '15')
        self.frm_box.entry_set(val=self.min_frm_no)
        forward_btn = Button(nav_frm, text=">", command=lambda: change_frm_func(int(self.frm_box.entry_get)+1))
        backward_btn = Button(nav_frm, text="<", command=lambda: change_frm_func(int(self.frm_box.entry_get)-1))
        forward_max_btn = Button(nav_frm, text=">>", command= lambda: change_frm_func(self.max_frm_no-1))
        backward_max_btn = Button(nav_frm, text="<<", command= lambda: change_frm_func(self.min_frm_no))
        jump_back = Button(jump_frame, text="<<", command=lambda: change_frm_func(int(self.frm_box.entry_get) - jump_size.get()))
        jump_forward = Button(jump_frame, text=">>", command= lambda: change_frm_func(int(self.frm_box.entry_get) + jump_size.get()))
        select_frm_btn = Button(nav_frm, text="VIEW SELECTED FRAME", command=change_frm_func(int(self.frm_box.entry_get)))

        nav_frm.grid(row=0, column=0)
        self.frm_box.grid(row=0, column=0)
        backward_max_btn.grid(row=0, column=1, sticky=NE, padx=Labelling.PADDING.value)
        backward_btn.grid(row=0, column=2, sticky=NE, padx=Labelling.PADDING.value)
        forward_btn.grid(row=0, column=3, sticky=NE, padx=Labelling.PADDING.value)
        forward_max_btn.grid(row=0, column=4, sticky=NE, padx=Labelling.PADDING.value)
        select_frm_btn.grid(row=1, column=0, sticky=NE, padx=Labelling.PADDING.value)

        jump_frame.grid(row=1, column=0)
        jump_lbl.grid(row=0, column=0, sticky=NE)
        jump_size.grid(row=0, column=1, sticky=SE)
        jump_back.grid(row=0, column=2, sticky=SE)
        jump_forward.grid(row=0, column=3, sticky=SE)

        return out_frm


    def v_navigation_pane_targeted_clips_version(self, parent: Frame) -> Frame:
        """ Create a vertical navigation pane for opening video and displaying keyboard shortcuts"""

        def bind_shortcut_keys(parent):
            parent.bind('<Control-s>', lambda x: self.__targeted_clips_save())
            parent.bind('<Right>', lambda x: self.change_frame_targeted_annotations(int(self.frm_box.entry_get)+1))
            parent.bind('<Left>', lambda x: self.change_frame_targeted_annotations(int(self.frm_box.entry_get)-1))
            parent.bind('<Control-l>', lambda x: self.change_frame_targeted_annotations(self.max_frm_no-1))
            parent.bind('<Control-o>', lambda x: self.change_frame_targeted_annotations(self.min_frm_no))

        video_player_frm = Frame(parent)
        play_video_btn = Button(video_player_frm, text='OPEN VIDEO', command= lambda: self.__play_video())
        play_video_btn.grid(sticky=N, pady=10)
        Label(video_player_frm, text='\n\n  Keyboard shortcuts for video navigation: \n p = Pause/Play'
                                                        '\n\n After pressing pause:'
                                                        '\n o = +2 frames \n e = +10 frames \n w = +1 second'
                                                        '\n\n t = -2 frames \n s = -10 frames \n x = -1 second'
                                                        '\n\n q = Close video window \n\n').grid(sticky=W)
        update_img_from_video = Button(video_player_frm, text='Show current video frame', command=None)
        update_img_from_video.grid(sticky=N)

        key_presses_lbl = Label(video_player_frm,
                            text='\n\n Keyboard shortcuts for frame navigation: \n Right Arrow = +1 frame'
                                 '\n Left Arrow = -1 frame'
                                 '\n Ctrl + s = Save annotations file'
                                 '\n Ctrl + l = Last frame'
                                 '\n Ctrl + o = First frame')
        key_presses_lbl.grid(sticky=S)
        bind_shortcut_keys(parent)
        return video_player_frm


    def targeted_clips_pane(self, parent: Frame) -> Frame:
        """ Create a pane for choosing bouts start and end and a radiobutton truth table for targeted clips annotations"""
        def set_selection():
            selection_txt.config(text=f"CURRENT SELECTION START: {start_frm_bx.entry_get}, END: {end_frm_bx.entry_get}")
            save_button.config(text=f"SAVE ANNOTATION CLIP \n START FRAME: {start_frm_bx.entry_get}, \n END FRAME: {end_frm_bx.entry_get}")
            self.set_start, self.set_end = start_frm_bx.entry_get, end_frm_bx.entry_get

        pane, self.set_start, self.set_end = Frame(parent, bd=2), 0, 1
        selection_txt = Label(pane, text=f"CURRENT SELECTION START: {0}, END: {1}", font=("Helvetica", 14, "bold"))
        start_frm_bx = Entry_Box(pane, 'SELECT START FRAME:', '20', validation='numeric', entry_box_width=10)
        start_frm_bx.entry_set('0')
        end_frm_bx = Entry_Box(pane, 'SELECT END FRAME:', '20', validation='numeric', entry_box_width=10)
        end_frm_bx.entry_set('1')
        confirm_btn = Button(pane, text="SET SELECTION", command= lambda: set_selection())
        selection_txt.grid(row=0, column=0, pady=30, sticky=N)
        start_frm_bx.grid(row=1, column=0, pady=30, sticky=S)
        end_frm_bx.grid(row=2, column=0, pady=10, sticky=S)
        confirm_btn.grid(row=3, column=0, sticky=S)

        Label(pane, text=f"PRESENT", font=("Helvetica", 12, "bold")).grid(row=4, column=1, pady=20, sticky=NW)
        Label(pane, text=f"ABSENT", font=("Helvetica", 12, "bold")).grid(row=4, column=2, pady=20, sticky=NW)
        self.clf_radio_btns = {}
        for target_cnt, target in enumerate(self.target_lst):
            self.clf_radio_btns[target] = StringVar(value=0)
            Label(pane, text=target, font=("Helvetica", 12, "bold")).grid(row=target_cnt+5, column=0, sticky=NW)
            present = Radiobutton(pane, text=None, variable=self.clf_radio_btns[target], value=1, command=None)
            absent = Radiobutton(pane, text=None, variable=self.clf_radio_btns[target], value=0, command=None)
            present.grid(row=target_cnt+5, column=1, pady=5, sticky=NW)
            absent.grid(row=target_cnt+5, column=2, pady=5, sticky=NW)

        save_button = Button(pane, text=f"SAVE ANNOTATION CLIP \n START FRAME: {start_frm_bx.entry_get}, \n END FRAME: {end_frm_bx.entry_get}", fg='green', font=("Helvetica", 14, "bold"), command= lambda: self.__targeted_clips_save())
        save_button.grid(row=5+len(self.target_lst), column=0, pady=40, sticky=S)

        return pane

    def __play_video(self):
        """Helper to play video when ``play video button`` is pressed """
        p = Popen('python {}'.format(Labelling.PLAY_VIDEO_SCRIPT_PATH.value), stdin=PIPE, stdout=PIPE, shell=True)
        p.stdin.write(bytes(self.video_path, 'utf-8'))
        p.stdin.close()
        temp_file = os.path.join(os.path.dirname(self.config_path), 'subprocess.txt')
        with open(temp_file, "w") as text_file: text_file.write(str(p.pid))

    def __targeted_clips_save(self):
        """Helper method called by targeted_clips_pane to save video and dataframe sliced during targeted clips annotations"""
        timer = SimbaTimer(start=True)
        print('Saving video clip...')
        start_idx, end_idx = int(self.set_start), int(self.set_end)
        if start_idx >= end_idx: raise FrameRangeError(msg=f'Start frame ({start_idx}) has to be before end frame ({end_idx}).', source=self.__class__.__name__)
        if (start_idx > self.max_frm_no) or (start_idx < self.min_frm_no) or (end_idx > self.max_frm_no) or (end_idx < self.min_frm_no):
            raise FrameRangeError(msg=f'Start frame ({start_idx}) or end frame ({end_idx}) cannot be smaller or larger than the minimum ({self.min_frm_no}) and maximum ({self.max_frm_no}) frame of the video', source=self.__class__.__name__)
        timestamps = find_time_stamp_from_frame_numbers(start_frame=start_idx, end_frame=end_idx, fps=self.video_meta_data['fps'])
        clip_video_in_range(file_path=self.video_path, start_time=timestamps[0], end_time=timestamps[1], out_dir=self.video_dir, include_clip_time_in_filename=True, overwrite=True)
        df_filename = os.path.join(self.video_name + f'_{timestamps[0].replace(":", "-")}_{timestamps[1].replace(":", "-")}.{self.file_type}')
        video_name = os.path.join(self.video_name + f'_{timestamps[0].replace(":", "-")}_{timestamps[1].replace(":", "-")}')
        print(f'Saving targets data {df_filename}...')
        df = self.data_df.iloc[start_idx:end_idx, :]
        for clf, btn in self.clf_radio_btns.items():
            df[clf] = btn.get()
        write_df(df=df, file_type=self.file_type, save_path=os.path.join(self.targets_folder, df_filename))
        self.add_video_to_video_info(video_name=video_name, fps=self.fps, width=self.video_meta_data['width'], height=self.video_meta_data['height'], pixels_per_mm=self.pixels_per_mm)
        timer.stop_timer()
        stdout_success(msg=f'Annotated clip {df_filename} saved into SimBA project', elapsed_time=timer.elapsed_time_str)


    def change_frame_targeted_annotations(self,
                                          new_frm_id: int):
        print(new_frm_id)
        check_int(name='FRAME NUMBER', value=new_frm_id)
        if new_frm_id > (self.max_frm_no -1):
            self.frm_box.entry_set(val=self.max_frm_no)
            raise FrameRangeError(msg=f"FRAME {new_frm_id} CANNOT BE SHOWN - YOU ARE VIEWING THE FINAL FRAME OF THE VIDEO (FRAME NUMBER {self.max_frm_no})", source=self.__class__.__name__)
        elif new_frm_id < 0:
            self.frm_box.entry_set(val=self.min_frm_no)
            raise FrameRangeError(msg=f"FRAME {new_frm_id} CANNOT BE SHOWN - YOU ARE VIEWING THE FIRST FRAME OF THE VIDEO (FRAME NUMBER {self.min_frm_no})", source=self.__class__.__name__)
        else:
            self.frm_box.entry_set(val=new_frm_id)
            self.video_frm_label(frm_number=int(new_frm_id))

from tkinter import *
import glob, os
from simba.misc_tools import get_fn_ext, get_video_meta_data
import tkinter.ttk as ttk
import platform
import datetime
import cv2
from simba.read_config_unit_tests import check_file_exist_and_readable
import json
from simba.batch_process_videos.batch_process_create_ffmpeg_commands import FFMPEGCommandCreator

def onMousewheel(event, canvas):
    try:
        scrollSpeed = event.delta
        if platform.system() == 'Darwin':
            scrollSpeed = event.delta
        elif platform.system() == 'Windows':
            scrollSpeed = int(event.delta / 120)
        canvas.yview_scroll(-1 * (scrollSpeed), "units")
    except:
        pass

def bindToMousewheel(event, canvas):
    canvas.bind_all("<MouseWheel>", lambda event: onMousewheel(event, canvas))

def unbindToMousewheel(event, canvas):
    canvas.unbind_all("<MouseWheel>")

def onFrameConfigure(canvas):
    canvas.configure(scrollregion=canvas.bbox("all"))

def hxtScrollbar(master):
    bg = master.cget("background")
    acanvas = Canvas(master, borderwidth=0, background=bg)
    frame = Frame(acanvas, background=bg)
    vsb = Scrollbar(master, orient="vertical", command=acanvas.yview)
    vsb2 = Scrollbar(master, orient='horizontal', command=acanvas.xview)
    acanvas.configure(yscrollcommand=vsb.set)
    acanvas.configure(xscrollcommand=vsb2.set)
    vsb.pack(side="right", fill="y")
    vsb2.pack(side="bottom", fill="x")
    acanvas.pack(side="left", fill="both", expand=True)

    acanvas.create_window((10, 10), window=frame, anchor="nw")
    acanvas.bind("<Configure>", lambda event, canvas=acanvas: onFrameConfigure(acanvas))
    acanvas.bind('<Enter>', lambda event: bindToMousewheel(event, acanvas))
    acanvas.bind('<Leave>', lambda event: unbindToMousewheel(event,acanvas))
    return frame

class BatchProcessFrame(object):
    """
    Class for creating interactive windows that collect user-inputs for batch processing videos (e.g., cropping,
    clipping etc.). User-selected output is stored in json file format within the user-defined `output_dir`

    Parameters
    ----------
    input_dir: str
        Input folder path containing videos for bath processing.
    output_dir: str
        Output folder path for where to store the processed videos.

    Notes
    ----------
    `Batch pre-process tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial_process_videos.md>`__.

    Examples
    ----------
    >>> batch_preprocessor = BatchProcessFrame(input_dir=r'MyInputVideosDir', output_dir=r'MyOutputVideosDir')
    >>> batch_preprocessor.create_main_window()
    >>> batch_preprocessor.create_video_table_headings()
    >>> batch_preprocessor.create_video_rows()
    >>> batch_preprocessor.create_execute_btn()
    >>> batch_preprocessor.batch_process_main_frame.mainloop()

    """




    def __init__(self,
                 input_dir: str,
                 output_dir: str):

        self.input_dir = input_dir
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.videos_in_dir_dict, self.crop_dict = {}, {}
        self.get_input_files()
        if len(self.input_dir) == 0:
            print('SIMBA WARNING: The input directory {} contains ZERO video files in either .avi, .mp4, .mov, .flv, or m4v format'.format(self.input_dir))
        else:
            self.max_char_vid_name = len(max(list(self.videos_in_dir_dict.keys()), key=len))



    def get_input_files(self):
        for file_path in glob.glob(self.input_dir + '/*'):
            lower_str_name = file_path.lower()
            if lower_str_name.endswith(('.avi', '.mp4', '.mov', '.flv', '.m4v')):
                _, video_name, ext = get_fn_ext(file_path)
                self.videos_in_dir_dict[video_name] = get_video_meta_data(file_path)
                self.videos_in_dir_dict[video_name]['extension'] = ext
                self.videos_in_dir_dict[video_name]['video_length'] = str(datetime.timedelta(seconds=int(self.videos_in_dir_dict[video_name]['frame_count'] / self.videos_in_dir_dict[video_name]['fps'])))
                self.videos_in_dir_dict[video_name]['video_length'] = '0' + self.videos_in_dir_dict[video_name]['video_length']
                self.videos_in_dir_dict[video_name]['file_path'] = file_path

    def create_main_window(self):
        self.batch_process_main_frame = Toplevel()
        self.batch_process_main_frame.minsize(1600, 600)
        self.batch_process_main_frame.wm_title("BATCH PRE-PROCESS VIDEOS IN SIMBA")
        self.batch_process_main_frame.lift()
        self.batch_process_main_frame = Canvas(hxtScrollbar(self.batch_process_main_frame))
        self.batch_process_main_frame.pack(fill="both", expand=True)
        self.quick_settings_frm = LabelFrame(self.batch_process_main_frame, text='QUICK SETTINGS', font=('Helvetica', 15, 'bold'), pady=5, padx=15)
        self.clip_video_settings_frm = LabelFrame(self.quick_settings_frm, text='Clip Videos Settings', padx=5)
        self.quick_clip_start_entry_lbl = Label(self.clip_video_settings_frm, text="Start Time: ")
        self.quick_clip_start_entry_box_val = StringVar()
        self.quick_clip_start_entry_box_val.set('00:00:00')
        self.quick_clip_start_entry_box = Entry(self.clip_video_settings_frm, width=15, textvariable=self.quick_clip_start_entry_box_val)
        self.quick_clip_end_entry_lbl = Label(self.clip_video_settings_frm, text="End Time: ")
        self.quick_clip_end_entry_box_val = StringVar()
        self.quick_clip_end_entry_box_val.set('00:00:00')
        self.quick_clip_end_entry_box = Entry(self.clip_video_settings_frm, width=15, textvariable=self.quick_clip_end_entry_box_val)
        self.quick_clip_apply = Button(self.clip_video_settings_frm, text='Apply', command=lambda: self.apply_trim_to_all())
        self.quick_downsample_frm = LabelFrame(self.quick_settings_frm,text='Downsample Videos',padx=5)
        self.quick_downsample_width_lbl = Label(self.quick_downsample_frm, text="Width: ")
        self.quick_downsample_width_val = IntVar()
        self.quick_downsample_width_val.set(400)
        self.quick_downsample_width = Entry(self.quick_downsample_frm, width=15, textvariable=self.quick_downsample_width_val)
        self.quick_downsample_height_lbl = Label(self.quick_downsample_frm, text="Height: ")
        self.quick_downsample_height_val = IntVar()
        self.quick_downsample_height_val.set(600)
        self.quick_downsample_height = Entry(self.quick_downsample_frm, width=15, textvariable=self.quick_downsample_height_val)
        self.quick_downsample_apply = Button(self.quick_downsample_frm,text='Apply',command=lambda: self.apply_resolution_to_all())
        self.quick_set_fps = LabelFrame(self.quick_settings_frm,text='Change FPS', padx=5)
        self.quick_fps_lbl = Label(self.quick_set_fps, text="FPS: ")
        self.quick_set_fps_val = IntVar()
        self.quick_set_fps_val.set(15)
        self.quick_fps_entry_box = Entry(self.quick_set_fps, width=15, textvariable=self.quick_set_fps_val)
        self.quick_set_fps_empty_row = Label(self.quick_set_fps, text=" ")
        self.quick_fps_apply = Button(self.quick_set_fps,text='Apply',command=lambda: self.apply_fps_to_all())
        self.quick_settings_frm.grid(row=0, column=0, sticky=W,padx=10)
        self.clip_video_settings_frm.grid(row=0, column=0, sticky=W)
        self.quick_clip_start_entry_lbl.grid(row=0, column=0, sticky=W)
        self.quick_clip_start_entry_box.grid(row=0, column=1, sticky=W)
        self.quick_clip_end_entry_lbl.grid(row=1, column=0, sticky=W)
        self.quick_clip_end_entry_box.grid(row=1, column=1, sticky=W)
        self.quick_clip_apply.grid(row=2, column=0)
        self.quick_downsample_frm.grid(row=0,column=1,sticky=W)
        self.quick_downsample_width_lbl.grid(row=0,column=0,sticky=W)
        self.quick_downsample_width.grid(row=0, column=1, sticky=W)
        self.quick_downsample_height_lbl.grid(row=1, column=0, sticky=W)
        self.quick_downsample_height.grid(row=1, column=1, sticky=W)
        self.quick_downsample_apply.grid(row=2, column=0, sticky=W)
        self.quick_set_fps.grid(row=0,column=2,sticky=W)
        self.quick_fps_lbl.grid(row=0, column=0, sticky=W)
        self.quick_fps_entry_box.grid(row=0, column=1, sticky=W)
        self.quick_set_fps_empty_row.grid(row=1, column=1, sticky=W)
        self.quick_fps_apply.grid(row=2, column=0, sticky=W)

    def inverse_all_cb_ticks(self, variable_name=None):
        if self.headings[variable_name].get():
            for video_name in self.videos.keys():
                self.videos[video_name][variable_name].set(True)
        if not self.headings[variable_name].get():
            for video_name in self.videos.keys():
                self.videos[video_name][variable_name].set(False)

    def apply_resolution_to_all(self):
        for video_name in self.videos.keys():
            self.videos[video_name]['width_var'].set(self.quick_downsample_width_val.get())
            self.videos[video_name]['height_var'].set(self.quick_downsample_height_val.get())

    def apply_trim_to_all(self):
        for video_name in self.videos.keys():
            self.videos[video_name]['start_time_var'].set(self.quick_clip_start_entry_box.get())
            self.videos[video_name]['end_time_var'].set(self.quick_clip_end_entry_box.get())

    def apply_fps_to_all(self):
        for video_name in self.videos.keys():
            self.videos[video_name]['fps_var'].set(self.quick_set_fps_val.get())

    def create_video_table_headings(self):
        self.headings = {}
        self.videos_frm = LabelFrame(self.batch_process_main_frame, text='VIDEOS', font=('Helvetica', 15, 'bold'), pady=5, padx=15)
        self.headings['video_name_col_head'] = Label(self.videos_frm, text='Video Name', width=self.max_char_vid_name + 5)
        self.headings['crop_video_col_head'] = Label(self.videos_frm, text='Crop Video', width=8)
        self.headings['start_time_col_head'] = Label(self.videos_frm, text='Start Time', width=8)
        self.headings['end_time_col_head'] = Label(self.videos_frm, text='End Time', width=8)
        self.headings['clip_cb_var'] = BooleanVar()
        self.headings['shorten_all_videos_cbox'] = Checkbutton(self.videos_frm, text='Clip All Videos',variable=self.headings['clip_cb_var'], command=lambda: self.inverse_all_cb_ticks(variable_name='clip_cb_var'))
        self.headings['video_width_col_head'] = Label(self.videos_frm, text='Video Width', width=8)
        self.headings['video_height_col_head'] = Label(self.videos_frm, text='Video Height', width=8)
        self.headings['downsample_cb_var'] = BooleanVar()
        self.headings['downsample_all_videos_cbox'] = Checkbutton(self.videos_frm,text='Downsample All Videos', variable=self.headings['downsample_cb_var'], command=lambda: self.inverse_all_cb_ticks(variable_name='downsample_cb_var'))
        self.headings['fps_col_head'] = Label(self.videos_frm, text='Video FPS', width=8)
        self.headings['fps_cb_var'] = BooleanVar()
        self.headings['change_fps_all_videos_cbox'] = Checkbutton(self.videos_frm, text='Change FPS', variable=self.headings['fps_cb_var'], command=lambda: self.inverse_all_cb_ticks(variable_name='fps_cb_var'))
        self.headings['grayscale_cb_var'] = BooleanVar()
        self.headings['grayscale_cbox'] = Checkbutton(self.videos_frm,text='Apply Greyscale',variable=self.headings['grayscale_cb_var'], command=lambda: self.inverse_all_cb_ticks(variable_name='grayscale_cb_var'))
        self.headings['frame_cnt_cb_var'] = BooleanVar()
        self.headings['frame_cnt_cbox'] = Checkbutton(self.videos_frm,text='Print Frame Count',variable=self.headings['frame_cnt_cb_var'], command=lambda: self.inverse_all_cb_ticks(variable_name='frame_cnt_cb_var'))
        self.headings['apply_clahe_cb_var'] = BooleanVar()
        self.headings['apply_clahe_cbox'] = Checkbutton(self.videos_frm,text='Apply CLAHE',variable=self.headings['apply_clahe_cb_var'], command=lambda: self.inverse_all_cb_ticks(variable_name='apply_clahe_cb_var'))

        self.videos_frm.grid(row=1, column=0, sticky=W, padx=5, pady=15)
        self.headings['video_name_col_head'].grid(row=0, column=1, sticky=W, padx=5)
        self.headings['crop_video_col_head'].grid(row=0, column=2, sticky=W, padx=5)
        self.headings['start_time_col_head'].grid(row=0, column=3, sticky=W, padx=5)
        self.headings['end_time_col_head'].grid(row=0, column=4, sticky=W, padx=5)
        self.headings['shorten_all_videos_cbox'].grid(row=0, column=5, sticky=W, padx=5)
        self.headings['video_width_col_head'].grid(row=0, column=6, sticky=W, padx=5)
        self.headings['video_height_col_head'].grid(row=0, column=7, sticky=W, padx=5)
        self.headings['downsample_all_videos_cbox'].grid(row=0, column=8, sticky=W, padx=5)
        self.headings['fps_col_head'].grid(row=0, column=9, sticky=W, padx=5)
        self.headings['change_fps_all_videos_cbox'].grid(row=0, column=10, sticky=W, padx=5)
        self.headings['grayscale_cbox'].grid(row=0, column=11, sticky=W, padx=5)
        self.headings['frame_cnt_cbox'].grid(row=0, column=12, sticky=W, padx=5)
        self.headings['apply_clahe_cbox'].grid(row=0, column=13, sticky=W, padx=5)

    def create_video_rows(self):
        self.videos = {}
        for video_cnt, (name, data) in enumerate(self.videos_in_dir_dict.items()):
            self.videos[name] = {}
            row = video_cnt + 1
            self.videos[name]['video_name_lbl'] = Label(self.videos_frm, text=name, width=self.max_char_vid_name + 5)
            self.videos[name]['crop_btn'] = Button(self.videos_frm, text='Crop', fg='black', command=lambda k=self.videos[name]['video_name_lbl']['text']: self.batch_process_crop_function(k))
            self.videos[name]['start_time_var'] = StringVar()
            self.videos[name]['start_time_var'].set('00:00:00')
            self.videos[name]['start_entry'] = Entry(self.videos_frm, width=6, textvariable=self.videos[name]['start_time_var'])
            self.videos[name]['end_time_var'] = StringVar()
            self.videos[name]['end_time_var'].set(data['video_length'])
            self.videos[name]['end_entry'] = Entry(self.videos_frm, width=6, textvariable=self.videos[name]['end_time_var'])
            self.videos[name]['clip_cb_var'] = BooleanVar()
            self.videos[name]['clip_cb'] = Checkbutton(self.videos_frm, variable=self.videos[name]['clip_cb_var'], command=None)
            self.videos[name]['width_var'] = IntVar()
            self.videos[name]['width_var'].set(data['width'])
            self.videos[name]['width_entry']=Entry(self.videos_frm, width=6, textvariable=self.videos[name]['width_var'])
            self.videos[name]['height_var'] = IntVar()
            self.videos[name]['height_var'].set(data['height'])
            self.videos[name]['height_entry']=Entry(self.videos_frm, width=6, textvariable=self.videos[name]['height_var'])
            self.videos[name]['downsample_cb_var'] = BooleanVar()
            self.videos[name]['downsample_cb'] = Checkbutton(self.videos_frm, variable=self.videos[name]['downsample_cb_var'], command=None)
            self.videos[name]['fps_var'] = IntVar()
            self.videos[name]['fps_var'].set(data['fps'])
            self.videos[name]['fps_entry'] = Entry(self.videos_frm, width=6, textvariable=self.videos[name]['fps_var'])
            self.videos[name]['fps_cb_var'] = BooleanVar()
            self.videos[name]['fps_cb'] = Checkbutton(self.videos_frm, variable=self.videos[name]['fps_cb_var'], command=None)
            self.videos[name]['grayscale_cb_var'] = BooleanVar()
            self.videos[name]['grayscale_cbox'] = Checkbutton(self.videos_frm, variable=self.videos[name]['grayscale_cb_var'], command=None)
            self.videos[name]['frame_cnt_cb_var'] = BooleanVar()
            self.videos[name]['frame_cnt_cbox'] = Checkbutton(self.videos_frm, variable=self.videos[name]['frame_cnt_cb_var'], command=None)
            self.videos[name]['apply_clahe_cb_var'] = BooleanVar()
            self.videos[name]['apply_clahe_cbox'] = Checkbutton(self.videos_frm, variable=self.videos[name]['apply_clahe_cb_var'], command=None)

            self.videos[name]['video_name_lbl'].grid(row=row, column=1, sticky=W, padx=5)
            self.videos[name]['crop_btn'].grid(row=row, column=2, padx=5)
            self.videos[name]['start_entry'].grid(row=row, column=3, padx=5)
            self.videos[name]['end_entry'].grid(row=row, column=4, padx=5)
            self.videos[name]['clip_cb'].grid(row=row, column=5, sticky=W, padx=5)
            self.videos[name]['width_entry'].grid(row=row, column=6, padx=5)
            self.videos[name]['height_entry'].grid(row=row, column=7, padx=5)
            self.videos[name]['downsample_cb'].grid(row=row, column=8, sticky=W, padx=5)
            self.videos[name]['fps_entry'].grid(row=row, column=9, padx=5)
            self.videos[name]['fps_cb'].grid(row=row, column=10, sticky=W, padx=5)
            self.videos[name]['grayscale_cbox'].grid(row=row, column=11, sticky=W, padx=5)
            self.videos[name]['frame_cnt_cbox'].grid(row=row, column=12, sticky=W, padx=5)
            self.videos[name]['apply_clahe_cbox'].grid(row=row, column=13, sticky=W, padx=5)

    def create_execute_btn(self):
        self.execute_frm = LabelFrame(self.batch_process_main_frame, text='EXECUTE', font=('Helvetica', 15, 'bold'), pady=5, padx=15)
        self.reset_all_btn = Button(self.execute_frm , text='RESET ALL', fg='red', command=lambda: self.create_video_rows())
        self.reset_crop_btn = Button(self.execute_frm, text='RESET CROP', fg='red', command=lambda: self.reset_crop())
        self.execute_btn = Button(self.execute_frm, text='EXECUTE', fg='red', command= lambda: self.execute())

        self.execute_frm.grid(row=2, column=0, sticky=W, padx=5, pady=30)
        self.reset_all_btn.grid(row=0, column=0, sticky=W, padx=5)
        self.reset_crop_btn.grid(row=0, column=1, sticky=W, padx=5)
        self.execute_btn.grid(row=0, column=2, sticky=W, padx=5)

    def reset_crop(self):
        self.crop_dict = {}
        for video_name, video_data in self.videos_in_dir_dict.items():
            self.videos[video_name]['crop_btn'].configure(fg="black")

    def batch_process_crop_function(self, video_name):
        check_file_exist_and_readable(self.videos_in_dir_dict[video_name]['file_path'])
        self.cap = cv2.VideoCapture(self.videos_in_dir_dict[video_name]['file_path'])
        self.cap.set(1, 0)
        _, self.frame = self.cap.read()
        cv2.namedWindow('CROP {}'.format(video_name), cv2.WINDOW_NORMAL)
        cv2.imshow('CROP {}'.format(video_name), self.frame)
        ROI = cv2.selectROI('CROP {}'.format(video_name), self.frame)
        self.crop_dict[video_name] = {}
        self.crop_dict[video_name]['top_left_x'] = ROI[0]
        self.crop_dict[video_name]['top_left_y'] = ROI[1]
        self.crop_dict[video_name]['width'] = (abs(ROI[0] - (ROI[2] + ROI[0])))
        self.crop_dict[video_name]['height'] = (abs(ROI[2] - (ROI[3] + ROI[2])))
        self.crop_dict[video_name]['bottom_right_x'] = self.crop_dict[video_name]['top_left_x'] + self.crop_dict[video_name]['width']
        self.crop_dict[video_name]['bottom_right_y'] = self.crop_dict[video_name]['top_left_y'] + self.crop_dict[video_name]['height']
        k = cv2.waitKey(20) & 0xFF
        cv2.destroyAllWindows()
        self.videos[video_name]['crop_btn'].configure(fg="red")

    def execute(self):
        out_video_dict = {}
        out_video_dict['meta_data'] = {}
        out_video_dict['video_data'] = {}
        out_video_dict['meta_data']['in_dir'] = self.input_dir
        out_video_dict['meta_data']['out_dir'] = self.output_dir
        for video_cnt, (name, data) in enumerate(self.videos_in_dir_dict.items()):
            out_video_dict['video_data'][name] = {}
            out_video_dict['video_data'][name]['video_info'] = self.videos_in_dir_dict[name]
            if name in self.crop_dict.keys():
                out_video_dict['video_data'][name]['crop'] = True
                out_video_dict['video_data'][name]['crop_settings'] = self.crop_dict[name]
            else:
                out_video_dict['video_data'][name]['crop'] = False
                out_video_dict['video_data'][name]['crop_settings'] = None
            if self.videos[name]['clip_cb_var'].get():
                out_video_dict['video_data'][name]['clip'] = True
                out_video_dict['video_data'][name]['clip_settings'] = {'start': self.videos[name]['start_time_var'].get(), 'stop':  self.videos[name]['end_time_var'].get()}
            else:
                out_video_dict['video_data'][name]['clip'] = False
                out_video_dict['video_data'][name]['clip_settings'] = None
            if self.videos[name]['downsample_cb_var'].get():
                out_video_dict['video_data'][name]['downsample'] = True
                out_video_dict['video_data'][name]['downsample_settings'] = {'width': self.videos[name]['width_var'].get(), 'height': self.videos[name]['height_var'].get()}
            else:
                out_video_dict['video_data'][name]['downsample'] = False
                out_video_dict['video_data'][name]['downsample_settings'] = None
            if self.videos[name]['fps_cb_var'].get():
                out_video_dict['video_data'][name]['fps'] = True
                out_video_dict['video_data'][name]['fps_settings'] = {'fps': self.videos[name]['fps_var'].get()}
            else:
                out_video_dict['video_data'][name]['fps'] = False
                out_video_dict['video_data'][name]['fps_settings'] = None
            if self.videos[name]['grayscale_cb_var'].get():
                out_video_dict['video_data'][name]['grayscale'] = True
                out_video_dict['video_data'][name]['grayscale_settings'] = None
            else:
                out_video_dict['video_data'][name]['grayscale'] = False
                out_video_dict['video_data'][name]['grayscale_settings'] = None
            if self.videos[name]['frame_cnt_cb_var'].get():
                out_video_dict['video_data'][name]['frame_cnt'] = True
                out_video_dict['video_data'][name]['frame_cnt_settings'] = None
            else:
                out_video_dict['video_data'][name]['frame_cnt'] = False
                out_video_dict['video_data'][name]['frame_cnt_settings'] = None
            if self.videos[name]['apply_clahe_cb_var'].get():
                out_video_dict['video_data'][name]['clahe'] = True
                out_video_dict['video_data'][name]['clahe_settings'] = None
            else:
                out_video_dict['video_data'][name]['clahe'] = False
                out_video_dict['video_data'][name]['clahe_settings'] = None


        self.save_path = os.path.join(self.output_dir, 'batch_process_log.json')
        with open(self.save_path, 'w') as fp:
            json.dump(out_video_dict, fp)
        self.perform_unit_tests(out_video_dict['video_data'])

    def perform_unit_tests(self, out_video_dict):
        for video_name, video_data in out_video_dict.items():
            if video_data['crop']:
                if not isinstance(video_data['crop_settings']['width'], int):
                    print('SIMBA ERROR: Crop width for video {} is not an integer'.format(video_name))
                    raise ValueError
                if not isinstance(video_data['crop_settings']['height'], int):
                    print('SIMBA ERROR: Crop height for video {} is not an integer'.format(video_name))
                    raise ValueError
            if video_data['clip']:
                r = re.compile('.{2}:.{2}:.{2}')
                for variable in ['start', 'stop']:
                    if len(video_data['clip_settings'][variable]) != 8:
                        print('SIMBA ERROR: Clip {} time for video {} is should be in the format XX:XX:XX where X is an integer between 0-9'.format(variable, video_name))
                    elif not r.match(video_data['clip_settings'][variable]):
                        print('SIMBA ERROR: Clip {} time for video {} is should be in the format XX:XX:XX where X is an integer between 0-9'.format(variable, video_name))
                    elif re.search('[a-zA-Z]', video_data['clip_settings'][variable]):
                        print('SIMBA ERROR: Clip {} time for video {} is should be in the format XX:XX:XX where X is an integer between 0-9'.format(variable, video_name))
            if video_data['downsample']:
                if not isinstance(video_data['downsample_settings']['width'], int):
                    print('SIMBA ERROR: Downsample width for video {} is not an integer'.format(video_name))
                if not isinstance(video_data['downsample_settings']['height'], int):
                    print('SIMBA ERROR: Downsample height for video {} is not an integer'.format(video_name))
            if video_data['fps']:
                if not isinstance(video_data['fps_settings']['fps'], int):
                    print('SIMBA ERROR: FPS settings for video {} is not an integer'.format(video_name))



        ffmpeg_runner = FFMPEGCommandCreator(json_path=self.save_path)
        ffmpeg_runner.crop_videos()
        ffmpeg_runner.clip_videos()
        ffmpeg_runner.downsample_videos()
        ffmpeg_runner.apply_fps()
        ffmpeg_runner.apply_grayscale()
        ffmpeg_runner.apply_frame_count()
        ffmpeg_runner.apply_clahe()
        ffmpeg_runner.move_all_processed_files_to_output_folder()
        print('SIMBA batch pre-process JSON saved at {}'.format(self.save_path))
        print('SIMBA COMPLETE: Video batch pre-processing complete, new videos stored in {}'.format(self.output_dir))

# test = BatchProcessFrame(input_dir=r'/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/videos', output_dir=r'/Users/simon/Desktop/test')
# test.create_main_window()
# test.create_video_table_headings()
# test.create_video_rows()
# test.create_execute_btn()
# test.batch_process_main_frame.mainloop()
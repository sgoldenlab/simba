__author__ = "Simon Nilsson", "JJ Choong"

from tkinter import *
import os, glob
import pandas as pd
from simba.misc_tools import (get_fn_ext,
                              get_video_meta_data)
from simba.read_config_unit_tests import (read_config_file,
                                          read_config_entry,
                                          read_project_path_and_file_type)
from simba.features_scripts.unit_tests import (read_video_info_csv,
                                               read_video_info)
from simba.enums import Paths, ReadConfig, Dtypes
from simba.get_coordinates_tools_v2 import get_coordinates_nilsson
from simba.tkinter_functions import hxtScrollbar
import collections

class VideoInfoTable(object):
    """
    Class for creating Tkinter GUI video meta data table. Allows users to modify resolutions, fps, and pixels-per-mm
    interactively of videos within the SimBA project. Data is stored within the project_folder/logs/video_info.csv
    file in the SimBA project.

    Parameters
    ----------
    config_file_path: str
        path to SimBA project config file in Configparser format

    Notes
    -----

    Examples
    -----

    >>> video_info_gui = VideoInfoTable(config_path='MyProjectConfig')
    >>> video_info_gui.create_window()
    >>> video_info_gui.main_frm.mainloop()
    """

    def __init__(self,
                 config_path: str):

        self.config = read_config_file(config_path)
        self.project_path, _ = read_project_path_and_file_type(config=self.config)
        self.logs_path = os.path.join(self.project_path, 'logs')
        self.video_folder = os.path.join(self.project_path, 'videos')
        self.vid_info_path = os.path.join(self.project_path, Paths.VIDEO_INFO.value)

        if os.path.isfile(self.vid_info_path):
            self.vid_info_df = read_video_info_csv(self.vid_info_path).reset_index(drop=True)
        else:
            self.vid_info_df = None
        self.distance_mm = read_config_entry(self.config, ReadConfig.FRAME_SETTINGS.value, ReadConfig.DISTANCE_MM.value, Dtypes.FLOAT.value, 0.00)
        self.find_video_files()
        self.video_basename_lst = []
        self.max_char_vid_name = len(max(self.video_paths))

    def find_video_files(self):
        self.video_paths = []
        file_paths_in_folder = [f for f in glob.glob(self.video_folder + '/*') if os.path.isfile(f)]
        for file_cnt, file_path in enumerate(file_paths_in_folder):
            _, file_name, file_ext = get_fn_ext(file_path)
            if (file_ext.lower() == '.mp4') or (file_ext.lower() == '.avi'):
                self.video_paths.append(file_name + file_ext)
        if len(self.video_paths) == 0:
            print('SIMBA WARNING: No videos in mp4 or avi format was found to be imported to SimBA project')
            raise FileNotFoundError

    def __check_that_value_is_numeric(self, value=None, value_name=None, video_name=None):
        if str(value).isdigit():
            return int(value)
        elif str(value).replace('.','',1).isdigit() and str(value).count('.') < 2:
            return float(value)
        else:
            print('SIMBA ERROR: The {} setting for video {} is set to {} in the project_folder/logs/video_info. Please set it to a numeric value.'.format(value_name, video_name, value))
            raise ValueError

    def __append_videos_from_video_folder(self):
        for cnt, name in enumerate(self.video_paths):
            _, video_basename, _ = get_fn_ext(name)
            self.video_basename_lst.append(video_basename)
            self.videos[name] = {}
            self.videos[name]['video_idx_lbl'] = Label(self.video_frm, text=str(cnt), width = 6)
            self.videos[name]['video_name_lbl'] = Label(self.video_frm, text=video_basename, width=self.max_char_vid_name)
            self.videos[name]['video_name_w_ext'] = name
            video_meta = get_video_meta_data(os.path.join(self.video_folder, name))
            self.videos[name]['fps_var'] = IntVar()
            self.videos[name]['fps_var'].set(video_meta['fps'])
            self.videos[name]['fps_entry'] = Entry(self.video_frm, width=20, textvariable=self.videos[name]['fps_var'])
            self.videos[name]['width_var'] = IntVar()
            self.videos[name]['width_var'].set(video_meta['width'])
            self.videos[name]['width_entry'] = Entry(self.video_frm, width=20, textvariable=self.videos[name]['width_var'])
            self.videos[name]['height_var'] = IntVar()
            self.videos[name]['height_var'].set(video_meta['height'])
            self.videos[name]['height_entry'] = Entry(self.video_frm, width=20, textvariable=self.videos[name]['height_var'])
            self.videos[name]['distance'] = StringVar()
            self.videos[name]['distance'].set(self.distance_mm)
            self.videos[name]['distance_entry'] = Entry(self.video_frm, width=20, textvariable=self.videos[name]['distance'])
            self.videos[name]['find_dist_btn'] = Button(self.video_frm, text='Calculate distance', fg='black', command=lambda k= (self.videos[name]['video_name_w_ext'], self.videos[name]['distance']): self.__initiate_find_distance(k))
            self.videos[name]['px_mm'] = StringVar()
            self.videos[name]['px_mm'].set(0)
            self.videos[name]['px_mm_entry'] = Entry(self.video_frm, width=20, textvariable=self.videos[name]['px_mm'])
            if isinstance(self.vid_info_df, pd.DataFrame):
                prior_data = None
                try:
                    prior_data = read_video_info(self.vid_info_df, video_basename)[0]
                except ValueError:
                    pass
                if prior_data is not None:
                    for value_name, set_name in zip(['fps', 'Resolution_width', 'Resolution_height', 'Distance_in_mm', 'pixels/mm'], ['fps_var', 'width_var', 'height_var', 'distance', 'px_mm']):
                        float_val = self.__check_that_value_is_numeric(value=prior_data[value_name].values[0], value_name=value_name, video_name=prior_data['Video'].values[0])
                        self.videos[name][set_name].set(float_val)

    def __check_for_duplicate_names(self):
        duplicate_video_names = [item for item, count in collections.Counter(self.video_basename_lst).items() if count > 1]
        if len(duplicate_video_names) > 0:
            for video_name in duplicate_video_names:
                print(video_name)
            print('SIMBA WARNING: SimBA found {} duplicate video name(s) in your SimBA project. The video(s) with duplicated names are printed above.'
                    ' This can happen if you have imported a video called MyVideo.mp4, and MyVideo.avi. Please avoid non-unique video '
                    'names with different file extensions / video formats.'.format(str(len(duplicate_video_names))))

    def __check_no_zero_px_per_mm_values(self):
        for index, row in self.video_df.iterrows():
            try:
                px_per_mm_float = float(row['pixels/mm'])
            except:
                print('SIMBA ERROR: The pixels/mm for video {} is not a numeric value. Please set pixels per mm using the "Calculate distance" button before proceding.'.format(row['Video']))
                raise ValueError
            if px_per_mm_float <= 0:
                print('SIMBA WARNING: The pixels/mm for video {} is set to zero. Please calculate pixels per mm using the "Calculate distance" button before proceding.'.format(row['Video']))

    def create_window(self):
        self.main_frm = Toplevel()
        self.main_frm.minsize(1550, 800)
        self.main_frm.wm_title("Video Info")
        self.main_frm = Canvas(hxtScrollbar(self.main_frm))
        self.main_frm.pack(expand=True,fill=BOTH)
        self.instructions_frm = LabelFrame(self.main_frm, text='INSTRUCTIONS', font=('Helvetica', 15, 'bold'))
        self.intructions_label_1 = Label(self.instructions_frm, text='1. Enter the known distance (millimeters) in the "DISTANCE IN MM" column. Consider using the "autopopulate" entry box in the main project window you have a lot of videos.', font=('Helvetica', 15))
        self.intructions_label_2 = Label(self.instructions_frm, text='2. Click on "Calculate distance" button(s) to calculate pixels/mm for each video.', font=('Helvetica', 15))
        self.intructions_label_3 = Label(self.instructions_frm, text='3. Click <SAVE DATA> when all the data are filled in.', font=('Helvetica', 15))
        self.instructions_frm.grid(row=0, column=0, sticky=W)
        self.intructions_label_1.grid(row=0, column=0, sticky=W)
        self.intructions_label_2.grid(row=1, column=0, sticky=W)
        self.intructions_label_3.grid(row=2, column=0, sticky=W)
        self.execute_frm = LabelFrame(self.main_frm, text='EXECUTE', font=('Helvetica', 15, 'bold'))
        self.save_data_btn = Button(self.execute_frm, text='SAVE DATA', fg='green', command=lambda: self.__save_data())
        self.execute_frm.grid(row=1, column=0, sticky=W)
        self.save_data_btn.grid(row=0, column=0, sticky=W)
        self.video_frm = LabelFrame(self.main_frm, text='PROJECT VIDEOS', font=('Helvetica', 15, 'bold'))
        self.column_names = ['INDEX', 'VIDEO', 'FPS', 'RESOLUTION WIDTH', 'RESOLUTION HEIGHT', 'DISTANCE IN MM', 'FIND DISTANCE', 'PIXELS PER MM']
        self.col_widths = ['5',self.max_char_vid_name,'18','18','18','18', '18', '18']
        self.video_frm.grid(row=6, column=0)
        for cnt, (col_name, col_width) in enumerate(zip(self.column_names, self.col_widths)):
            col_header_label = Label(self.video_frm, text=col_name, width=col_width, font=('Helvetica', 14, 'bold'))
            col_header_label.grid(row=0, column=cnt, sticky=W)
        self.videos = {}
        self.__append_videos_from_video_folder()
        self.__check_for_duplicate_names()
        self.duplicate_btn = Button(self.video_frm, text='Duplicate index 1 pixels/mm (CAUTION!)', fg='red', command=lambda: self.__duplicate_idx_1_px_mm())
        self.duplicate_btn.grid(row=1, column=7, sticky=W, padx=5)

        for vid_cnt, name in enumerate(self.videos.keys()):
            vid_cnt += 2
            self.videos[name]['video_idx_lbl'].grid(row=vid_cnt, column=0, sticky=W, padx=5)
            self.videos[name]['video_name_lbl'].grid(row=vid_cnt, column=1, sticky=W, padx=5)
            self.videos[name]['fps_entry'].grid(row=vid_cnt, column=2, sticky=W)
            self.videos[name]['width_entry'].grid(row=vid_cnt, column=3, sticky=W, padx=5)
            self.videos[name]['height_entry'].grid(row=vid_cnt, column=4, sticky=W, padx=5)
            self.videos[name]['distance_entry'].grid(row=vid_cnt, column=5, sticky=W, padx=5)
            self.videos[name]['find_dist_btn'].grid(row=vid_cnt, column=6, padx=5)
            self.videos[name]['px_mm_entry'].grid(row=vid_cnt, column=7, padx=5)

    def __initiate_find_distance(self, k):
        video_name, distance = k[0], k[1].get()
        try:
            distance = float(distance)
        except:
            print('SIMBA WARNING: The *DISTANCE IN MM* for video {} is not an integer or float value. The *DISTANCE IN MM* has to be a numerical value.'.format(video_name))
            raise ValueError
        if distance <= 0:
            print('SIMBA WARNING: The *DISTANCE IN MM* for video {} is <=0. The *DISTANCE IN MM* has to be a value above 0.'.format(video_name))
            raise ValueError
        video_pixels_per_mm = get_coordinates_nilsson(os.path.join(self.video_folder, video_name), distance)
        self.videos[video_name]['px_mm'].set(str(round(video_pixels_per_mm, 5)))

    def __duplicate_idx_1_px_mm(self):
        px_value = self.videos[list(self.videos.keys())[0]]['px_mm_entry'].get()
        for cnt, name in enumerate(self.video_paths):
            self.videos[name]['px_mm'].set(px_value)

    def __save_data(self):
        self.video_df = pd.DataFrame(columns=['Video', 'fps', 'Resolution_width', 'Resolution_height', 'Distance_in_mm', 'pixels/mm'])
        for name, data in self.videos.items():
            _, name, _ = get_fn_ext(name)
            lst = [name, data['fps_var'].get(), data['width_var'].get(), data['height_var'].get(), data['distance'].get(), data['px_mm'].get()]
            self.video_df.loc[len(self.video_df)] = lst
        self.__check_no_zero_px_per_mm_values()
        self.video_df.drop_duplicates(subset=['Video'], inplace=True)
        self.video_df = self.video_df.set_index('Video')
        try:
            self.video_df.to_csv(os.path.join(self.project_path, Paths.VIDEO_INFO.value))
        except PermissionError:
            print('SIMBA PERMISSION ERROR: SimBA tried to write to project_folder/logs/video_info.csv, but was not allowed. If this file is open in another program, tru closing it.')
            raise PermissionError()
        print('Video info saved at project_folder/logs/video_info.csv...')

# test = VideoInfoTable(config_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini')
# test.create_window()
# test.main_frm.mainloop()
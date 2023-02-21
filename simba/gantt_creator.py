__author__ = "Simon Nilsson", "JJ Choong"

from simba.misc_tools import (detect_bouts,
                              get_named_colors,
                              SimbaTimer)
from simba.read_config_unit_tests import (read_config_entry,
                                          read_config_file,
                                          check_if_filepath_list_is_empty,
                                          read_project_path_and_file_type)
from simba.features_scripts.unit_tests import (read_video_info_csv,
                                               read_video_info)
from simba.enums import Paths, ReadConfig, Formats, Dtypes
from simba.misc_visualizations import make_gantt_plot
from simba.train_model_functions import get_all_clf_names
from simba.drop_bp_cords import get_fn_ext
from simba.rw_dfs import read_df
import os
import numpy as np
import matplotlib.pyplot as plt
import io
import cv2
import PIL


class GanttCreatorSingleProcess(object):
    """
    Class for creating gantt chart videos and/or images using a single core.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format
    frame_setting: bool
        If True, creates individual frames
    video_setting: bool
        If True, creates videos
    files_found: list
        File paths representing files with machine predictions e.g., ['project_folder/csv/machine_results/My_results.csv']

    Notes
    ----------
    `GitHub gantt tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#gantt-plot>`__.

    Examples
    ----------
    >>> gantt_creator = GanttCreatorSingleProcess(config_path='tests/test_data/multi_animal_dlc_two_c57/project_folder/project_config.ini', frame_setting=False, video_setting=True, files_found=['tests/test_data/multi_animal_dlc_two_c57/project_folder/csv/machine_results/Together_1.csv'])
    >>> gantt_creator.create_gannt()

    """

    def __init__(self,
                 config_path: str,
                 frame_setting: bool,
                 video_setting: bool,
                 last_frm_setting: bool,
                 files_found: list,
                 style_attr: dict):

        self.frame_setting, self.video_setting, self.style_attr, self.last_frm_setting = frame_setting, video_setting, style_attr, last_frm_setting
        self.files_found = files_found
        if (self.frame_setting != True) and (self.video_setting != True) and (self.last_frm_setting != True):
            print('SIMBA ERROR: Please select gantt videos, frames, and/or last frame.')
            raise ValueError('SIMBA ERROR: Please select gantt videos, frames, and/or last frame.')
        self.config = read_config_file(config_path)
        self.project_path, self.file_type = read_project_path_and_file_type(config=self.config)
        self.data_in_dir = os.path.join(self.project_path, Paths.MACHINE_RESULTS_DIR.value)
        self.target_cnt = read_config_entry(self.config, ReadConfig.SML_SETTINGS.value, ReadConfig.TARGET_CNT.value, data_type=Dtypes.INT.value)
        self.vid_info_df = read_video_info_csv(os.path.join(self.project_path, Paths.VIDEO_INFO.value))
        check_if_filepath_list_is_empty(filepaths=self.files_found,
                                        error_msg='SIMBA ERROR: Zero files found in the project_folder/csv/machine_results directory. Create classification results before visualizing gantt charts')
        self.colours = get_named_colors()
        self.colour_tuple_x = list(np.arange(3.5, 203.5, 5))
        self.clf_names = get_all_clf_names(config=self.config, target_cnt=self.target_cnt)
        self.out_parent_dir = os.path.join(self.project_path, Paths.GANTT_PLOT_DIR.value)
        if not os.path.exists(self.out_parent_dir): os.makedirs(self.out_parent_dir)
        self.fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        print('Processing {} video(s)...'.format(str(len(self.files_found))))
        self.timer = SimbaTimer()
        self.timer.start_timer()

    def create_gannt(self):
        for file_cnt, file_path in enumerate(self.files_found):
            _, self.video_name, _ = get_fn_ext(file_path)
            self.data_df = read_df(file_path, self.file_type).reset_index(drop=True)
            self.video_info_settings, _, self.fps = read_video_info(self.vid_info_df, self.video_name)
            self.bouts_df = detect_bouts(data_df=self.data_df, target_lst=self.clf_names, fps=self.fps)
            if self.frame_setting:
                self.save_frame_folder_dir = os.path.join(self.out_parent_dir, self.video_name)
                if not os.path.exists(self.save_frame_folder_dir): os.makedirs(self.save_frame_folder_dir)
            if self.video_setting:
                self.save_video_path = os.path.join(self.out_parent_dir, self.video_name + '.mp4')
                self.writer = cv2.VideoWriter(self.save_video_path, self.fourcc, self.fps, (self.style_attr['width'], self.style_attr['height']))

            if self.last_frm_setting:
                _ = make_gantt_plot(data_df=self.data_df,
                                    bouts_df=self.bouts_df,
                                    clf_names=self.clf_names,
                                    fps=self.fps,
                                    style_attr=self.style_attr,
                                    video_name=self.video_name,
                                    save_path=os.path.join(self.out_parent_dir, self.video_name + '_final_image.png'))

            if self.frame_setting or self.video_setting:
                for image_cnt, k in enumerate(range(len(self.data_df))):
                    fig, ax = plt.subplots()
                    relevant_rows = self.bouts_df.loc[self.bouts_df['End_frame'] <= k]
                    for i, event in enumerate(relevant_rows.groupby("Event")):
                        for x in self.clf_names:
                            if event[0] == x:
                                ix = self.clf_names.index(x)
                                data_event = event[1][["Start_time", "Bout_time"]]
                                ax.broken_barh(data_event.values, (self.colour_tuple_x[ix], 3), facecolors=self.colours[ix])

                    x_ticks_locs = x_lbls = np.round(np.linspace(0, round((image_cnt / self.fps), 3), 6))
                    ax.set_xticks(x_ticks_locs)
                    ax.set_xticklabels(x_lbls)
                    ax.set_ylim(0, self.colour_tuple_x[len(self.clf_names)])
                    ax.set_yticks(np.arange(5, 5 * len(self.clf_names) + 1, 5))
                    ax.set_yticklabels(self.clf_names, rotation=self.style_attr['font rotation'])
                    ax.tick_params(axis='both', labelsize=self.style_attr['font size'])
                    plt.xlabel('Session (s)', fontsize=self.style_attr['font size'])
                    ax.yaxis.grid(True)
                    buffer_ = io.BytesIO()
                    plt.savefig(buffer_, format="png")
                    buffer_.seek(0)
                    image = PIL.Image.open(buffer_)
                    ar = np.asarray(image)
                    open_cv_image = cv2.cvtColor(ar, cv2.COLOR_RGB2BGR)
                    open_cv_image = cv2.resize(open_cv_image, (self.style_attr['width'], self.style_attr['height']))
                    frame = np.uint8(open_cv_image)
                    buffer_.close()
                    plt.close(fig)
                    if self.frame_setting:
                        frame_save_path = os.path.join(self.save_frame_folder_dir, str(k) + '.png')
                        cv2.imwrite(frame_save_path, frame)
                    if self.video_setting:
                        self.writer.write(frame)
                    print('Gantt frame: {} / {}. Video: {} ({}/{})'.format(str(image_cnt+1), str(len(self.data_df)),
                                                                     self.video_name, str(file_cnt + 1), len(self.files_found)))
                if self.video_setting:
                    self.writer.release()
                print('Gantt for video {} saved...'.format(self.video_name))
        self.timer.stop_timer()
        print('SIMBA COMPLETE: All gantt visualizations created in project_folder/frames/output/gantt_plots directory (elapsed time: {}s)'.format(self.timer.elapsed_time_str))


# style_attr = {'width': 640, 'height': 480, 'font size': 12, 'font rotation': 45}
# test = GanttCreatorSingleProcess(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                  frame_setting=False,
#                                  video_setting=True,
#                                  last_frm_setting=True,
#                                  style_attr=style_attr,
#                                  files_found=['/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/machine_results/Together_1.csv'])
# test.create_gannt()




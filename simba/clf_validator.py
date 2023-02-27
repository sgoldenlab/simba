from simba.read_config_unit_tests import (read_config_file,
                                          check_int,
                                          check_that_column_exist,
                                          check_if_filepath_list_is_empty,
                                          read_project_path_and_file_type)
from simba.features_scripts.unit_tests import read_video_info_csv
from simba.misc_tools import (find_video_of_file,
                              get_file_path_parts,
                              detect_bouts,
                              get_video_meta_data,
                              SimbaTimer)
from simba.enums import Paths, Formats

from simba.rw_dfs import read_df
import numpy as np
import os, glob
import cv2

class ClassifierValidationClips(object):
    """
    Class for creating video clips of classified events. Helpful for faster detection of false positive event bouts.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format
    window: int
        Number of seconds before and after the event bout that should be included in the output video.
    clf_name: str
        Name of the classifier to create validation videos for.
    clips: bool
        If True, creates individual video file clips for each validation bout.
    text_clr: tuple
        Color of text overlay in BGR
    concat_video: bool
        If True, creates a single video including all events bouts for each video.


    Notes
    ----------
    `GitHub tutorial/documentation <https://github.com/sgoldenlab/simba/blob/master/docs/classifier_validation.md#classifier-validation>`__.

    Examples
    ----------
    >>> clf_validator = ClassifierValidationClips(config_path='MyProjectConfigPath', window=5, clf_name='Attack', text_clr=(255,255,0), clips=False, concat_video=True)
    >>> clf_validator.create_clips()
    """

    def __init__(self,
                 config_path: str,
                 window: int,
                 clf_name: str,
                 clips: bool,
                 text_clr: tuple,
                 concat_video: bool):

        check_int(name='Time window', value=window)
        self.window, self.clf_name = int(window), clf_name
        self.clips, self.concat_video = clips, concat_video
        if (not self.clips) and (not self.concat_video):
            print('SIMBA ERROR: Please select to create clips and/or a concatenated video')
            raise ValueError()
        self.p_col = 'Probability_' + self.clf_name
        self.config, self.text_clr = read_config_file(config_path), text_clr
        self.project_path, self.file_type = read_project_path_and_file_type(config=self.config)
        self.data_in_dir = os.path.join(self.project_path, Paths.MACHINE_RESULTS_DIR.value)
        self.video_dir = os.path.join(self.project_path, 'videos')
        self.vid_info_df = read_video_info_csv(os.path.join(self.project_path, Paths.VIDEO_INFO.value))
        self.files_found = glob.glob(self.data_in_dir + '/*.' + self.file_type)
        self.fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.out_folder = os.path.join(self.project_path, Paths.CLF_VALIDATION_DIR.value)
        if not os.path.exists(self.out_folder): os.makedirs(self.out_folder)
        check_if_filepath_list_is_empty(filepaths=self.files_found,
                                        error_msg='SIMBA ERROR: No data found in the project_folder/csv/machine_results directory')
        print('Processing {} files...'.format(str(len(self.files_found))))
        self.timer = SimbaTimer()
        self.timer.start_timer()


    def __insert_inter_frms(self):
        """
        Helper to create N blank frames separating the classified event bouts.
        """

        for i in range(int(self.fps)):
            inter_frm = np.full((int(self.video_info['height']), int(self.video_info['width']), 3), (49, 32, 189)).astype(np.uint8)
            cv2.putText(inter_frm, 'Bout #{}'.format(str(self.bout_cnt + 1)), (10, (self.video_info['height'] - self.video_info['height']) + self.spacing_scale), self.font, self.font_size, (0, 0, 0), 2)
            self.concat_writer.write(inter_frm)

    def create_clips(self):
        """
        Method to generate clips. Results are saved in the ``project_folder/frames/output/classifier_validation directory``
        directory of the SimBA project.

        Returns
        -------
        None
        """

        for file_cnt, file_path in enumerate(self.files_found):
            self.data_df = read_df(file_path, self.file_type)
            check_that_column_exist(df=self.data_df, column_name=self.p_col, file_name=file_path)
            _, file_name, _ = get_file_path_parts(file_path)
            self.video_path = find_video_of_file(video_dir=self.video_dir,filename=file_name)
            self.video_info = get_video_meta_data(video_path=self.video_path)
            self.fps = int(self.video_info['fps'])
            self.space_scale, self.radius_scale, self.res_scale, self.font_scale = 60, 12, 1500, 1.5
            self.max_dim = max(self.video_info['width'], self.video_info['height'])
            self.circle_scale = int(self.radius_scale / (self.res_scale / self.max_dim))
            self.font_size = float(self.font_scale / (self.res_scale / self.max_dim))
            self.spacing_scale = int(self.space_scale / (self.res_scale / self.max_dim))
            clf_bouts = detect_bouts(data_df=self.data_df, target_lst=[self.clf_name], fps=self.fps).reset_index(drop=True)
            if self.concat_video:
                self.concat_video_save_path = os.path.join(self.out_folder, self.clf_name + '_' + file_name + '_all_events.mp4')
                self.concat_writer = cv2.VideoWriter(self.concat_video_save_path, self.fourcc, self.fps, (int(self.video_info['width']), int(self.video_info['height'])))
                self.bout_cnt = 0
                self.__insert_inter_frms()
            if len(clf_bouts) == 0:
                print('Skipping video {}: No classified behavior detected...'.format(file_name))
                continue
            for bout_cnt, bout in clf_bouts.iterrows():
                self.bout_cnt = bout_cnt
                event_start_frm, event_end_frm = bout['Start_frame'], bout['End_frame']
                start_window = int(event_start_frm - (int(self.video_info['fps']) * self.window))
                end_window =  int(event_end_frm + (int(self.video_info['fps']) * self.window))
                self.save_path = os.path.join(self.out_folder, self.clf_name + '_' + str(bout_cnt) + '_' + file_name + '.mp4')
                if start_window < 0:
                    start_window = 0
                if end_window > len(self.data_df):
                    end_window = len(self.data_df)
                if self.clips:
                    bout_writer = cv2.VideoWriter(self.save_path, self.fourcc, self.fps, (int(self.video_info['width']), int(self.video_info['height'])))
                cap = cv2.VideoCapture(self.video_path)
                event_frm_count = end_window - start_window
                for frm_cnt, frame_no in enumerate(list(range(start_window, end_window))):
                    p = self.data_df.loc[frame_no, self.p_col]
                    self.add_spacer = 2
                    cap.set(1, frame_no)
                    ret, img = cap.read()
                    cv2.putText(img, '{} event # {}'.format(self.clf_name, str(bout_cnt + 1)), (10, (self.video_info['height'] - self.video_info['height']) + self.spacing_scale * self.add_spacer), self.font, self.font_size, self.text_clr, 2)
                    self.add_spacer += 1
                    cv2.putText(img, 'Total frames of event: {}'.format(str(event_frm_count)), (10, (self.video_info['height'] - self.video_info['height']) + self.spacing_scale * self.add_spacer), self.font, self.font_size, self.text_clr, 2)
                    self.add_spacer += 1
                    cv2.putText(img, 'Frames of event {} to {}'.format(str(start_window), str(end_window)), (10, (self.video_info['height'] - self.video_info['height']) + self.spacing_scale * self.add_spacer), self.font, self.font_size, self.text_clr, 2)
                    self.add_spacer += 1
                    cv2.putText(img, 'Frame number: {}'.format(str(frame_no)), (10, (self.video_info['height'] - self.video_info['height']) + self.spacing_scale * self.add_spacer), self.font, self.font_size, self.text_clr, 2)
                    self.add_spacer += 1
                    cv2.putText(img, 'Frame {} probability: {}'.format(self.clf_name, str(p)), (10, (self.video_info['height'] - self.video_info['height']) + self.spacing_scale * self.add_spacer), self.font, self.font_size, self.text_clr, 2)
                    print('Frame {} / {}, Bout {}/{}, Video {}/{}...'.format(str(frm_cnt), str(event_frm_count), str(bout_cnt+1), str(len(clf_bouts)), str(file_cnt+1), str(len(self.files_found))))
                    if self.clips:
                        bout_writer.write(img)
                    if self.concat_video:
                        self.concat_writer.write(img)
                if self.clips:
                    bout_writer.release()
                if self.concat_video:
                    self.__insert_inter_frms()
            if self.concat_video:
                self.concat_writer.release()
        self.timer.stop_timer()
        print('SIMBA COMPLETE: All validation clips complete. Files are saved in the project_folder/frames/output/classifier_validation directory of the SimBA project (elapsed time: {}s)'.format(self.timer.elapsed_time_str))

# test = ClassifierValidationClips(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini', window=1, clf_name='Attack', clips=False, concat_video=True)
# test.create_clips()









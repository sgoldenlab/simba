from simba.read_config_unit_tests import (read_config_file,
                                          read_config_entry,
                                          check_int,
                                          check_that_column_exist)
from simba.features_scripts.unit_tests import read_video_info_csv
from simba.misc_tools import find_video_of_file, get_file_path_parts, detect_bouts, get_video_meta_data
from simba.rw_dfs import read_df
import os, glob
import cv2

class ClassifierValidationClips(object):
    """
    Class for creating video clips of classified events.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format

    Notes
    ----------
    `GitHub tutorial/documentation <https://github.com/sgoldenlab/simba/blob/master/docs/classifier_validation.md#classifier-validation>`__.

    Examples
    ----------
    >>> clf_validator = ClassifierValidationClips(config_path='MyProjectConfigPath', window=5, clf_name='Attack')
    >>> clf_validator.create_clips()
    """

    def __init__(self,
                 config_path: str,
                 window: int,
                 clf_name: str):

        check_int(name='Time window', value=window)
        self.window, self.clf_name = int(window), clf_name
        self.p_col = 'Probability_' + self.clf_name
        self.config = read_config_file(config_path)
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.data_in_dir = os.path.join(self.project_path, 'csv', 'machine_results')
        self.video_dir = os.path.join(self.project_path, 'videos')
        self.file_type = read_config_entry(self.config, 'General settings', 'workflow_file_type', 'str', 'csv')
        self.vid_info_df = read_video_info_csv(os.path.join(self.project_path, 'logs', 'video_info.csv'))
        self.files_found = glob.glob(self.data_in_dir + '/*.' + self.file_type)
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.out_folder = os.path.join(self.project_path, 'frames', 'output', 'classifier_validation')
        if not os.path.exists(self.out_folder): os.makedirs(self.out_folder)
        if len(self.files_found) == 0:
            print('SIMBA ERROR: No data found in the project_folder/csv/machine_results directory')
            raise ValueError
        print('Processing {} files...'.format(str(len(self.files_found))))


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
            video_info = get_video_meta_data(video_path=self.video_path)
            self.fps = int(video_info['fps'])
            self.space_scale, self.radius_scale, self.res_scale, self.font_scale = 60, 12, 1500, 1.5
            self.max_dim = max(video_info['width'], video_info['height'])
            self.circle_scale = int(self.radius_scale / (self.res_scale / self.max_dim))
            self.font_size = float(self.font_scale / (self.res_scale / self.max_dim))
            self.spacing_scale = int(self.space_scale / (self.res_scale / self.max_dim))
            clf_bouts = detect_bouts(data_df=self.data_df, target_lst=[self.clf_name], fps=self.fps).reset_index(drop=True)
            if len(clf_bouts) == 0:
                print('Skipping video {}: No classified behavior detected...'.format(file_name))
                continue
            for bout_cnt, bout in clf_bouts.iterrows():
                event_start_frm, event_end_frm = bout['Start_frame'], bout['End_frame']
                start_window = int(event_start_frm - (int(video_info['fps']) * self.window))
                end_window =  int(event_end_frm + (int(video_info['fps']) * self.window))
                self.save_path = os.path.join(self.out_folder, self.clf_name + '_' + str(bout_cnt) + '_' + file_name + '.mp4')
                if start_window < 0:
                    start_window = 0
                if end_window > len(self.data_df):
                    end_window = len(self.data_df)
                writer = cv2.VideoWriter(self.save_path, self.fourcc, self.fps, (int(video_info['width']), int(video_info['height'])))
                cap = cv2.VideoCapture(self.video_path)
                event_frm_count = end_window - start_window
                for frm_cnt, frame_no in enumerate(list(range(start_window, end_window))):
                    p = self.data_df.loc[frame_no, self.p_col]
                    self.add_spacer = 2
                    cap.set(1, frame_no)
                    ret, img = cap.read()
                    cv2.putText(img, '{} event # {}'.format(self.clf_name, str(bout_cnt)), (10, (video_info['height'] - video_info['height']) + self.spacing_scale * self.add_spacer), self.font, self.font_size, (255, 255, 0), 2)
                    self.add_spacer += 1
                    cv2.putText(img, 'Total frames of event: {}'.format(str(event_frm_count)), (10, (video_info['height'] - video_info['height']) + self.spacing_scale * self.add_spacer), self.font, self.font_size, (255, 255, 0), 2)
                    self.add_spacer += 1
                    cv2.putText(img, 'Frames of event {} to {}'.format(str(start_window), str(end_window)), (10, (video_info['height'] - video_info['height']) + self.spacing_scale * self.add_spacer), self.font, self.font_size, (255, 255, 0), 2)
                    self.add_spacer += 1
                    cv2.putText(img, 'Frame number: {}'.format(str(frame_no)), (10, (video_info['height'] - video_info['height']) + self.spacing_scale * self.add_spacer), self.font, self.font_size, (255, 255, 0), 2)
                    self.add_spacer += 1
                    cv2.putText(img, 'Frame {} probability: {}'.format(self.clf_name, str(p)), (10, (video_info['height'] - video_info['height']) + self.spacing_scale * self.add_spacer), self.font, self.font_size, (255, 255, 0), 2)
                    print('Frame {} / {}, Bout {}/{}, Video {}/{}...'.format(str(frm_cnt), str(event_frm_count), str(bout_cnt), str(len(clf_bouts)), str(file_cnt), str(len(self.files_found))))
                    writer.write(img)
                writer.release()

        print('SIMBA COMPLETE: All validation clips complete. Files are saved in the project_folder/frames/output/classifier_validation directory of the SimBA project')

# test = ClassifierValidationClips(config_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini', window=5, clf_name='Attack')
# test.create_clips()









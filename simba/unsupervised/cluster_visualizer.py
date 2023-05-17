__author__ = "Simon Nilsson"

import pandas as pd
import numpy as np
import os
import cv2

from simba.utils.checks import check_file_exist_and_readable
from simba.utils.read_write import find_all_videos_in_directory, get_video_meta_data
from simba.utils.enums import Paths, Formats
from simba.mixins.config_reader import ConfigReader
from simba.mixins.unsupervised_mixin import UnsupervisedMixin
from simba.unsupervised.enums import Clustering, Unsupervised
from simba.utils.printing import stdout_success
from simba.utils.warnings import NoFileFoundWarning


CLUSTER = 'CLUSTER'
FPS = 'fps'
VIDEO_SPEED = 'video_speed'
START_FRAME = 'START_FRAME'
POSE = 'pose'
CREATE = 'create'
CIRCLE_SIZE = 'circle_size'


class ClusterVisualizer(ConfigReader, UnsupervisedMixin):
    """
    Class for creating video examples of cluster assignments.

    :param str config_path: path to SimBA configparser.ConfigParser project_config.ini
    :param str data_path: path to pickle holding unsupervised results in ``data_map.yaml`` format.
    :param str video_dir: path to directory holding videos.
    :param dict settings: dict holding attributes of the videos

    :example:
    >>> settings = {'video_speed': 0.5, 'pose': {'create': True, 'circle_size': 5}}
    >>> visualizer = ClusterVisualizer(video_dir='unsupervised/project_folder/videos', data_path='unsupervised/cluster_models/quizzical_rhodes.pickle', settings=settings, config_path='unsupervised/project_folder/project_config.ini')
    >>> visualizer.run()
    """

    def __init__(self,
                 config_path: str,
                 video_dir: str,
                 data_path: str,
                 settings: dict):



        ConfigReader.__init__(self, config_path=config_path)
        UnsupervisedMixin.__init__(self)
        self.settings, self.video_dir, self.data_path = settings, video_dir, data_path
        self.save_parent_dir = os.path.join(self.project_path, Paths.CLUSTER_EXAMPLES.value)
        if not os.path.exists(self.save_parent_dir): os.makedirs(self.save_parent_dir)
        check_file_exist_and_readable(file_path=data_path)
        self.fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        self.video_files = find_all_videos_in_directory(directory=video_dir, as_dict=True)

    def run(self):
        self.video_counter = 0
        self.data = self.read_pickle(data_path=self.data_path)
        self.cluster_data = self.data[Clustering.CLUSTER_MODEL.value][Unsupervised.MODEL.value].labels_
        self.x_df = self.data[Unsupervised.METHODS.value][Unsupervised.SCALED_DATA.value].reset_index()
        self.x_y_df = pd.concat([self.x_df, pd.DataFrame(self.cluster_data, columns=[CLUSTER])], axis=1)
        self.bout_cnt = len(self.data[Unsupervised.DATA.value][Unsupervised.BOUTS_FEATURES.value])
        for cluster_id in np.unique(self.cluster_data):
            self.cluster_id = cluster_id
            for video_name in self.data[Unsupervised.DATA.value][Unsupervised.BOUTS_FEATURES.value][Unsupervised.VIDEO.value].unique():
                self.video_name = video_name
                if video_name not in self.video_files.keys():
                    NoFileFoundWarning(msg=f'Video {video_name} not found in video directory {self.video_dir}')
                    continue
                else:
                    self.cluster_video_df = self.x_y_df.loc[(self.x_y_df[Unsupervised.VIDEO.value] == video_name) & (self.x_y_df[CLUSTER] == cluster_id)]
                    if len(self.cluster_video_df) > 0:
                        self.__create_videos()
        self.timer.stop_timer()
        stdout_success(msg=f'Visualizations complete. Data saved at {self.save_parent_dir}', elapsed_time=self.timer.elapsed_time_str)


    def __create_videos(self):
        self.save_directory = os.path.join(self.save_parent_dir, str(self.cluster_id), self.video_name)
        if not os.path.exists(self.save_directory): os.makedirs(self.save_directory)
        video_meta_data = get_video_meta_data(video_path=self.video_files[self.cluster_video_df[Unsupervised.VIDEO.value].values[0]])
        output_fps = max(1, int(video_meta_data[FPS] * self.settings[VIDEO_SPEED]))
        cap = cv2.VideoCapture(self.video_files[self.cluster_video_df['VIDEO'].values[0]])
        cluster_frames = list(self.cluster_video_df.apply(lambda x: list(range(int(x[Unsupervised.START_FRAME.value]), int(x[Unsupervised.END_FRAME.value]) + 1)), 1))
        for cluster_event_cnt, cluster_event in enumerate(cluster_frames):
            save_path = os.path.join(self.save_directory, f'Event_{str(cluster_event_cnt)}.mp4')
            self.writer = cv2.VideoWriter(save_path, self.fourcc, output_fps, (video_meta_data['width'], video_meta_data['height']))
            start_frm, end_frm, current_frm = cluster_event[0], cluster_event[-1], cluster_event[0]
            cluster_event_frms = end_frm-start_frm
            cap.set(1, cluster_event[0])
            frame_cnt = 0
            while current_frm < end_frm:
                _, img = cap.read()
                if self.settings[POSE][CREATE]:
                    bp_data = self.data[Unsupervised.DATA.value]['FRAME_POSE'].iloc[current_frm]
                    for animal_cnt, (animal_name, animal_bps) in enumerate(self.animal_bp_dict.items()):
                        for bp_cnt, bp in enumerate(zip(animal_bps['X_bps'], animal_bps['Y_bps'])):
                            x_bp, y_bp = bp_data[bp[0]], bp_data[bp[1]]
                            cv2.circle(img, (int(x_bp), int(y_bp)), self.settings[POSE][CIRCLE_SIZE], self.clr_lst[animal_cnt][bp_cnt], -1)
                self.writer.write(img)
                print(f'Writing frame {str(frame_cnt)}/{str(cluster_event_frms)}, '
                      f'Bout {str(cluster_event_cnt+1)}/{len(cluster_frames)}, '
                      f''f'Cluster: {self.cluster_id}, '
                      f'Video: {self.video_name}, '
                      f'Total bout count: {self.video_counter}/{self.bout_cnt}...')
                current_frm += 1
                frame_cnt += 1
            cap.release()
            self.writer.release()
            self.video_counter += 1


# settings = {'video_speed': 0.5, 'pose': {'create': True, 'circle_size': 5}}
# test = ClusterVisualizer(video_dir='/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/videos',
#                          data_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/cluster_models/quizzical_rhodes.pickle',
#                          settings=settings,
#                          config_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/project_config.ini')
# test.run()

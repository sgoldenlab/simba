import numpy as np
import matplotlib.pyplot as plt
from simba.unsupervised.misc import (read_pickle,
                                     check_directory_exists)
import pandas as pd
from simba.misc_tools import (check_file_exist_and_readable,
                              get_video_meta_data,
                              find_all_videos_in_directory,
                              check_multi_animal_status,
                              SimbaTimer)
from simba.drop_bp_cords import getBpNames, create_body_part_dictionary, createColorListofList
from simba.read_config_unit_tests import read_config_file, read_config_entry, read_project_path_and_file_type
from simba.enums import Paths, Formats, ReadConfig, Dtypes
import os
import warnings
import cv2


class GridSearchClusterVisualizer(object):
    def __init__(self,
                 clusterers_path: str,
                 save_dir: str,
                 settings: dict):

        self.timer = SimbaTimer()
        self.timer.start_timer()
        check_directory_exists(save_dir)
        self.save_dir = save_dir
        self.settings = settings
        self.clusterers = None
        if clusterers_path:
            check_directory_exists(clusterers_path)
            self.clusterers = read_pickle(data_path=clusterers_path)

    def create_datasets(self):
        self.img_data = {}
        print('Retrieving models for visualization...')
        for k, v in self.clusterers.items():
            self.img_data[k] = {}
            self.img_data[k]['categorical_legends'] = set()
            self.img_data[k]['continuous_legends'] = set()
            embedder = v['EMBEDDER']
            cluster_data = v['MODEL'].labels_.reshape(-1, 1).astype(np.int8)
            embedding_data = embedder['MODEL'].embedding_
            data = np.hstack((embedding_data, cluster_data))
            self.img_data[k]['DATA'] = pd.DataFrame(data, columns=['X', 'Y', 'CLUSTER'])
            self.img_data[k]['HASH'] = v['HASH']
            self.img_data[k]['EMBEDDER'] = embedder
            self.img_data[k]['CLUSTERER_NAME'] = v['NAME']
            self.img_data[k]['categorical_legends'].add('CLUSTER')

        if self.settings['HUE']:
            for hue_id, hue_settings in self.settings['HUE'].items():
                field_type, field_name = hue_settings['FIELD_TYPE'], hue_settings['FIELD_NAME']
                for k, v in self.img_data.items():
                    embedder = v['EMBEDDER']
                    if not 'categorical_legends' in self.img_data[k].keys():
                        self.img_data[k]['categorical_legends'] = set()
                        self.img_data[k]['continuous_legends'] = set()
                    if (field_type == 'CLASSIFIER'):
                        self.img_data[k]['categorical_legends'].add(field_name)
                    if (field_type == 'VIDEO NAMES'):
                        self.img_data[k]['categorical_legends'].add(field_type)
                    elif (field_type == 'CLASSIFIER PROBABILITY') or (field_type == 'START FRAME'):
                        if field_name != 'None' and field_name != '':
                            self.img_data[k]['continuous_legends'].add(field_name)
                        else:
                            self.img_data[k]['continuous_legends'].add(field_type)
                    if field_name != 'None' and field_name != '':
                        self.img_data[k]['DATA'][field_name] = embedder[field_type][field_name]
                    else:
                        self.img_data[k]['DATA'][field_type] = embedder[field_type]


    def create_imgs(self):
        print('Creating plots...')
        plots = {}
        for k, v in self.img_data.items():
            for categorical in v['categorical_legends']:
                fig, ax = plt.subplots()
                colmap = {name: n for n, name in enumerate(set(list(v['DATA'][categorical].unique())))}
                scatter = ax.scatter(v['DATA']['X'], v['DATA']['Y'], c=[colmap[name] for name in v['DATA'][categorical]], cmap=self.settings['CATEGORICAL_PALETTE'], s=self.settings['SCATTER_SIZE'])
                plt.legend(*scatter.legend_elements()).set_title(categorical)
                plt.xlabel('X')
                plt.ylabel('Y')
                plt_key = v['HASH'] + '_' + v['CLUSTERER_NAME'] + '_' + categorical
                title = 'EMBEDDER: {} \n CLUSTERER: {}'.format(v['HASH'], v['CLUSTERER_NAME'])
                if categorical != 'CLUSTER':
                    title = 'EMBEDDER: {}'.format(v['HASH'])
                plt.title(title, ha="center", fontsize=15, bbox={"facecolor": "orange", "alpha": 0.5, "pad": 0})
                plots[plt_key] = fig
                plt.close('all')

            for continuous in v['continuous_legends']:
                fig, ax = plt.subplots()
                plt.xlabel('X')
                plt.ylabel('Y')
                points = ax.scatter(v['DATA']['X'], v['DATA']['Y'], c=v['DATA'][continuous], s=self.settings['SCATTER_SIZE'], cmap=self.settings['CONTINUOUS_PALETTE'])
                cbar = fig.colorbar(points)
                cbar.set_label(continuous, loc='center')
                title = 'EMBEDDER: {}'.format(v['HASH'])
                plt_key = v['HASH'] + v['CLUSTERER_NAME'] + '_' + continuous
                plt.title(title, ha="center", fontsize=15, bbox={"facecolor": "orange", "alpha": 0.5, "pad": 0})
                plots[plt_key] = fig
                plt.close('all')

        for plt_key, fig in plots.items():
            save_path = os.path.join(self.save_dir, f'{plt_key}.png')
            print(f'Saving scatterplot {plt_key} ...')
            fig.savefig(save_path)
        self.timer.stop_timer()
        print(f'SIMBA COMPLETE: {str(len(plots.keys()))} plots saved in {self.save_dir} (elapsed time: {self.timer.elapsed_time_str}s)')



class ClusterVisualizer(object):
    def __init__(self,
                 config_path: str,
                 video_dir: str,
                 data_path: str,
                 settings: dict):

        self.config, self.settings = read_config_file(ini_path=config_path), settings
        self.project_path, self.file_type = read_project_path_and_file_type(config=self.config)
        self.save_parent_dir = os.path.join(self.project_path, Paths.CLUSTER_EXAMPLES.value)
        if not os.path.exists(self.save_parent_dir): os.makedirs(self.save_parent_dir)
        check_file_exist_and_readable(file_path=data_path)
        self.x_cols, self.y_cols, self.pcols = getBpNames(config_path)
        self.no_animals = read_config_entry(self.config, ReadConfig.GENERAL_SETTINGS.value, ReadConfig.ANIMAL_CNT.value, Dtypes.INT.value)
        self.pose_colors = createColorListofList(self.no_animals, int(len(self.x_cols) + 1))
        self.multi_animal_status, self.multi_animal_id_lst = check_multi_animal_status(self.config, self.no_animals)
        self.animal_bp_dict = create_body_part_dictionary(self.multi_animal_status, self.multi_animal_id_lst, self.no_animals, self.x_cols, self.y_cols, [], self.pose_colors)
        self.fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        self.video_files = find_all_videos_in_directory(directory=video_dir, as_dict=True)
        self.data = read_pickle(data_path=data_path)
        self.video_dir, self.pose_df = video_dir, None
        self.cluster_ids = self.data['DATA']['CLUSTER'].unique()

    def create(self):
        for cluster_id in self.cluster_ids:
            self.cluster_id = cluster_id
            cluster_df = self.data['DATA'][self.data['DATA']['CLUSTER'] == cluster_id]
            for video_name in cluster_df['VIDEO'].unique():
                self.video_name = video_name
                if video_name not in self.video_files.keys():
                    warnings.warn(f'SIMBA WARNING: Video {video_name} not found in video directory {self.video_dir}')
                    continue
                else:
                    self.cluster_video_df = cluster_df[cluster_df['VIDEO'] == video_name]
                    self.__cluster_video_creator()


    def __cluster_video_creator(self):
        self.save_directory = os.path.join(self.save_parent_dir, str(self.cluster_id), self.video_name)
        if self.settings['pose']['include']:
            self.pose_df = self.data['POSE'][self.data['POSE']['VIDEO'] == self.video_name].drop(['FRAME', 'VIDEO'], axis=1).reset_index(drop=True)
        if not os.path.exists(self.save_directory): os.makedirs(self.save_directory)
        video_meta_data = get_video_meta_data(video_path=self.video_files[self.cluster_video_df['VIDEO'].values[0]])
        output_fps = int(video_meta_data['fps'] * self.settings['video_speed'])
        if output_fps < 1: output_fps = 1
        cluster_frames = list(self.cluster_video_df.apply(lambda x: list(range(int(x['START_FRAME']), int(x['END_FRAME']) + 1)), 1))
        cap = cv2.VideoCapture(self.video_files[self.cluster_video_df['VIDEO'].values[0]])
        for cluster_event_cnt, cluster_event in enumerate(cluster_frames):
            file_name = os.path.join(self.save_directory, f'Event_{str(cluster_event_cnt)}.mp4')
            self.writer = cv2.VideoWriter(file_name, self.fourcc, output_fps, (video_meta_data['width'], video_meta_data['height']))
            start_frm, end_frm, current_frm = cluster_event[0], cluster_event[-1], cluster_event[0]
            cluster_event_frms = end_frm-start_frm
            cap.set(1, cluster_event[0])
            frame_cnt = 0
            while current_frm < end_frm:
                _, img = cap.read()
                if self.settings['pose']['include']:
                    bp_data = self.pose_df.iloc[current_frm]
                    for cnt, (animal_name, animal_bps) in enumerate(self.animal_bp_dict.items()):
                        for bp in zip(animal_bps['X_bps'], animal_bps['Y_bps']):
                            x_bp, y_bp = bp_data[bp[0]], bp_data[bp[1]]
                            cv2.circle(img, (int(x_bp), int(y_bp)), self.settings['pose']['circle_size'], self.animal_bp_dict[animal_name]['colors'][cnt], -1)
                self.writer.write(img)
                print(f'Writing frame {str(frame_cnt)}/{str(cluster_event_frms)}, Bout {str(cluster_event_cnt+1)}/{len(cluster_frames)}, '
                      f'Cluster: {self.cluster_id}, Video: {self.video_name}...')
                current_frm += 1
                frame_cnt += 1
            cap.release()
            self.writer.release()





# settings = {'video_speed': 0.01, 'pose': {'include': True, 'circle_size': 5}}
# test = ClusterVisualizer(video_dir='/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/videos',
#                          data_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/dr_models/dreamy_spence_awesome_elion.pickle',
#                          settings=settings,
#                          config_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/project_config.ini')
# test.create()


# settings = {'HUE': {'FIELD_TYPE': 'VIDEO_NAMES', 'FIELD_NAME': None}}
# settings = {'SCATTER_SIZE': 50, 'CATEGORICAL_PALETTE': 'Set1', 'CONTINUOUS_PALETTE': 'magma', 'HUE': {0: {'FIELD_TYPE': 'CLASSIFIER PROBABILITY', 'FIELD_NAME': 'None'}, 1: {'FIELD_TYPE': 'CLASSIFIER', 'FIELD_NAME': 'Attack'}}}
# test = GridSearchClusterVisualizer(clusterers_path= '/Users/simon/Desktop/envs/troubleshooting/unsupervised/cluster_models',
#                                    save_dir='/Users/simon/Desktop/envs/troubleshooting/unsupervised/images',
#                                    settings=settings)
# test.create_datasets()
# test.create_imgs()

#{0: {'FIELD_TYPE': 'CLASSIFIER PROBABILITY', 'FIELD_NAME': 'None'}, 1: {'FIELD_TYPE': 'CLASSIFIER', 'FIELD_NAME': 'Attack'}}

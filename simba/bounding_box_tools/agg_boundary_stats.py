import pandas as pd
from simba.rw_dfs import read_df
from simba.read_config_unit_tests import (read_config_entry,
                                          read_config_file)
from simba.misc_tools import detect_bouts, plug_holes_shortest_bout
from simba.features_scripts.unit_tests import read_video_info_csv, read_video_info
from simba.misc_tools import get_fn_ext
from datetime import datetime
import os, glob

class AggBoundaryStatisticsCalculator(object):
    """
    Class computing aggregate boundary statistics

    Parameters
    ----------
    config_path: str
        Path to SimBA project config file in Configparser format
    measures: list
        Aggregate statistics measurements. OPTIONS: 'INTERACTION TIME (s)', 'INTERACTION BOUT COUNT', 'INTERACTION BOUT MEAN (s)', 'INTERACTION BOUT MEDIAN (s)'
    shortest_allowed_interaction: int
        The shortest allowed animal-anchored ROI intersection in millisecond.

    Notes
    ----------
    `Bounding boxes tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/anchored_rois.md__.

    Examples
    ----------
    >>> boundary_stats_calculator = AggBoundaryStatisticsCalculator('MyProjectConfig', measures=['INTERACTION TIME (s)'], shortest_allowed_interaction=200)
    >>> boundary_stats_calculator.run()
    >>> boundary_stats_calculator.save()
    """


    def __init__(self,
                 config_path: str,
                 measures: list,
                 shortest_allowed_interaction=int):

        self.config_path, self.config, self.measures = config_path, read_config_file(config_path), measures
        self.shortest_allowed_interaction_ms = shortest_allowed_interaction
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.anchored_roi_path = os.path.join(self.project_path, 'logs', 'anchored_rois.pickle')
        self.data_path = os.path.join(self.project_path, 'csv', 'anchored_roi_data')
        self.datetime = datetime.now().strftime('%Y%m%d%H%M%S')
        self.video_info = read_video_info_csv(file_path=os.path.join(self.project_path, 'logs', 'video_info.csv'))
        if not os.path.isdir(self.data_path):
            print('SIMBA ERROR: No anchored roi statistics found in {}'.format(self.data_path))
            raise ValueError
        self.files_found = glob.glob(self.data_path + '/*.pickle') + glob.glob(self.data_path + '/*.parquet') + glob.glob(self.data_path + '/*.csv')

    def run(self):
        self.results = {}
        for file_cnt, file_path in enumerate(self.files_found):
            _, self.file_name, ext = get_fn_ext(file_path)
            print('Creating aggregate statistics for video {}...'.format(self.file_name))
            _, _, fps = read_video_info(vid_info_df=self.video_info, video_name=self.file_name)

            if ext.lower() == '.csv': data_df = read_df(file_path=file_path, file_type='csv')
            elif ext.lower() == '.parquet': data_df = read_df(file_path=file_path, file_type='parquet')
            elif ext.lower() == '.pickle': data_df = pd.read_pickle(file_path)
            else:
                print('Data with extension {} is not supported (OPTIONS: csv, parquet or pickle)')
                raise ValueError()
            if (self.shortest_allowed_interaction_ms / fps) > 0:
                for column in data_df.columns:
                    data_df = plug_holes_shortest_bout(data_df=data_df, clf_name=column, fps=int(fps),shortest_bout=self.shortest_allowed_interaction_ms)
            bouts_df = detect_bouts(data_df=data_df, target_lst=list(data_df.columns), fps=int(fps))
            self.video_results, self.detailed_interactions_results = {}, {}
            if 'INTERACTION TIME (s)' in self.measures:
                self.video_results['INTERACTION TIME (s)'] = bouts_df.groupby(by='Event')['Bout_time'].sum().to_dict()
            if ('INTERACTION BOUT COUNT' in self.measures):
                self.video_results['INTERACTION BOUT COUNT'] = bouts_df.groupby(by='Event')['Bout_time'].count().to_dict()
            if ('INTERACTION BOUT TIME MEAN (s)' in self.measures):
                self.video_results['INTERACTION BOUT MEAN (s)'] = bouts_df.groupby(by='Event')['Bout_time'].mean().to_dict()
            if ('INTERACTION BOUT TIME MEDIAN (s)' in self.measures):
                self.video_results['INTERACTION BOUT MEDIAN (s)'] = bouts_df.groupby(by='Event')['Bout_time'].median().to_dict()
            if ('DETAILED INTERACTIONS TABLE' in self.measures):
                self.create_detailed_interactions_table(df=bouts_df)
            self.results[self.file_name] = self.video_results

    def save(self):
        save_path = os.path.join(self.project_path, 'logs', 'aggregate_statistics_anchored_rois_{}.csv'.format(self.datetime))
        out_df = pd.DataFrame(columns=['VIDEO', 'ANIMAL 1', 'ANIMAL 2', 'ANIMAL 2 KEYPOINT', 'MEASUREMENT', 'VALUE'])
        if len(self.results.keys()) > 0:
            for video, video_data in self.results.items():
                for measurement, measurement_data in video_data.items():
                    for animal_interaction, animal_interaction_value in measurement_data.items():
                        animal_names = animal_interaction.split(':')
                        if len(animal_names) == 2:
                            animal_names.append('None')
                        out_df.loc[len(out_df)] = [video, animal_names[0], animal_names[1], animal_names[2], measurement, animal_interaction_value]
            out_df['VALUE'] = out_df['VALUE'].round(4)
            out_df = out_df.sort_values(by=['VIDEO', 'MEASUREMENT']).set_index('VIDEO')
            out_df.to_csv(save_path)
            print('SIMBA COMPLETE: Aggregate animal-anchored ROI statistics saved at {}'.format(save_path))

        if len(self.detailed_interactions_results.keys()) > 0:
            save_path = os.path.join(self.project_path, 'logs', 'detailed_aggregate_statistics_anchored_rois_{}.csv'.format(self.datetime))
            out_df = pd.concat(self.detailed_interactions_results.values(), ignore_index=True)
            out_df = out_df.sort_values(by=['VIDEO']).set_index('VIDEO')
            out_df.to_csv(save_path)
            print('SIMBA COMPLETE: Detailed Aggregate animal-anchored ROI statistics saved at {}'.format(save_path))

    def create_detailed_interactions_table(self,
                                           df: pd.DataFrame):

        df = df.rename(columns={'Start_time': 'START TIME (s)', 'End Time': 'END TIME (s)', 'Start_frame': 'START FRAME', 'End_frame': 'END FRAME', 'Bout_time': 'BOUT TIME (s)'})
        df['ROI 1'], df['ROI 2'], df['KEY-POINT'] = df['Event'].str.split(':', 2).str
        df = df.drop(['Event'], axis=1)
        df['VIDEO'] = self.file_name
        df['BOUT FRAMES'] = (df['END FRAME'] + 1) - df['START FRAME']
        df = df[['VIDEO', 'ROI 1', 'ROI 2', 'KEY-POINT', 'START TIME (s)', 'END TIME (s)', 'START FRAME', 'END FRAME', 'BOUT FRAMES', 'BOUT TIME (s)']]
        self.detailed_interactions_results[self.file_name] = df

# boundary_stats_calculator = AggBoundaryStatisticsCalculator('/Users/simon/Desktop/troubleshooting/termites/project_folder/project_config.ini', measures=['INTERACTION TIME (s)', 'DETAILED INTERACTIONS TABLE'], shortest_allowed_interaction=200)
# boundary_stats_calculator.run()
# boundary_stats_calculator.save()
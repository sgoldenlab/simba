import os
from typing import Optional, Union

import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log,
    check_if_df_field_is_boolean, check_that_column_exist, check_valid_boolean)
from simba.utils.data import detect_bouts
from simba.utils.errors import CountError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import get_fn_ext, read_df


class AnnotationMetaDataExtractor(ConfigReader):

    """
    Extract annotation statistics (number of annotated frames, seconds, bouts etc.) for all classifiers in a SimBA project
    to MS Excel format.

    .. note::
       `Example expected output <https://github.com/sgoldenlab/simba/blob/master/misc/ANNOTATION_STATISTICS_20240713132805.xlsx>`__.

    :param Union[str, os.PathLike] config_path: path to SimBA configparser.ConfigParser project_config.ini
    :param Optional[bool] annotated_bouts: If True, includes information on annotated bouts (start and stop time and bout length). Default True.
    :param Optional[bool] split_by_video: If True, includes a worksheet where the annotation counts are split by video. Default True.

    :example:
    >>> annotation_meta_extractor = AnnotationMetaDataExtractor(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
    >>> annotation_meta_extractor.run()
    >>> annotation_meta_extractor.save()

    """
    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 split_by_video: Optional[bool] = True,
                 annotated_bouts: Optional[bool] = True):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=True)
        if len(self.clf_names) == 0:
            raise CountError(msg=f'No classifier names associated with SimBA project {config_path}', source=self.__class__.__name__)
        if len(self.target_file_paths) == 0:
            raise CountError(msg=f'No data files found inside the {self.targets_folder} directory', source=self.__class__.__name__)
        self.save_path = os.path.join(self.logs_path, f'ANNOTATION_STATISTICS_{self.datetime}.xlsx')
        check_valid_boolean(value=[annotated_bouts, split_by_video])
        self.annotated_bouts, self.split_by_video = annotated_bouts, split_by_video

    def run(self):
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.target_file_paths)
        self.results, self.bout_data = {}, []
        print(f'Analyzing annotations in {len(self.target_file_paths)} data file(s)...')
        for file_cnt, file_path in enumerate(self.target_file_paths):
            _, video_name, _ = get_fn_ext(filepath=file_path)
            self.results[video_name] = {}
            _, _, fps = self.read_video_info(video_name=video_name)
            print(f'Analyzing annotations in {video_name }... ')
            df = read_df(file_path=file_path, file_type=self.file_type)
            check_that_column_exist(df=df, column_name=self.clf_names, file_name=file_path)
            bouts = detect_bouts(data_df=df, target_lst=self.clf_names, fps=fps)
            bouts.columns = ['ANNOTATED CLASSIFIER', 'ANNOTATED START TIME', 'ANNOTATED END TIME', 'ANNOTATED START FRAME','ANNOTATED END FRAME','ANNOTATED BOUT TIME (S)']
            bouts['VIDEO'] = video_name
            bouts = bouts[['VIDEO', 'ANNOTATED CLASSIFIER', 'ANNOTATED START TIME', 'ANNOTATED END TIME', 'ANNOTATED START FRAME','ANNOTATED END FRAME','ANNOTATED BOUT TIME (S)']]
            self.bout_data.append(bouts)
            for clf in self.clf_names:
                check_if_df_field_is_boolean(df=df, field=clf, df_name=file_path)
                present_df, absent_df = df[df[clf] == 1], df[df[clf] == 0]
                self.results[video_name][clf] = {f'ANNOTATED PRESENT FRAME COUNT': len(present_df),
                                                 f'ANNOTATED PRESENT TIME (S)': round((len(present_df) / fps), 4),
                                                 f'ANNOTATED ABSENT FRAMES COUNT': len(absent_df),
                                                 f'ANNOTATED ABSENT TIME (S)': round((len(absent_df) / fps), 4)}


    def __aggregates(self):
        self.by_video = pd.DataFrame(columns=['VIDEO', 'CLASSIFIER', 'ANNOTATION MEASUREMENT', 'ANNOTATION STATISTIC'])
        for video_name, video_data in self.results.items():
            for clf in self.clf_names:
                for clf_key, clf_data in self.results[video_name][clf].items():
                    self.by_video.loc[len(self.by_video)] = [video_name, clf,  clf_key, clf_data]
        self.aggregates = pd.DataFrame(self.by_video.drop(['VIDEO'], axis=1).groupby(by=['CLASSIFIER', 'ANNOTATION MEASUREMENT'])['ANNOTATION STATISTIC'].sum())
        self.by_video = self.by_video.set_index(['VIDEO', 'CLASSIFIER'])

    def save(self):
        self.__aggregates()
        self.bout_data = pd.concat(self.bout_data, axis=0)
        with pd.ExcelWriter(self.save_path) as writer:
            self.aggregates.to_excel(writer, sheet_name='TOTAL ANNOTATION COUNTS', index=True)
            if self.split_by_video:
                self.by_video.to_excel(writer, sheet_name='VIDEO ANNOTATION COUNTS', index=True)
            if self.annotated_bouts:
                self.bout_data.to_excel(writer, sheet_name='VIDEO ANNOTATION BOUT DATA', index=False)
        self.timer.stop_timer()
        stdout_success(msg=f'Annotation data for {len(self.target_file_paths)} video(s) saved at {self.save_path}', source=self.__class__.__name__, elapsed_time=self.timer.elapsed_time)


# annotation_meta_extractor = AnnotationMetaDataExtractor(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
# annotation_meta_extractor.run()
# annotation_meta_extractor.save()
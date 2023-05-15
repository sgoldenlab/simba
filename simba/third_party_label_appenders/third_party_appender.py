import pandas as pd


from simba.utils.enums import Methods
import os, glob
from copy import deepcopy
from simba.mixins.config_reader import ConfigReader
from simba.utils.read_write import read_df, write_df, get_fn_ext
from simba.utils.checks import check_if_filepath_list_is_empty, check_if_dir_exists
from simba.utils.printing import stdout_success
from simba.third_party_label_appenders.tools import (read_boris_annotation_files,
                                                     read_deepethogram_files,
                                                     read_ethovision_files,
                                                     read_observer_files,
                                                     read_solomon_files,
                                                     read_bento_files,
                                                     fix_uneven_start_stop_count,
                                                     check_stop_events_prior_to_start_events)
from simba.utils.warnings import (ThirdPartyAnnotationFileNotFoundWarning,
                                  ThirdPartyAnnotationsAdditionalClfWarning,
                                  ThirdPartyAnnotationsMissingAnnotationsWarning,
                                  ThirdPartyAnnotationEventCountWarning,
                                  ThirdPartyAnnotationOverlapWarning,
                                  ThirdPartyAnnotationsOutsidePoseEstimationDataWarning)
from simba.utils.errors import (ThirdPartyAnnotationFileNotFoundError,
                                ThirdPartyAnnotationsAdditionalClfError,
                                ThirdPartyAnnotationsMissingAnnotationsError,
                                ThirdPartyAnnotationEventCountError,
                                ThirdPartyAnnotationOverlapError,
                                ThirdPartyAnnotationsOutsidePoseEstimationDataError)



BORIS = 'BORIS'
DEEPETHOGRAM = 'DEEPETHOGRAM'
ETHOVISION = 'ETHOVISION'
OBSERVER = 'OBSERVER'
SOLOMON = 'SOLOMON'
BENTO = 'BENTO'
BEHAVIOR = 'BEHAVIOR'

class ThirdPartyLabelAppender(ConfigReader):

    """
    Concatenate third-party annotations to featurized pose-estimation datasets in SimBA.

    :param str app: Third-party application. OPTIONS: ['BORIS', 'BENTO', 'DEEPETHOGRAM', 'ETHOVISION', 'OBSERVER', 'SOLOMON'].
    :param str config_path: path to SimBA project config file in Configparser format.
    :param str data_dir: Directory holding third-party annotation data files.
    :param dict settings: User-defined settings including how to handle errors, logging, and data file types associated with the third-party application.

    ... note::
       `Third-party import tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/third_party_annot.md>`__.

       `BENTO: expected input <https://github.com/sgoldenlab/simba/blob/master/misc/bento_example.annot`__.
       `BORIS: expected input <https://github.com/sgoldenlab/simba/blob/master/misc/boris_example.csv>`__..
       `DEEPETHOGRAM: expected input <https://github.com/sgoldenlab/simba/blob/master/misc/deep_ethogram_labels.csv>`__.
       `ETHOVISION: expected input <https://github.com/sgoldenlab/simba/blob/master/misc/ethovision_example.xlsx>`__.
       `OBSERVER: expected input I <https://github.com/sgoldenlab/simba/blob/master/misc/Observer_example_1.xlsx>`__.
       `OBSERVER: expected input II <https://github.com/sgoldenlab/simba/blob/master/misc/Observer_example_2.xlsx>`__.
       `SOLOMON: expected input II <https://github.com/sgoldenlab/simba/blob/master/misc/solomon_example.csv>`__.


    Examples
    ----------
    >>> settings = {'log': True,  'file_format': 'csv', 'errors': {'INVALID annotations file data format': 'WARNING',
    >>>                                                             'ADDITIONAL third-party behavior detected': 'NONE',
    >>>                                                             'Annotations EVENT COUNT conflict': 'WARNING',
    >>>                                                             'Annotations OVERLAP inaccuracy': 'WARNING',
    >>>                                                             'ZERO third-party video behavior annotations found': 'WARNING',
    >>>                                                             'Annotations and pose FRAME COUNT conflict': 'WARNING',
    >>>                                                             'Annotations data file NOT FOUND': 'WARNING'}}
    >>> third_party_importer = ThirdPartyLabelAppender(app='BORIS', config_path=r'MySimBAConfigPath', data_dir=r'MyDeepEthogramDir', settings=settings)
    >>> third_party_importer.run()

    References
    ----------

    .. [1] `DeepEthogram repo <https://github.com/jbohnslav/deepethogram>`__.
    .. [2]  Segalin et al., eLife, https://doi.org/10.7554/eLife.63720
    .. [3] `Behavioral Observation Research Interactive Software (BORIS) user guide <https://boris.readthedocs.io/en/latest/#>`__.
    .. [4] `Noldus Ethovision XT <https://www.noldus.com/ethovision-xt>`__.
    .. [5] `Noldus Observer XT <https://www.noldus.com/observer-xt>`__.
    .. [6] `Solomon coder user-guide (PDF) <https://solomon.andraspeter.com/Solomon%20Intro.pdf>`__.

    """
    def __init__(self,
                 app: str,
                 config_path: str,
                 data_dir: str,
                 settings: dict):

        super().__init__(config_path=config_path)
        check_if_dir_exists(in_dir=data_dir)
        self.data_dir, self.app, self.settings = data_dir, app, settings
        self.err_settings = self.settings['errors']
        if self.settings['log']:
            self.create_logger(path=os.path.join(self.logs_path, f'BORIS_append_{self.datetime}.log'))
        self.data_file_paths = glob.glob(self.data_dir + f'/*.{self.settings["file_format"]}')
        self.data_file_paths = [x for x in self.data_file_paths if '~$' not in x]
        check_if_filepath_list_is_empty(filepaths=self.data_file_paths,
                                        error_msg=f'SIMBA ERROR: ZERO {app} {self.settings["file_format"]} files found in {data_dir} directory')
        check_if_filepath_list_is_empty(filepaths=self.feature_file_paths,
                                        error_msg='SIMBA ERROR: ZERO files found in the project_folder/csv/features_extracted directory')
        print(f'Processing {str(len(self.feature_file_paths))} {app} file(s)...')

    def __check_annotation_clf_df_integrity(self, df=pd.DataFrame):
        clf_name = df['BEHAVIOR'].loc[0]
        if len(df[df['EVENT'] == 'START']) != len(df[df['EVENT'] == 'STOP']):
            if self.err_settings[Methods.THIRD_PARTY_EVENT_COUNT_CONFLICT.value] == Methods.WARNING.value:
                ThirdPartyAnnotationEventCountWarning(video_name=self.video_name, clf_name=clf_name,
                                                      start_event_cnt=len(df[df['EVENT'] == 'START']),
                                                      stop_event_cnt=len(df[df['EVENT'] == 'STOP']),
                                                      log_status=self.settings['log'])
                df = fix_uneven_start_stop_count(data=df)


            elif self.err_settings[Methods.THIRD_PARTY_EVENT_COUNT_CONFLICT.value] == Methods.ERROR.value:
                raise ThirdPartyAnnotationEventCountError(video_name=self.video_name, clf_name=clf_name,
                                                          start_event_cnt=len(df[df['EVENT'] == 'START']),
                                                          stop_event_cnt=len(df[df['EVENT'] == 'STOP']))

        else:
            start = df['FRAME'][df['EVENT'] == 'START'].reset_index(drop=True)
            stop = df['FRAME'][df['EVENT'] == 'STOP'].reset_index(drop=True)
            df = pd.concat([start, stop], axis=1)
            df.columns = ['START', 'STOP']

        overlaps_idx = check_stop_events_prior_to_start_events(df=df)
        if overlaps_idx:
            if self.err_settings[Methods.THIRD_PARTY_EVENT_OVERLAP.value] == Methods.WARNING.value:
                ThirdPartyAnnotationOverlapWarning(video_name=self.video_name, clf_name=clf_name, log_status=self.settings['log'])
                df = df.drop(index=overlaps_idx).reset_index(drop=True)

            elif self.err_settings[Methods.THIRD_PARTY_EVENT_OVERLAP.value] == Methods.ERROR.value:
                raise ThirdPartyAnnotationOverlapError(video_name=self.video_name, clf_name=clf_name)

        return df

    def run(self):
        data = None
        print(f'Reading in {str(len(self.data_file_paths))} {self.app} annotation files...')
        if self.app == BORIS:
            data = read_boris_annotation_files(data_paths=self.data_file_paths,
                                               error_setting=self.settings['errors'][Methods.INVALID_THIRD_PARTY_APPENDER_FILE.value],
                                               video_info_df=self.video_info_df,
                                               log_setting=self.settings['log'])
        elif self.app == DEEPETHOGRAM:
            data = read_deepethogram_files(data_paths=self.data_file_paths,
                                           error_setting=self.settings['errors'][Methods.INVALID_THIRD_PARTY_APPENDER_FILE.value],
                                           log_setting=self.settings['log'])
        elif self.app == ETHOVISION:
            data = read_ethovision_files(data_paths=self.data_file_paths,
                                               error_setting=self.settings['errors'][Methods.INVALID_THIRD_PARTY_APPENDER_FILE.value],
                                               video_info_df=self.video_info_df,
                                               log_setting=self.settings['log'])
        elif self.app == OBSERVER:
            data = read_observer_files(data_paths=self.data_file_paths,
                                         error_setting=self.settings['errors'][Methods.INVALID_THIRD_PARTY_APPENDER_FILE.value],
                                         video_info_df=self.video_info_df,
                                         log_setting=self.settings['log'])

        elif self.app == SOLOMON:
            data = read_solomon_files(data_paths=self.data_file_paths,
                                         error_setting=self.settings['errors'][Methods.INVALID_THIRD_PARTY_APPENDER_FILE.value],
                                         video_info_df=self.video_info_df,
                                         log_setting=self.settings['log'])

        elif self.app == BENTO:
            data = read_bento_files(data_paths=self.data_file_paths,
                                         error_setting=self.settings['errors'][Methods.INVALID_THIRD_PARTY_APPENDER_FILE.value],
                                         video_info_df=self.video_info_df,
                                         log_setting=self.settings['log'])

        for file_cnt, file_path in enumerate(self.feature_file_paths):
            _, self.video_name, _ = get_fn_ext(filepath=file_path)
            print(f'Processing annotations for {self.video_name} video...')
            if self.video_name not in data.keys():
                if self.err_settings[Methods.THIRD_PARTY_ANNOTATION_FILE_NOT_FOUND.value] == Methods.WARNING.value:
                    ThirdPartyAnnotationFileNotFoundWarning(video_name=self.video_name, log_status=self.settings['log'])
                    continue
                if self.err_settings[Methods.THIRD_PARTY_ANNOTATION_FILE_NOT_FOUND.value] == Methods.ERROR.value:
                    raise ThirdPartyAnnotationFileNotFoundError(video_name=self.video_name)
            annot_df = data[self.video_name].drop_duplicates().reset_index(drop=True)
            additional_clfs = list(set(annot_df[BEHAVIOR].unique()) - set(self.clf_names))
            if additional_clfs and self.err_settings[Methods.ADDITIONAL_THIRD_PARTY_CLFS.value] == Methods.WARNING.value:
                ThirdPartyAnnotationsAdditionalClfWarning(video_name=self.video_name, clf_names=additional_clfs, log_status=self.settings['log'])
            elif additional_clfs and self.err_settings[Methods.ADDITIONAL_THIRD_PARTY_CLFS.value] == Methods.ERROR.value:
                raise ThirdPartyAnnotationsAdditionalClfError(video_name=self.video_name, clf_names=additional_clfs)

            features_df = read_df(file_path=file_path, file_type=self.file_type)
            out_df = deepcopy(features_df)
            for clf in self.clf_names:
                clf_annot = annot_df[(annot_df[BEHAVIOR] == clf)].reset_index(drop=True)

                if len(clf_annot) == 0:
                    if self.err_settings[Methods.ZERO_THIRD_PARTY_VIDEO_BEHAVIOR_ANNOTATIONS.value] == Methods.WARNING.value:
                        ThirdPartyAnnotationsMissingAnnotationsWarning(video_name=self.video_name, clf_names=self.clf_names, log_status=self.settings['log'])
                        out_df[clf] = 0
                        continue
                    elif self.err_settings[Methods.ZERO_THIRD_PARTY_VIDEO_BEHAVIOR_ANNOTATIONS.value] == Methods.ERROR.value:
                        raise ThirdPartyAnnotationsMissingAnnotationsError(video_name=self.video_name, clf_names=self.clf_names)
                clf_annot = self.__check_annotation_clf_df_integrity(df=clf_annot)
                annot_idx = list(clf_annot.apply(lambda x: list(range(int(x['START']), int(x['STOP']) + 1)), 1))
                annot_idx = [x for xs in annot_idx for x in xs]
                idx_diff = list(set(annot_idx) - set(out_df.index))
                if len(idx_diff) > 0:
                    if self.err_settings[Methods.THIRD_PARTY_FRAME_COUNT_CONFLICT.value] == Methods.WARNING.value:
                        ThirdPartyAnnotationsOutsidePoseEstimationDataWarning(video_name=self.video_name,
                                                                              clf_name=clf,
                                                                              frm_cnt=out_df.index[-1],
                                                                              first_error_frm=idx_diff[0],
                                                                              ambiguous_cnt=len(idx_diff),
                                                                              log_status=self.settings['log'])
                    elif self.err_settings[Methods.THIRD_PARTY_FRAME_COUNT_CONFLICT.value] == Methods.ERROR.value:
                        raise ThirdPartyAnnotationsOutsidePoseEstimationDataError(video_name=self.video_name,
                                                                                  clf_name=clf,
                                                                                  frm_cnt=out_df.index[-1],
                                                                                  first_error_frm=idx_diff[0],
                                                                                  ambiguous_cnt=len(idx_diff))

                annot_idx = [x for x in annot_idx if x not in idx_diff]
                out_df[clf] = 0
                out_df.loc[annot_idx, clf] = 1
            save_path = os.path.join(self.targets_folder, self.video_name + '.' + self.file_type)
            write_df(out_df, self.file_type, save_path)
            print(f'Saved {self.app} annotations for video {self.video_name}...')
        self.timer.stop_timer()
        stdout_success(msg=f'{self.app} annotations appended to dataset and saved in project_folder/csv/targets_inserted directory', elapsed_time=self.timer.elapsed_time_str)

# settings = {'log': True,  'file_format': 'xlsx', 'errors': {'INVALID annotations file data format': 'WARNING',
#                                                            'ADDITIONAL third-party behavior detected': 'NONE',
#                                                            'Annotations EVENT COUNT conflict': 'WARNING',
#                                                            'Annotations OVERLAP inaccuracy': 'WARNING',
#                                                            'ZERO third-party video behavior annotations found': 'WARNING',
#                                                            'Annotations and pose FRAME COUNT conflict': 'WARNING',
#                                                            'Annotations data file NOT FOUND': 'WARNING'}}
# #
#
#
# test = ThirdPartyLabelAppender(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                data_dir='/Users/simon/Downloads/FIXED',
#                                settings=settings,
#                                app='BORIS')
# test.run()


# test = ThirdPartyLabelAppender(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                data_dir='/Users/simon/Desktop/envs/simba_dev/tests/test_data/deepethogram_example',
#                                settings=settings,
#                                app='DEEPETHOGRAM')
# test.run()


# test = ThirdPartyLabelAppender(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                data_dir=r'/Users/simon/Desktop/envs/simba_dev/tests/test_data/import_tests/ethovision_data',
#                                settings=settings,
#                                app='ETHOVISION')
# test.run()

# test = ThirdPartyLabelAppender(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                data_dir=r'/Users/simon/Desktop/envs/simba_dev/tests/test_data/solomon_import/solomon_import',
#                                settings=settings,
#                                app='SOLOMON')
# test.run()


# test = ThirdPartyLabelAppender(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                data_dir=r'/Users/simon/Desktop/envs/simba_dev/tests/test_data/bento_example',
#                                settings=settings,
#                                app='BENTO')
# test.run()
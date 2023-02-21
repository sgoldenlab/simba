import os, glob
from simba.features_scripts.unit_tests import read_video_info_csv, read_video_info
from simba.read_config_unit_tests import (read_config_file,
                                          read_project_path_and_file_type,
                                          read_config_entry,
                                          check_if_filepath_list_is_empty)
from simba.misc_tools import get_fn_ext
from simba.train_model_functions import get_all_clf_names
from simba.enums import Paths, ReadConfig, Dtypes
from simba.rw_dfs import read_df, save_df
import pandas as pd
from copy import deepcopy

class BentoAppender(object):
    """
    Class for appending BENTO behavioral annotation to SimBA featurized datasets.

    Notes
    ----------
    `Example BORIS input file <https://github.com/sgoldenlab/simba/blob/master/misc/boris_example.csv`__.

    Examples
    ----------
    >>> bento_dir = 'tests/test_data/bento_example'
    >>> bento_appender = BentoAppender(config_path='SimBAProjectConfig', bento_dir=bento_dir)
    >>> bento_appender.read_files()
    """

    def __init__(self,
                 config_path: str,
                 bento_dir: str):

        self.config, self.bento_dir = read_config_file(ini_path=config_path), bento_dir
        self.project_path, self.file_type = read_project_path_and_file_type(config=self.config)
        self.features_dir = os.path.join(self.project_path, Paths.FEATURES_EXTRACTED_DIR.value)
        self.targets_folder = os.path.join(self.project_path, Paths.TARGETS_INSERTED_DIR.value)
        self.video_info_df = read_video_info_csv(os.path.join(self.project_path, Paths.VIDEO_INFO.value))
        self.target_cnt = read_config_entry(self.config, ReadConfig.SML_SETTINGS.value, ReadConfig.TARGET_CNT.value, Dtypes.INT.value)
        self.clf_names = get_all_clf_names(config=self.config, target_cnt=self.target_cnt)
        self.feature_files = glob.glob(self.features_dir + '/*.' + self.file_type)
        self.bento_files = glob.glob(self.bento_dir + '/*.' + 'annot')
        check_if_filepath_list_is_empty(filepaths=self.feature_files, error_msg='SIMBA ERROR: No feature files found in project_folder/csv/features_extracted. Extract Features BEFORE appending BENTO annotations')
        check_if_filepath_list_is_empty(filepaths=self.bento_files, error_msg=f'SIMBA ERROR: No BENTO files with .annot extension found in {self.bento_dir}.')
        self.total_annotations = 0
        self.saved_files = []


    def read_files(self):
        for file_cnt, file_path in enumerate(self.feature_files):
            _, self.video_name, ext = get_fn_ext(filepath=file_path)
            print(f'Appending BENTO annotation to video {self.video_name}...')
            _, _, fps = read_video_info(vid_info_df=self.video_info_df, video_name=self.video_name)
            bento_path = os.path.join(self.bento_dir, self.video_name + '.annot')
            if bento_path not in self.bento_files:
                print('SIMBA WARNING: No BENTO annotations found for {}. SimBA expected the file path {} but could not find it. SimBA is skipping appending annotations to video {}'.format(video_name, bento_path, video_name))
                continue
            self.save_path = os.path.join(self.targets_folder, self.video_name + '.' + self.file_type)
            feature_df = read_df(file_path=file_path, file_type=self.file_type)
            video_frm_length = len(feature_df)
            self.results_df = deepcopy(feature_df)
            annotation_df = pd.read_csv(bento_path, delim_whitespace=True, index_col=False, low_memory=False)
            start_idx = annotation_df.index[annotation_df['Bento'] == 'Ch1----------'].values[0]
            sliced_annot = annotation_df.iloc[start_idx+1:]
            annotated_behaviors = sliced_annot[sliced_annot['Bento'].str.contains(">")]['Bento'].tolist()
            annotated_behavior_names = [x[1:] for x in annotated_behaviors]
            missing_annotation = set(self.clf_names) - set(annotated_behavior_names)
            missing_clf = set(annotated_behavior_names) - set(self.clf_names)
            annotation_intersection = [x for x in self.clf_names if x in annotated_behavior_names]
            self.total_annotations += len(annotation_intersection)
            if len(missing_annotation) > 0:
                print('SIMBA WARNING: BENTO annotation file for video {} has no annotations for the following behaviors {}. SimBA will set all frame labels in video {} as behavior ABSENT for these behaviors'.format(video_name, missing_annotation, video_name))
            if len(missing_clf) > 0:
                print('SIMBA WARNING: BENTO annotation file for video {} has annotations for the following behaviors {} that are not classifiers named in the SimBA project. SimBA will OMIT appending the data for these classifiers'.format(video_name, missing_clf))
            for missing_clf in missing_annotation:
                self.results_df[missing_clf] = 0

            for clf_name in annotation_intersection:
                self.results_df[clf_name] = 0
                clf_start_idx = sliced_annot.index[sliced_annot['Bento'] == f'>{clf_name}'].values[0]
                clf_df = sliced_annot.loc[clf_start_idx+2:, :]
                end_idx = clf_df.isnull()[clf_df.isnull().any(axis=1)].idxmax(axis=1).index
                if end_idx.values:
                    end_idx = end_idx.values[0]
                else:
                    end_idx = max(clf_df.index+1)
                clf_df = clf_df.loc[:end_idx-1, :].reset_index(drop=True)
                clf_df.columns = ['start_time', 'stop_time', 'duration']
                clf_df['start_frm'] = clf_df['start_time'].astype(float) * fps
                clf_df['end_frm'] = clf_df['stop_time'].astype(float) * fps
                clf_df['start_frm'] = clf_df['start_frm'].astype(int)
                clf_df['end_frm'] = clf_df['end_frm'].astype(int)
                annotations_idx = list(clf_df.apply(lambda x: list(range(int(x['start_frm']), int(x['end_frm']) + 1)), 1))
                annotations_idx = [i for s in annotations_idx for i in s]
                annotations_idx_outside_video = [x for x in annotations_idx if x > video_frm_length]
                valid_annotation_ids = [x for x in annotations_idx if x <= video_frm_length]
                if len(annotations_idx_outside_video):
                    print(f'SIMBA WARNING: SimBA found {str(len(annotations_idx_outside_video))} {clf_name} BENTO annotations that occured at frame number {str(min(annotations_idx_outside_video))} or later. The pose-estimated video is only {str(video_frm_length)} frames long so SimBA doesnt know what to do with these annotations and will SKIP them.')
                if len(valid_annotation_ids) > 0:
                    print(f'Appending {str(len(valid_annotation_ids))} {clf_name} frame annotations to video {self.video_name}...')
                    self.results_df.loc[valid_annotation_ids, clf_name] = 1
            self.save()
        print(f'SIMBA COMPLETE: {str(self.total_annotations)} frame annotations for {str(len(self.saved_files))} video(s) and saved in project_folder/csv/targets_inserted directory.')

    def save(self):
        save_df(df=self.results_df, file_type=self.file_type, save_path=self.save_path)
        self.saved_files.append(self.save_path)
        print(f'BENTO annotations appended to video {self.video_name} and saved in {self.save_path}')



















#
#
#
#
# def append_dot_ANNOTT(inifile, annotationsFolder):
#     configFile = str(inifile)
#     config = ConfigParser()
#     try:
#         config.read(configFile)
#     except MissingSectionHeaderError:
#         print('ERROR:  Not a valid project_config file. Please check the project_config.ini path.')
#     projectPath = config.get('General settings', 'project_path')
#     try:
#         wfileType = config.get('General settings', 'workflow_file_type')
#     except NoOptionError:
#         wfileType = 'csv'
#     featureFilesFolder = os.path.join(projectPath, 'csv', 'features_extracted')
#     targetFolder = os.path.join(projectPath, 'csv', 'targets_inserted')
#     videoLogDf = read_video_info_csv(os.path.join(projectPath, 'logs', 'video_info.csv'))
#     featureFiles = glob.glob(featureFilesFolder + '/*.' + wfileType)
#     noTargets = config.getint('SML settings', 'no_targets')
#     targetList = []
#
#     for i in range(noTargets):
#         currTargName = 'target_name_' + str(i + 1)
#         currTargVal = config.get('SML settings', currTargName)
#         targetList.append('>' + currTargVal)
#
#     for i in range(len(featureFiles)):
#         curVideoFileName = os.path.basename(featureFiles[i]).replace('.' + wfileType, '')
#         print('Processing file ' + str(curVideoFileName) + '...')
#         currVideoSettings, currPixPerMM, fps = read_video_info(vidinfDf=videoLogDf, currVidName=curVideoFileName)
#         currAnnotationFilePath = os.path.join(annotationsFolder, curVideoFileName + '.annot')
#         currFeatureFilePath = os.path.join(featureFilesFolder, curVideoFileName + '.' + wfileType)
#         try:
#             currAnnotFile = pd.read_csv(currAnnotationFilePath, delim_whitespace=True,index_col=False, low_memory=False)
#         except FileNotFoundError:
#             print('Could not find the ' + str(currAnnotationFilePath)+ ' and / or ' + str(currFeatureFilePath) +' files. Make sure the file exist.')
#         currFeatureFile = read_df(currFeatureFilePath, wfileType)
#         currFeatureFile = currFeatureFile.set_index([0])
#         StartIndex = currAnnotFile.index[currAnnotFile['Bento'] == 'Ch1----------'].tolist()
#         clippedAnnot = currAnnotFile.iloc[StartIndex[0]+1:]
#         behavsInFile = clippedAnnot[clippedAnnot['Bento'].str.contains(">")]
#         behavsInFile = behavsInFile.reset_index()
#         for behavs in range(len(targetList)):
#             behaviorName = targetList[behavs].replace('>', '')
#             foundCheck = behavsInFile.index[behavsInFile['Bento'].str.contains(targetList[behavs])]
#             if (len(foundCheck) == 1):
#                 foundChecker = int(foundCheck[0])
#                 if foundChecker < 0:
#                     foundChecker = 0
#                 if foundChecker != len(behavsInFile)-1:
#                     index1, index2 = behavsInFile.loc[foundChecker, 'index'], behavsInFile.loc[foundChecker+1, 'index']
#                 if foundChecker == len(behavsInFile)-1:
#                     index1, index2 = behavsInFile.loc[foundChecker, 'index'], len(currAnnotFile)
#                 currBehav = currAnnotFile.iloc[index1:index2]
#                 currBehav = currBehav.iloc[2:]
#                 currBehav.columns = ['Start', 'Stop', 'Duration']
#                 currFeatureFile[behaviorName] = 0
#                 currBehav = currBehav.apply(pd.to_numeric)
#                 boutLists = []
#                 for index, row in currBehav.iterrows():
#                     startFrame, endFrame = int((row['Start'] * fps)), int((row['Stop'] * fps))
#                     boutLists.append(list(range(startFrame, endFrame)))
#                 frames, seconds = (sum(map(len, boutLists))), int((sum(map(len, boutLists))) / fps)
#                 print('Appending. Video: ' + str(curVideoFileName) + '. Behaviour: ' + str(behaviorName) + '. # frames labelled with behavior present: ' + str(frames) + '. Seconds of behavior labelled as present: ' + str(seconds))
#                 for bout in boutLists:
#                     for frame in bout:
#                         currFeatureFile.loc[frame, behaviorName] = 1
#             else:
#                 print('No annotations in json for: ' + str(behaviorName) + ' in video ' + str(curVideoFileName) + '. Setting all ' + str(behaviorName) + ' annotations to 0 in video ' + str(curVideoFileName) + '.')
#                 currFeatureFile[behaviorName] = 0
#         savePath = os.path.join(targetFolder, curVideoFileName + '.' + str(wfileType))
#         print('Saving file...')
#         save_df(currFeatureFile, wfileType, savePath)
#         print('Saved annotations for file ' + str(curVideoFileName) + '.')
#     print('All BENTO annotations appended to respective feature files.')
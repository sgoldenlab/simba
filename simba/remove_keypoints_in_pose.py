import glob, os
import pandas as pd
from datetime import datetime
import warnings
from tables import NaturalNameWarning
from simba.misc_tools import SimbaTimer
from simba.read_config_unit_tests import check_if_filepath_list_is_empty
warnings.filterwarnings('ignore', category=NaturalNameWarning)

class KeypointRemover(object):
    """
    Class for removing pose-estimated keypoints from data in CSV or H5 format.

    Parameters
    ----------
    data_folder: str
        Path to directory containing pose-estiation CSV or H5 data
    pose_tool: str
        Tool used to perform pose-estimation.
    file_format: str
        File type of pose-estimation data.

    Notes
    ----------
    `GitHub tutorial/documentation <https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#remove-body-parts-from-tracking-data>`__.

    Examples
    ----------
    >>> keypoint_remover = KeypointRemover(data_folder="MyDataFolder", pose_tool='maDLC', file_format='h5')
    >>> keypoint_remover.run_bp_removal(bp_to_remove_list=['Nose_1, Nose_2'])
    """

    def __init__(self,
                 data_folder: str,
                 pose_tool: str,
                 file_format: str):

        if not os.path.isdir(data_folder):
            print('SIMBA ERROR: {} is not a valid directory.'.format(str(data_folder)))
            raise NotADirectoryError
        self.files_found = glob.glob(data_folder + '/*.' + file_format)
        check_if_filepath_list_is_empty(filepaths=self.files_found,
                                        error_msg='SIMBA ERROR: Zero files found of type {} in the {} directory'.format(file_format, data_folder))
        self.datetime = str(datetime.now().strftime('%Y%m%d%H%M%S'))
        self.pose_tool, self.data_folder = pose_tool, data_folder
        self.file_format = file_format
        if file_format == 'h5':
            first_df = pd.read_hdf(self.files_found[0])
        else:
            first_df = pd.read_csv(self.files_found[0], header=[0, 1, 2])
        header_list = list(first_df.columns)[1:]
        self.body_part_names, self.animal_names = [], []

        if pose_tool == 'DLC':
            for header_entry in header_list:
                if header_entry[1] not in self.body_part_names:
                    self.body_part_names.append(header_entry[1])

        else:
            for header_entry in header_list:
                if header_entry[1] not in self.body_part_names:
                    self.animal_names.append(header_entry[1])
                    self.body_part_names.append(header_entry[2])

        self.body_part_names, self.animal_names = list(set(self.body_part_names)), list(set(self.animal_names))

    def run_bp_removal(self, animal_names: list, bp_to_remove_list: list):
        self.timer = SimbaTimer()
        self.timer.start_timer()
        save_directory = os.path.join(self.data_folder, 'Reorganized_bp_{}'.format(self.datetime))
        if not os.path.exists(save_directory): os.makedirs(save_directory)
        print('Saving {} new pose-estimation files in {} directory...'.format(str(len(self.files_found)), save_directory))
        if (self.pose_tool == 'DLC') or (self.pose_tool == 'maDLC'):
            for file_cnt, file_path in enumerate(self.files_found):
                save_path = os.path.join(save_directory, os.path.basename(file_path))
                if self.file_format == 'csv':
                    self.df = pd.read_csv(file_path, header=[0, 1, 2], index_col=0)
                    for body_part in bp_to_remove_list:
                        if body_part not in self.df.columns._levels[1]:
                            print('SIMBA ERROR: {} key point is not present in file {}'.format(body_part, file_path))
                            raise KeyError
                        self.df = self.df.drop(body_part, axis=1, level=1)
                    self.df.to_csv(save_path)
                if self.file_format == 'h5':
                    self.df = pd.read_hdf(file_path)
                    try:
                        first_header_value = self.df.columns._levels[0].values[0]
                    except:
                        print('SIMBA ERROR: {} is not a valid maDLC pose-estimation file'.format(file_path))
                        raise ValueError
                    for (body_part, animal_name) in zip(bp_to_remove_list, animal_names):
                        for cord in ['x', 'y', 'likelihood']:
                            try:
                                self.df = self.df.drop((first_header_value, animal_name, body_part, cord), axis=1)
                            except:
                                print('SIMBA ERROR: Could not find body part {} in {}'.format(body_part, file_path))
                                raise ValueError
                    self.df.to_hdf(save_path, key='re-organized', format='table', mode='w')
                print('Saved {}, Video {}/{}.'.format(os.path.basename(file_path), str(file_cnt + 1), str(len(self.files_found))))
            self.timer.stop_timer()
            print('SIMBA COMPLETE: {} new data with {} body-parts removed saved in {} directory (elapsed time {}s)'.format(str(len(self.files_found)), str(len(bp_to_remove_list)), save_directory, self.timer.elapsed_time_str))
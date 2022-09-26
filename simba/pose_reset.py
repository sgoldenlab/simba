__author__ = "Simon Nilsson", "JJ Choong"

import os
import simba
import shutil
from simba.misc_tools import check_file_exist_and_readable
import pandas as pd

class PoseResetter(object):
    """
    Class for deleting all user-defined pose-estimation schematics, diagrams and other settings from the
    SimBA installation

    Parameters
    ----------
    master: tk
        tkinter master window, (default=None).

    Notes
    ----------

    Examples
    ----------
    >>> _ = PoseResetter(master=None)
    """

    def __init__(self,
                 master=None):
        self.default_pose_configs_cnt = 13
        self.simba_dir = os.path.dirname(simba.__file__)
        self.pose_configs_dir = os.path.join(self.simba_dir, 'pose_configurations')
        self.archive_dir = os.path.join(self.simba_dir, 'pose_configurations_archive')
        if not os.path.exists(self.archive_dir): os.makedirs(self.archive_dir)
        self.archive_folders_cnt = len(os.listdir(self.archive_dir))
        self.archive_subdir = os.path.join(self.archive_dir, 'pose_configurations_archive_{}'.format(str(self.archive_folders_cnt+1)))
        shutil.copytree(self.pose_configs_dir, self.archive_subdir)

        self.bp_names_csv_path = os.path.join(self.pose_configs_dir, 'bp_names','bp_names.csv')
        self.pose_config_names_csv_path = os.path.join(self.pose_configs_dir, 'configuration_names', 'pose_config_names.csv')
        self.no_animals_csv_path = os.path.join(self.pose_configs_dir, 'no_animals', 'no_animals.csv')
        self.schematics_path = os.path.join(self.pose_configs_dir, 'schematics')

        for file_path in [self.bp_names_csv_path, self.pose_config_names_csv_path, self.no_animals_csv_path, self.no_animals_csv_path]:
            check_file_exist_and_readable(file_path=file_path)
            df = pd.read_csv(file_path, header=None, error_bad_lines=False)
            df = df.iloc[0:self.default_pose_configs_cnt]
            df.to_csv(file_path, index=False, header=False)

        default_pic_list = []
        user_pic_lst = os.listdir(self.schematics_path)
        for idx in range(self.default_pose_configs_cnt):
            default_pic_list.append('Picture{}.png'.format(str(idx+1)))
        for i in list(set(user_pic_lst) - set(default_pic_list)):
            os.remove(os.path.join(self.schematics_path, i))

        print('SIMBA COMPLETE: Pose-estimation configuration reset. All user-defined poses removed.')
        master.destroy()

# test = PoseResetter()





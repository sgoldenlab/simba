__author__ = "Simon Nilsson"

import os
import simba
import shutil
import pandas as pd
from tkinter import *
from typing import Optional

from simba.utils.printing import stdout_trash
from simba.utils.lookups import get_bp_config_codes
from simba.utils.checks import check_file_exist_and_readable


class PoseResetter(object):
    """
    Launch GUI for deleting all **user-defined** pose-estimation schematics, diagrams and other settings from the
    SimBA installation.

    :param Optional[TopLevel] master: Tkinter TopLevel window. Default: None.

    Examples
    ----------
    >>> _ = PoseResetter(master=None)
    """

    def __init__(self,
                 master: Optional[Toplevel] = None):

        self.default_pose_configs_cnt = len(get_bp_config_codes().keys())
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
            default_pic_list.append('{}.png'.format(str(idx+1)))
        for i in list(set(user_pic_lst) - set(default_pic_list)):
            os.remove(os.path.join(self.schematics_path, i))

        stdout_trash(msg='User-defined pose-estimation configuration reset. User-defined poses removed.')
        master.destroy()

# test = PoseResetter()





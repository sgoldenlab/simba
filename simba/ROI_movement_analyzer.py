__author__ = "Simon Nilsson", "JJ Choong"


from datetime import datetime
from simba.read_config_unit_tests import read_config_entry, read_config_file
import pandas as pd
import os, glob
from simba.ROI_analyzer import ROIAnalyzer

class ROIMovementAnalyzer(object):
    """

    Class for computing movements of individual animals within individual user-defined ROIs.

    Parameters
    ----------
    config_path: str
        Path to SimBA project config file in Configparser format

    Notes
    ----------
    `ROI tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial_new.md>`__.

    Examples
    ----------
    >>> _ = ROIMovementAnalyzer(config_path='MyProjectConfig')
    """

    def __init__(self,
                 config_path: str):

        self.config = read_config_file(config_path)
        self.datetime = datetime.now().strftime('%Y%m%d%H%M%S')
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.file_type = read_config_entry(self.config, 'General settings', 'workflow_file_type', 'str', 'csv')
        self.roi_analyzer = ROIAnalyzer(ini_path=config_path, data_path='outlier_corrected_movement_location', calculate_distances=True)
        self.roi_analyzer.read_roi_dfs()
        self.roi_analyzer.analyze_ROIs()
        self.save_data()

    def save_data(self):
        """
        Save ROI movement analysis. Results are stored in the ``project_folder/logs`` directory
        of the SimBA project.

        Returns
        ----------
        None
        """


        save_df = pd.concat(self.roi_analyzer.dist_lst, axis=0).reset_index(drop=True)
        save_path = os.path.join(self.roi_analyzer.logs_path, 'ROI_movement_data_' + self.roi_analyzer.timestamp + '.csv')
        save_df.to_csv(save_path)
        print('SIMBA COMPLETE: ROI movement data saved in the "project_folder/logs/" directory')

#test = ROIMovementAnalyzer(config_path='/Users/simon/Desktop/train_model_project/project_folder/project_config.ini')
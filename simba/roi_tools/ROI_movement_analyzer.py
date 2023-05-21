__author__ = "Simon Nilsson"

import os

from simba.mixins.config_reader import ConfigReader
from simba.roi_tools.ROI_analyzer import ROIAnalyzer
from simba.utils.printing import stdout_success


class ROIMovementAnalyzer(ConfigReader):
    """

    Compute movements of individual animals within user-defined ROIs.

    :param str config_path: Path to SimBA project config file in Configparser format

    .. note::
       `ROI tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial_new.md>`__.

    Examples
    ----------
    >>> _ = ROIMovementAnalyzer(config_path='MyProjectConfig')
    """

    def __init__(self, config_path: str):
        ConfigReader.__init__(self, config_path=config_path)
        self.roi_analyzer = ROIAnalyzer(
            ini_path=config_path,
            data_path="outlier_corrected_movement_location",
            calculate_distances=True,
        )
        self.roi_analyzer.run()
        self.save_data()

    def save_data(self):
        """
        Save ROI movement analysis. Results are stored in the ``project_folder/logs`` directory
        of the SimBA project.

        Returns
        ----------
        None
        """
        save_path = os.path.join(
            self.roi_analyzer.logs_path,
            "ROI_movement_data_" + self.roi_analyzer.datetime + ".csv",
        )
        self.roi_analyzer.movements_df.to_csv(save_path)
        self.timer.stop_timer()
        stdout_success(
            'ROI movement data saved in the "project_folder/logs/" directory',
            elapsed_time=self.timer.elapsed_time_str,
        )


# test = ROIMovementAnalyzer(config_path='/Users/simon/Desktop/envs/troubleshooting/two_animals_16bp_032023/project_folder/project_config.ini')

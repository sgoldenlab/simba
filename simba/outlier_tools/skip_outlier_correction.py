__author__ = "Simon Nilsson"

import os
from simba.utils.printing import stdout_success, SimbaTimer
from simba.utils.read_write import get_fn_ext, read_df, write_df
from simba.utils.checks import check_if_filepath_list_is_empty
from simba.mixins.config_reader import ConfigReader


class OutlierCorrectionSkipper(ConfigReader):
    """
    Class for skipping outlier correction in SimBA projects.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format

    Notes
    ----------

    Examples
    ----------
    >>> outlier_correction_skipper = OutlierCorrectionSkipper(config_path='MyProjectConfig')
    >>> outlier_correction_skipper.skip_outlier_correction()

    """

    def __init__(self, config_path: str):

        super().__init__(config_path=config_path)
        if not os.path.exists(self.outlier_corrected_dir):
            os.makedirs(self.outlier_corrected_dir)
        check_if_filepath_list_is_empty(
            filepaths=self.input_csv_paths,
            error_msg=f"No files found in {self.input_csv_dir}.",
        )
        print(f"Processing {len(self.input_csv_paths)} file(s)...")

    def skip_outlier_correction(self):
        """
        Standardizes pose-estimation data (i.e., headers) from different pose-estimation packages.
        Results are stored in the project_folder/csv/outlier_corrected_movement_location` directory of
        the SimBA project
        """

        for file_cnt, file_path in enumerate(self.input_csv_paths):
            video_timer = SimbaTimer(start=True)
            _, video_name, ext = get_fn_ext(file_path)
            data_df = read_df(
                file_path=file_path, file_type=self.file_type, check_multiindex=True
            )
            if "scorer" in data_df.columns:
                data_df = data_df.set_index("scorer")
            data_df = self.insert_column_headers_for_outlier_correction(
                data_df=data_df, new_headers=self.bp_col_names, filepath=file_path
            )
            data_df.index.name = None
            save_path = os.path.join(
                self.outlier_corrected_dir, video_name + "." + self.file_type
            )
            write_df(df=data_df, file_type=self.file_type, save_path=save_path)
            video_timer.stop_timer()
            print(
                f"Skipped outlier correction for video {video_name} (elapsed time {video_timer.elapsed_time_str}s)..."
            )
        self.timer.stop_timer()
        stdout_success(
            msg=f"Skipped outlier correction for {len(self.input_csv_paths)} files",
            elapsed_time=self.timer.elapsed_time_str,
        )


# test = OutlierCorrectionSkipper(config_path='/Users/simon/Desktop/envs/troubleshooting/naresh/project_folder/project_config.ini')
# test.skip_outlier_correction()
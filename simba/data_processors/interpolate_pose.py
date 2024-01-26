__author__ = "Simon Nilsson"

import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader


class Interpolate(ConfigReader):
    """
    Interpolate missing body-parts in pose-estimation data.

    Parameters
    ----------
    config_file_path: str
        path to SimBA project config file in Configparser format
    in_file: pd.DataFrame
        Pose-estimation data

    Notes
    -----
    `Interpolation tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#to-import-multiple-dlc-csv-files>`__.

    Examples
    -----
    >>> body_part_interpolator = Interpolate(config_file_path='MyProjectConfig', in_file=input_df)
    >>> body_part_interpolator.detect_headers()
    >>> body_part_interpolator.fix_missing_values(method_str='Body-parts: Nearest')
    >>> body_part_interpolator.reorganize_headers()

    """

    def __init__(self, config_file_path: str, in_file: pd.DataFrame):
        super().__init__(config_path=config_file_path, read_video_info=False)
        self.in_df = in_file

    def detect_headers(self):
        """
        Method to detect multi-index headers and set values to numeric in input dataframe
        """
        self.multi_index_headers_list = []
        self.in_df.columns = self.bp_headers
        self.header_col_cnt = self.get_number_of_header_columns_in_df(df=self.in_df)
        self.current_df = (
            self.in_df.iloc[self.header_col_cnt :]
            .apply(pd.to_numeric)
            .reset_index(drop=True)
        )
        self.multi_index_headers = self.in_df.iloc[: self.header_col_cnt]
        if self.header_col_cnt == 2:
            self.idx_names = ["scorer", "bodyparts", "coords"]
            for column in self.multi_index_headers:
                self.multi_index_headers_list.append(
                    (
                        column,
                        self.multi_index_headers[column][0],
                        self.multi_index_headers[column][1],
                    )
                )
        else:
            self.idx_names = ["scorer", "individuals", "bodyparts", "coords"]
            for column in self.multi_index_headers:
                self.multi_index_headers_list.append(
                    (
                        column,
                        self.multi_index_headers[column][0],
                        self.multi_index_headers[column][1],
                        self.multi_index_headers[column][2],
                    )
                )

    def fix_missing_values(self, method_str: str):
        """
        Method to interpolate missing values in pose-estimation data.

        Parameters
        ----------
        method_str: str
            String representing interpolation method. OPTIONS: 'None','Animal(s): Nearest', 'Animal(s): Linear',
            'Animal(s): Quadratic','Body-parts: Nearest', 'Body-parts: Linear', 'Body-parts: Quadratic'

        """

        interpolation_type, interpolation_method = (
            method_str.split(":")[0],
            method_str.split(":")[1].replace(" ", "").lower(),
        )
        self.animal_df_list, self.header_list_p = [], []
        if interpolation_type == "Animal(s)":
            for animal in self.multi_animal_id_list:
                currentAnimalX, currentAnimalY, currentAnimalP = (
                    self.animal_bp_dict[animal]["X_bps"],
                    self.animal_bp_dict[animal]["Y_bps"],
                    self.animal_bp_dict[animal]["P_bps"],
                )
                header_list_xy = []
                for (
                    col1,
                    col2,
                    col3,
                ) in zip(currentAnimalX, currentAnimalY, currentAnimalP):
                    header_list_xy.extend((col1, col2))
                    self.header_list_p.append(col3)
                self.animal_df_list.append(self.current_df[header_list_xy])
            for loop_val, animal_df in enumerate(self.animal_df_list):
                repeat_bol = animal_df.eq(animal_df.iloc[:, 0], axis=0).all(
                    axis="columns"
                )
                indices_to_replace_animal = repeat_bol.index[repeat_bol].tolist()
                print(
                    f"Detected {str(len(indices_to_replace_animal))} missing pose-estimation frames for {str(self.multi_animal_id_list[loop_val])}..."
                )
                animal_df.loc[indices_to_replace_animal] = np.nan
                self.animal_df_list[loop_val] = (
                    animal_df.interpolate(method=interpolation_method, axis=0)
                    .ffill()
                    .bfill()
                )
            self.new_df = pd.concat(self.animal_df_list, axis=1).fillna(0)

        if interpolation_type == "Body-parts":
            for animal in self.animal_bp_dict:
                for x_bps_name, y_bps_name in zip(
                    self.animal_bp_dict[animal]["X_bps"],
                    self.animal_bp_dict[animal]["Y_bps"],
                ):
                    zero_indices = self.current_df[
                        (self.current_df[x_bps_name] == 0)
                        & (self.current_df[y_bps_name] == 0)
                    ].index.tolist()
                    self.current_df.loc[zero_indices, [x_bps_name, y_bps_name]] = np.nan
                    self.current_df[x_bps_name] = (
                        self.current_df[x_bps_name]
                        .interpolate(method=interpolation_method, axis=0)
                        .ffill()
                        .bfill()
                    )
                    self.current_df[y_bps_name] = (
                        self.current_df[y_bps_name]
                        .interpolate(method=interpolation_method, axis=0)
                        .ffill()
                        .bfill()
                    )
            self.new_df = self.current_df.fillna(0)

    def reorganize_headers(self):
        """
        Method to re-insert original multi-index headers
        """
        loop_val = 2
        for p_col_name in self.header_list_p:
            p_col = list(self.in_df[p_col_name].iloc[self.header_col_cnt :])
            self.new_df.insert(loc=loop_val, column=p_col_name, value=p_col)
            loop_val += 3
        self.new_df.columns = pd.MultiIndex.from_tuples(
            self.multi_index_headers_list, names=self.idx_names
        )


# config_file_path = r"/Users/simon/Desktop/envs/troubleshooting/Vince_time_bins/project_folder/project_config.ini"
# in_file = r"/Users/simon/Desktop/envs/troubleshooting/Vince_time_bins/project_folder/csv/input_csv/2022-06-15_15-26-48_ArC_H_1_1Hrcropped.csv"
# interpolation_method = 'Body-parts: Linear'
# csv_df = pd.read_csv(in_file, index_col=0)
# interpolate_body_parts = Interpolate(config_file_path, csv_df)
# interpolate_body_parts.detect_headers()
# interpolate_body_parts.fix_missing_values(interpolation_method)
# interpolate_body_parts.reorganize_headers()

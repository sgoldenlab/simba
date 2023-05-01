from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from simba.read_config_unit_tests import check_file_exist_and_readable, check_if_filepath_list_is_empty
from simba.read_config_unit_tests import read_config_file, read_config_entry
from simba.rw_dfs import read_df
from simba.drop_bp_cords import getBpNames
import pandas as pd
import numpy as np
from numba import jit, prange
import os, glob

COL_NAMES = ['Low_prob_detections_0.1',
            'Low_prob_detections_0.5',
            'Low_prob_detections_0.75']


def add_probability_features_to_annotated_data(config_path: str):

    @jit(nopython=True)
    def count_values_in_range(data: np.array, ranges: np.array):
        results = np.full((data.shape[0], ranges.shape[0]), 0)
        for i in prange(data.shape[0]):
            for j in prange(ranges.shape[0]):
                lower_bound, upper_bound = ranges[j][0], ranges[j][1]
                results[i][j] = data[i][np.logical_and(data[i] >= lower_bound, data[i] <= upper_bound)].shape[0]
        return results

    check_file_exist_and_readable(file_path=config_path)
    config = read_config_file(ini_path=config_path)
    project_path = read_config_entry(config, 'General settings', 'project_path', data_type='folder_path')
    targets_dir = os.path.join(project_path, 'csv', 'targets_inserted')
    files_found = glob.glob(targets_dir + '/*.csv')
    check_if_filepath_list_is_empty(filepaths=files_found, error_msg=f'ERROR: No files found in {targets_dir}')
    x_cols, y_cols, pcols = getBpNames(config_path)

    for file_path in files_found:
        df = read_df(file_path=file_path, file_type='csv').reset_index(drop=True)
        missing_cols = list(set(COL_NAMES) - set(df.columns))
        if len(missing_cols) == 0:
            print(f'SIMBA ERROR: {COL_NAMES} already exist in {file_path}.')
            continue
        results = pd.DataFrame(count_values_in_range(data=df.filter(pcols).values, ranges=np.array([[0.0, 0.1], [0.000000000, 0.5], [0.000000000, 0.75]])),
                               columns=['Low_prob_detections_0.1', 'Low_prob_detections_0.5', 'Low_prob_detections_0.75'])
        df = pd.concat([df, results], axis=1)
        df.to_csv(file_path)
        print(f'New features added to {file_path}...')

    print(f'COMPLETE: {str(len(COL_NAMES))} new features added to {len(files_found)} in the {targets_dir} directory')

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-dir", "--directory", help="path to SimBA project config file")
    args = vars(parser.parse_args())
    add_probability_features_to_annotated_data(config_path=args['directory'])
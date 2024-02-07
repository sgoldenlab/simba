from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import repeat

import numpy as np
import pandas as pd

from simba.misc_tools import SimbaTimer, find_core_cnt, get_fn_ext
from simba.rw_dfs import read_df
from simba.utils.errors import ColumnNotFoundError


def _read_data_file_helper(file_path, file_type, clf_names):
    timer = SimbaTimer()
    timer.start_timer()
    _, vid_name, _ = get_fn_ext(file_path)
    df = read_df(file_path, file_type).dropna(axis=0, how="all").fillna(0)
    if clf_names != None:
        for clf_name in clf_names:
            if not clf_name in df.columns:
                raise ColumnNotFoundError(column_name=clf_name, file_name=file_path)
    timer.stop_timer()
    print(f"Reading complete {vid_name} (elapsed time: {timer.elapsed_time_str}s)...")
    return df


def read_all_files_in_folder_mp(
    file_paths: list, file_type: str, classifier_names=None
):
    """

    Multiprocessing helper function to read in all data files in a folder to a single
    pd.DataFrame for downstream ML. Defaults to ceil(CPU COUNT / 2) cores. Asserts that all classifiers
    have annotation fields present in each dataframe.

    Parameters
    ----------
    file_paths: list
        List of file paths representing files to be read in.
    file_type: str
        Type of files in ``file_paths``. OPTIONS: csv or parquet.
    classifier_names: list or None
        List of classifier names representing fields of human annotations. If not None, then assert that classifier names
        are present in each data file.

    Returns
    -------
    df_concat: pd.DataFrame

    """
    print(f"Reading {len(file_paths)} files ...")
    cpu_cnt, _ = find_core_cnt()
    df_lst = []
    with ProcessPoolExecutor(int(np.ceil(cpu_cnt / 2))) as pool:
        for res in pool.map(
            _read_data_file_helper,
            file_paths,
            repeat(file_type),
            repeat(classifier_names),
        ):
            df_lst.append(res)
    df_concat = pd.concat(df_lst, axis=0)
    try:
        df_concat = df_concat.set_index("scorer")
    except KeyError:
        pass
    if len(df_concat) == 0:
        raise ValueError(
            "ANNOTATION ERROR: SimBA found 0 annotated frames in the project_folder/csv/targets_inserted directory"
        )
    df_concat = df_concat.loc[:, ~df_concat.columns.str.contains("^Unnamed")]
    return df_concat.reset_index(drop=True)


# IN_DIR = '/Users/simon/Desktop/envs/troubleshooting/locomotion/project_folder/csv/targets_inserted'
# file_paths = glob.glob(IN_DIR + '/*.csv')
# df = read_all_files_in_folder_mp(file_paths=file_paths, file_type='csv', classifier_names=['Thrown_Off_Wheel'])

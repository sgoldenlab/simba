__author__ = "Simon Nilsson", "JJ Choong"

import scipy.io
import pandas as pd
import ast
import os, glob
from configparser import ConfigParser, NoSectionError, NoOptionError
from simba.rw_dfs import save_df
from simba.drop_bp_cords import get_fn_ext

def read_config(config_path):
    config = ConfigParser()
    config.read(str(config_path))
    project_path = config.get('General settings', 'project_path')
    output_path = os.path.join(project_path, 'csv', 'input_csv')

    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'

    return output_path, wfileType

def read_data(file_path):
    dannce_dict = scipy.io.loadmat(file_path)
    dannce_pred = dannce_dict['predictions']
    bodypart_lst = [x[0] for x in ast.literal_eval(str(dannce_pred.dtype))][:-1]
    out_df_list = []
    for bp in range(0, len(bodypart_lst)):
        curr_pred = pd.DataFrame(dannce_pred[0][0][bp],
                                 columns=[bodypart_lst[bp] + '_x', bodypart_lst[bp] + '_y', bodypart_lst[bp] + '_z'])
        curr_pred[bodypart_lst[bp] + '_p'] = 1
        out_df_list.append(curr_pred)
    return pd.concat(out_df_list, axis=1)

def insert_multi_index_header(df):
    multiindex_cols = []
    for column in range(len(df.columns)):
        multiindex_cols.append(tuple(('DANNCE_3D_data', 'DANNCE_3D_data', df.columns[column])))
    df.columns = pd.MultiIndex.from_tuples(multiindex_cols, names=['scorer', 'bodypart', 'coords'])
    return df

def import_DANNCE_folder(config_path, folder_path, interpolation_method):
    output_path, wfileType = read_config(config_path)
    files_found = glob.glob(folder_path + '/*.mat')
    for file in files_found:
        dir_name, file_name, ext = get_fn_ext(file)
        out_df = read_data(file)
        out_df = insert_multi_index_header(out_df)
        out_path_name = os.path.join(output_path, file_name + '.' + wfileType)
        save_df(out_df, wfileType, out_path_name)
        print('Imported: ' + str(os.path.basename(file)))

def import_DANNCE_file(config_path, file_path, interpolation_method):
    output_path, wfileType = read_config(config_path)
    dir_name, file_name, ext = get_fn_ext(file_path)
    out_df = read_data(file_path)
    out_df = insert_multi_index_header(out_df)
    out_path_name = os.path.join(output_path, file_name + '.' + wfileType)
    save_df(out_df, wfileType, out_path_name)
    print('Imported: ' + str(os.path.basename(file_path)))

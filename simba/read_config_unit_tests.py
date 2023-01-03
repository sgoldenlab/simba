__author__ = "Simon Nilsson", "JJ Choong"

import trafaret as t
import os
import pandas as pd
from configparser import ConfigParser, MissingSectionHeaderError


def check_int(name, value, max_value=None, min_value=None):
    try:
        t.Int().check(value)
    except t.DataError as e:
        print('SIMBA VALUE ERROR: {} {} {}'.format(name, str(value), e.as_dict()))
        raise ValueError('SIMBA VALUE ERROR: {} should be an integer number in SimBA, but is set to {}'.format(name, str(value)))
    if (min_value != None):
        if int(value) < min_value:
            print(('SIMBA VALUE ERROR: {} should be more than, or equal {}. It is set to {}'.format(name, str(min_value),
                                                                                              str(value))))
            raise ValueError(
                'SIMBA VALUE ERROR: {} should be more than, or equal {}. It is set to {}'.format(name, str(min_value),
                                                                                           str(value)))
    if (max_value != None):
        if int(value) > max_value:
            print(('SIMBA VALUE ERROR: {} should be less than, or equal {}. It is set to {}'.format(name, str(max_value),
                                                                                              str(value))))
            raise ValueError(
                'SIMBA VALUE ERROR: {} should be less than, or equal {}. It is set to {}'.format(name, str(max_value),
                                                                                           str(value)))


def check_str(name, value, options=(), allow_blank=False):
    try:
        t.String(allow_blank=allow_blank).check(value)
    except t.DataError as e:
        print('{} {} {}'.format(name, str(value), e.as_dict()))
        raise ValueError('SIMBA VALUE ERROR: {} should be an string in SimBA, but is set to {}'.format(name, str(value)))
    if len(options) > 0:
        if value not in options:
            print(('SIMBA VALUE ERROR: {} is set to {} in SimBA, but this is not a valid option {}'.format(name, str(value),
                                                                                                     options)))
            raise ValueError(
                'SIMBA VALUE ERROR: {} is set to {} in SimBA, but this is not a valid option {}'.format(name, str(value),
                                                                                                  options))
        else:
            pass
    else:
        pass


def check_float(name: str,
                value=None,
                max_value=None,
                min_value=None):
    try:
        t.Float().check(value)
    except t.DataError as e:
        print('{} {} {}'.format(name, str(value), e.as_dict()))
        print('SIMBA VALUE ERROR: {} should be a float number in SimBA, but is set to {}'.format(name, str(value)))
        raise ValueError('SIMBA VALUE ERROR: {} should be a float in SimBA, but is set to {}'.format(name, str(value)))
    if (min_value != None):
        if float(value) < min_value:
            print('SIMBA VALUE ERROR: {} should be more than, or equal {}. It is set to {}'.format(name, str(min_value),
                                                                                             str(value)))
            raise ValueError('SIMBA VALUE ERROR: {} should be more than, or equal {}. It is set to {}'.format(name, str(min_value),
                                                                                             str(value)))
    if (max_value != None):
        if float(value) > max_value:
            print('SIMBA VALUE ERROR: {} should be less than, or equal {}. It is set to {}'.format(name, str(max_value),
                                                                                             str(value)))
            raise ValueError('SIMBA VALUE ERROR: {} should be less than, or equal {}. It is set to {}'.format(name, str(max_value),
                                                                                             str(value)))


def read_config_entry(config: ConfigParser,
                      section: str,
                      option: str,
                      data_type: str,
                      default_value=None,
                      options=None):
    try:
        if config.has_option(section, option):
            if data_type == 'float':
                value = config.getfloat(section, option)
            elif data_type == 'int':
                value = config.getint(section, option)
            elif data_type == 'str':
                value = config.get(section, option)
            elif data_type == 'folder_path':
                value = config.get(section, option)
                if not os.path.isdir(value):
                    print('VALUE ERROR: SimBA is looking for the folder {} but it does not exist.'.format(value))
                    raise ValueError
            if options != None:
                if value not in options:
                    print(('VALUE ERROR: {} is set to {} in SimBA, but this is not among the valid options ({})'.format(option, str(value), options)))
                    raise ValueError
                else:
                    return value
            return value
        elif default_value != None:
            return default_value
        else:
            print('VALUE ERROR: SimBA could not find an entry for option {} under section {} in the project_config.ini. Please specify the settings in the settings menu.'.format(option, section))
            raise ValueError
    except ValueError:
        if default_value != None:
            return default_value
        else:
            print(('VALUE ERROR: SimBA could not find an entry for option {} under section {} in the project_config.ini. Please specify the settings in the settings menu.'.format(option, section)))
            raise ValueError

def read_simba_meta_files(folder_path: str):
    meta_file_lst = []
    for i in os.listdir(folder_path):
        if i.__contains__("meta"):
            meta_file_lst.append(os.path.join(folder_path, i))
    if len(meta_file_lst) == 0:
        print(
            'SIMBA WARNING: The training meta-files folder (project_folder/configs) does not have any meta files inside it (no files in this folder has the "meta" substring in the filename')
    return meta_file_lst


def read_meta_file(meta_file_path):
    return pd.read_csv(meta_file_path, index_col=False).to_dict(orient='records')[0]


def check_file_exist_and_readable(file_path: str):
    if not os.path.isfile(file_path):
        print('SIMBA FILE NOT FOUND ERROR: {} is not a valid file path'.format(file_path))
        raise FileNotFoundError('SIMBA FILE NOT FOUND ERROR: {} is not a valid file path'.format(file_path))
    elif not os.access(file_path, os.R_OK):
        print('SIMBA FILE CORRUPTED ERROR: {} is not readable'.format(file_path))
        raise FileNotFoundError('SIMBA FILE CORRUPTED ERROR: {} is not readable'.format(file_path))


def read_config_file(ini_path: str):
    config = ConfigParser()
    try:
        config.read(str(ini_path))
    except MissingSectionHeaderError:
        print('ERROR:  Not a valid project_config file. Please check the project_config.ini path.')
        raise MissingSectionHeaderError
    return config


def insert_default_headers_for_feature_extraction(df: pd.DataFrame,
                                                  headers: list,
                                                  pose_config: str,
                                                  filename: str):
    if len(headers) != len(df.columns):
        raise ValueError('SIMBA ERROR: Your SimBA project is set to using the default {} pose-configuration. '
                         'SimBA therefore expects {} columns of data inside the files within the project_folder. However, '
                         'within file {} file, SimBA found {} columns.'.format(pose_config, str(len(headers)), filename,
                                                                               str(len(df.columns))))
    else:
        df.columns = headers
        return df


def check_that_column_exist(df: pd.DataFrame,
                            column_name: str,
                            file_name: str):
    if column_name not in df.columns:
        print('SIMBA ERROR: Could not find column named `{}` within the file representing {}'.format(column_name,
                                                                                                     file_name))
        raise ValueError(
            'SIMBA ERROR: Could not find column named `{}` within the file representing {}'.format(column_name,
                                                                                                   file_name))
    else:
        pass



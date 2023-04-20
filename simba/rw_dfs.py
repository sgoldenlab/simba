import pandas as pd
from simba.utils.errors import InvalidFileTypeError
from simba.enums import Formats
from simba.read_config_unit_tests import check_file_exist_and_readable
import pyarrow as pa
import numpy as np
import pickle
from pyarrow import csv


PARSE_OPTIONS = csv.ParseOptions(delimiter=',')
READ_OPTIONS = csv.ReadOptions(encoding='utf8')

def read_df(file_path: str,
            file_type: str,
            idx=0,
            remove_columns: list or None=None,
            usecols: list or None=None):
    """
    Helper function to read single data file into memory.

    Parameters
    ----------
    file_path: str
        Path to data file.
    file_type: str
        Type of data. OPTIONS: 'parquet' or 'csv'.
    idx: int,
        Index column location. Default: 0.
    remove_columns: list or None,
        If list, then remove columns
    usecols: list or None,
        If list, keep columns

    Returns
    -------
    df: pd.DataFrame

    """
    check_file_exist_and_readable(file_path=file_path)
    if file_type == Formats.CSV.value:
        try:
            df = csv.read_csv(file_path, parse_options=PARSE_OPTIONS, read_options=READ_OPTIONS)
            duplicate_headers = list(set([x for x in df.column_names if df.column_names.count(x) > 1]))
            if len(duplicate_headers) == 1:
                new_headers = [duplicate_headers[0] + f'_{x}' for x in range(len(df.column_names))]
                df = df.rename_columns(new_headers)
            df = df.to_pandas().iloc[:, 1:]
        except Exception as e:
            print(e, e.args)
            raise InvalidFileTypeError(msg=f'{file_path} is not a valid CSV file')
        if remove_columns:
            df = df[df.columns[~df.columns.isin(remove_columns)]]
        if usecols:
            df = df[df.columns[df.columns.isin(usecols)]]
    elif file_type == Formats.PARQUET.value:
        df = pd.read_parquet(file_path)
    else:
        raise InvalidFileTypeError(msg=f'{file_type} is not a valid filetype OPTIONS: [csv, parquet]')

    return df

def save_df(df: pd.DataFrame,
            file_type: str,
            save_path: str):
    """
    Helper function to save single data file from memory.

    Parameters
    ----------
    df: pd.DataFrame
        Pandas dataframe to save to disk.
    file_type: str
        Type of data. OPTIONS: 'parquet' or 'csv'.
    save_path: str,
        Location where to store the data.

    Returns
    -------
    None
    """

    if file_type == Formats.CSV.value:
        df = df.drop('scorer', axis=1, errors='ignore')
        idx = np.arange(len(df)).astype(str)
        df.insert(0, '', idx)
        df = pa.Table.from_pandas(df=df)
        csv.write_csv(df, save_path)
    elif file_type == Formats.PARQUET.value:
        df.to_parquet(save_path)
    elif file_type == Formats.PICKLE.value:
        try:
            with open(save_path, 'wb') as f:
                pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(e.args[0])
            raise InvalidFileTypeError(msg='Data could not be saved as a pickle.')
    else:
        raise InvalidFileTypeError(msg=f'{file_type} is not a valid filetype OPTIONS: [csv, pickle, parquet]')

# df = read_df(file_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/input_csv/Together_1.csv', file_type='csv')
# df = df.loc[2:]
# df = pd.concat([df, df, df,df, df, df,df, df, df]).reset_index(drop=True)
# df = pd.concat([df, df, df]).reset_index(drop=True)
# df = pd.concat([df, df,df, df, df]).reset_index(drop=True)
# df = pd.concat([df, df, df, df], axis=1).reset_index(drop=True)
# df.columns = [str(x) for x in range(len(df.columns))]
# print(len(df))
#
#
# start = time.time()
# save_df(df=df, file_type='csv', save_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/validation/test.csv')
# print(time.time() - start)
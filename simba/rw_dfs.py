import os.path
import pandas as pd

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
        If list, remove columns
    usecols: list or None,
        If list, keep columns

    Returns
    -------
    df: pd.DataFrame

    """
    try:
        if file_type == 'csv':
            try:
                df = pd.read_csv(file_path, index_col=idx, low_memory=False, sep=',')
            except Exception as e:
                if type(e).__name__ == 'ParserError':
                    print('SIMBA ERROR: SimBA tried to read {} as a comma delimited CSV file and failed. Make sure {} is a utf-8 encoded comma delimited CSV file.'.format(file_path, os.path.basename(file_path)))
                    raise ValueError(e)
                if type(e).__name__ == 'UnicodeDecodeError':
                    print('SIMBA ERROR: {} is not a valid CSV file'.format(file_path))
                    raise ValueError(e)
        elif file_type == 'parquet':
            df = pd.read_parquet(file_path)
        else:
            print('SIMBA ERROR: The file type ({}) is not recognized. Please set the workflow file type to either csv or parquet'.format(file_type))
            raise ValueError('SIMBA ERROR: The file type is not recognized. Please set the workflow file type to either csv or parquet')
        if remove_columns:
            df = df[df.columns[~df.columns.isin(remove_columns)]]
        if usecols:
            df = df[df.columns[df.columns.isin(usecols)]]
        else:
            pass
        return df
    except FileNotFoundError:
        print('The CSV file could not be located at the following path: ' + str(file_path) + ' . It may be that you missed a step in the analysis. Please generate the file before proceeding.')

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

    if file_type == 'csv':
        df = df.drop('scorer', axis=1, errors='ignore')
        df.to_csv(save_path, index=True)
    elif file_type == 'parquet':
        df.to_parquet(file_type)
    else:
        print('SIMBA ERROR: The file type is not recognized. Please set the workflow file type to either csv or parquet')
        raise ValueError('SIMBA ERROR: The file type is not recognized. Please set the workflow file type to either csv or parquet')
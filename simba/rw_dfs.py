import pandas as pd

def read_df(file_path: str,
            file_type: str,
            idx=0):
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

    Returns
    -------
    df: pd.DataFrame

    """
    try:
        if file_type == 'csv':
            df = pd.read_csv(file_path, index_col=idx, low_memory=False)
        elif file_type == 'parquet':
            df = pd.read_parquet(file_path)
        else:
            print('SIMBA ERROR: The file type is not recognized. Please set the workflow file type to either csv or parquet')
            raise ValueError('SIMBA ERROR: The file type is not recognized. Please set the workflow file type to either csv or parquet')
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

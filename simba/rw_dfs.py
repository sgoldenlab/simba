import pandas as pd

def read_df(currentFilePath, wfileType,idx=0):
    try:
        if wfileType == 'csv':
            currDf = pd.read_csv(currentFilePath, index_col=idx, low_memory=False)
        if wfileType == 'parquet':
            currDf = pd.read_parquet(currentFilePath)
        return currDf
    except FileNotFoundError:
        print('The CSV file could not be located at the following path: ' + str(currentFilePath) + ' . It may be that you missed a step in the analysis. Please generate the file before proceeding.')

def save_df(df=None, file_type=None, save_path=None):
    if file_type == 'csv':
        df = df.drop('scorer', axis=1, errors='ignore')
        df.to_csv(save_path, index=True)
    if file_type == 'parquet':
        df.to_parquet(file_type)
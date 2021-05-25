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

def save_df(currDf, wfileType, path):
    try:
        if wfileType == 'csv':
            currDf = currDf.drop('scorer', axis=1, errors='ignore')
            currDf.to_csv(path, index=True)
        if wfileType == 'parquet':
            print(currDf)
            currDf.to_parquet(path)
    except FileNotFoundError:
        print('The PARQUET file could not be located at the following path: ' + str(path) + ' . It may be that you missed a step in the analysis. Please generate the file before proceeding.')
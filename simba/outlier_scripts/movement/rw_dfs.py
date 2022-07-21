import pandas as pd

def read_df(currentFilePath, wfileType):
    if wfileType == 'csv':
        currDf = pd.read_csv(currentFilePath)
    if wfileType == 'parquet':
        currDf = pd.read_parquet(currentFilePath)
    return currDf

def save_df(currDf, wfileType, path):
    if wfileType == 'csv':
        currDf.to_csv(path, index=False)
    if wfileType == 'parquet':
        currDf.to_parquet(path, index=False)
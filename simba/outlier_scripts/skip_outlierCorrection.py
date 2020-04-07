import pandas as pd
import os
import numpy as np
import statistics
import math
from configparser import ConfigParser
import glob


def skip_outlier_c(configini):
    configFile = str(configini)
    config = ConfigParser()
    config.read(configFile)
    projectPath = config.get('General settings', 'project_path')
    currentBodyPartFile = os.path.join(projectPath, 'logs', 'measures', 'pose_configs', 'bp_names',
                                       'project_bp_names.csv')
    bodyPartsFile = pd.read_csv(os.path.join(currentBodyPartFile, currentBodyPartFile), header=None)
    bodyPartsList = list(bodyPartsFile[0])
    bodyPartHeaders = []
    xy_headers = []
    p_cols = []
    x_cols = []
    y_cols = []
    for i in bodyPartsList:
        col1, col2, col3 = (str(i) + '_x', str(i) + '_y', str(i) + '_p')
        p_cols.append(col3)
        x_cols.append(col1)
        y_cols.append(col2)
        bodyPartHeaders.extend((col1, col2, col3))
        xy_headers.extend((col1, col2))
    csv_dir = config.get('General settings', 'csv_path')
    csv_dir_in = os.path.join(csv_dir, 'input_csv')
    csv_dir_out = os.path.join(csv_dir, 'outlier_corrected_movement_location')
    if not os.path.exists(csv_dir_out):
        os.makedirs(csv_dir_out)

    filesFound = glob.glob(csv_dir_in + '/*.csv')

    for i in filesFound:
        currentFile = i
        baseNameFile = os.path.basename(currentFile).replace('.csv', '')
        csv_df = pd.read_csv(currentFile, names=bodyPartHeaders, index_col=None)
        csv_df = csv_df.drop(csv_df.index[[0, 1, 2]])
        csv_df = csv_df.apply(pd.to_numeric)

        csv_out = csv_df.rename_axis('scorer')
        fileOut = str(baseNameFile) + str('.csv')
        pathOut = os.path.join(csv_dir_out, fileOut)
        csv_out.to_csv(pathOut)

    print('CAUTION: Outlier corrections skipped. File headers corrected. Ready for the next step.')
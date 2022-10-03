__author__ = "Simon Nilsson", "JJ Choong"

import pandas as pd
import json
import numpy as np
import glob, os
from configparser import ConfigParser, NoSectionError, NoOptionError
from simba.rw_dfs import *
import pyarrow.parquet as pq
import pyarrow as pa
from simba.interpolate_pose import *

def json2csv_folder(configini, folderpath, interpolation_method, smoothing_method):
    print('Converting JSON files...')
    config = ConfigParser()
    config.read(configini)
    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'
    projectPath = config.get('General settings', 'project_path')
    csv_dir_out = os.path.join(projectPath, 'csv', 'input_csv')
    filesFound = glob.glob(folderpath + '/*.json')
    for file in filesFound:
        jsonfile = json.load(open(file))
        basename = os.path.basename(file).replace('.json', '.' + str(wfileType))
        savePath = os.path.join(csv_dir_out, basename)
        keypoints, scores = jsonfile['keypoints'], jsonfile['scores']
        keypoints, scores = np.array(keypoints), np.array(scores)
        keypointsDf = pd.DataFrame({'Nose_1_x': keypoints[:, 0, 0, 0], 'Nose_1_y': keypoints[:, 0, 1, 0], 'Nose_1_p': scores[:, 0, 0],
                                  'Ear_left_1_x': keypoints[:, 0, 0, 1], 'Ear_left_1_y': keypoints[:, 0, 1, 1], 'Ear_left_1_p': scores[:, 0, 1],
                                  'Ear_right_1_x': keypoints[:, 0, 0, 2], 'Ear_right_1_y': keypoints[:, 0, 1, 2], 'Ear_right_1_p': scores[:, 0, 2],
                                  'Neck_1_x': keypoints[:, 0, 0, 3], 'Neck_1_y': keypoints[:, 0, 1, 3], 'Neck_1_p': scores[:, 0, 3],
                                  'Hip_left_1_x': keypoints[:, 0, 0, 4], 'Hip_left_1_y': keypoints[:, 0, 1, 4], 'Hip_left_1_p': scores[:, 0, 4],
                                  'Hip_right_1_x': keypoints[:, 0, 0, 5], 'Hip_right_1_y': keypoints[:, 0, 1, 5], 'Hip_right_1_p': scores[:, 0, 5],
                                  'Tail_1_x': keypoints[:, 0, 0, 6], 'Tail_1_y': keypoints[:, 0, 1, 6], 'Tail_1_p': scores[:, 0, 6],
                                  'Nose_2_x': keypoints[:, 1, 0, 0], 'Nose_2_y': keypoints[:, 1, 1, 0], 'Nose_2_p': scores[:, 1, 0],
                                  'Ear_left_2_x': keypoints[:, 1, 0, 1], 'Ear_left_2_y': keypoints[:, 1, 1, 1], 'Ear_left_2_p': scores[:, 1, 1],
                                  'Ear_right_2_x': keypoints[:, 1, 0, 2], 'Ear_right_2_y': keypoints[:, 1, 1, 2], 'Ear_right_2_p': scores[:, 1, 2],
                                  'Neck_2_x': keypoints[:, 1, 0, 3], 'Neck_2_y': keypoints[:, 1, 1, 3], 'Neck_2_p': scores[:, 1, 3],
                                  'Hip_left_2_x': keypoints[:, 1, 0, 4], 'Hip_left_2_y': keypoints[:, 1, 1, 4], 'Hip_left_2_p': scores[:, 1, 4],
                                  'Hip_right_2_x': keypoints[:, 1, 0, 5], 'Hip_right_2_y': keypoints[:, 1, 1, 5], 'Hip_right_2_p': scores[:, 1, 5],
                                  'Tail_2_x': keypoints[:, 1, 0, 6], 'Tail_2_y': keypoints[:, 1, 1, 6], 'Tail_2_p': scores[:, 1, 6]})
        MultiIndexCol = []
        for column in range(len(keypointsDf.columns)):
            MultiIndexCol.append(tuple(('MARS', 'MARS', keypointsDf.columns[column])))
        keypointsDf.columns = pd.MultiIndex.from_tuples(MultiIndexCol, names=['scorer', 'bodypart', 'coords'])
        if wfileType == 'parquet':
            table = pa.Table.from_pandas(keypointsDf)
            pq.write_table(table, savePath)
        if wfileType == 'csv':
            keypointsDf.to_csv(savePath)
        if interpolation_method != 'None':
            perform_interpolation(savePath, wfileType, configini, interpolation_method)
        print('JSON file ' + basename + ' imported')
    print('All MARS Json files imported in SimBA')

def json2csv_file(configini, filename, interpolation_method, smoothing_method):
    config = ConfigParser()
    config.read(configini)
    projectPath = config.get('General settings', 'project_path')
    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'
    csv_dir_out = os.path.join(projectPath, 'csv', 'input_csv')
    jsonfile = json.load(open(filename))
    basename = os.path.basename(filename).replace('.json', '.csv')
    savePath = os.path.join(csv_dir_out, basename)
    keypoints, scores = jsonfile['keypoints'], jsonfile['scores']
    keypoints, scores = np.array(keypoints), np.array(scores)
    keypointsDf = pd.DataFrame(
        {'Nose_1_x': keypoints[:, 0, 0, 0], 'Nose_1_y': keypoints[:, 0, 1, 0], 'Nose_1_p': scores[:, 0, 0],
         'Ear_left_1_x': keypoints[:, 0, 0, 1], 'Ear_left_1_y': keypoints[:, 0, 1, 1], 'Ear_left_1_p': scores[:, 0, 1],
         'Ear_right_1_x': keypoints[:, 0, 0, 2], 'Ear_right_1_y': keypoints[:, 0, 1, 2],
         'Ear_right_1_p': scores[:, 0, 2],
         'Neck_1_x': keypoints[:, 0, 0, 3], 'Neck_1_y': keypoints[:, 0, 1, 3], 'Neck_1_p': scores[:, 0, 3],
         'Hip_left_1_x': keypoints[:, 0, 0, 4], 'Hip_left_1_y': keypoints[:, 0, 1, 4], 'Hip_left_1_p': scores[:, 0, 4],
         'Hip_right_1_x': keypoints[:, 0, 0, 5], 'Hip_right_1_y': keypoints[:, 0, 1, 5],
         'Hip_right_1_p': scores[:, 0, 5],
         'Tail_1_x': keypoints[:, 0, 0, 6], 'Tail_1_y': keypoints[:, 0, 1, 6], 'Tail_1_p': scores[:, 0, 6],
         'Nose_2_x': keypoints[:, 1, 0, 0], 'Nose_2_y': keypoints[:, 1, 1, 0], 'Nose_2_p': scores[:, 1, 0],
         'Ear_left_2_x': keypoints[:, 1, 0, 1], 'Ear_left_2_y': keypoints[:, 1, 1, 1], 'Ear_left_2_p': scores[:, 1, 1],
         'Ear_right_2_x': keypoints[:, 1, 0, 2], 'Ear_right_2_y': keypoints[:, 1, 1, 2],
         'Ear_right_2_p': scores[:, 1, 2],
         'Neck_2_x': keypoints[:, 1, 0, 3], 'Neck_2_y': keypoints[:, 1, 1, 3], 'Neck_2_p': scores[:, 1, 3],
         'Hip_left_2_x': keypoints[:, 1, 0, 4], 'Hip_left_2_y': keypoints[:, 1, 1, 4], 'Hip_left_2_p': scores[:, 1, 4],
         'Hip_right_2_x': keypoints[:, 1, 0, 5], 'Hip_right_2_y': keypoints[:, 1, 1, 5],
         'Hip_right_2_p': scores[:, 1, 5],
         'Tail_2_x': keypoints[:, 1, 0, 6], 'Tail_2_y': keypoints[:, 1, 1, 6], 'Tail_2_p': scores[:, 1, 6]})
    MultiIndexCol = []
    for column in range(len(keypointsDf.columns)):
        MultiIndexCol.append(tuple(('MARS', 'MARS', keypointsDf.columns[column])))
    keypointsDf.columns = pd.MultiIndex.from_tuples(MultiIndexCol, names=['scorer', 'bodypart', 'coords'])
    save_df(keypointsDf, wfileType, savePath)
    if interpolation_method != 'None':
        print('performing interpolation...')
        perform_interpolation(savePath, wfileType, configini, interpolation_method)
    print('JSON file ' + basename + ' imported')
    print('All MARS Json files imported in SimBA')

def perform_interpolation(file_path, wfileType, configini, interpolation_method):
    if wfileType == 'parquet': csv_df = pd.read_parquet(file_path)
    if wfileType == 'csv': csv_df = pd.read_csv(file_path, index_col=0)
    interpolate_body_parts = Interpolate(configini, csv_df)
    interpolate_body_parts.detect_headers()
    interpolate_body_parts.fix_missing_values(interpolation_method)
    interpolate_body_parts.reorganize_headers()
    if wfileType == 'parquet':
        table = pa.Table.from_pandas(interpolate_body_parts.new_df)
        pq.write_table(table, file_path)
    if wfileType == 'csv':
        interpolate_body_parts.new_df.to_csv(file_path)



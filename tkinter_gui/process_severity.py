import pandas as pd
import re
import os
import numpy as np
from configparser import ConfigParser
from datetime import datetime
import statistics
dateTime = datetime.now().strftime('%Y%m%d%H%M%S')

def analyze_process_severity(configini):

    config = ConfigParser()
    configFile = str(configini)
    config.read(configFile)
    filesFound = []
    configFilelist = []
    csv_dir = config.get('General settings', 'csv_path')
    csv_dir_in = os.path.join(csv_dir, 'machine_results')
    use_master = config.get('General settings', 'use_master_config')
    severity_brackets = config.getint('Path plot settings', 'severity_brackets')
    vidInfPath = config.get('General settings', 'project_path')
    vidInfPath = os.path.join(vidInfPath, 'logs')
    vidInfPath = os.path.join(vidInfPath, 'video_info.csv')
    vidinfDf = pd.read_csv(vidInfPath)

    severityGrades = list(np.arange(0, 1.0, ((10/severity_brackets)/10)))
    severityGrades.append(10)
    severityLogFrames = [0] * severity_brackets
    severityLogTime = [0] * severity_brackets

    ########### logfile path ###########
    log_fn = config.get('General settings', 'project_name')
    log_fn = 'severity_' + dateTime + '.csv'
    log_path = config.get('General settings', 'project_path')
    log_path = os.path.join(log_path, 'logs')
    log_fn = os.path.join(log_path, log_fn)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    loopy=0

    headers = ['Video']
    for i in range(severity_brackets):
        currStr = 'Grade' + str(loopy) + '_frames'
        headers.append(currStr)
        loopy+=1
    loopy=0
    for i in range(severity_brackets):
        currStr = 'Grade' + str(loopy) + '_time'
        headers.append(currStr)
        loopy += 1
    log_df = pd.DataFrame(columns=headers)




    ########### FIND CSV FILES ###########
    if use_master == 'yes':
        for i in os.listdir(csv_dir_in):
            if i.__contains__(".csv"):
                file = os.path.join(csv_dir_in, i)
                filesFound.append(file)
    if use_master == 'no':
        config_folder_path = config.get('General settings', 'config_folder')
        for i in os.listdir(config_folder_path):
            if i.__contains__(".ini"):
                configFilelist.append(os.path.join(config_folder_path, i))
                iniVidName = i.split(".")[0]
                csv_fn = iniVidName + '.csv'
                file = os.path.join(csv_dir_in, csv_fn)
                filesFound.append(file)

    for i in filesFound:
        currentFile = i
        CurrentVideoName = os.path.basename(currentFile)
        videoSettings = vidinfDf.loc[vidinfDf['Video'] == str(CurrentVideoName.replace('.csv', ''))]
        fps = int(videoSettings['fps'])
        csv_df = pd.read_csv(currentFile, index_col=[0])
        for i in range(severity_brackets):
            lowerBound = severityGrades[i]
            upperBound = severityGrades[i + 1]
            currGrade = len(csv_df[(csv_df['aggression_prediction'] == 1) & (csv_df['Scaled_movement_M1_M2'] > lowerBound) & (csv_df['Scaled_movement_M1_M2'] <= upperBound)])
            severityLogFrames[i] = currGrade
        log_list = []
        log_list.append(str(CurrentVideoName.replace('.csv', '')))
        for bb in range(len(severityLogFrames)):
            severityLogTime[bb] = severityLogFrames[bb] / fps
        log_list.extend(severityLogFrames)
        log_list.extend(severityLogTime)
        log_df.loc[loopy] = log_list
        loopy += 1
    log_df = log_df.replace('NaN', 0)
    log_df.to_csv(log_fn, index=False)













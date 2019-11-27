import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from configparser import ConfigParser

configFile = r"Z:\DeepLabCut\DLC_extract\New_082119\project_folder\project_config.ini"

config = ConfigParser()
config.read(configFile)
frames_dir = config.get('Frame settings', 'frames_dir_out')
frames_dir_out = os.path.join(frames_dir, 'line_plot')
vidInfPath = config.get('General settings', 'project_path')
vidInfPath = os.path.join(vidInfPath, 'project_folder', 'logs')
vidInfPath = os.path.join(vidInfPath, 'video_info.csv')
vidinfDf = pd.read_csv(vidInfPath)
if not os.path.exists(frames_dir_out):
    os.makedirs(frames_dir_out)
csv_dir = config.get('General settings', 'csv_path')
csv_dir_in = os.path.join(csv_dir, 'machine_results')
use_master = config.get('General settings', 'use_master_config')

filesFound = []
loopy = 0
configFilelist = []


POI_1 = config.get('Distance plot', 'POI_1')
POI_2 = config.get('Distance plot', 'POI_2')
columnNames = (str(POI_1) + str('_x'), str(POI_1) + str('_y'), str(POI_2) + str('_x'), str(POI_2) + str('_y'))

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
    loop = 0
    currentFile = i
    if use_master == 'no':
        configFile = configFilelist[loopy]
        config = ConfigParser()
        config.read(configFile)
        fps = config.getint('Frame settings', 'fps')
        OneMMpixel = config.getint('Frame settings', 'mm_per_pixel') * 10
    CurrentVideoName = os.path.basename(currentFile)
    print(currentFile)
    videoSettings = vidinfDf.loc[vidinfDf['Video'] == str(CurrentVideoName.replace('.csv', ''))]
    fps = int(videoSettings['fps'])
    OneMMpixel = int(videoSettings['pixels/mm'])
    loopy+=1
    distanceOI_mm = []
    xTick = []
    xLabel = []
    csv_df = pd.read_csv(currentFile, usecols=[columnNames[0], columnNames[1], columnNames[2], columnNames[3]])
    csv_df_shifted = csv_df.shift(periods=1)
    csv_df_shifted = csv_df_shifted.rename(columns={columnNames[0]: str(columnNames[0] + '_shifted'), columnNames[1]: str(columnNames[1] + '_shifted'), columnNames[2]: str(columnNames[2] + '_shifted'), columnNames[3]: str(columnNames[3] + '_shifted')})
    csv_df_combined = pd.concat([csv_df, csv_df_shifted], axis=1, join='inner')
    csv_df_combined = csv_df_combined.fillna(0)
    csv_df_combined.Nose_1_x_shifted.iloc[[0]] = csv_df_combined[columnNames[0]]
    csv_df_combined.Nose_1_y_shifted.iloc[[0]] = csv_df_combined[columnNames[1]]
    csv_df_combined.Nose_2_x_shifted.iloc[[0]] = csv_df_combined[columnNames[2]]
    csv_df_combined.Nose_2_y_shifted.iloc[[0]] = csv_df_combined[columnNames[3]]
    csv_df_combined["distanceOI"] = np.sqrt((csv_df[columnNames[0]] - csv_df[columnNames[2]]) ** 2 + (csv_df[columnNames[1]] - csv_df[columnNames[3]]) ** 2)
    VideoNo = os.path.basename(currentFile)
    VideoNo = 'Video' + str(re.sub("[^0-9]", "", VideoNo))
    imagesDirOut = str(str(VideoNo) + str('_distance_plot'))
    savePath = os.path.join(frames_dir_out, imagesDirOut)
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    maxYaxis = int(((csv_df_combined['distanceOI'].max() / 10) + 10))
    yTicks = list(range(0, maxYaxis+1))
    yLabels = yTicks
    yTicks = yTicks[0::10]
    yLabels = yLabels[0::10]

    fig, ax = plt.subplots()
    for index, row in csv_df_combined.iterrows():
        distanceOI_mm.append((row["distanceOI"]) / 10)
        plt.plot(distanceOI_mm)
        plt.ylabel('distance (mm)')
        xTick.append(loop)
        xLabel.append(str(loop/fps))
        if len(xLabel) > fps * 60:
            xLabels = xLabel[0::250]
            xTicks = xTick[0::250]
        if len(xLabel) > ((fps * 60) * 10):
            xLabels = xLabel[0::150]
            xTicks = xTick[0::150]
        if len(xLabel) < fps * 60:
            xLabels = xLabel[0::75]
            xTicks = xTick[0::75]
        ax.set_xlabel('time (s)')
        ax.yaxis.grid(True)
        plt.xticks(xTicks, xLabels, rotation='vertical',fontsize=8)
        plt.yticks(yTicks, yLabels, fontsize=8)
        figName = str(str(loop) + '.png')
        figSavePath = os.path.join(savePath, figName)
        plt.savefig(figSavePath)
        loop+=1
        plt.close('all')
        print('figure saved: ' + str(figSavePath))
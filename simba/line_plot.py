import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from configparser import ConfigParser, NoSectionError, NoOptionError
import glob
from simba.rw_dfs import *

def line_plot_config(configini):
    configFile = str(configini)
    config = ConfigParser()
    config.read(configFile)
    projectPath = config.get('General settings', 'project_path')
    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'
    frames_dir_out = os.path.join(projectPath, 'frames', 'output', 'line_plot')
    if not os.path.exists(frames_dir_out):
        os.makedirs(frames_dir_out)
    vidInfPath = os.path.join(projectPath, 'logs', 'video_info.csv')
    vidinfDf = pd.read_csv(vidInfPath)
    csv_dir_in = os.path.join(projectPath, 'csv', 'machine_results')
    loopy = 0

    POI_1 = config.get('Distance plot', 'POI_1')
    POI_2 = config.get('Distance plot', 'POI_2')

    columnNames = (str(POI_1) + str('_x'), str(POI_1) + str('_y'), str(POI_2) + str('_x'), str(POI_2) + str('_y'))

    ########### FIND CSV FILES ###########
    filesFound = glob.glob(csv_dir_in + "/*." + wfileType)
    print('Generating line plots for ' + str(len(filesFound)) + ' video(s)...')

    for currentFile in filesFound:
        loop = 0
        CurrentVideoName = os.path.basename(currentFile)
        videoSettings = vidinfDf.loc[vidinfDf['Video'] == str(CurrentVideoName.replace('.' + wfileType, ''))]
        try:
            fps = int(videoSettings['fps'])
        except TypeError:
            print('Error: make sure all the videos that are going to be analyzed are represented in the project_folder/logs/video_info.csv file')
        loopy += 1
        distanceOI_mm, xTick, xLabel = [], [], []
        csv_df = read_df(currentFile, wfileType)
        csv_df = csv_df[[columnNames[0], columnNames[1], columnNames[2], columnNames[3]]]
        csv_df_shifted = csv_df.shift(periods=1)
        csv_df_shifted = csv_df_shifted.rename(columns={columnNames[0]: str(columnNames[0] + '_shifted'), columnNames[1]: str(columnNames[1] + '_shifted'), columnNames[2]: str(columnNames[2] + '_shifted'), columnNames[3]: str(columnNames[3] + '_shifted')})
        csv_df_combined = pd.concat([csv_df, csv_df_shifted], axis=1, join='inner')
        csv_df_combined = csv_df_combined.fillna(0)
        csv_df_combined["distanceOI"] = np.sqrt((csv_df[columnNames[0]] - csv_df[columnNames[2]]) ** 2 + (csv_df[columnNames[1]] - csv_df[columnNames[3]]) ** 2)
        savePath = os.path.join(frames_dir_out, CurrentVideoName.replace('.csv', ''))
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        maxYaxis = int(((csv_df_combined['distanceOI'].max() / 10) + 10))
        yTicks = list(range(0, maxYaxis + 1))
        yLabels = yTicks
        yTicks = yTicks[0::10]
        yLabels = yLabels[0::10]

        fig, ax = plt.subplots()

        for index, row in csv_df_combined.iterrows():
            distanceOI_mm.append((row["distanceOI"]) / 10)
            plt.plot(distanceOI_mm)
            plt.ylabel('distance (mm)')
            xTick.append(loop)
            xLabel.append(str(round((loop / fps), 2)))


            if len(xLabel) > fps * 60:
                xLabels, xTicks = xLabel[0::250], xTick[0::250]
            if len(xLabel) > ((fps * 60) * 10):
                xLabels, xTicks = xLabel[0::150], xTick[0::150]
            if len(xLabel) < fps * 60:
                xLabels, xTicks= xLabel[0::75], xTick[0::75]

            ax.set_xlabel('time (s)')
            ax.yaxis.grid(True)
            plt.xticks(xTicks, xLabels, rotation='vertical', fontsize=8)
            plt.yticks(yTicks, yLabels, fontsize=8)
            plt.suptitle(str(POI_1) + ' vs. ' + str(POI_2), x=0.5,y=0.92, fontsize=12)
            figName = str(str(loop) + '.png')
            figSavePath = os.path.join(savePath, figName)
            plt.savefig(figSavePath)
            loop += 1
            plt.close('all')
            print('Line plot ' + str(loop) + '/' + str(len(csv_df_combined)) + ' for video ' + str(loopy) + '/' + str(len(filesFound)))
    print('Finished generating line plots. Plots are saved @ project_folder/frames/output/line_plot')
import os
import pandas as pd
import matplotlib.pyplot as plt
from configparser import ConfigParser, NoSectionError, NoOptionError
import glob
from simba.rw_dfs import *
from simba.features_scripts.unit_tests import read_video_info_csv, read_video_info
from simba.misc_tools import get_file_path_parts

def plot_threshold(configini, behavior):
    configFile = str(configini)
    config = ConfigParser()
    config.read(configFile)
    projectPath = config.get('General settings', 'project_path')
    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'
    fileFolder = os.path.join(projectPath, 'csv', 'machine_results')
    videoLog = read_video_info_csv(os.path.join(projectPath, 'logs', 'video_info.csv'))
    filesFound = glob.glob(fileFolder +"/*." + wfileType)

    fileCounter = 0
    for file in filesFound:
        probabilityList, xTick, xLabel = [], [], []
        fileCounter += 1
        colName = 'Probability_' + behavior
        currDf = read_df(file, wfileType)
        currDf = currDf[colName]
        _, currVidName, _ = get_file_path_parts(file)
        currVidInfo, currPixPerMM, fps = read_video_info(vidinfDf=videoLog, currVidName=str(currVidName))
        highestTheshold = float(currDf.max())
        framesDirOut = os.path.join(projectPath, 'frames', 'output', 'probability_plots', currVidName)
        if not os.path.exists(framesDirOut):
            os.makedirs(framesDirOut)
        for index, row in currDf.iteritems():
            print(row)
            probabilityList.append(row)
            plt.ylim([0,highestTheshold])
            plt.plot(probabilityList, color="m", linewidth=6)
            plt.plot(index, probabilityList[-1], "o", markersize=20, color="m")
            plt.ylabel('probability ' + behavior)
            xTick.append(index)
            xLabel.append(str(round((index / fps), 1)))
            if len(xLabel) > fps * 60:
                xLabels, xTicks  = xLabel[0::250], xTick[0::250]
            if len(xLabel) > ((fps * 60) * 10):
                xLabels, xTicks = xLabel[0::150], xTick[0::150]
            if len(xLabel) < fps * 60:
                xLabels, xTicks = xLabel[0::75], xTick[0::75]
            plt.xlabel('time (s)')
            plt.grid()
            plt.xticks(xTicks, xLabels, rotation='vertical', fontsize=8)
            #plt.yticks(yTicks, yLabels, fontsize=8)
            plt.suptitle('probability ' + behavior, x=0.5, y=0.92, fontsize=12)
            figSavePath = os.path.join(framesDirOut, str(index) + '.png')
            plt.savefig(figSavePath)
            plt.close('all')
            print('Probability plot ' + str(index) + '/' + str(len(currDf)) + ' for video ' + str(fileCounter) + '/' + str(len(filesFound)))
        print('Finished generating line plots. Plots are saved @ project_folder/frames/output/probability_plots')
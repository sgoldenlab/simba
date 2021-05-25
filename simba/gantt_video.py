import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from configparser import ConfigParser, NoSectionError, NoOptionError
import glob
from simba.rw_dfs import *

def ganntplot_config(configini):
    config = ConfigParser()
    config.read(configini)
    projectPath = config.get('General settings', 'project_path')
    frames_dir_out = os.path.join(projectPath, 'frames', 'output', 'gantt_plots')
    if not os.path.exists(frames_dir_out):
        os.makedirs(frames_dir_out)
    csv_dir_in = os.path.join(projectPath, 'csv', 'machine_results')
    no_targets = config.getint('SML settings', 'No_targets')
    vidInfPath = os.path.join(projectPath, 'logs', 'video_info.csv')
    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'
    vidinfDf = pd.read_csv(vidInfPath)
    boutStart_list, target_names, VideoCounter = [], [], 0
    colours = ['red', 'green', 'pink', 'orange', 'blue', 'purple', 'lavender', 'grey', 'sienna', 'tomato', 'azure',
               'crimson', 'aqua', 'plum', 'teal', 'maroon', 'lime', 'coral']
    colourTupleX = list(np.arange(3.5, 203.5, 5))

    ########### FIND CSV FILES ###########
    filesFound = glob.glob(csv_dir_in + "/*." + wfileType)
    print('Generating gantt plots for ' + str(len(filesFound)) + ' video(s)...')

    ########### GET TARGET COLUMN NAMES ###########
    for ff in range(no_targets):
        currentModelNames = 'target_name_' + str(ff + 1)
        currentModelNames = config.get('SML settings', currentModelNames)
        target_names.append(currentModelNames)
    colours = colours[:len(target_names)]

    for currentFile in filesFound:
        CurrentVideoName = os.path.basename(currentFile.replace('.' + wfileType, ''))
        print('Analyzing file ' + str(CurrentVideoName) + '...')
        videoSettings = vidinfDf.loc[vidinfDf['Video'] == str(CurrentVideoName)]
        try:
            fps = int(videoSettings['fps'])
        except TypeError:
            print('Error: make sure all the videos that are going to be analyzed are represented in the project_folder/logs/video_info.csv file')
        VideoCounter += 1
        dataDf = read_df(currentFile, wfileType)
        dataDf['frames'] = np.arange(len(dataDf))
        rowCount = dataDf.shape[0]
        saveDir = os.path.join(frames_dir_out, CurrentVideoName)
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        boutsList, nameList, startTimeList, endTimeList,endFrameList = [], [], [], [], []
        for currTarget in target_names:
            groupDf = pd.DataFrame()
            v = (dataDf[currTarget] != dataDf[currTarget].shift()).cumsum()
            u = dataDf.groupby(v)[currTarget].agg(['all', 'count'])
            m = u['all'] & u['count'].ge(1)
            groupDf['groups'] = dataDf.groupby(v).apply(lambda x: (x.index[0], x.index[-1]))[m]
            for indexes, rows in groupDf.iterrows():
                currBout = list(rows['groups'])
                boutTime = ((currBout[-1] - currBout[0]) + 1) / fps
                startTime = (currBout[0] + 1) / fps
                endTime = (currBout[1]) / fps
                endFrame = (currBout[1])
                endTimeList.append(endTime)
                startTimeList.append(startTime)
                boutsList.append(boutTime)
                nameList.append(currTarget)
                endFrameList.append(endFrame)
        boutsDf = pd.DataFrame(list(zip(nameList, startTimeList, endTimeList, endFrameList, boutsList)),columns=['Event', 'Start_time', 'End Time', 'End_frame', 'Bout_time'])

        ################### PLOT #######################
        loop = 0
        for k in range(rowCount):
            fig, ax = plt.subplots()
            ylabels = ([s.replace('_prediction', '') for s in target_names])
            relRows = boutsDf.loc[boutsDf['End_frame'] <= k]
            for i, event in enumerate(relRows.groupby("Event")):
                for x in target_names:
                    if event[0] == x:
                        ix = target_names.index(x)
                        data_event = event[1][["Start_time", "Bout_time"]]
                        ax.broken_barh(data_event.values, (colourTupleX[ix], 3), facecolors=colours[ix])
                        loop += 1
            xLength = (round(k / fps)) + 1
            if xLength < 10:
                xLength = 10
            loop = 0
            ax.set_xlim(0, xLength)
            ax.set_ylim(0, colourTupleX[len(target_names)])
            setYtick = np.arange(5, 5 * no_targets + 1, 5)
            ax.set_yticks(setYtick)
            ax.set_yticklabels(ylabels, rotation=45, fontsize=8)
            ax.yaxis.grid(True)
            filename = (str(k) + '.png')
            savePath = os.path.join(saveDir, filename)
            if os.path.isfile(savePath):
                os.remove(savePath)
            plt.savefig(savePath)
            print('Gantt plot ' + str(k) + '/' + str(rowCount) + ' for video ' + str(VideoCounter) + '/' + str(len(filesFound)))
            plt.close('all')
        loop += 1






    print('Finished generating gantt plots. Plots are saved @ project_folder/frames/output/gantt')
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from configparser import ConfigParser

def ganntplot_config(configini):
    config = ConfigParser()
    config.read(configini)
    frames_dir_out = config.get('Frame settings', 'frames_dir_out')
    frames_dir_out = os.path.join(frames_dir_out, 'gantt_plots')
    if not os.path.exists(frames_dir_out):
        os.makedirs(frames_dir_out)
    csv_dir = config.get('General settings', 'csv_path')
    csv_dir_in = os.path.join(csv_dir, 'machine_results')
    no_targets = config.getint('SML settings', 'No_targets')
    use_master = config.get('General settings', 'use_master_config')
    vidInfPath = config.get('General settings', 'project_path')
    vidInfPath = os.path.join(vidInfPath, 'logs')
    vidInfPath = os.path.join(vidInfPath, 'video_info.csv')
    vidinfDf = pd.read_csv(vidInfPath)
    boutEnd = 0
    boutEnd_list = [0]
    boutStart_list = []
    filesFound = []
    target_names = []
    colours = ['red', 'green', 'pink', 'orange', 'blue', 'purple', 'lavender', 'grey', 'sienna', 'tomato', 'azure',
               'crimson', 'aqua', 'plum', 'teal', 'maroon', 'lime', 'coral']
    colourTupleX = list(np.arange(3.5, 203.5, 5))
    configFilelist = []
    loopy = 0

    ########### FIND CSV FILES ###########
    if use_master == 'yes':
        for i in os.listdir(csv_dir_in):
            if i.__contains__(".csv"):
                file = os.path.join(csv_dir_in, i)
                filesFound.append(file)
    print('Generating gantt plots for ' + str(len(filesFound)) + ' video(s)...')

    ########### GET TARGET COLUMN NAMES ###########
    for ff in range(no_targets):
        currentModelNames = 'target_name_' + str(ff + 1)
        currentModelNames = config.get('SML settings', currentModelNames)
        currentModelNames = currentModelNames
        target_names.append(currentModelNames)
    colours = colours[:len(target_names)]

    for i in filesFound:
        boutsDf = pd.DataFrame(columns=['Event', 'Start_frame', 'End_frame'])
        currentFile = i
        if use_master == 'no':
            configFile = configFilelist[loopy]
            config = ConfigParser()
            config.read(configFile)
            fps = config.getint('Frame settings', 'fps')
        CurrentVideoName = os.path.basename(currentFile)
        videoSettings = vidinfDf.loc[vidinfDf['Video'] == str(CurrentVideoName.replace('.csv', ''))]
        fps = int(videoSettings['fps'])
        loopy += 1
        dataDf = pd.read_csv(currentFile)
        dataDf['frames'] = np.arange(len(dataDf))
        rowCount = dataDf.shape[0]
        folderNm = os.path.basename(currentFile)
        folderName = str(folderNm.split('.')[0]) + str('_gantt')
        saveDir = os.path.join(frames_dir_out, folderName)
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        for bb in target_names:
            currTarget = bb
            for indexes, rows in dataDf[dataDf['frames'] >= boutEnd].iterrows():
                if rows[currTarget] == 1:
                    boutStart = rows['frames']
                    for index, row in dataDf[dataDf['frames'] >= boutStart].iterrows():
                        if row[currTarget] == 0:
                            boutEnd = row['frames']
                            if boutEnd_list[-1] != boutEnd:
                                boutStart_list.append(boutStart)
                                boutEnd_list.append(boutEnd)
                                values = [currTarget, boutStart, boutEnd]
                                boutsDf.loc[(len(boutsDf))] = values
                                break
                            break
            boutStart_list = [0]
            boutEnd_list = [0]
            boutEnd = 0

        # Convert to time
        boutsDf['Start_time'] = boutsDf['Start_frame'] / fps
        boutsDf['End_time'] = boutsDf['End_frame'] / fps
        boutsDf['Bout_time'] = boutsDf['End_time'] - boutsDf['Start_time']

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
            print('Gantt plot ' + str(k) + '/' + str(rowCount) + ' for video ' + str(loopy) + '/' + str(len(filesFound)))
            plt.close('all')
        loop += 1
    print('Finished generating gantt plots. Plots are saved @ project_folder/frames/output/gantt')
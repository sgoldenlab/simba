import pandas as pd
import os
from configparser import ConfigParser
from datetime import datetime
import numpy as np


def analyze_process_data_log(configini,chosenlist):

    dateTime = datetime.now().strftime('%Y%m%d%H%M%S')
    config = ConfigParser()
    configFile = str(configini)
    config.read(configFile)
    csv_dir = config.get('General settings', 'csv_path')
    csv_dir_in = os.path.join(csv_dir, 'machine_results')
    no_targets = config.getint('SML settings', 'No_targets')
    boutEnd = 0
    boutEnd_list = [0]
    boutStart_list = []
    filesFound = []
    target_names = []
    vidInfPath = config.get('General settings', 'project_path')
    vidInfPath = os.path.join(vidInfPath, 'logs')
    vidInfPath = os.path.join(vidInfPath, 'video_info.csv')
    vidinfDf = pd.read_csv(vidInfPath)
    loop = 0
    loopy = 0

    ########### FIND CSV FILES ###########
    for i in os.listdir(csv_dir_in):
        if i.endswith(".csv"):
            file = os.path.join(csv_dir_in, i)
            filesFound.append(file)

    ########### GET TARGET COLUMN NAMES ###########
    for ff in range(no_targets):
        currentModelNames = 'target_name_' + str(ff+1)
        currentModelNames = config.get('SML settings', currentModelNames)
        target_names.append(currentModelNames)
    print('Analyzing ' + str(len(target_names)) + ' classifier result(s) in ' + str(len(filesFound)) + ' video file(s).')

    ########### logfile path ###########
    log_fn = 'sklearn_' + str(dateTime) + '.csv'
    log_path = config.get('General settings', 'project_path')
    log_path = os.path.join(log_path, 'logs')
    log_fn = os.path.join(log_path, log_fn)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    headers = ['Video']
    for i in target_names:
        head1 = str(i) + ' events'
        head2 = str(i) + ' sum duration (s)'
        head3 = str(i) + ' mean duration (s)'
        head4 = str(i) + ' median duration (s)'
        head5 = str(i) + ' first occurance (s)'
        head6 = str(i) + ' mean interval (s)'
        head7 = str(i) + ' median interval (s)'
        headers.extend([head1, head2, head3, head4, head5, head6, head7])
    log_df = pd.DataFrame(columns=headers)

    for i in filesFound:
        boutsDf = pd.DataFrame(columns=['Event', 'Start_frame', 'End_frame'])
        currentFile = i
        currVidName = os.path.basename(currentFile)
        currVidName = currVidName.replace('.csv', '')
        fps = vidinfDf.loc[vidinfDf['Video'] == currVidName]
        try:
            fps = int(fps['fps'])
        except TypeError:
            print('Error: make sure all the videos that are going to be analyzed are represented in the project_folder/logs/video_info.csv file')
        loopy+=1
        print('Analyzing video ' + str(loopy) + '/' + str(len(filesFound)) + '...')
        dataDf = pd.read_csv(currentFile)
        dataDf['frames'] = np.arange(len(dataDf))
        folderNm = os.path.basename(currentFile)
        logFolderNm = str(folderNm.split('.')[0])
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

        #Convert to time
        boutsDf['Start_time'] = boutsDf['Start_frame'] / fps
        boutsDf['End_time'] = boutsDf['End_frame'] / fps
        boutsDf['Bout_time'] = boutsDf['End_time'] - boutsDf['Start_time']

        #record logs
        log_list = []
        log_list.append(logFolderNm)
        for i in target_names:
            currDf = boutsDf.loc[boutsDf['Event'] == i]
            try:
                firstOccur = round(currDf['Start_time'].iloc[0], 4)
            except IndexError:
                firstOccur = 0
            eventNOs = len(currDf)
            TotEventDur = round(currDf['Bout_time'].sum(), 4)
            try:
                MeanEventDur = round(TotEventDur / eventNOs, 4)
            except ZeroDivisionError:
                MeanEventDur = 0
            try:
                MedianEventDur = round(currDf['Bout_time'].median(), 10)

            except ZeroDivisionError:
                MedianEventDur = 0
            currDf_shifted = currDf.shift(periods=-1)
            currDf_shifted = currDf_shifted.drop(columns=['Event', 'Start_frame', 'End_frame', 'End_time', 'Bout_time'])
            currDf_shifted = currDf_shifted.rename(columns={'Start_time': 'Start_time_shifted'})
            currDf_combined = pd.concat([currDf, currDf_shifted], axis=1, join='inner')
            currDf_combined['Event_interval'] = currDf_combined['Start_time_shifted'] - currDf_combined['End_time']
            meanEventInterval = currDf_combined["Event_interval"].mean()
            medianEventInterval = currDf_combined['Event_interval'].median()
            log_list.append(eventNOs)
            log_list.append(TotEventDur)
            log_list.append(MeanEventDur)
            log_list.append(MedianEventDur)
            log_list.append(firstOccur)
            log_list.append(meanEventInterval)
            log_list.append(medianEventInterval)
        log_df.loc[loop] = log_list
        loop += 1
        print('File # processed for machine predictions: ' + str(loop) + '/' + str(len(filesFound)))
    log_df.fillna(0, inplace=True)
    # drop columns not chosen
    for i in chosenlist:
        log_df = log_df[log_df.columns.drop(list(log_df.filter(regex=str(i))))]


    log_df.to_csv(log_fn, index=False)
    print('All files processed for machine predictions: data file saved @' + str(log_fn))




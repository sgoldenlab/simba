import pandas as pd
import os
from configparser import ConfigParser
from datetime import datetime
import numpy as np
import glob


def analyze_process_data_log(configini,chosenlist):
    dateTime = datetime.now().strftime('%Y%m%d%H%M%S')
    config = ConfigParser()
    configFile = str(configini)
    config.read(configFile)
    projectPath = config.get('General settings', 'project_path')
    csv_dir_in = os.path.join(projectPath, 'csv', 'machine_results')
    no_targets = config.getint('SML settings', 'No_targets')
    filesFound, target_names = [], []
    vidinfDf = pd.read_csv(os.path.join(projectPath, 'logs', 'video_info.csv'))
    loop, videoCounter = 0, 0

    filesFound = glob.glob(csv_dir_in + '/*.csv')

    ########### GET TARGET COLUMN NAMES ###########
    for ff in range(no_targets):
        currentModelNames = 'target_name_' + str(ff+1)
        currentModelNames = config.get('SML settings', currentModelNames)
        target_names.append(currentModelNames)
    print('Analyzing ' + str(len(target_names)) + ' classifier result(s) in ' + str(len(filesFound)) + ' video file(s).')

    ########### logfile path ###########
    log_fn = 'sklearn_' + str(dateTime) + '.csv'
    log_fn = os.path.join(projectPath, 'logs', log_fn)

    headers = ['Video']
    headers_to_insert = [' # bout events', ' total events duration (s)',  ' mean bout duration (s)', ' median bout duration (s)', ' first occurance (s)', ' mean interval (s)', ' median interval (s)']
    for headerVar in headers_to_insert:
        for target in target_names:
            currHead = str(target) + headerVar
            headers.extend([currHead])
    log_df = pd.DataFrame(columns=headers)

    for currentFile in filesFound:
        videoCounter += 1
        currVidName = os.path.basename(currentFile).replace('.csv', '')
        fps = vidinfDf.loc[vidinfDf['Video'] == currVidName]
        try:
            fps = int(fps['fps'])
        except TypeError:
            print('Error: make sure all the videos that are going to be analyzed are represented in the project_folder/logs/video_info.csv file')
        print('Analyzing video ' + str(videoCounter) + '/' + str(len(filesFound)) + '...')
        dataDf = pd.read_csv(currentFile)
        boutsList, nameList, startTimeList, endTimeList = [], [], [], []
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
                endTimeList.append(endTime)
                startTimeList.append(startTime)
                boutsList.append(boutTime)
                nameList.append(currTarget)
        boutsDf = pd.DataFrame(list(zip(nameList, startTimeList, endTimeList, boutsList)), columns=['Event', 'Start Time', 'End Time', 'Duration'])
        boutsDf['Shifted start'] = boutsDf['Start Time'].shift(-1)
        boutsDf['Interval duration'] = boutsDf['Shifted start'] - boutsDf['End Time']

        firstOccurList, eventNOsList, TotEventDurList, MeanEventDurList, MedianEventDurList, TotEventDurList, meanIntervalList, medianIntervalList = [], [], [], [], [], [], [], []
        for targets in target_names:
            currDf = boutsDf.loc[boutsDf['Event'] == targets]
            try:
                firstOccurList.append(round(currDf['Start Time'].min(), 3))
            except IndexError:
                firstOccurList.append(0)
            eventNOsList.append(len(currDf))
            TotEventDurList.append(round(currDf['Duration'].sum(), 3))
            try:
                if len(currDf) > 1:
                    intervalDf = currDf[:-1].copy()
                    meanIntervalList.append(round(intervalDf['Interval duration'].mean(), 3))
                    medianIntervalList.append(round(intervalDf['Interval duration'].median(), 3))
                else:
                    meanIntervalList.append(0)
                    medianIntervalList.append(0)
            except ZeroDivisionError:
                meanIntervalList.append(0)
                medianIntervalList.append(0)
            try:
                MeanEventDurList.append(round(currDf["Duration"].mean(), 3))
                MedianEventDurList.append(round(currDf['Duration'].median(), 3))
            except ZeroDivisionError:
                MeanEventDurList.append(0)
                MedianEventDurList.append(0)
        currentVidList = [currVidName] + eventNOsList + TotEventDurList + MeanEventDurList + MedianEventDurList + firstOccurList + meanIntervalList + medianIntervalList

        log_df.loc[loop] = currentVidList
        loop += 1
        print('File # processed for machine predictions: ' + str(loop) + '/' + str(len(filesFound)))
    log_df.fillna(0, inplace=True)
    log_df.replace(0, np.NaN)

    # drop columns not chosen
    for target in target_names:
        for col2drop in chosenlist:
            currCol = target + ' ' + col2drop
            print(currCol)
            log_df = log_df.drop(currCol, 1)
    print(log_df.columns)
    log_df.to_csv(log_fn, index=False)
    print('All files processed for machine predictions: data file saved @' + str(log_fn))
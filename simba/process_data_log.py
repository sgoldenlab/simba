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
        head1 = str(i) + ' first occurance (s)'
        head2 = str(i) + ' # bout events'
        head3 = str(i) + ' total events duration (s)'
        head4 = str(i) + ' mean bout duration duration (s)'
        head5 = str(i) + ' median bout duration duration (s))'
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
                firstOccurList.append(round(boutsDf['Start Time'].min(), 3))
            except IndexError:
                firstOccurList.append(0)
            eventNOsList.append(len(currDf))
            TotEventDurList.append(round(currDf['Duration'].sum(), 3))
            try:
                meanIntervalList.append(round(currDf['Interval duration'].mean(), 3))
            except ZeroDivisionError:
                meanIntervalList.append(0)
            try:
                medianIntervalList.append(round(currDf['Interval duration'].median(), 3))
            except ZeroDivisionError:
                medianIntervalList.append(0)
            try:
                MeanEventDurList.append(round(currDf["Duration"].mean(), 3))
            except ZeroDivisionError:
                MeanEventDurList.append(0)
            try:
                MedianEventDurList.append(round(currDf['Duration'].median(), 3))
            except ZeroDivisionError:
                MedianEventDurList.append(0)
        currentVidList = [currVidName] + firstOccurList + eventNOsList + TotEventDurList + MeanEventDurList + MedianEventDurList + meanIntervalList + medianIntervalList
        log_df.loc[loop] = currentVidList
        loop += 1
        print('File # processed for machine predictions: ' + str(loop) + '/' + str(len(filesFound)))
    log_df.fillna(0, inplace=True)
    # drop columns not chosen
    for i in chosenlist:
        log_df = log_df[log_df.columns.drop(list(log_df.filter(regex=str(i))))]


    log_df.to_csv(log_fn, index=False)
    print('All files processed for machine predictions: data file saved @' + str(log_fn))




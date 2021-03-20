import os, glob
import pandas as pd
from configparser import ConfigParser, NoOptionError
from datetime import datetime
from simba.rw_dfs import *
from datetime import datetime


def time_bins_classifier(inifile,binLength):
    config = ConfigParser()
    configFile = str(inifile)
    config.read(configFile)
    projectPath = config.get('General settings', 'project_path')
    machineResultsFolder = os.path.join(projectPath, 'csv', 'machine_results')
    no_targets = config.getint('SML settings', 'No_targets')
    filesFound, target_names, fileCounter = [], [], 0
    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'
    vidinfDf = pd.read_csv(os.path.join(projectPath, 'logs', 'video_info.csv'))
    filesFound = glob.glob(machineResultsFolder + '/*.' + wfileType)
    dateTime = datetime.now().strftime('%Y%m%d%H%M%S')
    logList = []
    logDf = pd.DataFrame(columns=['Videos_omitted_from_time_bin_analysis'])

    ########### GET TARGET COLUMN NAMES ###########
    for ff in range(no_targets):
        currentModelNames = 'target_name_' + str(ff+1)
        currentModelNames = config.get('SML settings', currentModelNames)
        target_names.append(currentModelNames)
    print('Analyzing ' + str(len(target_names)) + ' classifier result(s) in ' + str(len(filesFound)) + ' video file(s).')

    for i in filesFound:
        outputList = []
        currDf = read_df(i, wfileType)
        currDf = currDf.drop(['scorer'], axis=1, errors='ignore')
        CurrentVideoName = os.path.basename(i)
        CurrentVideoRow = vidinfDf.loc[vidinfDf['Video'] == str(CurrentVideoName.replace('.' +wfileType, ''))]
        try:
            fps = int(CurrentVideoRow['fps'])
        except TypeError:
            print('Error: make sure all the videos that are going to be analyzed are represented in the project_folder/logs/video_info.csv file')
        binFrameLength = int(binLength * fps)
        currListDf = [currDf[i:i + binFrameLength] for i in range(0, currDf.shape[0], binFrameLength)]
        if fileCounter == 0:
            outputDfHeaders, outputDfHeadersDfTimeHeaders = ['Video'], []
            setBins = len(currListDf)

            for i in range(setBins):
                outputDfHeadersDfTimeHeaders.append('Bin_length_' + str(i + 1) + '_s')
            for target in range(len(target_names)):
                currTarget = target_names[target]
                for bin in range(setBins):
                    currHead1 = str(currTarget) + '_bin_no_' + str(bin + 1) + '_number_of_events'
                    currHead2 = str(currTarget) + '_bin_no_' + str(bin + 1) + '_total_event_duration'
                    currHead3 = str(currTarget) + '_bin_no_' + str(bin + 1) + '_mean_event_duration'
                    currHead4 = str(currTarget) + '_bin_no_' + str(bin + 1) + '_median_event_duration'
                    currHead5 = str(currTarget) + '_bin_no_' + str(bin + 1) + '_time_first_occurance'
                    currHead6 = str(currTarget) + '_bin_no_' + str(bin + 1) + '_mean_interval_duration'
                    currHead7 = str(currTarget) + '_bin_no_' + str(bin + 1) + '_median_interval_duration'
                    outputDfHeaders.extend((currHead1,currHead2,currHead3,currHead4,currHead5,currHead6,currHead7))
                dfHeaders = outputDfHeaders
                outputDf = pd.DataFrame(columns=dfHeaders)

        timebin = 0
        for currDf in currListDf[:setBins]:
            boutsList, nameList, startTimeList, endTimeList, timeBinList = [], [], [], [], []
            timebin +=1
            for currTarget in target_names:
                groupDf = pd.DataFrame()
                v = (currDf[currTarget] != currDf[currTarget].shift()).cumsum()
                u = currDf.groupby(v)[currTarget].agg(['all', 'count'])
                m = u['all'] & u['count'].ge(1)
                groupDf['groups'] = currDf.groupby(v).apply(lambda x: (x.index[0], x.index[-1]))[m]
                for indexes, rows in groupDf.iterrows():
                    currBout = list(rows['groups'])
                    boutTime = ((currBout[-1] - currBout[0]) + 1) / (fps + 2)
                    startTime = (currBout[0] + 1) / fps
                    endTime = (currBout[1]) / fps
                    endTimeList.append(endTime)
                    startTimeList.append(startTime)
                    boutsList.append(boutTime)
                    nameList.append(currTarget)
                    timeBinList.append(timebin)
            boutsDf = pd.DataFrame(list(zip(nameList, startTimeList, endTimeList, boutsList, timeBinList)), columns=['Event', 'Start Time', 'End Time', 'Duration', 'Time_bin'])
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
            currentVidList = eventNOsList + TotEventDurList + MeanEventDurList + MedianEventDurList + firstOccurList + meanIntervalList + medianIntervalList
            outputList.append(currentVidList)
        outputList = [item for sublist in outputList for item in sublist]
        currListDf = currListDf[:setBins]
        currListDf.reverse()
        outputList.insert(0, CurrentVideoName)
        if fileCounter == 0:
            listLength = len(outputList)
        try:
            outputDf.loc[len(outputDf)] = outputList
        except ValueError:
            targetVals, currentVals = listLength, len(outputList)
            difference = currentVals - targetVals
            if difference > 0:
                outputList = outputList[:-difference]
                print(CurrentVideoName + ' does not contain the same number of time bins as your other, previously analysed videos (it contains more). We shaved of these additional data bins to fit the dataframe.')
            if difference < 0:
                print(CurrentVideoName + ' does not contain the same number of time bins as your other, previously analysed videos (it contains less). We added a few zeros to this video to fit the dataframe.')
                addList = [0] * abs(difference)
                outputList.extend((addList))
            outputDf.loc[len(outputDf)] = outputList
            logList.append(CurrentVideoName)
        fileCounter += 1
        print('Processed time-bins for file ' + str(fileCounter) + '/' + str(len(filesFound)))
    log_fn = os.path.join(projectPath, 'logs', 'Time_bins_ML_results_' + dateTime + '.csv')
    outputDf = outputDf.fillna(0)
    outputDf.to_csv(log_fn, index=False)
    print('Time-bin analysis for machine results complete.')
    if len(logList) > 0:
        logDf['Videos_omitted_from_time_bin_analysis'] = logList
        log_fn = os.path.join(projectPath, 'logs', 'Time_bins_machine_results_omitted_videos_' + dateTime + '.csv')
        logDf.to_csv(log_fn)
        print('WARNING: Some of the videos you attempted to analyze contains an unequal number of time-bins and we had to omit / add some zeros to pad it out. To see which videos where had omitted times / added times, check the logfile in project_folder/logs or the SimBA GitHub repository for more information')

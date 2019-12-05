import pandas as pd
import os
from configparser import ConfigParser
from datetime import datetime

def analyze_process_data_log(configini):

    dateTime = datetime.now().strftime('%Y%m%d%H%M%S')

    config = ConfigParser()
    configFile = str(configini)
    config.read(configFile)
    frames_dir_out = config.get('Frame settings', 'frames_dir_out')
    frames_dir_out = os.path.join(frames_dir_out, 'gantt_plots')
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
        if i.__contains__(".csv"):
            file = os.path.join(csv_dir_in, i)
            filesFound.append(file)

    ########### GET TARGET COLUMN NAMES ###########
    for ff in range(no_targets):
        currentModelNames = 'target_name_' + str(ff+1)
        currentModelNames = config.get('SML settings', currentModelNames)
        target_names.append(currentModelNames)
    for i in range((len(target_names))):
        if '_prediction' in target_names[i]:
            continue
        else:
            b = target_names[i] + '_prediction'
            target_names[i] = b

    ########### logfile path ###########
    log_fn = 'sklearn_' + str(dateTime) + '.csv'
    log_path = config.get('General settings', 'project_path')
    log_path = os.path.join(log_path, 'logs')
    log_fn = os.path.join(log_path, log_fn)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    headers = ['Video']
    for i in target_names:
        head1 = str(i) + '_event_Nos'
        head2 = str(i) + '_Tot_event_duration'
        head3 = str(i) + '_mean_event_duration'
        head4 = str(i) + '_median_event_duration'
        head5 = str(i) + '_time_to_first_occurance'
        head6 = str(i) + '_mean_event_interval'
        head7 = str(i) + '_median_event_interval'
        headers.extend([head1, head2, head3, head4, head5, head6, head7])
    log_df = pd.DataFrame(columns=headers)

    for i in filesFound:
        boutsDf = pd.DataFrame(columns=['Event', 'Start_frame', 'End_frame'])
        currentFile = i
        currVidName = os.path.basename(currentFile)
        currVidName = currVidName.replace('.csv', '')
        fps = vidinfDf.loc[vidinfDf['Video'] == currVidName]
        fps = int(fps['fps'])
        loopy+=1
        readDf = pd.read_csv(currentFile)
        currCol = [col for col in readDf.columns if 'prediction' in col]
        currCol = [x for x in currCol if "Probability" not in x]
        dataDf = readDf.filter(currCol, axis=1)
        dataDf['frames'] = readDf['frames']
        folderNm = os.path.basename(currentFile)
        logFolderNm = 'Video' + str(folderNm.split('.')[0])
        folderName = str(folderNm.split('.')[0]) + str('_gantt')
        saveDir = os.path.join(frames_dir_out, folderName)
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        for bb in currCol:
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
                firstOccur = currDf['Start_time'].iloc[0]
            except IndexError:
                firstOccur = 0
            eventNOs = len(currDf)
            TotEventDur = round(currDf['Bout_time'].sum(), 2)
            try:
                MeanEventDur = TotEventDur / eventNOs
                MedianEventDur = currDf['Bout_time'].median()
            except ZeroDivisionError:
                MeanEventDur = 0
                MedianEventDur = 0
            currDf_shifted = currDf.shift(periods=-1)
            currDf_shifted = currDf_shifted.drop(columns=['Event', 'Start_frame', 'End_frame', 'End_time', 'Bout_time'])
            currDf_shifted = currDf_shifted.rename(columns={'Start_time': 'Start_time_shifted'})
            currDf_combined = pd.concat([currDf, currDf_shifted], axis=1, join='inner')
            currDf_combined['Event_interval'] = currDf_combined['Start_time_shifted'] - currDf_combined['End_time']
            meanEventInterval = currDf_combined["Event_interval"].mean()
            medianEventInterval = currDf_combined['Bout_time'].median()
            log_list.append(eventNOs)
            log_list.append(TotEventDur)
            log_list.append(MeanEventDur)
            log_list.append(MedianEventDur)
            log_list.append(firstOccur)
            log_list.append(meanEventInterval)
            log_list.append(medianEventInterval)
        log_df.loc[loop] = log_list
        loop += 1
        print('Files# processed for bout data: ' + str(loop))

    log_df.to_csv(log_fn, index=False)




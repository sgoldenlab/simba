import os, glob
import pandas as pd
from configparser import ConfigParser
from datetime import datetime


def time_bins_classifier(inifile,binLength):
    config = ConfigParser()
    configFile = str(inifile)
    config.read(configFile)
    projectPath = config.get('General settings', 'project_path')
    machineResultsFolder = os.path.join(projectPath, 'csv', 'machine_results')
    no_targets = config.getint('SML settings', 'No_targets')
    filesFound, target_names, fileCounter = [], [], 0
    vidinfDf = pd.read_csv(os.path.join(projectPath, 'logs', 'video_info.csv'))
    filesFound = glob.glob(machineResultsFolder + '/*.csv')
    dateTime = datetime.now().strftime('%Y%m%d%H%M%S')

    ########### GET TARGET COLUMN NAMES ###########
    for ff in range(no_targets):
        currentModelNames = 'target_name_' + str(ff+1)
        currentModelNames = config.get('SML settings', currentModelNames)
        target_names.append(currentModelNames)
    print('Analyzing ' + str(len(target_names)) + ' classifier result(s) in ' + str(len(filesFound)) + ' video file(s).')

    for i in filesFound:
        outputList = []
        currDf = pd.read_csv(i, index_col=0)
        CurrentVideoName = os.path.basename(i)
        CurrentVideoRow = vidinfDf.loc[vidinfDf['Video'] == str(CurrentVideoName.replace('.csv', ''))]
        try:
            fps = int(CurrentVideoRow['fps'])
        except TypeError:
            print('Error: make sure all the videos that are going to be analyzed are represented in the project_folder/logs/video_info.csv file')
        binFrameLength = int(binLength * fps)
        currListDf = [currDf[i:i + binFrameLength] for i in range(0, currDf.shape[0], binFrameLength)]
        if fileCounter == 0:
            outputDfHeaders, outputDfHeadersDfTimeHeaders = [], ['Video']
            setBins = len(currListDf)
            for i in range(setBins):
                outputDfHeadersDfTimeHeaders.append('Bin_length_' + str(i + 1) + '_s')
            for target in range(len(target_names)):
                currTarget = target_names[target]
                for bin in range(setBins):
                    currHead = str(currTarget) + '_bin_no_' + str(bin + 1) + '_s'
                    outputDfHeaders.append(currHead)
                dfHeaders = outputDfHeadersDfTimeHeaders + outputDfHeaders
                outputDf = pd.DataFrame(columns=dfHeaders)
        binLengthList = []
        for currBin in currListDf[:setBins]:
            behavFramesList = []
            for currBehav in target_names:
                behavFramesList.append(len(currBin.loc[currBin[currBehav] == 1]))
            binLengthList.append(len(currBin))
            outputList.extend(behavFramesList)
        outputListMerged = binLengthList + outputList
        outputListMerged = [x / fps for x in outputListMerged]
        outputListMerged = [round(num, 4) for num in outputListMerged]
        outputListMerged.insert(0, CurrentVideoName)
        outputDf.loc[len(outputDf)] = outputListMerged
        fileCounter += 1
        print('Processed time-bins for file ' + str(fileCounter) + '/' + str(len(filesFound)))

    log_fn = os.path.join(projectPath, 'logs', 'Time_bins_ML_results_' + dateTime + '.csv')
    outputDf.to_csv(log_fn, index=False)
    print('Time-bin analysis for machine results complete.')

























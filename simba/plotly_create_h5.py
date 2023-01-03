__author__ = "Simon Nilsson", "JJ Choong"

import os, glob
import pandas as pd
from configparser import ConfigParser, NoOptionError, NoOptionError
from datetime import datetime
from simba.rw_dfs import *

def create_plotly_container(configini, inputList):

    print('Generating SimBA project plotly/dash file container...')
    config = ConfigParser()
    configFile = str(configini)
    config.read(configFile)
    projectPath = config.get('General settings', 'project_path')
    machine_results_dir = os.path.join(projectPath, 'csv', 'machine_results')
    logsFolder = os.path.join(projectPath, 'logs')
    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'
    dateTime = datetime.now().strftime('%Y%m%d%H%M%S')
    storageFilePath = os.path.join(logsFolder, 'SimBA_dash_file_' + str(dateTime) + '.h5')
    storageFile = pd.HDFStore(storageFilePath, table=True, complib='blosc:zlib', complevel=9)
    vidInfoFilePath = os.path.join(projectPath, 'logs', 'video_info.csv')
    vidInfoDf = pd.read_csv(vidInfoFilePath, index_col=0)
    storageFile['Video_info'] = vidInfoDf
    model_nos = config.getint('SML settings', 'No_targets')
    target_names, probabilityColNames = [], []
    for i in range(model_nos):
        currentModelName = 'target_name_' + str(i+1)
        currentModelName = config.get('SML settings', currentModelName)
        currProbColName = 'Probability_' + currentModelName
        target_names.append(currentModelName)
        probabilityColNames.append(currProbColName)
    cols2Keep = target_names + probabilityColNames

    if inputList[0] == 1:
        sklearnFiles = glob.glob(logsFolder + '/sklearn_*')
        if not sklearnFiles:
            print('No sklearn machine data calculations could be found in your projects log folder')
        for file in sklearnFiles:
            currDf = read_df(file, wfileType)
            fileDfName = os.path.basename(file).replace('.' + wfileType, '')
            identifier = 'SklearnData/' + fileDfName
            storageFile[identifier] = currDf

    if inputList[1] == 1:
        timeBinsFiles = glob.glob(logsFolder + "/Time_bins_ML_results_*")
        if not timeBinsFiles:
            print('No time bins data calculations could be found in your projects log folder')
        for file in timeBinsFiles:
            timeBinsDf = pd.read_csv(file, index_col=0)
            fileDfName = os.path.basename(file).replace('.' + wfileType, '')
            identifier = 'TimeBins/' + fileDfName
            storageFile[identifier] = timeBinsDf

    if inputList[2] == 1:
        probabilityFiles = glob.glob(machine_results_dir + '/*.' + wfileType)
        if not probabilityFiles:
            print('No machine probability calculations could be found in your projects machine results folder')
        else:
            print('Compressing machine probability calculations for ' + str(len(probabilityFiles)) + ' files...')
        for file in probabilityFiles:
            currDf = pd.read_csv(file, index_col=0)
            currDf = currDf[cols2Keep]
            vidName = os.path.basename(file).replace('.' + wfileType, '')
            identifier = 'VideoData/' + vidName
            storageFile[identifier] = currDf

    if inputList[3] == 1:
        severityFiles = glob.glob(logsFolder + "/severity_*")
        if not severityFiles:
            print('No severity calculations could be found in your projects log folder')
        for file in severityFiles:
            severityDf = pd.read_csv(file, index_col=0)
            fileDfName = os.path.basename(file).replace('.' + wfileType, '')
            identifier = 'Severity/' + fileDfName
            storageFile[identifier] = severityDf

    if inputList[4] == 1:
        machineResultsFiles = glob.glob(machine_results_dir + "/*." + wfileType)
        if not machineResultsFiles:
            print('No machine results calculations could be found in your project folder')
        for file in machineResultsFiles:
            currDf = read_df(file, wfileType)
            fileDfName = os.path.basename(file).replace('.' + wfileType, '')
            identifier = 'Entire_data/' + fileDfName
            storageFile[identifier] = currDf

    infoDf = pd.DataFrame(columns=['Classifier_names'])
    for classifier in target_names:
        infoDf.loc[len(infoDf)] = [classifier]
    storageFile['Classifier_names'] = infoDf

    storageFile.close()

    print('SimBA project plotly/dash file container saved in project_folder/logs directory')































#
# timeBinsFile = r"Z:\Classifiers\Attack\project_folder\logs\Time_bins_ML_results_20200615145809.csv"
#
# sklearnFile = r"Z:\Classifiers\Attack\project_folder\logs\sklearn_20200615100537.csv"
# sklearnDf = pd.read_csv(sklearnFile, index_col=0)
# severityFile = r"Z:\Classifiers\Attack\project_folder\logs\severity_20200615145538.csv"
# severityDf = pd.read_csv(severityFile, index_col=0)
#
#
# ExampleStorage['sklearn_results'] = sklearnDf
# ExampleStorage['time_bins_results'] = timeBinsDf
#
# infoDf = pd.DataFrame(columns = ['Classifier_names', "Used_threshold", "fps", "Classifier_link"])
# infoDf.loc[len(infoDf)] = ['Attack', '0.5', "30", 'https://osf.io/stvhr/' ]
# infoDf.loc[len(infoDf)] = ['Sniffing', '0.5', "30", 'https://osf.io/sn72j/']
#
# ExampleStorage['Dataset_info'] = infoDf
#
# for csvfile in filesFound:
#     currDf = pd.read_csv(csvfile, index_col=0)
#     currDf = currDf[['Attack', 'Probability_Attack']]
#     currDf['Sniffing'] = np.random.randint(0, 1, currDf.shape[0])
#     currDf['Probability_Sniffing'] = np.random.uniform(low=0.00, high=1.00, size=(currDf.shape[0],))
#     vidName = os.path.basename(csvfile).replace('.csv', '')
#     identifier = 'VideoData/' + vidName
#     ExampleStorage[identifier] = currDf
#     print(ExampleStorage[identifier])
# ExampleStorage.close()

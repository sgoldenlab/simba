"""
Many thanks to GitHub @neurowookie

"""

import os, glob
import pandas as pd
from configparser import ConfigParser, MissingSectionHeaderError, NoSectionError, NoOptionError
from simba.rw_dfs import *
from simba.misc_tools import get_file_path_parts


def append_Boris_annot(configini, BorisPath):
    configFile = str(configini)
    config = ConfigParser()
    try:
        config.read(configFile)
    except MissingSectionHeaderError:
        print('ERROR:  Not a valid project_config file. Please check the project_config.ini path.')
    projectPath = config.get('General settings', 'project_path')
    featureFilePath = os.path.join(projectPath, 'csv', 'features_extracted')
    outputPath = os.path.join(projectPath, 'csv', 'targets_inserted')
    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'
    featFilesfound = glob.glob(featureFilePath + '/*.' + wfileType)
    borisFilesFound = glob.glob(BorisPath + '/*.csv')
    if len(borisFilesFound) == 0:
        print('No CSV files found in the BORIS file path: {}'.format(featureFilePath))
    model_nos = config.getint('SML settings', 'No_targets')
    behaviors = []
    for bb in range(model_nos):
        currentModelNames = 'target_name_' + str(bb+1)
        currentModelNames = config.get('SML settings', currentModelNames)
        behaviors.append(currentModelNames)
    combinedDf = pd.DataFrame()

    for file in borisFilesFound:
        currDf = pd.read_csv(file)
        try:
            index = (currDf[currDf['Observation id'] == "Time"].index.values)
            currDf = pd.read_csv(file, skiprows=range(0, int(index + 1)))
            currDf = currDf.loc[:, ~currDf.columns.str.contains('^Unnamed')]
            currDf.drop(['Behavioral category', 'Comment', 'Subject'], axis=1, inplace=True)
            for index, row in currDf.iterrows():
                _, currBase, _ = get_file_path_parts(row['Media file path'])
                currDf.at[index, 'Media file path'] = currBase
            combinedDf = pd.concat([combinedDf, currDf])
        except:
            print(str(file) + ' is not a BORIS annotation CSV file.')

    found_boris_behaviours = list(combinedDf['Behavior'].unique())
    print('Detected BORIS annotations:')
    for behavior in found_boris_behaviours:
        print(behavior)
    print(' ')
    print('DETECTED SimBA-project classifiers:')
    for behavior in behaviors:
        print(behavior)

    for featureFile in featFilesfound:
        featFileBaseName = os.path.basename(featureFile)
        print('Appending BORIS annotations to ' + featFileBaseName + '...')
        currfeatDf = read_df(featureFile, wfileType)
        _, videoName, _ = get_file_path_parts(featureFile)
        currVidAnnots = combinedDf.loc[combinedDf['Media file path'] == videoName]
        currVidAnnotsPoints = currVidAnnots.loc[currVidAnnots['Status'] == 'POINT']
        currVidAnnotsStart = currVidAnnots[(currVidAnnots.Status == 'START')]
        currVidAnnotsStop = currVidAnnots[(currVidAnnots.Status == 'STOP')]
        currVidAnnotsEvent = pd.concat([currVidAnnotsStart, currVidAnnotsStop], axis=0, join='inner', ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, copy=True).sort_index()


        for behavior in behaviors:
            currfeatDf[behavior] = 0

        ########### INSERT POINTS ###################
        for index, row in currVidAnnotsPoints.iterrows():
            currTime, currFps = float(row['Time']), int(row['FPS'])
            currFrame = int(currTime * currFps)
            currfeatDf.at[currFrame, behavior] = 1

        ########### INSERT EVENTS ###################
        for behavior in behaviors:
            currBehavDf = currVidAnnotsEvent[(currVidAnnotsEvent.Behavior == behavior)]
            currBehavDfStart = currBehavDf[(currBehavDf.Status == 'START')].reset_index(drop=True)
            currBehavDfStop = currBehavDf[(currBehavDf.Status == 'STOP')].reset_index(drop=True)
            for index, row in currBehavDfStart.iterrows():
                startTime, FPS = float(row['Time']), int(row['FPS'])
                try:
                    endTime = currBehavDfStop.iloc[index, 0]
                except IndexError:
                    total_length = float(row['Total length'])
                    endTime = int(total_length*FPS)
                startFrame, endFrame = int(startTime*FPS), int(endTime*FPS)
                print(startFrame, endFrame)
                FrameRange = list(range(startFrame, endFrame+1))




                currfeatDf.loc[FrameRange[0]:FrameRange[-1], behavior] = 1
        outputFilePath = os.path.join(outputPath, featFileBaseName)
        save_df(currfeatDf, wfileType, outputFilePath)
        print('BORIS annotations appended to ' + featFileBaseName)
    print('All BORIS annotations appended. The files are saved @' + outputPath)



#append_Boris_annot(r"Z:\DeepLabCut\DLC_extract\Troubleshooting\test_slp_import\project_folder\project_config.ini", r"Z:\DeepLabCut\DLC_extract\Troubleshooting\test_slp_import\data\boris")




import os, glob
import pandas as pd
from configparser import ConfigParser, NoSectionError, NoOptionError, MissingSectionHeaderError
from simba.rw_dfs import *


def append_dot_ANNOTT(inifile, annotationsFolder):
    configFile = str(inifile)
    config = ConfigParser()
    try:
        config.read(configFile)
    except MissingSectionHeaderError:
        print('ERROR:  Not a valid project_config file. Please check the project_config.ini path.')
    projectPath = config.get('General settings', 'project_path')
    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'
    featureFilesFolder = os.path.join(projectPath, 'csv', 'features_extracted')
    targetFolder = os.path.join(projectPath, 'csv', 'targets_inserted')
    videoLogPath = os.path.join(projectPath, 'logs', 'video_info.csv')
    videoLogDf = pd.read_csv(videoLogPath)
    featureFiles = glob.glob(featureFilesFolder + '/*.' + wfileType)
    noTargets = config.getint('SML settings', 'no_targets')
    targetList = []

    for i in range(noTargets):
        currTargName = 'target_name_' + str(i + 1)
        currTargVal = config.get('SML settings', currTargName)
        targetList.append('>' + currTargVal)

    for i in range(len(featureFiles)):
        curVideoFileName = os.path.basename(featureFiles[i]).replace('.' + wfileType, '')
        print('Processing file ' + str(curVideoFileName) + '...')
        currVideoSettings = videoLogDf.loc[videoLogDf['Video'] == curVideoFileName]
        fps = float(currVideoSettings['fps'])
        currAnnotationFilePath = os.path.join(annotationsFolder, curVideoFileName + '.annot')
        currFeatureFilePath = os.path.join(featureFilesFolder, curVideoFileName + '.' + wfileType)
        try:
            currAnnotFile = pd.read_csv(currAnnotationFilePath, delim_whitespace=True,index_col=False, low_memory=False)
        except FileNotFoundError:
            print('Could not find the ' + str(currAnnotationFilePath)+ ' and / or ' + str(currFeatureFilePath) +' files. Make sure the file exist.')
        currFeatureFile = read_df(currFeatureFilePath, wfileType)
        currFeatureFile = currFeatureFile.set_index([0])
        StartIndex = currAnnotFile.index[currAnnotFile['Bento'] == 'Ch1----------'].tolist()
        clippedAnnot = currAnnotFile.iloc[StartIndex[0]+1:]
        behavsInFile = clippedAnnot[clippedAnnot['Bento'].str.contains(">")]
        behavsInFile = behavsInFile.reset_index()
        for behavs in range(len(targetList)):
            behaviorName = targetList[behavs].replace('>', '')
            foundCheck = behavsInFile.index[behavsInFile['Bento'].str.contains(targetList[behavs])]
            if (len(foundCheck) == 1):
                foundChecker = int(foundCheck[0])
                if foundChecker < 0:
                    foundChecker = 0
                if foundChecker != len(behavsInFile)-1:
                    index1, index2 = behavsInFile.loc[foundChecker, 'index'], behavsInFile.loc[foundChecker+1, 'index']
                if foundChecker == len(behavsInFile)-1:
                    index1, index2 = behavsInFile.loc[foundChecker, 'index'], len(currAnnotFile)
                currBehav = currAnnotFile.iloc[index1:index2]
                currBehav = currBehav.iloc[2:]
                currBehav.columns = ['Start', 'Stop', 'Duration']
                currFeatureFile[behaviorName] = 0
                currBehav = currBehav.apply(pd.to_numeric)
                boutLists = []
                for index, row in currBehav.iterrows():
                    startFrame, endFrame = int((row['Start'] * fps)), int((row['Stop'] * fps))
                    boutLists.append(list(range(startFrame, endFrame)))
                frames, seconds = (sum(map(len, boutLists))), int((sum(map(len, boutLists))) / fps)
                print('Appending. Video: ' + str(curVideoFileName) + '. Behaviour: ' + str(behaviorName) + '. # frames labelled with behavior present: ' + str(frames) + '. Seconds of behavior labelled as present: ' + str(seconds))
                for bout in boutLists:
                    for frame in bout:
                        currFeatureFile.loc[frame, behaviorName] = 1
            else:
                print('No annotations in json for: ' + str(behaviorName) + ' in video ' + str(curVideoFileName) + '. Setting all ' + str(behaviorName) + ' annotations to 0 in video ' + str(curVideoFileName) + '.')
                currFeatureFile[behaviorName] = 0
        savePath = os.path.join(targetFolder, curVideoFileName + '.' + str(wfileType))
        print('Saving file...')
        save_df(currFeatureFile, wfileType, savePath)
        print('Saved annotations for file ' + str(curVideoFileName) + '.')
    print('All BENTO annotations appended to respective feature files.')
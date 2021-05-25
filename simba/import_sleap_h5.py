import h5py
import pandas as pd
import json
import cv2
import os, glob
from pylab import *
import numpy as np
import operator
from functools import reduce
from configparser import ConfigParser, MissingSectionHeaderError, NoOptionError
import errno
import simba.rw_dfs


#def importSLEAPbottomUP(inifile, dataFolder, currIDList):

data_folder = r'Z:\DeepLabCut\DLC_extract\Troubleshooting\Sleap_h5\import_folder'

configFile = str(r"Z:\DeepLabCut\DLC_extract\Troubleshooting\Sleap_h5\project_folder\project_config.ini")
config = ConfigParser()
try:
    config.read(configFile)
except MissingSectionHeaderError:
    print('ERROR:  Not a valid project_config file. Please check the project_config.ini path.')
projectPath = config.get('General settings', 'project_path')
animalIDs = config.get('Multi animal IDs', 'id_list')
currIDList = animalIDs.split(",")
currIDList = [x.strip(' ') for x in currIDList]
filesFound = glob.glob(data_folder + '/*.analysis.h5')
videoFolder = os.path.join(projectPath, 'videos')
outputDfFolder = os.path.join(projectPath, 'csv', 'input_csv')
try:
    wfileType = config.get('General settings', 'workflow_file_type')
except NoOptionError:
     wfileType = 'csv'
animalsNo = len(currIDList)
bpNamesCSVPath = os.path.join(projectPath, 'logs', 'measures', 'pose_configs', 'bp_names', 'project_bp_names.csv')
poseEstimationSetting = config.get('create ensemble settings', 'pose_estimation_body_parts')
print('Converting sleap h5 into dataframes...')
csvPaths = []

for filename in filesFound:
    video_save_name = os.path.basename(filename).replace('analysis.h5', wfileType)
    savePath = os.path.join(outputDfFolder, video_save_name)
    bpNames, orderVarList, OrderedBpList, MultiIndexCol, dfHeader, csvFilesFound, xy_heads, bp_cord_names, bpNameList, projBpNameList = [], [], [], [], [], [], [], [], [], []
    print('Processing ' + str(os.path.basename(filename)) + '...')
    hf = h5py.File(filename, 'r')
    bp_name_list, track_list, = [], [],
    for bp in hf.get('node_names'): bp_name_list.append(bp.decode('UTF-8'))
    for track in hf.get('track_names'): track_list.append(track.decode('UTF-8'))
    track_occupancy = hf.get('track_occupancy')
    with track_occupancy.astype('int16'):
        track_occupancy = track_occupancy[:]
    tracks = hf.get('tracks')
    with tracks.astype('int16'):
        tracks = tracks[:]
    frames = tracks.shape[3]

    animal_df_list = []
    for animals in range(len(track_list)):
        animal_x_array, animal_y_array = np.transpose(tracks[animals][0]), np.transpose(tracks[animals][1])
        animal_p_array = np.zeros(animal_x_array.shape)
        animal_array = np.ravel([animal_x_array, animal_y_array, animal_p_array], order="F").reshape(frames, len(bp_name_list) * 3)
        animal_df_list.append(pd.DataFrame(animal_array))
    video_df = pd.concat(animal_df_list, axis=1)

    for animal in range(len(currIDList)):
        for bp in bp_name_list:
            colName1, colName2, colName3 = str('Animal_' + str(animal+1) + '_' + bp + '_x'), ('Animal_' + str(animal+1) + '_' + bp + '_y'), ('Animal_' + str(animal+1) + '_' + bp + '_p')
            xy_heads.extend((colName1, colName2))
            bp_cord_names.append('_' + bp + '_x')
            bp_cord_names.append('_' + bp + '_y')
            bpNameList.extend((colName1, colName2, colName3))
            dfHeader.extend((colName1, colName2, colName3))
    if poseEstimationSetting == 'user_defined':
        config.set("General settings", "animal_no", str(animalsNo))
        with open(configFile, "w+") as f:
            config.write(f)
        f.close()

    bpNameListGrouped = [tuple(bpNameList[i:i + 3]) for i in range(0, len(bpNameList), 3)]

    video_df.columns = dfHeader
    video_df.fillna(0, inplace=True)
    simba.rw_dfs.save_df(video_df, wfileType, savePath)
    csvPaths.append(savePath)
    print('Saved file ' + savePath + '...')


###### ASSIGN IDENTITIES
global currIDcounter
def define_ID(event, x, y, flags, param):
    global currIDcounter
    if (event == cv2.EVENT_LBUTTONDBLCLK):
        centerX, centerY, currID = (int(x), int(y), currIDList[currIDcounter])
        ID_user_cords.append([centerX, centerY, currIDList[currIDcounter]])
        cv2.putText(overlay, str(currID), (centerX, centerY), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 255, 0), 5)
        currIDcounter += 1

cmap, colorList = cm.get_cmap(str('tab10'), animalsNo + 1), []
for i in range(cmap.N):
    rgb = list((cmap(i)[:3]))
    rgb = [i * 255 for i in rgb]
    rgb.reverse()
    colorList.append(rgb)

for csvFile in csvPaths:
    indBpCordList, frameNumber, addSpacer, EuclidDistanceList, changeList = [], 0, 2, [], []
    ID_user_cords, currIDcounter = [], 0
    assigningIDs, completePromt, chooseFrame, assignBpCords = False, False, True, True
    currDf = simba.rw_dfs.read_df(csvFile, wfileType)
    vidFname = os.path.join(videoFolder, os.path.basename(csvFile).replace('.csv', '.mp4'))
    vidBasename = os.path.basename(vidFname)
    if not os.path.exists(vidFname):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), vidFname)
    cap = cv2.VideoCapture(vidFname)
    if not cap.isOpened():
        raise Exception('Con\'t open video file ' + vidFname)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mySpaceScale, myRadius, myResolution, myFontScale = 40, 10, 1500, 1.2
    maxResDimension = max(width, height)
    circleScale, fontScale, spacingScale = int(myRadius / (myResolution / maxResDimension)), float(myFontScale / (myResolution / maxResDimension)), int(mySpaceScale / (myResolution / maxResDimension))
    cv2.namedWindow('Define animal IDs', cv2.WINDOW_NORMAL)

    while (1):
        if (chooseFrame == True) and (assignBpCords == True):
            cv2.namedWindow('Define animal IDs', cv2.WINDOW_NORMAL)
            cap.set(1, frameNumber)
            ret, frame = cap.read()
            if not ret:
                raise Exception('Can\'t read video file ' + vidFname)
            overlay = frame.copy()
            for animal_bps in range(len(bpNameListGrouped)):
                currCols = bpNameListGrouped[animal_bps]
                currcolor = tuple(colorList[animal_bps])
                x_cord = currDf.at[frameNumber, currCols[0]]
                y_cord = currDf.at[frameNumber, currCols[1]]
                indBpCordList.append([x_cord, y_cord, currCols[2]])
                cv2.circle(overlay, (int(x_cord), int(y_cord)), circleScale, currcolor, -1, lineType=cv2.LINE_AA)
            for loop, name in enumerate(indBpCordList):
                currstring = name[2]
                for substring in bp_cord_names:
                    if substring in currstring:
                        newstring = currstring.replace(substring, '')
                        indBpCordList[loop][2] = newstring
            imWithCordsOnly = overlay.copy()
            chooseFrame = False
        if (chooseFrame == False) and (assignBpCords == True):
            sideImage = np.ones((int(height / 2), width, 3))
            cv2.putText(sideImage, 'Current video: ' + str(vidBasename), (10, int(spacingScale)),cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 255), 3)
            cv2.putText(sideImage, 'Can you assign identities based on the displayed frame ?', (10, int(spacingScale * (addSpacer * 2))), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 255), 3)
            cv2.putText(sideImage, 'Press "x" to display new, random, frame', (10, int(spacingScale * (addSpacer * 3))), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 0), 3)
            cv2.putText(sideImage, 'Press "c" to continue to start assigning identities using this frame', (10, int(spacingScale * (addSpacer * 4))), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 255), 3)
            imageConcat = np.concatenate((overlay, sideImage), axis=0)
            imageConcat = np.uint8(imageConcat)
            cv2.imshow('Define animal IDs', imageConcat)
            k = cv2.waitKey(10)
            if k == ord('x'):
                cv2.destroyWindow('Define animal IDs')
                chooseFrame, assignBpCords = True, True
                frameNumber += 50
            elif k == ord('c'):
                chooseFrame, assignBpCords = False, False
                assigningIDs, completePromt, assigningIDs = True, False, True

        if assigningIDs == True:
            sideImage = np.ones((int(height / 2), width, 3))
            cv2.putText(sideImage, 'Double left mouse click on:', (10,  int(spacingScale)), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 255), 3)
            cv2.putText(sideImage, str(currIDList[currIDcounter]), (10, int(spacingScale * (addSpacer*2))), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 0), 3)
            imageConcat = np.concatenate((overlay, sideImage), axis=0)
            imageConcat = np.uint8(imageConcat)
            cv2.setMouseCallback('Define animal IDs', define_ID)
            cv2.imshow('Define animal IDs', imageConcat)
            cv2.waitKey(10)
            if currIDcounter >= len(currIDList):
                cv2.destroyWindow('Define animal IDs')
                assigningIDs, completePromt = False, True

        if completePromt == True:
            cv2.namedWindow('Define animal IDs', cv2.WINDOW_NORMAL)
            sideImage = np.ones((int(height/2), width, 3))
            cv2.putText(sideImage, 'Current video: ' + str(vidBasename), (10, int(spacingScale)), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 255), 3)
            cv2.putText(sideImage, 'Are you happy with your assigned identities ?', (10, int(spacingScale * (addSpacer*2))), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 255), 3)
            cv2.putText(sideImage, 'Press "c" to continue (to finish, or proceed to the next video)', (10, int(spacingScale * (addSpacer*3))), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 0), 3)
            cv2.putText(sideImage, 'Press "x" to re-start assigning identities', (10, int(spacingScale * (addSpacer*4))), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 255, 255), 3)
            imageConcat = np.concatenate((overlay, sideImage), axis=0)
            imageConcat = np.uint8(imageConcat)
            cv2.imshow('Define animal IDs', imageConcat)
            k = cv2.waitKey(10)
            if k == ord('c'):
                cv2.destroyWindow('Define animal IDs')
                break
            if k == ord('x'):
                overlay = imWithCordsOnly.copy()
                ID_user_cords, currIDcounter = [], 0
                assigningIDs, completePromt = True, False

    print('Re-organizing pose data-frame based on user-assigned identities: ' + str(os.path.basename(vidFname)) + '....')

    for values in ID_user_cords:
        currClickedX, currClickedY, currClickedID = values[0], values[1], values[2]
        for bpCords in indBpCordList:
            currX, currY, ID = bpCords[0], bpCords[1], bpCords[2]
            currEuclidian = np.sqrt((currClickedX - currX) ** 2 + (currClickedY - currY) ** 2)
            EuclidDistanceList.append([currEuclidian, currClickedID, ID])
    euclidDf = pd.DataFrame(EuclidDistanceList)
    euclidDf.columns = ['Distance', 'clickID', 'pose_ID']
    for i in currIDList:
        minDistance = euclidDf.loc[euclidDf['clickID'] == i, 'Distance'].min()
        animalPoseID = euclidDf.loc[euclidDf['Distance'] == minDistance, 'pose_ID'].iloc[0]
        changeList.append([animalPoseID, i])
    for animal in changeList:
        currPoseName, newName = animal[0], animal[1]
        loop = 0
        for header in bpNameList:
            if header.startswith(currPoseName):
                newHeader = header.replace(currPoseName, newName)
                bpNameList[loop] = newHeader
            loop += 1
    currDf.columns = bpNameList
    outDf = pd.DataFrame()
    for name in currIDList:
        currCols = [col for col in currDf.columns if name in col]
        sliceDf = currDf[currCols]
        outDf = pd.concat([outDf, sliceDf], axis=1)
    outDfcols = list(outDf.columns)
    toBpCSVlist = []
    if poseEstimationSetting == 'user_defined':
        for i in outDfcols:
            currBpName = i[:-2]
            for identityNo in range(len(currIDList)):
                if str(currIDList[identityNo]) in currBpName:
                    currBpName = currBpName + '_' + str(identityNo+1)
            toBpCSVlist.append(currBpName) if currBpName not in toBpCSVlist else toBpCSVlist
        f = open(bpNamesCSVPath, 'w+')
        for i in toBpCSVlist:
            f.write(i + '\n')
        f.close
    MultiIndexCol = []

    print('@@@@@@@@@@@@@@@@@@@',len(outDf.columns))
    for column in range(len(outDf.columns)):
        MultiIndexCol.append(tuple(('SLEAP_multi', 'SLEAP_multi', outDf.columns[column])))
    outDf.columns = pd.MultiIndex.from_tuples(MultiIndexCol, names=['scorer', 'bodypart', 'coords'])
    outputCSVname = os.path.basename(vidFname).replace('.mp4', '.csv')
    outDf.to_csv(os.path.join(outputDfFolder, outputCSVname))
    print('Imported ', outputCSVname, 'to project.')
print('All multi-animal SLEAP .slp tracking files ordered and imported into SimBA project in CSV file format')



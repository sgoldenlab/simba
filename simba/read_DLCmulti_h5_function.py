import os, glob
import pandas as pd
import cv2
import numpy as np
from configparser import ConfigParser, MissingSectionHeaderError
from pylab import *
import json

def importMultiDLCpose(inifile, dataFolder, filetype, idlist):
    global currIDcounter

    def define_ID(event, x, y, flags, param):
        global currIDcounter
        if (event == cv2.EVENT_LBUTTONDBLCLK):
            centerX, centerY, currID = (int(x), int(y), currIDList[currIDcounter])
            ID_user_cords.append([centerX, centerY, currIDList[currIDcounter]])
            cv2.putText(overlay, str(currID), (centerX, centerY), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 255, 0), 6)
            currIDcounter += 1

    configFile = str(inifile)
    config = ConfigParser()
    try:
        config.read(configFile)
    except MissingSectionHeaderError:
        print('ERROR:  Not a valid project_config file. Please check the project_config.ini path.')
    projectPath = config.get('General settings', 'project_path')
    if filetype == 'skeleton':
        filesFound = glob.glob(dataFolder + '/*sk.h5')
    if filetype == 'box':
        filesFound = glob.glob(dataFolder + '/*bx.h5')
    videoFolder = os.path.join(projectPath, 'videos')
    outputDfFolder = os.path.join(projectPath, 'csv', 'input_csv')
    bpNamesCSVPath = os.path.join(projectPath, 'logs', 'measures', 'pose_configs', 'bp_names', 'project_bp_names.csv')
    poseEstimationSetting = config.get('create ensemble settings', 'pose_estimation_body_parts')
    currIDList = idlist
    print('Importing ' + str(len(filesFound)) + ' multi-animal DLC h5 files to teh current project')

    for file in filesFound:
        bpNameList, x_heads, y_heads, xy_heads, indBpCordList, EuclidDistanceList, colorList, bp_cord_names, changeList, projBpNameList = [], [], [], [], [], [], [], [], [], []
        assigningIDs, completePromt, chooseFrame, assignBpCords = False, False, True, True
        addSpacer, ID_user_cords, currIDcounter, frameNumber = 2, [], 0, 0
        vidFname = os.path.join(videoFolder, os.path.basename(file).split('DLC_resnet50')[0] + '.mp4')
        vidBasename = os.path.basename(vidFname)
        currDf = pd.read_hdf(file)
        bpNames = [lis[2] for lis in list(currDf.columns)]
        uniqueBpNames = pd.Series(bpNames).drop_duplicates().tolist()
        animalIDs = [lis[1] for lis in list(currDf.columns)]
        UniqueIDs = pd.Series(animalIDs).drop_duplicates().tolist()
        cmap = cm.get_cmap(str('tab10'), len(UniqueIDs) + 1)
        for i in range(cmap.N):
            rgb = list((cmap(i)[:3]))
            rgb = [i * 255 for i in rgb]
            rgb.reverse()
            colorList.append(rgb)
        for animal in range(len(UniqueIDs)):
            for bp in uniqueBpNames:
                x_head, y_head, p_head = str('Animal' + str(animal) + '_' + bp + '_x'),  str('Animal' + str(animal) + '_' + bp + '_y'),  str('Animal' + str(animal) + '_' + bp + '_p')
                bp_cord_names.append('_' + bp + '_x')
                bp_cord_names.append('_' + bp + '_y')
                x_heads.append(x_head)
                y_heads.append(y_head)
                xy_heads.extend((x_head, y_head))
                bpNameList.extend((x_head, y_head, p_head))
        if poseEstimationSetting == 'user_defined':
            config.set("General settings", "animal_no", str(len(UniqueIDs)))
            with open(inifile, "w+") as f:
                config.write(f)
            f.close

        bpNameListGrouped = [xy_heads[x:x + len(uniqueBpNames)*2] for x in range(0, len(xy_heads) - 2, len(uniqueBpNames)*2)]
        currDf.columns = bpNameList
        currDf = currDf.fillna(0)
        cap = cv2.VideoCapture(vidFname)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        mySpaceScale, myRadius, myResolution, myFontScale = 40, 10, 1500, 1.2
        maxResDimension = max(width, height)
        circleScale = int(myRadius / (myResolution / maxResDimension))
        fontScale = float(myFontScale / (myResolution / maxResDimension))
        spacingScale = int(mySpaceScale / (myResolution / maxResDimension))
        cv2.namedWindow('Define animal IDs', cv2.WINDOW_NORMAL)

        while (1):
            if (chooseFrame == True) and (assignBpCords == True):
                cv2.namedWindow('Define animal IDs', cv2.WINDOW_NORMAL)
                cap.set(1, frameNumber)
                ret, frame = cap.read()
                overlay = frame.copy()
                cv2.namedWindow('Define animal IDs', cv2.WINDOW_NORMAL)
                for animal_bps in range(len(bpNameListGrouped)):
                    currCols = bpNameListGrouped[animal_bps]
                    currcolor = tuple(colorList[animal_bps])
                    for ind_bp_cords in range(0, len(currCols), 2):
                        y_cord = currDf.loc[currDf.index[frameNumber], currCols[ind_bp_cords + 1]]
                        x_cord = currDf.loc[currDf.index[frameNumber], currCols[ind_bp_cords]]
                        indBpCordList.append([x_cord, y_cord, currCols[ind_bp_cords]])
                        cv2.circle(overlay, (int(x_cord), int(y_cord)), circleScale, currcolor, -1, lineType=cv2.LINE_AA)
                    loop =0
                    for name in indBpCordList:
                        currstring = name[2]
                        for substring in bp_cord_names:
                            if substring in currstring:
                                newstring = currstring.replace(substring, '')
                                indBpCordList[loop][2] = newstring
                        loop+=1
                imWithCordsOnly = overlay.copy()
                chooseFrame = False
            if (chooseFrame == False) and (assignBpCords == True):
                sideImage = np.ones((int(height / 2), width, 3))
                cv2.putText(sideImage, 'Current video: ' + str(vidBasename), (10, int(spacingScale)),cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 255), 3)
                cv2.putText(sideImage, 'Can you assign identities based on the displayed frame ?', (10, int(spacingScale * (addSpacer * 2))), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 255), 3)
                cv2.putText(sideImage, 'Press "x" to display new, random, frame', (10, int(spacingScale * (addSpacer * 3))), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 0), 3)
                cv2.putText(sideImage, 'Press "c" to continue to start assigning identities using this frame', (10, int(spacingScale * (addSpacer * 4))), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 255, 0), 3)
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
                if currIDcounter >= len(UniqueIDs):
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
                toBpCSVlist.append(currBpName) if currBpName not in toBpCSVlist else toBpCSVlist
            f = open(bpNamesCSVPath, 'w+')
            for i in toBpCSVlist:
                f.write(i + '\n')
            f.close
        MultiIndexCol = []
        for column in range(len(outDf.columns)):
            MultiIndexCol.append(tuple(('DLC_multi', 'DLC_multi', outDf.columns[column])))
        outDf.columns = pd.MultiIndex.from_tuples(MultiIndexCol, names=['scorer', 'bodypart', 'coords'])
        outputCSVname = os.path.basename(vidFname).replace('.mp4', '.csv')
        outDf.to_csv(os.path.join(outputDfFolder, outputCSVname))
        print('Imported ', outputCSVname, 'to current project.')
    print('All multi-animal DLC .h5 tracking files ordered and imported into SimBA project in CSV file format')
__author__ = "Simon Nilsson", "JJ Choong"

import pandas as pd
import scipy.io as sio
import numpy as np
from configparser import ConfigParser, MissingSectionHeaderError, NoSectionError, NoOptionError
import os, glob
from simba.drop_bp_cords import *
import h5py
import cv2
import pyarrow.parquet as pq
import pyarrow as pa
from simba.interpolate_pose import *
from simba.drop_bp_cords import get_fn_ext, get_workflow_file_format
from simba.misc_tools import check_multi_animal_status, smooth_data_gaussian
import tables
from pathlib import Path


def import_trk(inifile, dataFolder, idlist, interpolation_method, smooth_settings_dict):
    global currIDcounter

    def define_ID(event, x, y, flags, param):
        global currIDcounter
        if (event == cv2.EVENT_LBUTTONDBLCLK):
            centerX, centerY, currID = (int(x), int(y), currIDList[currIDcounter])
            ID_user_cords.append([centerX, centerY, currIDList[currIDcounter]])
            cv2.putText(overlay, str(currID), (centerX, centerY), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 255, 0), 2)
            currIDcounter += 1

    configFile = str(inifile)
    config = ConfigParser()
    try:
        config.read(configFile)
    except MissingSectionHeaderError:
        print('ERROR:  Not a valid project_config file. Please check the project_config.ini path.')

    project_path = config.get('General settings', 'project_path')
    animalIDs = config.get('Multi animal IDs', 'id_list')
    noAnimals = config.getint('General settings', 'animal_no')
    currIDList = animalIDs.split(",")

    filesFound = glob.glob(dataFolder + '/*.trk')
    if len(filesFound) == 0: print('No TRK files found in ' + str(dataFolder))

    videoFolder = os.path.join(project_path, 'videos')
    outputDfFolder = os.path.join(project_path, 'csv', 'input_csv')

    wfileType = get_workflow_file_format(config)

    # ADD CORRECTION IF ONLY ONE ANIMAL
    if noAnimals < 2:
        idlist = ['Animal_1']

    multiAnimalStatus, multiAnimalIDList = check_multi_animal_status(config, noAnimals)
    Xcols, Ycols, Pcols = getBpNames(inifile)
    currIDList = idlist
    cMapSize = int(len(Xcols) / noAnimals) + 1
    colorListofList = createColorListofList(noAnimals, cMapSize)
    animalBpDict = create_body_part_dictionary(multiAnimalStatus, multiAnimalIDList, noAnimals, Xcols, Ycols, Pcols, colorListofList)


    for filename in filesFound:
        bpNameList, x_heads, y_heads, xy_heads, indBpCordList, EuclidDistanceList, colorList, bp_cord_names, changeList, projBpNameList = [], [], [], [], [], [], [], [], [], []
        assigningIDs, completePromt, chooseFrame, assignBpCords = False, False, True, True
        addSpacer, ID_user_cords, currIDcounter, frameNumber = 2, [], 0, 0

        print('Processing ' + str(os.path.basename(filename)) + '...')

        _, file_name_wo_ext, VideoExtension = get_fn_ext(filename)
        vidBasename = file_name_wo_ext + str(VideoExtension)
        if os.path.exists(os.path.join(videoFolder, file_name_wo_ext + '.mp4')):
            video_file = os.path.join(videoFolder, file_name_wo_ext + '.mp4')
        elif os.path.exists(os.path.join(videoFolder, file_name_wo_ext + '.avi')):
            video_file = os.path.join(videoFolder, file_name_wo_ext + '.avi')
        elif os.path.exists(os.path.join(videoFolder, file_name_wo_ext + '.AVI')):
            video_file = os.path.join(videoFolder, file_name_wo_ext + '.AVI')
        elif os.path.exists(os.path.join(videoFolder, file_name_wo_ext + '.MP4')):
            video_file = os.path.join(videoFolder, file_name_wo_ext + '.MP4')

        try:
            trk_dict = sio.loadmat(filename)
            trk_coordinates = trk_dict['pTrk']
            animals_tracked = trk_coordinates.shape[3]
            animals_tracked_list = [trk_coordinates[..., i] for i in range(animals_tracked)]


        except NotImplementedError:
            with h5py.File(filename, 'r') as trk_dict:
                trk_list = list(trk_dict['pTrk'])
                t_second = np.array(trk_list)
                if len(t_second.shape) > 3:
                    t_third = np.swapaxes(t_second, 0, 3)
                    trk_coordinates = np.swapaxes(t_third, 1, 2)
                    animals_tracked = trk_coordinates.shape[3]
                    animals_tracked_list = [trk_coordinates[..., i] for i in range(animals_tracked)]
                else:
                    trk_coordinates = np.swapaxes(t_second, 0, 2)
                    animals_tracked = 1
            print('Number of animals detected in TRK: ' + str(animals_tracked))

        animal_dfs = []
        if animals_tracked != 1:
            for animal in animals_tracked_list:
                m,n,r = animal.shape
                out_arr = np.column_stack((np.repeat(np.arange(m), n), animal.reshape(m * n, -1)))
                animal_dfs.append(pd.DataFrame(out_arr).T.iloc[1:].reset_index(drop=True))
            animal_dfs = pd.concat(animal_dfs, axis=1).fillna(0)
        else:
            m, n, r = trk_coordinates.shape
            out_arr = np.column_stack((np.repeat(np.arange(m), n), trk_coordinates.reshape(m * n, -1)))
            animal_dfs = pd.DataFrame(out_arr).T.iloc[1:].reset_index(drop=True)
        animal_dfs.columns = np.arange(len(animal_dfs.columns))
        p_cols = pd.DataFrame(1, index=animal_dfs.index, columns= animal_dfs.columns[1::2] + .5)
        currDf = pd.concat([animal_dfs, p_cols], axis=1).sort_index(axis=1)
        animal_dfs.columns = np.arange(len(animal_dfs.columns))
        new_headers = []

        for animal in animalBpDict.keys():
            for currXcol, currYcol, currPcol in zip(animalBpDict[animal]['X_bps'], animalBpDict[animal]['Y_bps'], animalBpDict[animal]['P_bps']):
                new_headers.extend((animal + '_' + currXcol, animal + '_' + currYcol, animal + '_' + currPcol))

        currDf.columns = new_headers

        cap = cv2.VideoCapture(video_file)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        mySpaceScale, myRadius, myResolution, myFontScale = 40, 10, 1500, 1.2
        maxResDimension = max(width, height)
        if maxResDimension == 0:
            print('Make sure you have imported the correct video(s) into your SimBA project.')
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
                for animal in animalBpDict.keys():
                    for currXcol, currYcol, currColor in zip(animalBpDict[animal]['X_bps'],
                                                             animalBpDict[animal]['Y_bps'],
                                                             animalBpDict[animal]['colors']):
                        y_cord = currDf.loc[currDf.index[frameNumber], animal + '_' + currYcol]
                        x_cord = currDf.loc[currDf.index[frameNumber], animal + '_' + currXcol]
                        indBpCordList.append([x_cord, y_cord, animal])
                        cv2.circle(overlay, (int(x_cord), int(y_cord)), circleScale, currColor, -1,
                                   lineType=cv2.LINE_AA)
                    loop = 0
                    for name in indBpCordList:
                        currstring = name[2]
                        for substring in bp_cord_names:
                            if substring in currstring:
                                newstring = currstring.replace(substring, '')
                                indBpCordList[loop][2] = newstring
                        loop += 1
                imWithCordsOnly = overlay.copy()
                chooseFrame = False
            if (chooseFrame == False) and (assignBpCords == True):
                sideImage = np.ones((int(height / 2), width, 3))
                cv2.putText(sideImage, 'Current video: ' + str(file_name_wo_ext), (10, int(spacingScale)),
                            cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 255), 2)
                cv2.putText(sideImage, 'Can you assign identities based on the displayed frame ?',
                            (10, int(spacingScale * (addSpacer * 2))), cv2.FONT_HERSHEY_SIMPLEX, fontScale,
                            (255, 255, 255), 2)
                cv2.putText(sideImage, 'Press "x" to display new, random, frame',
                            (10, int(spacingScale * (addSpacer * 3))), cv2.FONT_HERSHEY_SIMPLEX, fontScale,
                            (255, 255, 0), 2)
                cv2.putText(sideImage, 'Press "c" to continue to start assigning identities using this frame',
                            (10, int(spacingScale * (addSpacer * 4))), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 255, 0),
                            2)
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
                cv2.putText(sideImage, 'Double left mouse click on:', (10, int(spacingScale)), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale, (255, 255, 255), 2)
                cv2.putText(sideImage, str(currIDList[currIDcounter]), (10, int(spacingScale * (addSpacer * 2))),
                            cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 0), 2)
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
                sideImage = np.ones((int(height / 2), width, 3))
                cv2.putText(sideImage, 'Current video: ' + str(file_name_wo_ext), (10, int(spacingScale)),
                            cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 255), 3)
                cv2.putText(sideImage, 'Are you happy with your assigned identities ?',
                            (10, int(spacingScale * (addSpacer * 2))), cv2.FONT_HERSHEY_SIMPLEX, fontScale,
                            (255, 255, 255), 2)
                cv2.putText(sideImage, 'Press "c" to continue (to finish, or proceed to the next video)',
                            (10, int(spacingScale * (addSpacer * 3))), cv2.FONT_HERSHEY_SIMPLEX, fontScale,
                            (255, 255, 0), 2)
                cv2.putText(sideImage, 'Press "x" to re-start assigning identities',
                            (10, int(spacingScale * (addSpacer * 4))), cv2.FONT_HERSHEY_SIMPLEX, fontScale,
                            (0, 255, 255), 2)
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

        print('Re-organizing pose data-frame based on user-assigned identities: ' + str(
            os.path.basename(file_name_wo_ext)) + '....')

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
        currDf.columns = new_headers
        outDf = pd.DataFrame()

        for name in currIDList:
            currCols = [col for col in currDf.columns if name in col]
            sliceDf = currDf[currCols]
            outDf = pd.concat([outDf, sliceDf], axis=1)

        MultiIndexCol = []
        for column in range(len(outDf.columns)):
            MultiIndexCol.append(tuple(('APT_multi', 'APT_multi', outDf.columns[column])))
        outDf.columns = pd.MultiIndex.from_tuples(MultiIndexCol, names=('scorer', 'bodypart', 'coords'))
        outputCSVname = os.path.basename(vidBasename).replace(VideoExtension, '.' + wfileType)
        if wfileType == 'parquet':
            table = pa.Table.from_pandas(outDf)
            pq.write_table(table, os.path.join(outputDfFolder, outputCSVname))
        if wfileType == 'csv':
            outDf.to_csv(os.path.join(outputDfFolder, outputCSVname))

        if interpolation_method != 'None':
            print('Interpolating missing values (Method: ' + str(interpolation_method) + ') ...')
            if wfileType == 'parquet': csv_df = pd.read_parquet(os.path.join(outputDfFolder, outputCSVname))
            if wfileType == 'csv': csv_df = pd.read_csv(os.path.join(outputDfFolder, outputCSVname), index_col=0)
            interpolate_body_parts = Interpolate(inifile, csv_df)
            interpolate_body_parts.detect_headers()
            interpolate_body_parts.fix_missing_values(interpolation_method)
            interpolate_body_parts.reorganize_headers()
            if wfileType == 'parquet':
                table = pa.Table.from_pandas(interpolate_body_parts.new_df)
                pq.write_table(table, os.path.join(outputDfFolder, outputCSVname))
            if wfileType == 'csv':
                interpolate_body_parts.new_df.to_csv(os.path.join(outputDfFolder, outputCSVname))

        if smooth_settings_dict['Method'] == 'Gaussian':
            time_window = smooth_settings_dict['Parameters']['Time_window']
            smooth_data_gaussian(config=config, file_path=os.path.join(outputDfFolder, outputCSVname), time_window_parameter=time_window)

        print('Imported', outputCSVname, 'to current project.')
    print(
        'All APT TRK tracking files ordered and imported into SimBA project in the chosen workflow file format')


# import_trk(r"Z:\DeepLabCut\DLC_extract\Troubleshooting\TRK_test\project_folder\project_config.ini",
#                    r"Z:\DeepLabCut\DLC_extract\Troubleshooting\TRK_test\import\data",
#                    ['Bob', 'Ben', 'Bill'],
#                    'None',
#                    {'Method': 'Gaussian', 'Parameters': {'Time_window': '200'}})


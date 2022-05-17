import os, glob
import pandas as pd
import cv2
import numpy as np
from configparser import ConfigParser, MissingSectionHeaderError, NoSectionError, NoOptionError
from pylab import *
from simba.rw_dfs import *
from simba.drop_bp_cords import *
import pyarrow.parquet as pq
import pyarrow as pa
from simba.interpolate_pose import *
import itertools
from simba.drop_bp_cords import get_fn_ext
from simba.misc_tools import smooth_data_gaussian

def importMultiDLCpose(inifile, dataFolder, filetype, idlist, interpolation_method, smooth_settings_dict):

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
        dlc_file_ending, dlc_filtered_file_ending = 'sk.h5', 'sk_filtered.h5'
    elif filetype == 'box':
        dlc_file_ending, dlc_filtered_file_ending = 'bx.h5', 'bx_filtered.h5'
    elif filetype == 'ellipse':
        dlc_file_ending, dlc_filtered_file_ending = 'el.h5', 'el_filtered.h5'
    print('Searching ' + str(dataFolder) + ' for file-endings: ' + '"' + dlc_file_ending + '"' + ' and ' + '"' +dlc_filtered_file_ending + '"')

    filesFound = glob.glob(dataFolder + '/*' + dlc_file_ending) + glob.glob(dataFolder + '/*' + dlc_filtered_file_ending)
    all_files_in_folder = glob.glob(dataFolder + '/*')

    if len(filesFound) == 0:
        print('Found 0 files in ' + str(dataFolder) + ' for ' + filetype + ' tracking method.')
        if len(all_files_in_folder) == 0:
            print(str(dataFolder) + ' contains 0 files.')
        else:
            print('SimBA found other, non-' + filetype + ' files in ' + dataFolder + ' which are listed below.')
            print(all_files_in_folder)
        raise FileNotFoundError()



    videoFolder = os.path.join(projectPath, 'videos')
    outputDfFolder = os.path.join(projectPath, 'csv', 'input_csv')
    poseEstimationSetting = config.get('create ensemble settings', 'pose_estimation_body_parts')
    noAnimals = config.getint('General settings', 'animal_no')

    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'

    try:
        multiAnimalIDList = config.get('Multi animal IDs', 'id_list')
        multiAnimalIDList = multiAnimalIDList.split(",")
        if multiAnimalIDList[0] != '':
            multiAnimalStatus = True
            print('Applying settings for multi-animal tracking...')
        else:
            multiAnimalStatus = False
            for animal in range(noAnimals):
                multiAnimalIDList.append('Animal_' + str(animal + 1) + '_')
            print('Applying settings for classical tracking...')
    except NoSectionError:
        multiAnimalIDList = []
        for animal in range(noAnimals):
            multiAnimalIDList.append('Animal_' + str(animal + 1) + '_')
        multiAnimalStatus = False
        print('Applying settings for classical tracking...')

    print('Importing ' + str(len(filesFound)) + ' multi-animal DLC h5 files to the current project')

    Xcols, Ycols, Pcols = getBpNames(inifile)
    currIDList = idlist

    split_p_and_file_exts = [['DLC_resnet50', 'DLC_resnet_50', 'DLC_dlcrnetms5', 'DLC_effnet_b0'], ['.mp4', '.MP4', '.avi', '.AVI']]
    split_p_and_file_exts = list(itertools.product(*split_p_and_file_exts))

    for file in filesFound:
        bpNameList, x_heads, y_heads, xy_heads, indBpCordList, EuclidDistanceList, colorList, bp_cord_names, changeList, projBpNameList = [], [], [], [], [], [], [], [], [], []
        assigningIDs, completePromt, chooseFrame, assignBpCords = False, False, True, True
        addSpacer, ID_user_cords, currIDcounter, frameNumber = 2, [], 0, 0
        currVidName = os.path.basename(file)

        vidFname = 'None'
        searched_for_list = []
        for c in split_p_and_file_exts:
            _, file_base_wo_ext, _ = get_fn_ext(file)
            possible_vid_name = file_base_wo_ext.split(c[0])[0] + c[1]
            if os.path.exists(os.path.join(videoFolder, possible_vid_name)):
                vidFname = os.path.join(videoFolder, possible_vid_name)
            else:
                searched_for_list.append(possible_vid_name)

        if vidFname == 'None':
            print(searched_for_list)
            print('ERROR: SimBA searched your project_folder/videos directory for a video file representing the ' + str(currVidName) + ' and could not find a match. Above is a list of possible video filenames that SimBA searched for within your projects video directory without success.')
            raise AttributeError

        vidBasename, VideoExtension = os.path.basename(vidFname), os.path.splitext(vidFname)[1]
        currDf = pd.read_hdf(file)
        bpNames, idNames = [lis[2] for lis in list(currDf.columns)] , [lis[1] for lis in list(currDf.columns)]
        uniqueIds = list(unique(idNames))

        cMapSize = int(len(Xcols) / noAnimals) + 1
        colorListofList = createColorListofList(noAnimals, cMapSize)
        animalBpDict = create_body_part_dictionary(multiAnimalStatus, multiAnimalIDList, noAnimals, Xcols, Ycols, Pcols, colorListofList)
        for animal in animalBpDict.keys():
            for currXcol, currYcol, currPcol in zip(animalBpDict[animal]['X_bps'], animalBpDict[animal]['Y_bps'], animalBpDict[animal]['P_bps']):
                bpNameList.extend((animal + '_' + currXcol, animal + '_' + currYcol, animal + '_' + currPcol))
        if poseEstimationSetting == 'user_defined':
            config.set("General settings", "animal_no", str(len(uniqueIds)))
            with open(inifile, "w+") as f:
                config.write(f)
            f.close
        try:
            currDf.columns = bpNameList
        except ValueError as err:
            print(err)
            print('The number of body-parts in the input files do not match the number of body-parts in your SimBA project. Make sure you have specified the correct number of animals and body-parts in your project.')
        currDf.replace([np.inf, -np.inf], np.nan, inplace=True)
        currDf = currDf.fillna(0)
        cap = cv2.VideoCapture(vidFname)
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
                    for currXcol, currYcol, currColor in zip(animalBpDict[animal]['X_bps'], animalBpDict[animal]['Y_bps'], animalBpDict[animal]['colors']):
                        y_cord = currDf.loc[currDf.index[frameNumber], animal + '_' + currYcol]
                        x_cord = currDf.loc[currDf.index[frameNumber], animal + '_' + currXcol]
                        indBpCordList.append([x_cord, y_cord, animal])
                        try:
                            cv2.circle(overlay, (int(x_cord), int(y_cord)), circleScale, currColor, -1, lineType=cv2.LINE_AA)
                        except Exception as err:
                            if type(err) == OverflowError:
                                print('ERROR: SimBA encountered a pose-estimated body-part located at pixel position ' + str(x_cord) + ' ' + str(y_cord) + '. This value is too large to be converted to an integer. Please check your pose-estimation to make sure that it is accurate.')
                                print(err.args)
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
                cv2.putText(sideImage, 'Can you assign identities based on the displayed frame ?', (10, int(spacingScale * (addSpacer * 2))), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 255), 2)
                cv2.putText(sideImage, 'Press "x" to display new, random, frame', (10, int(spacingScale * (addSpacer * 3))), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 0), 3)
                cv2.putText(sideImage, 'Press "c" to continue to start assigning identities using this frame', (10, int(spacingScale * (addSpacer * 4))), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 255, 0), 2)
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
                cv2.putText(sideImage, 'Double left mouse click on:', (10,  int(spacingScale)), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 255), 2)
                cv2.putText(sideImage, str(currIDList[currIDcounter]), (10, int(spacingScale * (addSpacer*2))), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 0), 2)
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
                cv2.putText(sideImage, 'Are you happy with your assigned identities ?', (10, int(spacingScale * (addSpacer*2))), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 255), 2)
                cv2.putText(sideImage, 'Press "c" to continue (to finish, or proceed to the next video)', (10, int(spacingScale * (addSpacer*3))), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 0), 2)
                cv2.putText(sideImage, 'Press "x" to re-start assigning identities', (10, int(spacingScale * (addSpacer*4))), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 255, 255), 2)
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
        MultiIndexCol = []
        for column in range(len(outDf.columns)):
            MultiIndexCol.append(tuple(('DLC_multi', 'DLC_multi', outDf.columns[column])))
        outDf.columns = pd.MultiIndex.from_tuples(MultiIndexCol, names=('scorer', 'bodypart', 'coords'))
        outputCSVname = os.path.basename(vidFname).replace(VideoExtension, '.' + wfileType)
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


        print('Imported ', outputCSVname, 'to current project.')
    print('All multi-animal DLC .h5 tracking files ordered and imported into SimBA project in the chosen workflow file format')


# importMultiDLCpose(r"Z:\DeepLabCut\DLC_extract\Troubleshooting\DLC_two_mice\project_folder\project_config.ini",
#                    r"Z:\DeepLabCut\DLC_extract\Troubleshooting\DLC_two_mice\project_folder\import\test_2",
#                    'ellipse',
#                    ['mouse1', 'mouse2'],
#                    'None',
#                    {'Method': 'Gaussian', 'Parameters': {'Time_window': '200'}})

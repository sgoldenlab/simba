import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=DeprecationWarning)
import pickle
from configparser import ConfigParser, NoOptionError, NoSectionError
import os
import pandas as pd
import math
import cv2
import warnings
from simba.drop_bp_cords import *
import matplotlib.pyplot as plt
import numpy as np
import io
import PIL
from PIL import Image
import imutils
from simba.rw_dfs import *
from pylab import cm
from simba.features_scripts.unit_tests import *

plt.interactive(True)
plt.ioff()
warnings.simplefilter(action='ignore', category=FutureWarning)


# inifile = r"Z:\DeepLabCut\DLC_extract\Troubleshooting\Zebrafish\Zebrafish\project_folder\project_config.ini"
# featuresPath = r"Z:\DeepLabCut\DLC_extract\Troubleshooting\Zebrafish\Zebrafish\project_folder\csv\features_extracted\20200730_AB_7dpf_850nm_0002.csv"
# modelPath = r"Z:\DeepLabCut\DLC_extract\Troubleshooting\Zebrafish\Zebrafish\models\generated_models\Rheotaxis.sav"
# savfile = ''
# dt = 0.4
# sb = 67
# generategantt = 0

def validate_model_one_vid(inifile,featuresPath,modelPath,dt,sb,generategantt):

    def create_gantt_img(boutsDf, image_index, fps, gantt_img_title):
        fig, ax = plt.subplots()
        fig.suptitle(gantt_img_title, fontsize=24)
        relRows = boutsDf.loc[boutsDf['End_frame'] <= image_index]
        for i, event in enumerate(relRows.groupby("Event")):
            data_event = event[1][["Start_time", "Bout_time"]]
            ax.broken_barh(data_event.values, (4, 4), facecolors='red')
        xLength = (round(k / fps)) + 1
        if xLength < 10: xLength = 10
        ax.set_xlim(0, xLength)
        ax.set_ylim([0, 12])
        plt.ylabel(classifier_name, fontsize=12)
        plt.yticks([])
        plt.xlabel('time(s)', fontsize=12)
        ax.yaxis.set_ticklabels([])
        ax.grid(True)
        buffer_ = io.BytesIO()
        plt.savefig(buffer_, format="png")
        buffer_.seek(0)
        image = PIL.Image.open(buffer_)
        ar = np.asarray(image)
        open_cv_image = ar[:, :, ::-1]
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
        open_cv_image = cv2.resize(open_cv_image, (640, 480))
        open_cv_image = np.uint8(open_cv_image)
        buffer_.close()
        plt.close(fig)

        return open_cv_image

    def resize_gantt(gantt_img, img_height):
        return imutils.resize(gantt_img, height=img_height)

    config = ConfigParser()
    config.read(str(inifile))
    sample_feature_file = str(featuresPath)
    sample_feature_file_Name = os.path.basename(sample_feature_file).split('.', 1)[0]
    discrimination_threshold = float(dt)
    projectFolder = config.get('General settings', 'project_path')
    videoInputFolder = os.path.join(projectFolder, 'videos')
    noAnimals = config.getint('General settings', 'animal_no')
    outputPath = os.path.join(projectFolder, 'frames', 'output', 'validation')
    csv_dir_out_validation = config.get('General settings', 'csv_path')
    for path in [outputPath, csv_dir_out_validation]:
        if not os.path.exists(path):
            os.makedirs(outputPath)
    vid_info_df = pd.read_csv(os.path.join(projectFolder, 'logs', 'video_info.csv'))
    vid_info_df["Video"] = vid_info_df["Video"].astype(str)
    currVideoSettings, currPixPerMM, fps = read_video_info(vid_info_df, sample_feature_file_Name)
    classifier_name = os.path.basename(modelPath).replace('.sav','')
    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'
    outputFileName = os.path.join(outputPath, os.path.basename(sample_feature_file.replace('.' + wfileType, '_' + classifier_name + '.avi')))
    inputFile = read_df(sample_feature_file, wfileType)
    outputDf = inputFile.copy()
    inputFileOrganised = drop_bp_cords(inputFile, inifile)
    print('Running model...')
    clf = pickle.load(open(modelPath, 'rb'))
    ProbabilityColName = 'Probability_' + classifier_name
    predictions = clf.predict_proba(inputFileOrganised)
    outputDf[ProbabilityColName] = predictions[:, 1]
    outputDf[classifier_name] = np.where(outputDf[ProbabilityColName] > discrimination_threshold, 1, 0)


    # CREATE LIST OF GAPS BASED ON SHORTEST BOUT AND FILL/REMOVE

    framesToPlug = int(int(fps) * int(sb) / 1000)
    framesToPlugList = list(range(1, framesToPlug + 1))
    framesToPlugList.reverse()
    patternListofLists, negPatternListofList = [], []
    for k in framesToPlugList:
        zerosInList, oneInlist = [0] * k, [1] * k
        currList = [1]
        currList.extend(zerosInList)
        currList.extend([1])
        currListNeg = [0]
        currListNeg.extend(oneInlist)
        currListNeg.extend([0])
        patternListofLists.append(currList)
        negPatternListofList.append(currListNeg)
    fillPatterns = np.asarray(patternListofLists)
    remPatterns = np.asarray(negPatternListofList)

    for currPattern in fillPatterns:
        n_obs = len(currPattern)
        outputDf['rolling_match'] = (outputDf[classifier_name].rolling(window=n_obs, min_periods=n_obs)
                                     .apply(lambda x: (x == currPattern).all())
                                     .mask(lambda x: x == 0)
                                     .bfill(limit=n_obs - 1)
                                     .fillna(0)
                                     .astype(bool)
                                     )
        outputDf.loc[outputDf['rolling_match'] == True, classifier_name] = 1
        outputDf = outputDf.drop(['rolling_match'], axis=1)
    for currPattern in remPatterns:
        n_obs = len(currPattern)
        outputDf['rolling_match'] = (outputDf[classifier_name].rolling(window=n_obs, min_periods=n_obs)
                                     .apply(lambda x: (x == currPattern).all())
                                     .mask(lambda x: x == 0)
                                     .bfill(limit=n_obs - 1)
                                     .fillna(0)
                                     .astype(bool)
                                     )
        outputDf.loc[outputDf['rolling_match'] == True, classifier_name] = 0
        outputDf = outputDf.drop(['rolling_match'], axis=1)

    outFname = os.path.join(csv_dir_out_validation, sample_feature_file_Name + '.' + wfileType)
    save_df(outputDf, wfileType, outFname)
    print('Predictions created for ' + sample_feature_file_Name)

    #generate the video based on the just generated classification
    target_counter = 0
    inputFile = read_df(outFname, wfileType)
    currentDf = inputFile.fillna(0).astype(int)

    try:
        animalIDlist = config.get('Multi animal IDs', 'id_list')
        animalIDlist = animalIDlist.split(",")
        if animalIDlist[0] != '':
            multiAnimalStatus = True
            print('Applying settings for multi-animal tracking...')
        else:
            multiAnimalStatus = False
            print('Applying settings for classical tracking...')
    except NoSectionError:
        animalIDlist = []
        for animal in range(noAnimals):
            animalIDlist.append('Animal' + str(animal + 1))
        multiAnimalStatus = False
        print('Applying settings for classical tracking...')

    x_cols, y_cols, p_cols = getBpNames(inifile)
    targetColumn = classifier_name

    if os.path.exists(os.path.join(videoInputFolder, sample_feature_file_Name + '.mp4')):
        currVideoFile = os.path.join(videoInputFolder, sample_feature_file_Name + '.mp4')
    elif os.path.exists(os.path.join(videoInputFolder, sample_feature_file_Name + '.avi')):
        currVideoFile = os.path.join(videoInputFolder, sample_feature_file_Name + '.avi')
    else:
        print('ERROR: Could not find the video file the project_folder/videos directory.')

    cap = cv2.VideoCapture(currVideoFile)

    ## find vid size and fps
    fps, width, height = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    if height < width:
        videoHeight, videoWidth = width, height
    else:
        videoHeight, videoWidth = height, width
    if generategantt == 'None':
        writer = cv2.VideoWriter(outputFileName, fourcc, fps, (videoWidth, videoHeight))
    mySpaceScale, myRadius, myResolution, myFontScale = 60, 20, 1500, 1.5
    maxResDimension = max(width, height)
    circleScale = int(myRadius / (myResolution / maxResDimension))
    fontScale = float(myFontScale / (myResolution / maxResDimension))
    spacingScale = int(mySpaceScale / (myResolution / maxResDimension))
    currRow = 0
    colorListofList = []
    cmaps = ['spring', 'summer', 'autumn', 'cool', 'Wistia', 'Pastel1', 'Set1', 'winter']
    cMapSize = int(len(x_cols)/noAnimals) + 1
    for colormap in range(noAnimals):
        currColorMap = cm.get_cmap(cmaps[colormap], cMapSize)
        currColorList = []
        for i in range(currColorMap.N):
            rgb = list((currColorMap(i)[:3]))
            rgb = [i * 255 for i in rgb]
            rgb.reverse()
            currColorList.append(rgb)
        colorListofList.append(currColorList)
    animalBpDict = create_body_part_dictionary(multiAnimalStatus, animalIDlist, noAnimals, x_cols, y_cols, p_cols, colorListofList)

    if generategantt != 'None':
        boutsList, nameList, startTimeList, endTimeList, endFrameList = [], [], [], [], []
        groupDf = pd.DataFrame()
        v = (currentDf[classifier_name] != currentDf[classifier_name].shift()).cumsum()
        u = currentDf.groupby(v)[classifier_name].agg(['all', 'count'])
        m = u['all'] & u['count'].ge(1)
        groupDf['groups'] = currentDf.groupby(v).apply(lambda x: (x.index[0], x.index[-1]))[m]
        for indexes, rows in groupDf.iterrows():
            currBout = list(rows['groups'])
            boutTime = ((currBout[-1] - currBout[0]) + 1) / fps
            startTime = (currBout[0] + 1) / fps
            endTime = (currBout[1]) / fps
            endFrame = (currBout[1])
            endTimeList.append(endTime)
            startTimeList.append(startTime)
            boutsList.append(boutTime)
            nameList.append(classifier_name)
            endFrameList.append(endFrame)
        boutsDf = pd.DataFrame(list(zip(nameList, startTimeList, endTimeList, endFrameList, boutsList)), columns=['Event', 'Start_time', 'End Time', 'End_frame', 'Bout_time'])

        if generategantt != 'None':
            gantt_img_title = 'Behavior gantt chart (entire session)'
            gantt_img = create_gantt_img(boutsDf, len(currentDf), fps, gantt_img_title)
            gantt_img = resize_gantt(gantt_img, videoHeight)
            writer = cv2.VideoWriter(outputFileName, fourcc, fps, (int(videoWidth + gantt_img.shape[1]), int(videoHeight)))

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            IDlabelLoc = []
            for currAnimal in range(noAnimals):
                currentDictID = list(animalBpDict.keys())[currAnimal]
                currentDict = animalBpDict[currentDictID]
                currNoBps = len(currentDict['X_bps'])
                IDappendFlag = False
                animalArray = np.empty((currNoBps, 2), dtype=int)

                for bp in range(currNoBps):
                    hullColor = currentDict['colors'][bp]
                    currXheader, currYheader, currColor = currentDict['X_bps'][bp], currentDict['Y_bps'][bp], currentDict['colors'][bp]
                    currAnimal = currentDf.loc[currentDf.index[currRow], [currXheader, currYheader]]
                    cv2.circle(frame, (currAnimal[0], currAnimal[1]), 0, hullColor, circleScale)
                    animalArray[bp] = [currAnimal[0], currAnimal[1]]
                    if ('Centroid' in currXheader) or ('Center' in currXheader) or ('centroid' in currXheader) or ('center' in currXheader):
                        IDlabelLoc.append([currAnimal[0], currAnimal[1]])
                        IDappendFlag = True
                if IDappendFlag == False:
                    IDlabelLoc.append([currAnimal[0], currAnimal[1]])
            target_timer = (1/fps) * target_counter
            target_timer = round(target_timer, 2)
            if height < width:
                frame = np.array(Image.fromarray(frame).rotate(90,Image.BICUBIC, expand=True))

            cv2.putText(frame, str('Timer'), (10, ((height-height)+spacingScale)), cv2.FONT_HERSHEY_COMPLEX, fontScale, (0, 255, 0), 2)
            addSpacer = 2
            cv2.putText(frame, (str(classifier_name) + ' ' + str(target_timer) + str('s')), (10, (height-height)+spacingScale*addSpacer), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 255), 2)
            addSpacer+=1
            cv2.putText(frame, str('ensemble prediction'), (10, (height-height)+spacingScale*addSpacer), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 255, 0), 2)
            addSpacer += 2
            if currentDf.loc[currentDf.index[currRow], targetColumn] == 1:
                cv2.putText(frame, str(classifier_name), (10, (height - height) + spacingScale * addSpacer), cv2.FONT_HERSHEY_TRIPLEX, fontScale, (2, 166, 249), 2)
                target_counter += 1
                addSpacer += 1
            if generategantt == 'Gantt chart: final frame only (slightly faster)':
                frame = np.concatenate((frame, gantt_img), axis=1)
            if generategantt == 'Gantt chart: video':
                gantt_img = create_gantt_img(boutsDf, currRow, fps, 'Behavior gantt chart')
                gantt_img = resize_gantt(gantt_img, videoHeight)
                frame = np.concatenate((frame, gantt_img), axis=1)
            if generategantt != 'None':
                frame = cv2.resize(frame, (int(videoWidth + gantt_img.shape[1]), int(videoHeight)))
            writer.write(frame)
            currRow += 1
            print('Frame created: ' + str(currRow) + '/' + str(frames) + '.')
        if frame is None:
            print('Validation video saved @ ' + outputFileName)
            cap.release()
            writer.release()
            break
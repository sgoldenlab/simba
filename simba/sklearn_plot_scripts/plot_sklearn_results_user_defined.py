import cv2
import os
import pandas as pd
from scipy import ndimage
from configparser import ConfigParser, MissingSectionHeaderError
import glob

def plotsklearnresult_user_defined(configini):
    config = ConfigParser()
    configFile = str(configini)
    try:
        config.read(configFile)
    except MissingSectionHeaderError:
        print('ERROR:  Not a valid project_config file. Please check the project_config.ini path.')
    csv_dir = config.get('General settings', 'csv_path')
    csv_dir_in = os.path.join(csv_dir, "machine_results")
    frames_dir_in = config.get('Frame settings', 'frames_dir_in')
    frames_dir_out = config.get('Frame settings', 'frames_dir_out')
    frames_dir_out = os.path.join(frames_dir_out, 'sklearn_results')
    if not os.path.exists(frames_dir_out):
        os.makedirs(frames_dir_out)
    counters_no = config.getint('SML settings', 'No_targets')
    vidInfPath = config.get('General settings', 'project_path')
    logsPath = os.path.join(vidInfPath, 'logs')
    vidInfPath = os.path.join(vidInfPath, 'logs', 'video_info.csv')
    vidinfDf = pd.read_csv(vidInfPath)
    poseConfigPath = os.path.join(logsPath, 'measures', 'pose_configs', 'bp_names', 'project_bp_names.csv')
    poseConfigDf = pd.read_csv(poseConfigPath, header=None)
    bodypartNames = list(poseConfigDf[0])
    x_cols, y_cols, p_cols = ([], [], [])
    for bodypart in bodypartNames:
        x_cols.append(bodypart + '_x')
        y_cols.append(bodypart + '_y')
        p_cols.append(bodypart + '_p')
    loopy = 0

    ########### FIND CSV FILES ###########
    filesFound = glob.glob(csv_dir_in + '/*.csv')
    print('Generating frames for ' + str(len(filesFound)) + ' video(s)...')

    ########### GET MODEL NAMES ###########
    target_names = []
    for i in range(counters_no):
        currentModelNames = 'target_name_' + str(i + 1)
        currentModelNames = config.get('SML settings', currentModelNames)
        target_names.append(currentModelNames)

    ########### FIND PREDICTION COLUMNS ###########
    for i in filesFound:
        target_counters, target_timers = ([0] * counters_no, [0] * counters_no)
        currentVideo = i
        loopy += 1
        CurrentVideoName = os.path.basename(currentVideo)
        fps = vidinfDf.loc[vidinfDf['Video'] == str(CurrentVideoName.replace('.csv', ''))]
        try:
            fps = int(fps['fps'])
        except TypeError:
            print('Error: make sure all the videos that are going to be analyzed are represented in the project_folder/logs/video_info.csv file')
        currentDf = pd.read_csv(currentVideo)
        currentDf = currentDf.fillna(0)
        currentDf = currentDf.astype(int)
        loop = 0
        videoPathNm = str(CurrentVideoName.replace('.csv', ''))
        videoPathNmOut = str(videoPathNm) + '_frames'
        imagesDirIn = os.path.join(frames_dir_in, videoPathNm)
        imagesDirOut = os.path.join(frames_dir_out, videoPathNmOut)
        if not os.path.exists(imagesDirOut):
            os.makedirs(imagesDirOut)
        rowCount = currentDf.shape[0]

        for index, row in currentDf.iterrows():
            imageName = str(loop) + '.png'
            imageNameSave = str(loop) + '.bmp'
            image = os.path.join(imagesDirIn, imageName)
            imageSaveName = os.path.join(imagesDirOut, imageNameSave)
            im = cv2.imread(image)
            try:
                (height, width) = im.shape[:2]
            except AttributeError:
                print('ERROR: SimBA cannot find the appropriate frames. Please check the project_folder/frames/input folder.')
            fscale = 0.03
            cscale = 0.2
            space_scale = 0.8
            fontScale = min(width, height) / (25 / fscale)
            circleScale = int(min(width, height) / (25 / cscale))
            spacingScale = int(min(width, height) / (25 / space_scale))
            colors = [(255, 0, 0), (255, 191, 0), (255, 255, 0), (255, 165, 0), (0, 255, 0), (255, 0, 255), (0, 128, 0),
                      (255, 20, 147), (139, 0, 139), (127, 255, 212), (210, 105, 30), (255, 127, 80), (64, 224, 208),
                      (255, 105, 180)]
            for i in range(len(x_cols)):
                cv2.circle(im, (row[x_cols[i]], row[y_cols[i]]), circleScale, colors[i], thickness=-1, lineType=8, shift=0)

            if height < width:
               im = ndimage.rotate(im, 90)

            # draw event timers
            for b in range(counters_no):
                target_timers[b] = (1 / fps) * target_counters[b]
                target_timers[b] = round(target_timers[b], 2)

            cv2.putText(im, str('Timers'), (10, ((height - height) + spacingScale)), cv2.FONT_HERSHEY_COMPLEX, fontScale, (0, 255, 0), 2)
            addSpacer = 2
            for k in range(counters_no):
                cv2.putText(im, (str(target_names[k]) + ' ' + str(target_timers[k]) + str('s')),
                            (10, (height - height) + spacingScale * addSpacer), cv2.FONT_HERSHEY_SIMPLEX, fontScale,
                            (0, 0, 255), 2)
                addSpacer += 1

            cv2.putText(im, str('ensemble prediction'), (10, (height - height) + spacingScale * addSpacer),
                        cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 255, 0), 2)
            addSpacer += 1
            for p in range(counters_no):
                if row[target_names[p]] == 1:
                    cv2.putText(im, str(target_names[p]), (10, (height - height) + spacingScale * addSpacer), cv2.FONT_HERSHEY_TRIPLEX, fontScale, colors[p], 2)
                    target_counters[p] += 1
                    addSpacer += 1
            cv2.imwrite(imageSaveName, im)
            print('Frame ' + str(loop) + '/' + str(rowCount) + ' for video ' + str(loopy) + '/' + str(len(filesFound)))
            loop += 1
    print('Complete: Frames generated with machine predictions. Frames are saved @ project_folder/frames/output/sklearn_results')
import numpy as np
import cv2
import os
import pandas as pd
import re
from scipy import ndimage
from configparser import ConfigParser

def plotsklearnresult(configini):
    config = ConfigParser()
    configFile = str(configini)
    config.read(configFile)
    use_master = config.get('General settings', 'use_master_config')
    csv_dir = config.get('General settings', 'csv_path')
    csv_dir_in = os.path.join(csv_dir, "machine_results")
    frames_dir_in = config.get('Frame settings', 'frames_dir_in')
    frames_dir_out = config.get('Frame settings', 'frames_dir_out')
    frames_dir_out = os.path.join(frames_dir_out, 'sklearn_results')
    if not os.path.exists(frames_dir_out):
        os.makedirs(frames_dir_out)
    counters_no = config.getint('SML settings', 'No_targets')
    vidInfPath = config.get('General settings', 'project_path')
    vidInfPath = os.path.join(vidInfPath, 'logs')
    vidInfPath = os.path.join(vidInfPath, 'video_info.csv')
    vidinfDf = pd.read_csv(vidInfPath)
    currentDir = os.getcwd()
    filesFound = []
    firstRattle = True
    loop = 0
    target_names = []
    configFilelist = []
    original_target_names = []
    loopy = 0
    bodyPartCircleSize = 5

    ########### FIND CSV FILES ###########
    if use_master == 'yes':
        for i in os.listdir(csv_dir_in):
            if i.__contains__(".csv"):
                file = os.path.join(csv_dir_in, i)
                filesFound.append(file)
    if use_master == 'no':
        config_folder_path = config.get('General settings', 'config_folder')
        for i in os.listdir(config_folder_path):
            if i.__contains__(".ini"):
                configFilelist.append(os.path.join(config_folder_path, i))
                iniVidName = i.split(".")[0]
                csv_fn = iniVidName + '.csv'
                file = os.path.join(csv_dir_in, csv_fn)
                filesFound.append(file)

    ########### GET MODEL NAMES ###########
    for i in range(counters_no):
        currentModelNames = 'target_name_' + str(i + 1)
        currentModelNames = config.get('SML settings', currentModelNames)
        target_names.append(currentModelNames)
        original_target_names.append(currentModelNames)
    for i in range((len(target_names))):
        if '_prediction' in target_names[i]:
            continue
        else:
            b = target_names[i] + '_prediction'
            target_names[i] = b
    if any("tail_rattle" in s for s in target_names):
        tailRattleIndex = [i for i, x in enumerate(target_names) if x == 'tail_rattle_prediction'][0]

    ########### FIND PREDICTION COLUMNS ###########
    for i in filesFound:
        target_counters = [0] * counters_no
        target_timers = [0] * counters_no
        currentVideo = i
        if use_master == 'no':
            configFile = configFilelist[loopy]
            config = ConfigParser()
            config.read(configFile)
            fps = config.getint('Frame settings', 'fps')
        loopy += 1
        CurrentVideoName = os.path.basename(currentVideo)
        fps = vidinfDf.loc[vidinfDf['Video'] == str(CurrentVideoName.replace('.csv', ''))]
        fps = int(fps['fps'])
        CurrentVideoNumber = re.sub("\D", "", CurrentVideoName)
        currentDf = pd.read_csv(currentVideo)
        currentDf = currentDf.fillna(0)
        currentDf = currentDf.astype(int)
        for pp in range(len(original_target_names)):
            original_name = original_target_names[pp]
            new_name = target_names[pp]
            currentDf = currentDf.rename(columns={original_name: new_name})
        print(currentDf.columns)
        targetColumns = [col for col in currentDf.columns if 'prediction' in col]
        targetColumns = [x for x in targetColumns if "Probability" not in x]
        loop = 0
        videoPathNm = str(CurrentVideoName.replace('.csv', ''))
        videoPathNmOut = 'Video' + str(CurrentVideoNumber) + '_frames'
        imagesDirIn = os.path.join(frames_dir_in, videoPathNm)
        imagesDirOut = os.path.join(frames_dir_out, videoPathNmOut)
        if not os.path.exists(imagesDirOut):
            os.makedirs(imagesDirOut)

        for index, row in currentDf.iterrows():
            imageName = str(loop) + '.png'
            imageNameSave = str(loop) + '.bmp'
            image = os.path.join(imagesDirIn, imageName)
            imageSaveName = os.path.join(imagesDirOut, imageNameSave)
            im = cv2.imread(image)
            (height, width) = im.shape[:2]

            if height <= 400:
                LfontSize = 0.5
                MfontSize = 0.5
                SfontSize = 0.5
                textSpacing = 16
                bodyPartCircleSize = 5
            if height > 400:
                LfontSize = 1.5
                MfontSize = 1.2
                SfontSize = 1.0
                textSpacing = 32
                bodyPartCircleSize = 16

            M1polyglon_array = np.array(
                [[row['Ear_left_1_x'], row["Ear_left_1_y"]], [row['Ear_right_1_x'], row["Ear_right_1_y"]],
                 [row['Nose_1_x'], row["Nose_1_y"]], [row['Lat_left_1_x'], row["Lat_left_1_y"]], \
                 [row['Lat_right_1_x'], row["Lat_right_1_y"]], [row['Tail_base_1_x'], row["Tail_base_1_y"]],
                 [row['Center_1_x'], row["Center_1_y"]]]).astype(int)
            M2polyglon_array = np.array(
                [[row['Ear_left_2_x'], row["Ear_left_2_y"]], [row['Ear_right_2_x'], row["Ear_right_2_y"]],
                 [row['Nose_2_x'], row["Nose_2_y"]], [row['Lat_left_2_x'], row["Lat_left_2_y"]], \
                 [row['Lat_right_2_x'], row["Lat_right_2_y"]], [row['Tail_base_2_x'], row["Tail_base_2_y"]],
                 [row['Center_2_x'], row["Center_2_y"]]]).astype(int)
            M1polyglon_array_hull = cv2.convexHull((M1polyglon_array.astype(int)))
            M2polyglon_array_hull = cv2.convexHull((M2polyglon_array.astype(int)))

            # Draw DLC circles
            cv2.circle(im, (row['Ear_left_1_x'], row['Ear_left_1_y']), bodyPartCircleSize, (255, 0, 0), thickness=-1,
                       lineType=8, shift=0)
            cv2.circle(im, (row['Ear_right_1_x'], row['Ear_right_1_y']), bodyPartCircleSize, (255, 191, 0),
                       thickness=-1, lineType=8, shift=0)
            cv2.circle(im, (row['Nose_1_x'], row['Nose_1_y']), bodyPartCircleSize, (255, 255, 0), thickness=-1,
                       lineType=8, shift=0)
            cv2.circle(im, (row['Center_1_x'], row['Center_1_y']), bodyPartCircleSize, (255, 165, 0), thickness=-1,
                       lineType=8, shift=0)
            cv2.circle(im, (row['Lat_left_1_x'], row['Lat_left_1_y']), bodyPartCircleSize, (0, 255, 0), thickness=-1,
                       lineType=8, shift=0)
            cv2.circle(im, (row['Lat_right_1_x'], row['Lat_right_1_y']), bodyPartCircleSize, (255, 0, 255),
                       thickness=-1, lineType=8, shift=0)
            cv2.circle(im, (row['Tail_base_1_x'], row['Tail_base_1_y']), bodyPartCircleSize, (0, 128, 0), thickness=-1,
                       lineType=8, shift=0)

            cv2.circle(im, (row['Ear_left_2_x'], row['Ear_left_2_y']), bodyPartCircleSize, (255, 20, 147), thickness=-1,
                       lineType=8, shift=0)
            cv2.circle(im, (row['Ear_right_2_x'], row['Ear_right_2_y']), bodyPartCircleSize, (139, 0, 139),
                       thickness=-1, lineType=8, shift=0)
            cv2.circle(im, (row['Nose_2_x'], row['Nose_2_y']), bodyPartCircleSize, (127, 255, 212), thickness=-1,
                       lineType=8, shift=0)
            cv2.circle(im, (row['Center_2_x'], row['Center_2_y']), bodyPartCircleSize, (210, 105, 30), thickness=-1,
                       lineType=8, shift=0)
            cv2.circle(im, (row['Lat_left_2_x'], row['Lat_left_2_y']), bodyPartCircleSize, (255, 127, 80), thickness=-1,
                       lineType=8, shift=0)
            cv2.circle(im, (row['Lat_right_2_x'], row['Lat_right_2_y']), bodyPartCircleSize, (64, 224, 208),
                       thickness=-1, lineType=8, shift=0)
            cv2.circle(im, (row['Tail_base_2_x'], row['Tail_base_2_y']), bodyPartCircleSize, (255, 105, 180),
                       thickness=-1, lineType=8, shift=0)

            # get angles
            angle1 = (row['Mouse_1_angle'])
            angle2 = (row['Mouse_2_angle'])

            # draw centre to tailbase
            cv2.line(im, (M1polyglon_array[6][0], M1polyglon_array[6][1]),
                     (M1polyglon_array[5][0], M1polyglon_array[5][1]), (0, 0, 255), 2)
            cv2.line(im, (M2polyglon_array[6][0], M2polyglon_array[6][1]),
                     (M2polyglon_array[5][0], M2polyglon_array[5][1]), (0, 0, 255), 2)

            # draw nose to midpoint
            cv2.line(im, (M1polyglon_array[2][0], M1polyglon_array[2][1]),
                     (M1polyglon_array[6][0], M1polyglon_array[6][1]), (255, 0, 0), 2)
            cv2.line(im, (M2polyglon_array[2][0], M2polyglon_array[2][1]),
                     (M2polyglon_array[6][0], M2polyglon_array[6][1]), (255, 0, 0), 2)

            # draw hull
            cv2.drawContours(im, [M1polyglon_array_hull.astype(int)], 0, (255, 255, 255), 2)
            cv2.drawContours(im, [M2polyglon_array_hull.astype(int)], 0, (255, 165, 0), 2)

            # draw angle
            cv2.putText(im, str(angle1), (M1polyglon_array[6][0], M1polyglon_array[6][1]), cv2.FONT_HERSHEY_TRIPLEX,
                        0.8, (255, 255, 255), 2)
            cv2.putText(im, str(angle2), (M2polyglon_array[6][0], M2polyglon_array[6][1]), cv2.FONT_HERSHEY_TRIPLEX,
                        0.8, (255, 255, 255), 2)

            # draw event list
            x, y = 10, 50
            offset = 25

            if height < width:
               im = ndimage.rotate(im, 90)

            # draw event timers
            for b in range(counters_no):
                target_timers[b] = (1 / fps) * target_counters[b]
                target_timers[b] = round(target_timers[b], 2)

            cv2.putText(im, str('Timers'), (10, ((height - height) + textSpacing)), cv2.FONT_HERSHEY_COMPLEX, MfontSize,
                        (0, 255, 0), 2)
            addSpacer = 2
            for k in range(counters_no):
                cv2.putText(im, (str(target_names[k]) + ' ' + str(target_timers[k]) + str('s')),
                            (10, (height - height) + textSpacing * addSpacer), cv2.FONT_HERSHEY_SIMPLEX, SfontSize,
                            (0, 0, 255), 2)
                addSpacer += 1

            cv2.putText(im, str('ensemble prediction'), (10, (height - height) + textSpacing * addSpacer),
                        cv2.FONT_HERSHEY_SIMPLEX, MfontSize, (0, 255, 0), 2)
            addSpacer += 2
            colors = [(2, 166, 249), (47, 255, 173), (0, 165, 255), (60, 20, 220), (193, 182, 255), (238, 130, 238),
                      (144, 128, 112), (32, 165, 218), (0, 0, 128), (209, 206, 0)]
            for p in range(counters_no):
                if row[targetColumns[p]] == 1:
                    cv2.putText(im, str(target_names[p]), (10, (height - height) + textSpacing * addSpacer),
                                cv2.FONT_HERSHEY_TRIPLEX, LfontSize, colors[p], 2)
                    target_counters[p] += 1
                    addSpacer += 1
            if "tail_rattle" in target_names:
                if row[targetColumns[tailRattleIndex]] == 1:
                    if firstRattle == True:
                        dotCoordinateX = (row['Tail_end_1_x'] - 150)
                        dotCoordinateY = (row['Tail_end_1_y'] + 150)
                    cv2.circle(im, (dotCoordinateX, dotCoordinateY), 63, (0, 0, 255), -1)
                    cv2.putText(im, target_names[tailRattleIndex], ((dotCoordinateX + 50), (dotCoordinateY + 50)),
                                cv2.FONT_HERSHEY_TRIPLEX, LfontSize, (255, 255, 255), 2)
                    firstRattle = False
                if row[targetColumns[tailRattleIndex]] == 0:
                    firstRattle = True
            cv2.imwrite(imageSaveName, im)
            print(str(imageSaveName))
            loop += 1
    print('Complete: Frames generated with machine predictions')
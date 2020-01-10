import pickle
from configparser import ConfigParser
import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import subprocess
from scipy import ndimage
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def validate_model_one_vid(inifile,csvfile,savfile,dt,sb):
    configFile = str(inifile)
    config = ConfigParser()
    config.read(configFile)
    loop = 0
    sample_feature_file = str(csvfile)
    sample_feature_file_Name = os.path.basename(sample_feature_file)
    sample_feature_file_Name = sample_feature_file_Name.split('.', 1)[0]
    discrimination_threshold = float(dt)
    classifier_path = savfile
    classifier_name = os.path.basename(classifier_path).replace('.sav','')
    framefolder = config.get('Frame settings','frames_dir_in')
    inputFile = pd.read_csv(sample_feature_file)
    outputDf = inputFile
    inputFileOrganised = inputFile.drop(
        ["Ear_left_1_x", "Ear_left_1_y", "Ear_left_1_p", "Ear_right_1_x", "Ear_right_1_y", "Ear_right_1_p", "Nose_1_x",
         "Nose_1_y", "Nose_1_p", "Center_1_x", "Center_1_y", "Center_1_p", "Lat_left_1_x", "Lat_left_1_y",
         "Lat_left_1_p", "Lat_right_1_x", "Lat_right_1_y", "Lat_right_1_p", "Tail_base_1_x", "Tail_base_1_y",
         "Tail_base_1_p", "Tail_end_1_x", "Tail_end_1_y", "Tail_end_1_p", "Ear_left_2_x",
         "Ear_left_2_y", "Ear_left_2_p", "Ear_right_2_x", "Ear_right_2_y", "Ear_right_2_p", "Nose_2_x", "Nose_2_y",
         "Nose_2_p", "Center_2_x", "Center_2_y", "Center_2_p", "Lat_left_2_x", "Lat_left_2_y",
         "Lat_left_2_p", "Lat_right_2_x", "Lat_right_2_y", "Lat_right_2_p", "Tail_base_2_x", "Tail_base_2_y",
         "Tail_base_2_p", "Tail_end_2_x", "Tail_end_2_y", "Tail_end_2_p"], axis=1)
    print('Running model...')
    feature_list = list(inputFileOrganised.columns)
    feature_list = pd.DataFrame(feature_list)
    feature_list.to_csv("mylist.csv")
    clf = pickle.load(open(classifier_path, 'rb'))
    ProbabilityColName = 'Probability_' + classifier_name
    predictions = clf.predict_proba(inputFileOrganised)
    outputDf[ProbabilityColName] = predictions[:, 1]
    outputDf[classifier_name] = np.where(outputDf[ProbabilityColName] > discrimination_threshold, 1, 0)

    # CREATE LIST OF GAPS BASED ON SHORTEST BOUT
    shortest_bout = int(sb)
    vidInfPath = config.get('General settings', 'project_path')
    vidInfPath = os.path.join(vidInfPath, 'logs', 'video_info.csv')
    vidinfDf = pd.read_csv(vidInfPath)
    fps = vidinfDf.loc[vidinfDf['Video'] == str(sample_feature_file_Name.replace('.csv', ''))]
    try:
        fps = int(fps['fps'])
    except TypeError:
        print('Error: make sure all the videos that are going to be analyzed are represented in the project_folder/logs/video_info.csv file')
    framesToPlug = int(fps * (shortest_bout / 1000))
    framesToPlugList = list(range(1, framesToPlug + 1))
    framesToPlugList.reverse()
    patternListofLists = []
    for k in framesToPlugList:
        zerosInList = [0] * k
        currList = [1]
        currList.extend(zerosInList)
        currList.extend([1])
        patternListofLists.append(currList)
    patternListofLists.append([0, 1, 1, 0])
    patternListofLists.append([0, 1, 0])
    patterns = np.asarray(patternListofLists)
    for l in patterns:
        currPattern = l
        n_obs = len(currPattern)
        outputDf['rolling_match'] = (outputDf[classifier_name].rolling(window=n_obs, min_periods=n_obs)
                                     .apply(lambda x: (x == currPattern).all())
                                     .mask(lambda x: x == 0)
                                     .bfill(limit=n_obs - 1)
                                     .fillna(0)
                                     .astype(bool)
                                     )
        if (currPattern == patterns[-2]) or (currPattern == patterns[-1]):
            outputDf.loc[outputDf['rolling_match'] == True, classifier_name] = 0
        else:
            outputDf.loc[outputDf['rolling_match'] == True, classifier_name] = 1
        outputDf = outputDf.drop(['rolling_match'], axis=1)

    outFname = sample_feature_file_Name + '.csv'
    csv_dir_out_validation = config.get('General settings', 'csv_path')
    csv_dir_out_validation = os.path.join(csv_dir_out_validation,'validation')
    if not os.path.exists(csv_dir_out_validation):
        os.makedirs(csv_dir_out_validation)
    outFname = os.path.join(csv_dir_out_validation, outFname)
    outputDf.to_csv(outFname)
    print('Predictions generated...')


    #generate the frames based on the just generated file
    target_counter = 0
    frames_dir_in = os.path.join(framefolder, sample_feature_file_Name)
    frames_dir_out_validation = config.get('Frame settings', 'frames_dir_out')
    frames_dir_out_validation_sklearn = os.path.join(frames_dir_out_validation, 'validation', sample_feature_file_Name, 'sklearn')
    if not os.path.exists(frames_dir_out_validation_sklearn):
        os.makedirs(frames_dir_out_validation_sklearn)
    currentDf = pd.read_csv(outFname)
    currentDf = currentDf.fillna(0)
    currentDf = currentDf.astype(int)
    targetColumn = classifier_name
    frames_dir_out_merged = os.path.join(frames_dir_out_validation, 'validation', sample_feature_file_Name)
    if not os.path.exists(frames_dir_out_merged):
        os.makedirs(frames_dir_out_merged)

    for index, row in currentDf.iterrows():
        imageName = str(loop) + '.png'
        image = os.path.join(frames_dir_in, imageName)
        im = cv2.imread(image)
        try:
            (height, width) = im.shape[:2]
        except AttributeError:
            print('ERROR: SimBA cannot find the appropriate frames. Please check the project_folder/frames/input folder.')
        fscale = 0.05
        cscale = 0.2
        space_scale = 1.1
        fontScale = min(width, height) / (25 / fscale)
        circleScale = int(min(width, height) / (25 / cscale))
        spacingScale = int(min(width, height) / (25 / space_scale))
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
        cv2.circle(im, (row['Ear_left_1_x'], row['Ear_left_1_y']), circleScale, (255, 0, 0), thickness=-1, lineType=8, shift=0)
        cv2.circle(im, (row['Ear_right_1_x'], row['Ear_right_1_y']), circleScale, (255, 191, 0), thickness=-1, lineType=8, shift=0)
        cv2.circle(im, (row['Nose_1_x'], row['Nose_1_y']), circleScale, (255, 255, 0), thickness=-1, lineType=8, shift=0)
        cv2.circle(im, (row['Center_1_x'], row['Center_1_y']), circleScale, (255, 165, 0), thickness=-1, lineType=8, shift=0)
        cv2.circle(im, (row['Lat_left_1_x'], row['Lat_left_1_y']), circleScale, (0, 255, 0), thickness=-1, lineType=8, shift=0)
        cv2.circle(im, (row['Lat_right_1_x'], row['Lat_right_1_y']), circleScale, (255, 0, 255), thickness=-1, lineType=8,shift=0)
        cv2.circle(im, (row['Tail_base_1_x'], row['Tail_base_1_y']), circleScale, (0, 128, 0), thickness=-1, lineType=8, shift=0)
        cv2.circle(im, (row['Ear_left_2_x'], row['Ear_left_2_y']), circleScale, (255, 20, 147), thickness=-1, lineType=8, shift=0)
        cv2.circle(im, (row['Ear_right_2_x'], row['Ear_right_2_y']), circleScale, (139, 0, 139), thickness=-1, lineType=8, shift=0)
        cv2.circle(im, (row['Nose_2_x'], row['Nose_2_y']), circleScale, (127, 255, 212), thickness=-1, lineType=8, shift=0)
        cv2.circle(im, (row['Center_2_x'], row['Center_2_y']), circleScale, (210, 105, 30), thickness=-1, lineType=8, shift=0)
        cv2.circle(im, (row['Lat_left_2_x'], row['Lat_left_2_y']), circleScale, (255, 127, 80), thickness=-1, lineType=8, shift=0)
        cv2.circle(im, (row['Lat_right_2_x'], row['Lat_right_2_y']), circleScale, (64, 224, 208), thickness=-1, lineType=8,shift=0)
        cv2.circle(im, (row['Tail_base_2_x'], row['Tail_base_2_y']), circleScale, (255, 105, 180), thickness=-1, lineType=8, shift=0)
        angle1 = (row['Mouse_1_angle'])
        angle2 = (row['Mouse_2_angle'])
        cv2.line(im, (M1polyglon_array[6][0], M1polyglon_array[6][1]), (M1polyglon_array[5][0], M1polyglon_array[5][1]), (0, 0, 255), 2)
        cv2.line(im, (M2polyglon_array[6][0], M2polyglon_array[6][1]), (M2polyglon_array[5][0], M2polyglon_array[5][1]), (0, 0, 255), 2)
        cv2.line(im, (M1polyglon_array[2][0], M1polyglon_array[2][1]), (M1polyglon_array[6][0], M1polyglon_array[6][1]), (255, 0, 0), 2)
        cv2.line(im, (M2polyglon_array[2][0], M2polyglon_array[2][1]), (M2polyglon_array[6][0], M2polyglon_array[6][1]), (255, 0, 0), 2)
        cv2.drawContours(im, [M1polyglon_array_hull.astype(int)], 0, (255, 255, 255), 2)
        cv2.drawContours(im, [M2polyglon_array_hull.astype(int)], 0, (255, 165, 0), 2)
        cv2.putText(im, str(angle1), (M1polyglon_array[6][0], M1polyglon_array[6][1]), cv2.FONT_HERSHEY_TRIPLEX, fontScale, (255, 255, 255), 2)
        cv2.putText(im, str(angle2), (M2polyglon_array[6][0], M2polyglon_array[6][1]), cv2.FONT_HERSHEY_TRIPLEX, fontScale, (255, 255, 255), 2)
        target_timer = (1/fps) * target_counter
        target_timer = round(target_timer, 2)
        if height < width:
            im = ndimage.rotate(im, 90)

        cv2.putText(im, str('Timer'), (10, ((height-height)+spacingScale)), cv2.FONT_HERSHEY_COMPLEX, fontScale, (0, 255, 0), 2)
        addSpacer = 2
        cv2.putText(im, (str(classifier_name) + ' ' + str(target_timer) + str('s')), (10, (height-height)+spacingScale*addSpacer), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 255), 2)
        addSpacer+=1
        cv2.putText(im, str('ensemble prediction'), (10, (height-height)+spacingScale*addSpacer), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 255, 0), 2)
        addSpacer += 2
        if row[targetColumn] == 1:
            cv2.putText(im, str(classifier_name), (10, (height - height) + spacingScale * addSpacer), cv2.FONT_HERSHEY_TRIPLEX, fontScale, (2, 166, 249), 2)
            target_counter += 1
            addSpacer += 1

        imFrameHeight, imFrameWidth = im.shape[0:2]


        boutEnd = 0
        boutEnd_list = [0]
        boutStart_list = []
        boutsDf = pd.DataFrame(columns=['Event', 'Start_frame', 'End_frame'])
        frames_dir_out_validation_gantt = os.path.join(frames_dir_out_validation, 'validation', sample_feature_file_Name, 'gantt')
        if not os.path.exists(frames_dir_out_validation_gantt):
            os.makedirs(frames_dir_out_validation_gantt)
        rowCount = currentDf.shape[0]

        for indexes, rows in currentDf[currentDf['Unnamed: 0'] >= boutEnd].iterrows():
            if rows[classifier_name] == 1:
                boutStart = rows['Unnamed: 0']
                for index, row in currentDf[currentDf['Unnamed: 0'] >= boutStart].iterrows():
                    if row[classifier_name] == 0:
                        boutEnd = row['Unnamed: 0']
                        if boutEnd_list[-1] != boutEnd:
                            boutStart_list.append(boutStart)
                            boutEnd_list.append(boutEnd)
                            values = [classifier_name, boutStart, boutEnd]
                            boutsDf.loc[(len(boutsDf))] = values
                            break
                        break
                boutStart_list = [0]
                boutEnd_list = [0]
        boutsDf['Start_time'] = boutsDf['Start_frame'] / fps
        boutsDf['End_time'] = boutsDf['End_frame'] / fps
        boutsDf['Bout_time'] = boutsDf['End_time'] - boutsDf['Start_time']
        loop = 0
        for k in range(rowCount):
            fig, ax = plt.subplots()
            currentDf = currentDf.iloc[:k]
            relRows = boutsDf.loc[boutsDf['End_frame'] <= k]
            for i, event in enumerate(relRows.groupby("Event")):
                data_event = event[1][["Start_time", "Bout_time"]]
                ax.broken_barh(data_event.values, (4, 4), facecolors='red')
                loop+=1
            xLength = (round(k / fps)) + 1
            if xLength < 10:
                xLength = 10
            loop=0
            ax.set_xlim(0, xLength)
            ax.set_ylim([0, 12])
            plt.ylabel(classifier_name, fontsize=12)
            plt.yticks([])
            plt.xlabel('time(s)', fontsize=12)
            ax.yaxis.set_ticklabels([])
            ax.grid(True)
            filename = (str(k) + '.png')
            savePath = os.path.join(frames_dir_out_validation_gantt, filename)
            plt.savefig(savePath, dpi=200)
            print('Gantt plot: ' + str(k) + '/' + str(rowCount))
            plt.close('all')
            img = cv2.imread(savePath)
            img = imutils.resize(img, height=imFrameHeight)

        horizontalConcat = np.concatenate((im, img), axis=1)
        horizontalConcat = imutils.resize(horizontalConcat, height=600)
        outPath = os.path.join(frames_dir_out_merged, str(loop) + '.png')
        cv2.imwrite(outPath, horizontalConcat)
        print('Merged frame: ' + str(loop) + '/' + str(rowCount))
        loop += 1

    #merge frames to movie
    imageForSize = cv2.imread(outPath)
    (height, width) = imageForSize.shape[:2]
    movie_dir_path_out = os.path.join(frames_dir_out_validation, "validation", sample_feature_file_Name, sample_feature_file_Name + '.mp4')
    ffmpegFileName = os.path.join(frames_dir_out_merged, '%d.png')
    print('Generating video...')
    command = str('ffmpeg -r ' + str(fps) + str(' -f image2 -s ') + str(height) + 'x' + str(width) + ' -i ' + str(ffmpegFileName) + ' -vcodec libx264 -b ' + str(2400) + 'k ' + str(movie_dir_path_out))
    subprocess.call(command, shell=True)
    print('Validation video saved @' + 'project_folder/frames/output/validation')












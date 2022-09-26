__author__ = "Simon Nilsson", "JJ Choong"

import glob
import pandas as pd
from pylab import *
from configparser import ConfigParser, MissingSectionHeaderError, NoSectionError, NoOptionError
import os
import cv2
import numpy as np
from simba.drop_bp_cords import getBpHeaders
import math
from simba.rw_dfs import *
from simba.features_scripts.unit_tests import read_video_info_csv, read_video_info
from simba.misc_tools import get_file_path_parts

def plotHeatMap(inifile, animalbp1, mmSize, noIncrements, colorPalette, targetBehav, lastImageOnlyBol):
    config = ConfigParser()
    configFile = str(inifile)
    try:
        config.read(configFile)
    except MissingSectionHeaderError:
        print('ERROR:  Not a valid project_config file. Please check the project_config.ini path.')
    projectPath = config.get('General settings', 'project_path')
    csv_dir_in = os.path.join(projectPath, 'csv', 'machine_results')
    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'
    videoLog = read_video_info_csv(os.path.join(projectPath, 'logs', 'video_info.csv'))
    trackedBodyParts = [str(animalbp1)+'_x', str(animalbp1)+ '_y']
    frames_dir_out = os.path.join(projectPath, 'frames', 'output', 'heatmap_behavior')
    if not os.path.exists(frames_dir_out):
        os.makedirs(frames_dir_out)
    vidInfPath = os.path.join(projectPath, 'logs', 'video_info.csv')
    vidinfDf = pd.read_csv(vidInfPath)
    colorList, loopCounter = [], 0

    filesFound = glob.glob(csv_dir_in + '/*.' + wfileType)
    for currVid in filesFound:
        calcFlag = False
        loopCounter += 1
        file_directory, currVidBaseName, extension = get_file_path_parts(currVid)
        currVidInfo, pxPerMM, fps = read_video_info(vidinfDf=vidinfDf, currVidName=currVidBaseName)
        width, height = int(currVidInfo['Resolution_width']), int(currVidInfo['Resolution_height'])
        binInput = float(mmSize * pxPerMM)
        outputfilename = os.path.join(frames_dir_out, currVidBaseName.replace('.csv', '.mp4'))
        binWidth, binHeight = (binInput, binInput)
        try:
            NbinsX, NbinsY, NbinsXCheck, NbinsYCheck = (int(width / binWidth), int(height / binHeight), width / binWidth, height / binHeight)
        except ZeroDivisionError:
            print('Could not find the resolution information for ' + str(currVidBaseName) + '. Make sure the video is represented in your video_info.csv file')
        targetCountArrayFrames = np.zeros((NbinsY, NbinsX))
        im = np.zeros((height, width, 3))
        im.fill(0)
        currDf = read_df(currVid, wfileType)
        count_row = currDf.shape[0]
        im_scale_bar_width = int(width / 5)
        im_scale_bar = np.zeros((height, im_scale_bar_width, 3))
        im_scale_bar_digits = np.zeros((height, im_scale_bar_width, 3))
        im_scale_bar_digits_jumps = int(im_scale_bar_digits.shape[0] / 4)
        im_scale_bar_digits_jumps_List = np.arange(0, im_scale_bar_digits.shape[0], im_scale_bar_digits_jumps).tolist()
        im_scale_bar_digits_jumps_List.append(int(im_scale_bar_digits.shape[0] - 10))
        mySpaceScale, myRadius, myResolution, myFontScale = 60, 12, 1500, 1.5
        maxResDimension = max(width, height)
        circleScale, fontScale, spacingScale = int(myRadius / (myResolution / maxResDimension)), float(myFontScale / (myResolution / maxResDimension)), int(mySpaceScale / (myResolution / maxResDimension))
        if noIncrements != 'auto':
            noIncrements = int(noIncrements)
            cmap = cm.get_cmap(str(colorPalette), noIncrements + 1)
            for i in range(cmap.N):
                rgb = list((cmap(i)[:3]))
                rgb = [i * 255 for i in rgb]
                rgb.reverse()
                colorList.append(rgb)

        if noIncrements == 'auto':
            calcFlag = True
            print('auto')
            for index, row in currDf.iterrows():
                if row[targetBehav] == 1:
                    cordX, cordY = (row[trackedBodyParts[0]], row[trackedBodyParts[1]])
                    binX, binY = (int((cordX * NbinsX) / width), int((cordY * NbinsY) / height))
                    targetCountArrayFrames[binY, binX] = targetCountArrayFrames[binY, binX] + 1
                    targetCountArray = ((targetCountArrayFrames / fps) - 0.1) / 1
                    targetCountArray[targetCountArray < 0] = 0
            noIncrements = math.ceil(np.max(targetCountArray))
        cmap = cm.get_cmap(str(colorPalette), noIncrements + 1)

        for i in range(cmap.N):
            rgb = list((cmap(i)[:3]))
            rgb = [i * 255 for i in rgb]
            rgb.reverse()
            colorList.append(rgb)
        increments_digits_jump = noIncrements / 4
        increments_digits_jump_list = np.arange(0, noIncrements, increments_digits_jump).tolist()
        increments_digits_jump_list.append(noIncrements)

        if (calcFlag == False) and (lastImageOnlyBol == 1):
            for index, row in currDf.iterrows():
                if row[targetBehav] == 1:
                    cordX, cordY = (row[trackedBodyParts[0]], row[trackedBodyParts[1]])
                    binX, binY = (int((cordX * NbinsX) / width), int((cordY * NbinsY) / height))
                    targetCountArrayFrames[binY, binX] = targetCountArrayFrames[binY, binX] + 1
                    targetCountArray = ((targetCountArrayFrames / fps) - 0.1) / 1
                    targetCountArray[targetCountArray < 0] = 0

        for i in range(cmap.N):
            topX = 0
            try:
                topY = int((height / (noIncrements)) * i)
            except ZeroDivisionError:
                topY = 0
            bottomX, bottomY = int(im_scale_bar_width), int(height / noIncrements) * cmap.N
            cv2.rectangle(im_scale_bar, (topX, topY - 4), (bottomX, bottomY), colorList[i], -1)
            if NbinsYCheck.is_integer():
                pass
            else:
                frac = int((NbinsYCheck - int(NbinsYCheck)) * 100)
                newHeight = int(height - (frac*0.5))
                im_scale_bar = im_scale_bar[0:newHeight, 0:width]
                im_scale_bar_digits = im_scale_bar_digits[0:newHeight, 0:width]
            if NbinsXCheck.is_integer():
                pass
            else:
                frac = int((NbinsXCheck - int(NbinsXCheck)) * 100)
                newWidth = int(width - (frac*0.5))
                im_scale_bar = im_scale_bar[0:height, 0:newWidth]
                im_scale_bar_digits = im_scale_bar_digits[0:height, 0:newWidth]

        for i in range(len(increments_digits_jump_list)):
            currY = im_scale_bar_digits_jumps_List[i]
            # if i == 0:
            #     currY = currY + 15
            # if i == len(increments_digits_jump_list)-1:
            #     currY = currY - 15
            if (i != 0) and (i != len(increments_digits_jump_list)-1):
                currText = increments_digits_jump_list[i]
                cv2.putText(im_scale_bar_digits, str(currText), (10, currY), cv2.FONT_HERSHEY_COMPLEX, fontScale, (255, 255, 255), 1)
        im_scale_bar_text = np.concatenate((im_scale_bar, im_scale_bar_digits), axis=1)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        outputWidth = int(width + im_scale_bar_text.shape[0])
        writer = cv2.VideoWriter(outputfilename, fourcc, fps, (outputWidth, height))

        print('Calculating heatmap from file: ' + str(currVidBaseName) + '...')

        if lastImageOnlyBol == 1:
            print('last_only')
            for i in range(NbinsY):
                for j in range(NbinsX):
                    if targetCountArray[i, j] >= noIncrements:
                        targetCountArray[i, j] = noIncrements - 1
                    colorValue = colorList[int(targetCountArray[i, j])]
                    topX = int(binWidth * j)
                    topY = int(binHeight * i)
                    bottomX = int(binWidth * (j + 1))
                    bottomY = int(binHeight * (i + 1))
                    currColor = tuple(colorValue)
                    cv2.rectangle(im, (topX, topY), (bottomX, bottomY), currColor, -1)
            if NbinsYCheck.is_integer():
                pass
            else:
                frac = int((NbinsYCheck - int(NbinsYCheck)) * 100)
                newHeight = int(height - (frac * 0.5))
                im = im[0:newHeight, 0:width]
            if NbinsXCheck.is_integer():
                pass
            else:
                frac = int((NbinsXCheck - int(NbinsXCheck)) * 100)
                newWidth = int(width - (frac * 0.5))
                im = im[0:height, 0:newWidth]

            im = cv2.blur(im, (int(binWidth * 1.5), int(binHeight * 1.5)))
            imageConcat = np.concatenate((im, im_scale_bar_text), axis=1)
            imageConcat = cv2.resize(imageConcat, (outputWidth, height))
            imageConcat = np.uint8(imageConcat)
            imageSaveName = os.path.join(frames_dir_out, currVidBaseName.replace('.csv', '.png'))
            cv2.imwrite(imageSaveName, imageConcat)
            print('Heatmap saved @: project_folder/frames/output/' + str(currVidBaseName.replace('.csv', '.png')))

        if lastImageOnlyBol == 0:
            print('not_last_image')
            targetCountArrayFrames = np.zeros((NbinsY, NbinsX))
            targetCountArray = targetCountArrayFrames.copy()
            for index, row in currDf.iterrows():
                if row[targetBehav] == 1:
                    cordX, cordY = (row[trackedBodyParts[0]], row[trackedBodyParts[1]])
                    binX, binY = (int((cordX * NbinsX) / width), int((cordY * NbinsY) / height))
                    targetCountArrayFrames[binY, binX] = targetCountArrayFrames[binY, binX] + 1
                    targetCountArray = ((targetCountArrayFrames / fps) - 0.1) / 1
                    targetCountArray[targetCountArray < 0] = 0
                for i in range(NbinsY):
                    for j in range(NbinsX):
                        if targetCountArray[i, j] >= noIncrements:
                            targetCountArray[i, j] = noIncrements-1
                        colorValue = colorList[int(targetCountArray[i, j])]
                        topX = int(binWidth * j)
                        topY = int(binHeight * i)
                        bottomX = int(binWidth * (j + 1))
                        bottomY = int(binHeight * (i + 1))
                        currColor = tuple(colorValue)
                        cv2.rectangle(im, (topX, topY), (bottomX, bottomY), currColor, -1)
                if NbinsYCheck.is_integer():
                    pass
                else:
                    frac = int((NbinsYCheck - int(NbinsYCheck)) * 100)
                    newHeight = int(height - (frac*0.5))
                    im = im[0:newHeight, 0:width]
                if NbinsXCheck.is_integer():
                    pass
                else:
                    frac = int((NbinsXCheck - int(NbinsXCheck)) * 100)
                    newWidth = int(width - (frac*0.5))
                    im = im[0:height, 0:newWidth]
                    im = cv2.blur(im, (int(binWidth*1.5), int(binHeight*1.5)))
                    imageConcat = np.concatenate((im, im_scale_bar_text), axis=1)
                    imageConcat = cv2.resize(imageConcat, (outputWidth, height))
                    imageConcat = np.uint8(imageConcat)
                    writer.write(imageConcat)
                    print('Image ' + str(index) + '/' + str(count_row) + '. Video ' + str(loopCounter) + '/' + str(len(filesFound)))
        print('All heatmaps generated in folder ' + 'project_folder/frames/output/heatmap_behavior')
        cv2.destroyAllWindows()
        writer.release()
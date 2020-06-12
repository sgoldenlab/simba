import glob
import pandas as pd
from pylab import *
from configparser import ConfigParser, MissingSectionHeaderError
import os
import cv2
import numpy as np
from drop_bp_cords import getBpHeaders

def plotHeatMapLocation(inifile, animalbp1, mmSize, noIncrements, secIncrements, colorPalette, lastImageOnlyBol):
    config = ConfigParser()
    configFile = str(inifile)
    try:
        config.read(configFile)
    except MissingSectionHeaderError:
        print('ERROR:  Not a valid project_config file. Please check the project_config.ini path.')
    projectPath = config.get('General settings', 'project_path')
    csv_dir_in = os.path.join(projectPath, 'csv', 'outlier_corrected_movement_location')
    vidLogFilePath = os.path.join(projectPath, 'logs', 'video_info.csv')
    videoLog = pd.read_csv(vidLogFilePath)
    trackedBodyParts = [str(animalbp1)+'_x', str(animalbp1)+ '_y']
    frames_dir_out = os.path.join(projectPath, 'frames', 'output', 'heatmaps_location')
    if not os.path.exists(frames_dir_out):
        os.makedirs(frames_dir_out)
    vidInfPath = os.path.join(projectPath, 'logs', 'video_info.csv')
    vidinfDf = pd.read_csv(vidInfPath)
    colorList, loopCounter = [], 0
    cmap = cm.get_cmap(str(colorPalette), noIncrements+1)
    for i in range(cmap.N):
        rgb = list((cmap(i)[:3]))
        rgb = [i * 255 for i in rgb]
        rgb.reverse()
        colorList.append(rgb)

    filesFound = glob.glob(csv_dir_in + '/*.csv')
    for currVid in filesFound:
        loopCounter += 1
        currVidBaseName = os.path.basename(currVid)
        currVidInfo = videoLog.loc[videoLog['Video'] == str(currVidBaseName.replace('.csv', ''))]
        fps = int(currVidInfo['fps'])
        width = int(currVidInfo['Resolution_width'])
        height = int(currVidInfo['Resolution_height'])
        pxPerMM = int(currVidInfo['pixels/mm'])
        binInput = int(mmSize * pxPerMM)
        outputfilename = os.path.join(frames_dir_out, currVidBaseName.replace('.csv', '.mp4'))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(outputfilename, fourcc, fps, (int(width + (width / 5)), height))
        outputWidth = int(width + (width / 5))
        binWidth, binHeight = (binInput, binInput)
        NbinsX, NbinsY, NbinsXCheck, NbinsYCheck = (int(width / binWidth), int(height / binHeight), width / binWidth, height / binHeight)
        targetCountArrayFrames = np.zeros((NbinsY, NbinsX))
        im = np.zeros((height, width, 3))
        im.fill(0)
        currDf = pd.read_csv(currVid)
        newColHeads = getBpHeaders(inifile)
        newColHeads.insert(0, "scorer")
        currDf.columns = newColHeads
        count_row = currDf.shape[0]
        vidInfo = vidinfDf.loc[vidinfDf['Video'] == str(currVidBaseName.replace('.csv', ''))]
        fps = int(vidInfo['fps'])
        im_scale_bar_width = int(width / 5)
        im_scale_bar = np.zeros((height, im_scale_bar_width, 3))
        for i in range(cmap.N):
            textAdd = ''
            topX = 0
            try:
                topY = int((height / (noIncrements)) * i)
            except ZeroDivisionError:
                topY = 0
            bottomX = int(im_scale_bar_width)
            bottomY = int(height / noIncrements) * (i + 1)
            cv2.rectangle(im_scale_bar, (topX, topY - 4), (bottomX, bottomY), colorList[i], -1)
            textValue = round(0 + (secIncrements * i), 2)
            if i == (noIncrements - 1):
                textAdd = '>'
            text = textAdd + ' ' + str(textValue) + 's'
            cv2.putText(im_scale_bar, str(text), (int((topX + bottomX) / 10), int((topY + bottomY) / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if NbinsYCheck.is_integer():
                pass
            else:
                frac = int((NbinsYCheck - int(NbinsYCheck)) * 100)
                newHeight = int(height - (frac*0.5))
                im_scale_bar = im_scale_bar[0:newHeight, 0:width]
            if NbinsXCheck.is_integer():
                pass
            else:
                frac = int((NbinsXCheck - int(NbinsXCheck)) * 100)
                newWidth = int(width - (frac*0.5))
                im_scale_bar = im_scale_bar[0:height, 0:newWidth]

        print('Calculating heatmap from file: ' + str(currVidBaseName) + '...')
        for index, row in currDf.iterrows():
            cordX, cordY = (row[trackedBodyParts[0]], row[trackedBodyParts[1]])
            binX, binY = (int((cordX * NbinsX) / width), int((cordY * NbinsY) / height))
            targetCountArrayFrames[binY, binX] = targetCountArrayFrames[binY, binX] + 1
            targetCountArray = ((targetCountArrayFrames / fps) - 0.1) / secIncrements
            targetCountArray[targetCountArray < 0] = 0
            if lastImageOnlyBol == 1:
                print('Analyzing image ' + str(index) + '/' + str(len(currDf)-1))
            if ((lastImageOnlyBol == 0) or ((lastImageOnlyBol == 1) and (index == (len(currDf)-1)))):
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
                imageConcat = np.concatenate((im, im_scale_bar), axis=1)
                imageConcat = cv2.resize(imageConcat, (outputWidth, height))
                imageConcat = np.uint8(imageConcat)
                if ((lastImageOnlyBol == 1) and (index == (len(currDf)-1))):
                    imageSaveName = os.path.join(frames_dir_out, currVidBaseName.replace('.csv', '.png'))
                    cv2.imwrite(imageSaveName, imageConcat)
                    print('Heatmap saved @: project_folder/frames/output/' + str(currVidBaseName.replace('.csv', '.png')))
                if (lastImageOnlyBol == 0):
                    writer.write(imageConcat)
                    print('Image ' + str(index) + '/' + str(count_row) + '. Video ' + str(loopCounter) + '/' + str(len(filesFound)))
        print('All heatmaps generated.')
        cv2.destroyAllWindows()
        writer.release()
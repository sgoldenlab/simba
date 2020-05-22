import glob
import pandas as pd
from pylab import *
from configparser import ConfigParser, MissingSectionHeaderError
import os
import cv2

def plotHeatMap(inifile):
    config = ConfigParser()
    configFile = str(inifile)
    try:
        config.read(configFile)
    except MissingSectionHeaderError:
        print('ERROR:  Not a valid project_config file. Please check the project_config.ini path.')
    projectPath = config.get('General settings', 'project_path')
    csv_dir_in = os.path.join(projectPath, 'csv', 'machine_results')
    animalbp1 = config.get('Heatmap settings','body_part')
    trackedBodyParts = [str(animalbp1)+'_x', str(animalbp1)+ '_y']
    frames_dir_out = os.path.join(projectPath, 'frames', 'output', 'heatmaps')
    if not os.path.exists(frames_dir_out):
        os.makedirs(frames_dir_out)
    binInput = config.getint('Heatmap settings', 'bin_size_pixels')
    vidInfPath = os.path.join(projectPath, 'logs', 'video_info.csv')
    vidinfDf = pd.read_csv(vidInfPath)
    maxColorScale = config.getint('Heatmap settings', 'Scale_max_seconds')
    increments_in_seconds = config.getfloat('Heatmap settings', 'Scale_increments_seconds')
    colorPalette = config.get('Heatmap settings', 'palette')
    targetcolumnName = config.get('Heatmap settings', 'target')
    colorList, loopCounter = [], 0
    cmap = cm.get_cmap(str(colorPalette), maxColorScale+1)
    for i in range(cmap.N):
        rgb = list((cmap(i)[:3]))
        rgb = [i * 255 for i in rgb]
        rgb.reverse()
        colorList.append(rgb)

    filesFound = glob.glob(csv_dir_in + '/*.csv')
    print(filesFound)
    for currVid in filesFound:
        loopCounter += 1
        currVidBaseName = os.path.basename(currVid)
        if os.path.exists(os.path.join(projectPath,'videos', currVidBaseName.replace('.csv', '.mp4'))):
            currVideo = os.path.join(projectPath,'videos', currVidBaseName.replace('.csv', '.mp4'))
        elif os.path.exists(os.path.join(projectPath,'videos', currVidBaseName.replace('.csv', '.avi'))):
            currVideo = os.path.join(projectPath,'videos', currVidBaseName.replace('.csv', '.avi'))
        else:
            print('Cannot locate video ' + str(currVidBaseName.replace('.csv', '')) + 'in mp4 or avi format')
            break
        cap = cv2.VideoCapture(currVideo)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        outputfilename = os.path.join(frames_dir_out, currVidBaseName.replace('.csv', '.mp4'))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(outputfilename, fourcc, fps, (int(width + (width /5)), height))
        outputWidth = int(width + (width /5))
        binWidth, binHeight = (binInput, binInput)
        NbinsX, NbinsY, NbinsXCheck, NbinsYCheck = (int(width / binWidth), int(height / binHeight), width / binWidth, height / binHeight)
        targetCountArrayFrames = np.zeros((NbinsY, NbinsX))
        im = np.zeros((height, width, 3))
        im.fill(0)
        currDf = pd.read_csv(currVid)
        count_row = currDf.shape[0]
        vidInfo = vidinfDf.loc[vidinfDf['Video'] == str(currVidBaseName.replace('.csv', ''))]
        fps = int(vidInfo['fps'])
        im_scale_bar_width = int(width / 5)
        im_scale_bar = np.zeros((height, im_scale_bar_width, 3))
        for i in range(cmap.N):
            textAdd = ''
            topX = 0
            try:
                topY = int((height / (maxColorScale)) * i)
            except ZeroDivisionError:
                topY = 0
            bottomX = int(im_scale_bar_width)
            bottomY = int(height / maxColorScale) * (i + 1)
            cv2.rectangle(im_scale_bar, (topX, topY - 4), (bottomX, bottomY), colorList[i], -1)
            textValue = round(0 + (increments_in_seconds * i), 2)
            if i == (maxColorScale - 1):
                textAdd = '>'
            text = textAdd + ' ' + str(textValue) + 's'
            cv2.putText(im_scale_bar, str(text), (int((topX + bottomX) / 10), int((topY + bottomY) / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if NbinsYCheck.is_integer():
                pass
            else:
                frac = int((NbinsYCheck - int(NbinsYCheck)) * 100)
                newHeight = height - frac
                im_scale_bar = im_scale_bar[0:newHeight, 0:width]
            if NbinsXCheck.is_integer():
                pass
            else:
                frac = int((NbinsXCheck - int(NbinsXCheck)) * 100)
                newWidth = width - frac
                im_scale_bar = im_scale_bar[0:height, 0:newWidth]
        for index, row in currDf.iterrows():
            if row[targetcolumnName] == 1:
                cordX, cordY = (row[trackedBodyParts[0]], row[trackedBodyParts[1]])
                binX, binY = (int((cordX * NbinsX) / width), int((cordY * NbinsY) / height))
                targetCountArrayFrames[binY, binX] = targetCountArrayFrames[binY, binX] + 1
            targetCountArray = targetCountArrayFrames / (fps * increments_in_seconds)
            # CREATE COLOR ARRAY
            for i in range(NbinsY):
                for j in range(NbinsX):
                    if targetCountArray[i, j] >= maxColorScale:
                        targetCountArray[i, j] = maxColorScale-1
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
                newHeight = height - frac
                im = im[0:newHeight, 0:width]
            if NbinsXCheck.is_integer():
                pass
            else:
                frac = int((NbinsXCheck - int(NbinsXCheck)) * 100)
                newWidth = width - frac
                im = im[0:height, 0:newWidth]

            im = cv2.blur(im, (int(binWidth*1.5), int(binHeight*1.5)))
            imageConcat = np.concatenate((im, im_scale_bar), axis=1)
            imageConcat = cv2.resize(imageConcat, (outputWidth, height))
            imageConcat = np.uint8(imageConcat)
            writer.write(imageConcat)
            print('Image ' + str(index) + '/' + str(count_row) + '. Video ' + str(loopCounter) + '/' + str(len(filesFound)))
        print('All heatmaps generated')
        cv2.destroyAllWindows()
        writer.release()
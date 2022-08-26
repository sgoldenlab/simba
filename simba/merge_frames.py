import os
import cv2
import numpy as np
from configparser import ConfigParser

def merge_frames_config(configini):
    configFile = str(configini)
    config = ConfigParser()
    config.read(configFile)
    loop=1
    frames_dir_in = config.get('Frame settings', 'frames_dir_out')
    frames_dir_out = os.path.join(frames_dir_in, "merged")
    if not os.path.exists(frames_dir_out):
        os.makedirs(frames_dir_out)
    sklearnDir = os.path.join(frames_dir_in, "sklearn_results")
    gantt_plot_dirs = os.path.join(frames_dir_in, "gantt_plots")
    path_plot_dirs = os.path.join(frames_dir_in, "path_plots")
    data_plot_dirs = os.path.join(frames_dir_in, "live_data_table")
    line_plot_dirs = os.path.join(frames_dir_in, "line_plot")

    sklearnDir = [f.path for f in os.scandir(sklearnDir) if f.is_dir()]
    gantt_plot_dirs = [f.path for f in os.scandir(gantt_plot_dirs) if f.is_dir()]
    path_plot_dirs = [f.path for f in os.scandir(path_plot_dirs) if f.is_dir()]
    data_plot_dirs = [f.path for f in os.scandir(data_plot_dirs) if f.is_dir()]
    line_plot_dirs = [f.path for f in os.scandir(line_plot_dirs) if f.is_dir()]

    frameDirs = [s for s in sklearnDir if "_frames" in s]
    ganttDirs = [s for s in gantt_plot_dirs if "gantt" in s]
    dataDirs = [s for s in data_plot_dirs if "_data_plots" in s]
    pathDirs = [s for s in path_plot_dirs if "_path_dot" in s]
    lineDirs = [s for s in line_plot_dirs if "_distance_plot" in s]
    videoNos = len(frameDirs)
    print('Merging frames for ' + str(videoNos) + ' videos...')

    def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]
        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))
        resized = cv2.resize(image, dim, interpolation=inter)
        return resized

    for i in range(videoNos):
        loopy = 0
        frameList = os.listdir(frameDirs[i])
        frameList = sorted(frameList, key=lambda x: int(x.split('.')[0]))
        ganttList = os.listdir(ganttDirs[i])
        ganttList = sorted(ganttList, key=lambda x: int(x.split('.')[0]))
        dataList = os.listdir(dataDirs[i])
        dataList = sorted(dataList, key=lambda x: int(x.split('.')[0]))
        pathList = os.listdir(pathDirs[i])
        pathList = sorted(pathList, key=lambda x: int(x.split('.')[0]))
        lineList = os.listdir(lineDirs[i])
        lineList = sorted(lineList, key=lambda x: int(x.split('.')[0]))

        for y in frameList:
            currentFrame = os.path.join(frameDirs[i], frameList[loopy])
            currentGantt = os.path.join(ganttDirs[i], ganttList[loopy])
            currentData = os.path.join(dataDirs[i], dataList[loopy])
            currentPath = os.path.join(pathDirs[i], pathList[loopy])
            currentLine = os.path.join(lineDirs[i], lineList[loopy])
            imageFrame = cv2.imread(currentFrame)
            ganttFrame = cv2.imread(currentGantt)
            dataFrame = cv2.imread(currentData)
            pathFrame = cv2.imread(currentPath)
            lineFrame = cv2.imread(currentLine)
            try:
                imageSize = imageFrame.shape
            except AttributeError:
                print('ERROR: SimBA cannot find the appropriate frames. Please check the project_folder/frames/sklearn_results folder.')
            resizedGantt = image_resize(ganttFrame, height=int(imageSize[0]))
            resizedGantSize = resizedGantt.shape
            resizedData = image_resize(dataFrame, width=int(resizedGantSize[1]))
            resizedData = image_resize(resizedData, height=int(imageSize[0]))
            resizedPath = image_resize(pathFrame, width=int(resizedGantSize[1]))
            resizedPath = image_resize(resizedPath, height=int(imageSize[0]))
            resizedLine = image_resize(lineFrame, width=int(resizedGantSize[1]))
            resizedLine = image_resize(resizedLine, height=int(imageSize[0]))
            horizontalConcatTop = np.concatenate((resizedData, resizedPath), axis=1)
            horizontalConcatBottom = np.concatenate((resizedGantt, resizedLine), axis=1)
            horizontalConcatTop = image_resize(horizontalConcatTop, width=int(horizontalConcatBottom.shape[1]))
            verticalConcat = np.concatenate((horizontalConcatTop, horizontalConcatBottom), axis=0)
            verticalConcat = image_resize(verticalConcat, height=int(imageSize[0]))
            final_horizontalConcat = np.concatenate((imageFrame, verticalConcat), axis=1)
            imageSaveName = str(loopy) + str('.bmp')
            savePath = os.path.basename(frameDirs[i]) + str('_merged')
            outPath = os.path.join(frames_dir_out, savePath)
            if not os.path.exists(outPath):
                os.makedirs(outPath)
            saveFramePath = os.path.join(outPath, imageSaveName)
            cv2.imwrite(saveFramePath, final_horizontalConcat)
            print('Merged frame ' + str(loopy) + '/' + str(len(frameList)) + ' for video ' + str(loop) + '/' + str(videoNos))
            loopy += 1
        loop+=1
    print('Merge frames complete. Videos saved @ project_folder/frames/output/merged')
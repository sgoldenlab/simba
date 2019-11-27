import subprocess
import os
import cv2
from os import listdir
from os.path import isfile, join
from configparser import ConfigParser
import pandas as pd

configFile = r"Z:\DeepLabCut\DLC_extract\New_082119\project_folder\project_config.ini"
config = ConfigParser()
config.read(configFile)
frames_dir_in = config.get('Frame settings', 'frames_dir_out')
frames_dir_in = os.path.join(frames_dir_in, 'merged')
allDirs = [f.path for f in os.scandir(frames_dir_in) if f.is_dir() ]
fps = config.getint('Frame settings', 'fps')
fileformat = config.get('Create movie settings', 'file_format')
bitrate = config.getint('Create movie settings', 'bitrate')
vidInfPath = config.get('General settings', 'project_path')
vidInfPath = os.path.join(vidInfPath, 'project_folder', 'logs')
vidInfPath = os.path.join(vidInfPath, 'video_info.csv')
vidinfDf = pd.read_csv(vidInfPath)


for dir in allDirs:
    currentDir = dir
    VideoName = os.path.basename(currentDir)
    VideoName = VideoName.split('_')[0]
    videoInfoDf = vidinfDf.loc[vidinfDf['Video'] == VideoName]
    fps = int(videoInfoDf['fps'])
    width = int(videoInfoDf['Resolution_width'])
    height = int(videoInfoDf['Resolution_height'])
    fileOut = str(currentDir) + str(fileformat)
    currentFileList = [f for f in listdir(currentDir) if isfile(join(currentDir, f))]
    imgPath = os.path.join(currentDir, currentFileList[0])
    img = cv2.imread(imgPath)
    ffmpegFileName = os.path.join(currentDir, '%d.bmp')
    command = str('ffmpeg -r ' + str(fps) + str(' -f image2 -s ') + str(height) + 'x' + str(width) + ' -i ' + str(ffmpegFileName) + ' -vcodec libx264 -b ' + str(bitrate) + 'k ' + str(fileOut))
    print(command)
    subprocess.call(command, shell=True)




import subprocess
import os
from configparser import ConfigParser
import pandas as pd

def generatevideo_config_ffmpeg(configini):
    config = ConfigParser()
    configFile = str(configini)
    config.read(configFile)
    frames_dir_in = config.get('Frame settings', 'frames_dir_out')
    frames_dir_in = os.path.join(frames_dir_in, 'merged')
    allDirs = [f.path for f in os.scandir(frames_dir_in) if f.is_dir()]
    fileformat = config.get('Create movie settings', 'file_format')
    bitrate = config.getint('Create movie settings', 'bitrate')
    vidInfPath = config.get('General settings', 'project_path')
    vidInfPath = os.path.join(vidInfPath, 'logs')
    vidInfPath = os.path.join(vidInfPath, 'video_info.csv')
    vidinfDf = pd.read_csv(vidInfPath)
    print('Creating ' + str(len(allDirs)) + ' video(s)...')

    for dir in allDirs:
        currentDir = dir
        VideoName = os.path.basename(currentDir)
        VideoName = VideoName.split('_')[0]
        videoInfoDf = vidinfDf.loc[vidinfDf['Video'] == VideoName]
        fps = int(videoInfoDf['fps'])
        width = int(videoInfoDf['Resolution_width'])
        height = int(videoInfoDf['Resolution_height'])
        fileOut = str(currentDir) + str(fileformat)
        ffmpegFileName = os.path.join(currentDir, '%d.bmp')
        print('Creating video ' + '"' + str(VideoName) + '"' + ' @' + str(fileOut))
        command = str('ffmpeg -r ' + str(fps) + str(' -f image2 -s ') + str(height) + 'x' + str(width) + ' -i ' + str(ffmpegFileName) + ' -vcodec libx264 -b ' + str(bitrate) + 'k ' + str(fileOut))
        subprocess.call(command, shell=True)
        print('Video ' + str(VideoName) + ' created.')
    print('All video(s) created.')





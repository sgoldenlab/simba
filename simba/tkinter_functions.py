__author__ = "Simon Nilsson", "JJ Choong"

import subprocess
import numpy as np
import os
import cv2
from os import listdir
from os.path import isfile, join
import yaml
from PIL import Image
import glob
import pathlib
import csv
import shutil
from datetime import datetime
import glob
import pandas as pd
from simba.extract_frames_fast import *
from simba.drop_bp_cords import get_fn_ext
from simba.features_scripts.unit_tests import read_video_info_csv

# def colorized(filename):
#
#     def execute(command):
#         print(command)
#         subprocess.call(command, shell=True, stdout = subprocess.PIPE)
#
#     ########### DEFINE COMMAND ###########
#
#     currentFile = filename
#     outFile = currentFile.replace('.mp4', '')
#     outFile = str(outFile) + '_colorized.mp4'
#     command = (str('python bw2color_video3.py --prototxt colorization_deploy_v2.prototxt --model colorization_release_v2.caffemodel --points pts_in_hull.npy --input ' )+ str(currentFile))
#     execute(command)

def shortenvideos1(filename,starttime,endtime):
    if starttime =='' or endtime =='':
        print('Please enter the time')

    elif filename != '' and filename != 'No file selected':

        def execute(command):
            subprocess.call(command, shell=True, stdout = subprocess.PIPE)

        ########### DEFINE COMMAND ###########

        currentFile = filename
        outFile, fileformat = currentFile.split('.')
        outFile = str(outFile) + '_clipped.mp4'
        output = os.path.basename(outFile)

        command = (str('ffmpeg -i ') +'"'+ str(currentFile) +'"'+ ' -ss ' + starttime + ' -to ' + endtime + ' -async 1 '+'"'+ outFile+'"')

        file = pathlib.Path(outFile)
        if file.exists():
            print(output, 'already exist')
        else:
            print('Clipping video....')
            execute(command)
            print(output,' generated!')
        return output


    else:
        print('Please select a video to trim')

def mergemovebatch(dir,framespersec,vidformat,bit,imgformat):
    currDir = os.listdir(dir)
    fps = str(framespersec)
    fileformat = str('.' + vidformat)
    bitrate = str(bit)
    imageformat = str(imgformat)

    for i in currDir:
        directory = os.path.join(dir,i)
        fileOut = str(directory) + str(fileformat)
        currentDirPath = directory
        currentFileList = [f for f in listdir(currentDirPath) if isfile(join(currentDirPath, f))]
        imgPath = os.path.join(currentDirPath, currentFileList[0])
        img = cv2.imread(imgPath)
        print(imgPath)
        ffmpegFileName = os.path.join(currentDirPath, '%d.' + str(imageformat))
        imgShape = img.shape
        height = imgShape[0]
        width = imgShape[1]
        command = str('ffmpeg -r ' + str(fps) + str(' -f image2 -s ') + str(height) + 'x' + str(width) + ' -i ' +'"'+ str(
            ffmpegFileName) +'"'+ ' -vcodec libx264 -b ' + str(bitrate) + 'k ' +'"'+ str(fileOut)+'"')
        print(command)
        subprocess.call(command, shell=True)
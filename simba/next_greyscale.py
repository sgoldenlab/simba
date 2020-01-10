import subprocess
import os
import cv2
from os import listdir
from os.path import isfile, join
import yaml
from PIL import Image
import glob


filesFound = []
most_recent_folder = max(glob.glob(os.path.join(os.getcwd(), '*/')), key=os.path.getmtime)

########### FIND MP4 FILES ###########
for i in os.listdir(most_recent_folder):
    if i.__contains__(".mp4"):
        filesFound.append(i)
print(most_recent_folder)


def execute(command):
    print(command)
    subprocess.call(command, shell=True, stdout=subprocess.PIPE)

for i in filesFound:
    currentFile = i
    outFile = currentFile.replace('.mp4', '')
    outFile = str(outFile) + '_greyscale.mp4'
    outFile = os.path.basename(outFile)

    command = (str('ffmpeg -i ') + str(most_recent_folder) + str(currentFile) + ' -vf format=gray '+ str(os.getcwd())+ '\\greyscale_videos\\' + outFile)

    execute(command)


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
    outFile = str(outFile) + '_frame_no.mp4'
    outFile = os.path.basename(outFile)

    command = (str('ffmpeg -i ') +str(most_recent_folder) + str(currentFile) + ' -vf "drawtext=fontfile=Arial.ttf: text=\'%{frame_num}\': start_number=1: x=(w-tw)/2: y=h-(2*lh): fontcolor=black: fontsize=20: box=1: boxcolor=white: boxborderw=5" -c:a copy '+ str(os.getcwd()) + '\\withFrames_videos\\' +outFile)
    execute(command)
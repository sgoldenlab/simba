__author__ = "Simon Nilsson", "JJ Choong"

import cv2
import sys
import os
import signal
import pandas as pd
from simba.features_scripts.unit_tests import read_video_info_csv


currentVideoPath = sys.stdin.readline()
currentVideoPath = currentVideoPath.encode().decode()
mainprojectdir = os.path.dirname(os.path.dirname(currentVideoPath))
capture = cv2.VideoCapture(currentVideoPath)
frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(capture.get(3))
height = int(capture.get(4))
space_scale, radius_scale, res_scale, font_scale = 60, 12, 1500, 1.5
max_dim = max(width, height)
font_size = float(font_scale / (res_scale / max_dim))
f = open(os.path.join(mainprojectdir,'labelling_info.txt'), 'w')
video_info_path = os.path.join(mainprojectdir, 'logs','video_info.csv')
vidinfDf = read_video_info_csv(video_info_path)
videoName = os.path.basename(currentVideoPath)
videoName = videoName.split('.', 2)[0]
videoSettings = vidinfDf.loc[vidinfDf['Video'] == str(videoName)]
fps = int(videoSettings['fps'])

timeBetFrames = int(1000/fps)

def printOnFrame(currentFrame):
    currentTime = currentFrame / fps
    currentTime = round(currentTime, 2)
    cv2.putText(frame, 'F~ ' + str(currentFrame), (10, (height - 20)), cv2.FONT_HERSHEY_SIMPLEX, int(font_size), (0, 255, 255), 2)
    cv2.putText(frame, 'T~ ' + str(currentTime), (10, (height - 80)), cv2.FONT_HERSHEY_SIMPLEX, int(font_size), (0, 255, 255), 2)

while True:
    ret, frame = capture.read()
    key = cv2.waitKey(timeBetFrames) & 0xff
    if key == ord('p'):
        while True:
            key2 = cv2.waitKey(1) or 0xff
            ### THE VIDEO IS PAUSED
            currDisplayFrameNo = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
            f.seek(0)
            f.write(str(currDisplayFrameNo-1))
            f.truncate()
            f.flush()
            os.fsync(f.fileno())
            ### BACK UP TWO FRAME
            if key2 == ord('t'):
                capture.set(cv2.CAP_PROP_POS_FRAMES, (currDisplayFrameNo - 2))
                ret, frame = capture.read()
                currentFrame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
                printOnFrame(currentFrame)
                cv2.imshow('Video', frame)
                f.seek(0)
                f.write(str(currentFrame))
                f.truncate()
                f.flush()
                os.fsync(f.fileno())
            ### BACK UP TEN FRAME
            if key2 == ord('s'):
                capture.set(cv2.CAP_PROP_POS_FRAMES, (currDisplayFrameNo - 11))
                ret, frame = capture.read()
                currentFrame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
                printOnFrame(currentFrame)
                cv2.imshow('Video', frame)
                f.seek(0)
                f.write(str(currentFrame))
                f.truncate()
                f.flush()
                os.fsync(f.fileno())
            ### BACK UP 1s
            if key2 == ord('x'):
                capture.set(cv2.CAP_PROP_POS_FRAMES, (currDisplayFrameNo-fps))
                ret, frame = capture.read()
                currentFrame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
                printOnFrame(currentFrame)
                cv2.imshow('Video', frame)
                f.seek(0)
                f.write(str(currentFrame))
                f.truncate()
                f.flush()
                os.fsync(f.fileno())
            ### FORWARD 1s
            if key2 == ord('w'):
                capture.set(cv2.CAP_PROP_POS_FRAMES, (currDisplayFrameNo+fps))
                ret, frame = capture.read()
                currentFrame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
                printOnFrame(currentFrame)
                cv2.imshow('Video', frame)
                f.seek(0)
                f.write(str(currentFrame))
                f.truncate()
                f.flush()
                os.fsync(f.fileno())
            ### FORWARD TWO FRAMES
            if key2 == ord('o'):
                capture.set(cv2.CAP_PROP_POS_FRAMES, (currDisplayFrameNo + 1))
                ret, frame = capture.read()
                currentFrame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
                printOnFrame(currentFrame)
                cv2.imshow('Video', frame)
                f.seek(0)
                f.write(str(currentFrame))
                f.truncate()
                f.flush()
                os.fsync(f.fileno())
            ### FORWARD TEN FRAMES
            if key2 == ord('e'):
                capture.set(cv2.CAP_PROP_POS_FRAMES, (currDisplayFrameNo + 9))
                ret, frame = capture.read()
                currentFrame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
                printOnFrame(currentFrame)
                cv2.imshow('Video', frame)
                f.seek(0)
                f.write(str(currentFrame))
                f.truncate()
                f.flush()
                os.fsync(f.fileno())
            if key2 == ord('p'):
                break
            if key2 == ord('q'):
                capture.release()
                cv2.destroyAllWindows()
                path = os.path.join(mainprojectdir,'subprocess.txt')
                txtFile = open(path)
                line = txtFile.readline()
                os.kill(int(line), signal.SIGTERM)
                break
            if cv2.getWindowProperty('Video', 1) == -1:
                capture.release()
                cv2.destroyAllWindows()
                path = os.path.join(mainprojectdir,'subprocess.txt')
                txtFile = open(path)
                line = txtFile.readline()
                try:
                    os.kill(int(line), signal.SIGTERM)
                except OSError:
                    print('OSError: Cannot save/read latest image file CSV. Please try again')

    currentFrame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
    printOnFrame(currentFrame)
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.imshow('Video', frame)
    if key == ord('q'):
        break
    if cv2.getWindowProperty('Video', 1) == -1:
        break

capture.release()
f.close()
cv2.destroyAllWindows()
path = os.path.join(mainprojectdir,'subprocess.txt')
txtFile = open(path)
line = txtFile.readline()
os.kill(int(line), signal.SIGTERM)












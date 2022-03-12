import cv2
import sys
import os
import signal
import pandas as pd

currentVideoPath = sys.stdin.read()
projpath = os.path.dirname(os.path.dirname(currentVideoPath))
capture = cv2.VideoCapture(currentVideoPath)
frame_count, width, height = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), int(capture.get(3)), int(capture.get(4))
f = open(os.path.join(projpath,'labelling_info.txt'), 'w')
video_info_path = os.path.join((projpath), 'logs', 'video_info.csv')
print(video_info_path)
vidinfDf = pd.read_csv(video_info_path)
vidinfDf["Video"] = vidinfDf["Video"].astype(str)
videoName = os.path.basename(currentVideoPath)
videoName = videoName.split('.', 2)[0]
fps = capture.get(cv2.CAP_PROP_FPS)
timeBetFrames = int(1000/fps)


def printOnFrame(currentFrame):
    currentTime = currentFrame / fps
    currentTime = round(currentTime, 2)
    cv2.putText(frame, 'F~ ' + str(currentFrame), (10, (height - 20)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2)
    cv2.putText(frame, 'T~ ' + str(currentTime), (10, (height - 80)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2)

while True:
    ret, frame = capture.read()
    key = cv2.waitKey(timeBetFrames) & 0xff
    if key == ord('p'):
        while True:
            key2 = cv2.waitKey(1) or 0xff
            ### THE VIDEO IS PAUSED
            currDisplayFrameNo = int(capture.get(cv2.CAP_PROP_POS_FRAMES))-1
            f.seek(0)
            f.write(str(currDisplayFrameNo-1))

            f.truncate()
            f.flush()
            os.fsync(f.fileno())
            ### BACK UP TWO FRAME

            if key2 == ord('o'):
                capture.set(cv2.CAP_PROP_POS_FRAMES, (currDisplayFrameNo - 2))
                ret, frame = capture.read()
                currentFrame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))-2
                printOnFrame(currentFrame)
                cv2.imshow('Video', frame)
                f.seek(0)
                f.write(str(currentFrame))
                f.truncate()
                f.flush()
                os.fsync(f.fileno())

            ### BACK UP TEN FRAME
            if key2 == ord('i'):
                capture.set(cv2.CAP_PROP_POS_FRAMES, (currDisplayFrameNo - 10))
                ret, frame = capture.read()
                currentFrame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))-2
                printOnFrame(currentFrame)
                cv2.imshow('Video', frame)
                f.seek(0)
                f.write(str(currentFrame))
                f.truncate()
                f.flush()
                os.fsync(f.fileno())
            ### BACK UP 1s
            if key2 == ord('u'):
                capture.set(cv2.CAP_PROP_POS_FRAMES, (currDisplayFrameNo-fps))
                ret, frame = capture.read()
                currentFrame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))-2
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
                currentFrame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))-2
                printOnFrame(currentFrame)
                cv2.imshow('Video', frame)
                f.seek(0)
                f.write(str(currentFrame))
                f.truncate()
                f.flush()
                os.fsync(f.fileno())
            ### FORWARD TWO FRAMES
            if key2 == ord('r'):
                capture.set(cv2.CAP_PROP_POS_FRAMES, (currDisplayFrameNo + 2))
                ret, frame = capture.read()
                currentFrame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))-2
                printOnFrame(currentFrame)
                cv2.imshow('Video', frame)
                f.seek(0)
                f.write(str(currentFrame))
                f.truncate()
                f.flush()
                os.fsync(f.fileno())
            ### FORWARD TEN FRAMES
            if key2 == ord('e'):
                capture.set(cv2.CAP_PROP_POS_FRAMES, (currDisplayFrameNo + 10))
                ret, frame = capture.read()
                currentFrame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))-2
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
                path = os.path.join(projpath,'subprocess.txt')
                txtFile = open(path)
                line = txtFile.readline()
                os.kill(int(line), signal.SIGTERM)
                break
            if cv2.getWindowProperty('Video', 1) == -1:
                capture.release()
                cv2.destroyAllWindows()
                path = os.path.join(projpath,'subprocess.txt')
                txtFile = open(path)
                line = txtFile.readline()
                try:
                    os.kill(int(line), signal.SIGTERM)
                except OSError:
                    print('OSError: Cannot save/read latest image file CSV. Please try again')

    currentFrame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))-1
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
path = os.path.join(projpath,'subprocess.txt')
txtFile = open(path)
line = txtFile.readline()
os.kill(int(line), signal.SIGTERM)












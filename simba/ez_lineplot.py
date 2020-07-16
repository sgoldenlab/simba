import os
import cv2
import numpy as np
import pandas as pd

def draw_line_plot(configini,video,bodypart):
    configdir = os.path.dirname(configini)
    csvname = video.split('.')[0] + '.csv'
    tracking_csv = os.path.join(configdir,'csv','outlier_corrected_movement_location',csvname)
    inputDf = pd.read_csv(tracking_csv)
    videopath = os.path.join(configdir,'videos',video)
    outputvideopath = os.path.join(configdir,'videos','path_plot')
    if not os.path.exists(outputvideopath):
        os.mkdir(outputvideopath)

    #datacleaning
    colHeads = [bodypart + '_x', bodypart + '_y', bodypart + '_p']
    df = inputDf[colHeads].copy()

    widthlist = df[colHeads[0]].astype(float).astype(int)
    heightlist = df[colHeads[1]].astype(float).astype(int)
    circletup = tuple(zip(widthlist,heightlist))

    # get resolution of video
    vcap = cv2.VideoCapture(videopath)
    if vcap.isOpened():
        width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
        height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
        fps = int(vcap.get(cv2.CAP_PROP_FPS))
        totalFrameCount = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))

    # make white background
    img = np.zeros([height, width, 3])
    img.fill(255)
    img = np.uint8(img)


    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(os.path.join(outputvideopath,video), fourcc, fps, (width,height))
    counter=0
    while (vcap.isOpened()):
        ret,frame = vcap.read()
        if ret == True:
            if counter !=0:
                cv2.line(img,circletup[counter-1],circletup[counter],5)

            lineWithCircle = img.copy()
            cv2.circle(lineWithCircle, circletup[counter],5,[0,0,255],-1)



            out.write(lineWithCircle)
            counter+=1
            print('Frame ' + str(counter) + '/' + str(totalFrameCount))

        else:
            break

    vcap.release()
    cv2.destroyAllWindows()
    print('Video generated.')
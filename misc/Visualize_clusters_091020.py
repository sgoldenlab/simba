import numpy as np
import pandas as pd
import os, glob
import cv2

plot_skeleton = 'yes'
save_frames = 'yes'
videoFn = 'SA_02_156'

dataFilePath =
videoFile = 
outputFolder =
skeletonDfpath = 
animal1HeadersDf =


animal1Headers, animal2Headers = list(animal1HeadersDf['Animal_1']), list(animal1HeadersDf['Animal_2'])
dataFile = pd.read_csv(dataFilePath, index_col=0)
dataFile = dataFile[dataFile['Video'] == os.path.basename(videoFile).replace('.mp4', '')]
cap = cv2.VideoCapture(videoFile)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
width, height, noFrames = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
clusterVals = list(dataFile['Cluster'].unique())

if plot_skeleton == 'yes':
    skeletonDf = pd.read_csv(os.path.join(dataFilePath, videoFn + '.csv'), index_col=0)

if save_frames == 'yes':
    dirPath = os.path.join(outputFolder, videoFn)
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

for currCluster in clusterVals:
    if save_frames == 'yes':
        dirPathCluster = os.path.join(outputFolder, videoFn, 'Cluster_' + str(currCluster))
        if not os.path.exists(dirPathCluster):
            os.makedirs(dirPathCluster)

    cap = cv2.VideoCapture(videoFile)
    vidBasename = os.path.basename(videoFile).replace('.mp4', '.avi')
    outputPath = os.path.join(outputFolder, 'Cluster_' + str(currCluster) + '_' + vidBasename)
    writer = cv2.VideoWriter(outputPath, fourcc, 30, (width, height))
    outputDf = pd.DataFrame(columns=['Cluster'])
    outputDf['Cluster'] = [0] * noFrames
    currDf = dataFile[dataFile['Cluster'] == currCluster]
    currDf = currDf[['Frame_start', 'Frame_end']]
    for index, row in currDf.iterrows():
        frameList = list(range(row['Frame_start'], row['Frame_end']))
        for frame in frameList:
            outputDf['Cluster'][frame] = 1

    outDf = pd.DataFrame(columns=['Start', 'End'])
    groupDf = pd.DataFrame()
    v = (outputDf['Cluster'] != outputDf['Cluster'].shift()).cumsum()
    u = outputDf.groupby(v)['Cluster'].agg(['all', 'count'])
    m = u['all'] & u['count'].ge(1)
    groupDf['groups'] = outputDf.groupby(v).apply(lambda x: (x.index[0], x.index[-1]))[m]
    for row in groupDf.itertuples():
        start, end = row[1][0] - 15, row[1][1] + 15
        if start < 0: start = 0
        if end > len(outputDf): end = len(outputDf)
        appendList = [start, end]
        outDf.loc[len(outDf)] = appendList

    clusterCounter = 0
    for index, row in outDf.iterrows():
        clusterCounter += 1
        if save_frames == 'yes':
            dirPathClusterExample = os.path.join(dirPathCluster, 'Example_' + str(clusterCounter))
            if not os.path.exists(dirPathClusterExample):
                os.makedirs(dirPathClusterExample)
        frames = list(range(row['Start'], row['End']+1))
        behaviorStart, behaviorEnd = frames[0] + 15, frames[-1] - 15
        frameCounter = 0
        for frameNo in frames:
            frameCounter += 1
            currAnimal1 = list(skeletonDf.loc[skeletonDf.index[frameNo], animal1Headers])
            currAnimal1 = np.array([currAnimal1[i:i + 2] for i in range(0, len(currAnimal1), 2)]).astype(int)
            currAnimal2 = list(skeletonDf.loc[skeletonDf.index[frameNo], animal2Headers])
            currAnimal2 = np.array([currAnimal2[i:i + 2] for i in range(0, len(currAnimal2), 2)]).astype(int)
            currFrame = np.zeros((height, width , 3), dtype = "uint8")
            currAnimal1_hull = cv2.convexHull((currAnimal1.astype(int)))
            currAnimal2_hull = cv2.convexHull((currAnimal2.astype(int)))

            cv2.drawContours(currFrame, [currAnimal1_hull.astype(int)], 0, (255, 255, 255), 1)
            cv2.drawContours(currFrame, [currAnimal2_hull.astype(int)], 0, (0, 255, 0), 1)
            for anim1, anim2 in zip(currAnimal1, currAnimal2):
                cv2.circle(currFrame, (int(anim1[0]), int(anim1[1])), 4, (147, 20, 255), thickness=-1, lineType=8, shift=0)
                cv2.circle(currFrame, (int(anim2[0]), int(anim2[1])), 4, (0, 255, 255), thickness=-1, lineType=8, shift=0)

            cv2.line(currFrame, (currAnimal1[0][0], currAnimal1[0][1]), (currAnimal1[1][0], currAnimal1[1][1]), (0, 0, 255), 2)
            cv2.line(currFrame, (currAnimal2[0][0], currAnimal2[0][1]), (currAnimal2[1][0], currAnimal2[1][1]), (0, 0, 255), 2)
            cv2.line(currFrame, (currAnimal1[4][0], currAnimal1[4][1]), (currAnimal1[5][0], currAnimal1[5][1]), (0, 0, 255), 2)
            cv2.line(currFrame, (currAnimal2[4][0], currAnimal2[4][1]), (currAnimal2[5][0], currAnimal2[5][1]), (0, 0, 255), 2)
            cv2.line(currFrame, (currAnimal1[2][0], currAnimal1[2][1]), (currAnimal1[6][0], currAnimal1[6][1]), (0, 0, 255), 2)
            cv2.line(currFrame, (currAnimal2[2][0], currAnimal2[2][1]), (currAnimal2[6][0], currAnimal2[6][1]), (0, 0, 255), 2)


            if (frameNo < behaviorStart):
                cv2.putText(currFrame, str('Cluster ' + str(currCluster)) + ' coming up...', (30, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
            elif (frameNo > behaviorEnd):
                cv2.putText(currFrame, str('Cluster ' + str(currCluster)) + ' happened.', (30, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(currFrame, str(currCluster) + '!!', (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)

            if save_frames == 'yes':
                filename = os.path.join(dirPathClusterExample, str(frameCounter) + '.png')
                cv2.imwrite(filename, currFrame)


            writer.write(currFrame)

        for i in range(30):
            blueFrame = np.zeros((currFrame.shape[0], currFrame.shape[1], 3))
            blueFrame[:] = (255, 0, 0)
            blueFrame = blueFrame.astype(np.uint8)
            writer.write(blueFrame)

    print('saved')
    cap.release()
































currRow = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        currVals = clusterFramesDf.loc[clusterFramesDf['Frame'] == currRow]
        if currVals.empty:
            cv2.putText(frame, str('Cluster: None'), (30, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2)
        else:
            clustervalue = list(currVals['Cluster'])
            cv2.putText(frame, str(clustervalue[0]), (30, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2)




        frame = np.uint8(frame)
        cv2.imshow('image', frame)
        cv2.waitKey(30)
        #cv2.destroyAllWindows()
        writer.write(frame)
    if frame is None:
        print('Video saved.')
        cap.release()
        break
    currRow+=1
    print(currRow)








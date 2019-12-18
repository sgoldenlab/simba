import os
import pandas as pd
currDir = os.getcwd()

# Enter the path to the folder that contains both the feature extracted files (.csv) and the caltech annotation files (.txt).
# The matching txt and csv files need to have the same file names (e.g., 'video_1.csv' and 'video_1.txt, 'video_2.csv and 'video_2.txt').
data_files_folder = 'Z:\DeepLabCut\misc\LinLab\data'

#If the video files were shortened after Caltech Behavior Annotator, but before DLC tracking, enter the path to a csv that contains the #frames that were removed here.
#If the videos are not cut, set the 'Initial frames removed' in this csv file to be 0. For an example of the layout of this csv file,go to this link: https://github.com/sgoldenlab/simba/blob/master/video_information_updated_CRIM13.csv
videoInfoFilePath = r"Z:\DeepLabCut\misc\LinLab\info_files\video_information_updated.csv"

# Specify a list of behaviour names that were annotated
behaviourList = ['attack_intruder', 'sniff_investigation', 'grooming_themselves', 'rearing', 'stay_corner']


videoInfoDf = pd.read_csv(videoInfoFilePath, index_col=False)
#READ IN THE DLC and annotation files
annotatorFilesList = []
outPutDir = os.path.join(currDir, 'output_files')
if not os.path.exists(outPutDir):
    os.makedirs(outPutDir)
DLCfilesList = []

for i in os.listdir(data_files_folder):
    if i.__contains__(".csv"):
        DLCFilePath = os.path.join(data_files_folder, i)
        DLCfilesList.append(DLCFilePath)
    if i.__contains__(".txt"):
        AnnotationFilePath = os.path.join(data_files_folder, i)
        annotatorFilesList.append(AnnotationFilePath)

for i in range(len(DLCfilesList)):
    curVideoFileNamePath = os.path.basename(DLCfilesList[i])
    curVideoFileName = curVideoFileNamePath.replace('.csv', '')
    currentVidInf = videoInfoDf.loc[videoInfoDf['Video_name'] == curVideoFileName].astype('str')
    print(currentVidInf)
    cutoffValue = int(currentVidInf['Initial frames removed'])
    filename = DLCfilesList[i]
    currentDLCfile = pd.read_csv(DLCfilesList[i], low_memory=False)
    framesInDLCfile = len(currentDLCfile)
    for name in behaviourList:
        currentDLCfile[name] = 0
    currAnnotationFilePath = DLCfilesList[i].replace('.csv', '.txt')
    print(DLCfilesList[i], currAnnotationFilePath, cutoffValue)
    currentAnnotationFile = pd.read_csv(currAnnotationFilePath, delim_whitespace=True,index_col=False, low_memory=False)
    currentAnnotationFile = currentAnnotationFile.iloc[19:]
    currentAnnotationFile = currentAnnotationFile.drop(['File', 'Annotation', '-'], axis=1)
    currentAnnotationFile.columns = ['Start_frame', 'End_frame', "Behaviour"]
    currentAnnotationFile.reset_index(drop=True, inplace=True)
    currentAnnotationFile[["Start_frame", "End_frame"]] = currentAnnotationFile[["Start_frame", "End_frame"]].apply(pd.to_numeric)
    for index, row in currentAnnotationFile.iterrows():
        currBehaviour = row['Behaviour']
        if currBehaviour in behaviourList:
            boutList = range((row['Start_frame']-1), row['End_frame'])
            boutList = [x - cutoffValue for x in boutList]
            boutList = sorted(i for i in boutList if i >= 0)
            boutList = sorted(i for i in boutList if i < framesInDLCfile)
            for b in boutList:
                currentDLCfile.loc[b, currBehaviour] = 1
    fileSaveName = os.path.basename(filename)
    fileSaveName = os.path.join(outPutDir, fileSaveName)
    currentDLCfile.rename(columns={'Unnamed: 0': 'scorer'}, inplace=True)
    currentDLCfile.to_csv(fileSaveName, index=False)
    print('File saved: ' + str(fileSaveName))
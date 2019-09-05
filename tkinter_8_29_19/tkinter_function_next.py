import subprocess
import os
import cv2
from os import listdir
from os.path import isfile, join
import yaml
from PIL import Image
import glob


def downsamplevideo_next(width,height,directory,maindirectory):
    filesFound = []
    most_recent_folder = directory

    ########### FIND MP4 FILES ###########
    for i in os.listdir(most_recent_folder):
        if i.__contains__(".mp4"):
            filesFound.append(i)
    print(most_recent_folder)

    for i in filesFound:
        currentFile = i
        outFile = currentFile.replace('.mp4', '')
        outFile = str(outFile) + '_downsampled.mp4'
        outFile = os.path.basename(outFile)
        print(outFile)

        command = (str('ffmpeg -i ') +str(most_recent_folder) +str(currentFile) + ' -vf scale='+str(width)+':'+ str(height) + ' '+str(directory)+'\\downsampled_videos\\' + outFile + ' -hide_banner')

        cfilename = str(maindirectory)+'\\'+'process_video_define.txt'

        if os.path.exists(cfilename):
            append_write = 'a'  # append if already exists
        else:
            append_write = 'w'  # make a new file if not

        highscore = open(cfilename, append_write)
        highscore.write(command + '\n')
        highscore.close()

        print(command, 'is added to ', cfilename)

def greyscale_next(directory,maindirectory):
    filesFound = []
    most_recent_folder = directory

    ########### FIND MP4 FILES ###########
    for i in os.listdir(most_recent_folder):
        if i.__contains__(".mp4"):
            filesFound.append(i)
    print(most_recent_folder)

    for i in filesFound:
        currentFile = i
        outFile = currentFile.replace('.mp4', '')
        outFile = str(outFile) + '_greyscale.mp4'
        outFile = os.path.basename(outFile)

        command = (str('ffmpeg -i ') + str(most_recent_folder) + str(currentFile) + ' -vf format=gray '+ str(directory)+ '\\greyscale_videos\\' + outFile)

        cfilename = str(maindirectory)+'\\'+'process_video_define.txt'

        if os.path.exists(cfilename):
            append_write = 'a'  # append if already exists
        else:
            append_write = 'w'  # make a new file if not

        highscore = open(cfilename, append_write)
        highscore.write(command + '\n')
        highscore.close()

        print(command, 'is added to ', cfilename)

def superimposeframe_next(directory,maindirectory):
    filesFound = []
    most_recent_folder = directory

    ########### FIND MP4 FILES ###########
    for i in os.listdir(most_recent_folder):
        if i.__contains__(".mp4"):
            filesFound.append(i)
    print(most_recent_folder)

    for i in filesFound:

        currentFile = i
        outFile = currentFile.replace('.mp4', '')
        outFile = str(outFile) + '_frame_no.mp4'
        outFile = os.path.basename(outFile)

        command = (str('ffmpeg -i ') + str(currentFile) + ' -vf "drawtext=fontfile=Arial.ttf: text=\'%{frame_num}\': start_number=1: x=(w-tw)/2: y=h-(2*lh): fontcolor=black: fontsize=20: box=1: boxcolor=white: boxborderw=5" -c:a copy '+ str(directory) + '\\withFrames_videos\\' +outFile)

        cfilename = str(maindirectory)+'\\'+'process_video_define.txt'

        if os.path.exists(cfilename):
            append_write = 'a'  # append if already exists
        else:
            append_write = 'w'  # make a new file if not

        highscore = open(cfilename, append_write)
        highscore.write(command + '\n')
        highscore.close()

        print(command, 'is added to ', cfilename)

def colorized_text(filename):

    currentFile = filename
    outFile = currentFile.replace('.mp4', '')
    outFile = str(outFile) + '_colorized.mp4'
    command = (str('python bw2color_video3.py --prototxt colorization_deploy_v2.prototxt --model colorization_release_v2.caffemodel --points pts_in_hull.npy --input ' )+ str(currentFile))

    filename = 'process_video_define.txt'

    if os.path.exists(filename):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not

    highscore = open(filename, append_write)
    highscore.write(command + '\n')
    highscore.close()
    print(command, 'is added to ', filename)

def shortenvideos_text(videofile,starttime,endtime,directory):

    currentFile = videofile
    outFile = currentFile.replace('.mp4', '')
    outFile = str(outFile) + '_shorten.mp4'
    outFile = os.path.basename(outFile)

    print('Cutting video....')
    command = (str('ffmpeg -i ') + str(currentFile) + ' -ss ' + starttime +' -to ' + endtime + ' -c:v copy -c:a copy '+ str(directory)+'\\shoten_videos\\' +outFile)

    filename = 'process_video_define.txt'

    if os.path.exists(filename):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not

    highscore = open(filename, append_write)
    highscore.write(command + '\n')
    highscore.close()

    print(command, 'is added to ', filename)


def extract_allframescommand_text(filename):
    pathDir1 = os.path.basename(str(filename[:-4]))
    pathDir = str((os.path.dirname(filename)+'\\frames\\')+ pathDir1)
    # print(pathDir)
    # if not os.path.exists(pathDir):
    #     os.makedirs(pathDir)
    picFname = '%d.png'

    saveDirFilenames = os.path.join(pathDir, picFname)

    fname = str(filename)
    cap = cv2.VideoCapture(fname)
    fps = cap.get(cv2.CAP_PROP_FPS)
    amount_of_frames = cap.get(7)
    command = str('ffmpeg -i ' + str(fname) + str(' -r ') + str(fps) + ' ' + str(saveDirFilenames))
    print(command)
    filename = 'process_video_define.txt'

    if os.path.exists(filename):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not

    highscore = open(filename, append_write)
    highscore.write(command + '\n')
    highscore.close()

    print(command, 'is added to ', filename)

def mergemovieffmpeg_text(directory,framespersec,vidformat,bit,imgformat):

    currDir = directory

    fps = str(framespersec)
    fileformat = str('.'+vidformat)
    bitrate = str(bit)
    imageformat = str(imgformat)

    currentDir = directory
    fileOut = str(directory)+ str(fileformat)
    currentDirPath = directory
    currentFileList = [f for f in listdir(currentDirPath) if isfile(join(currentDirPath, f))]
    imgPath = os.path.join(currentDirPath, currentFileList[0])
    img = cv2.imread(imgPath)
    print(imgPath)
    ffmpegFileName = os.path.join(currentDirPath, '%d.' + str(imageformat))
    imgShape = img.shape
    height = imgShape[0]
    width = imgShape[1]
    command = str('ffmpeg -r ' + str(fps) + str(' -f image2 -s ') + str(height) + 'x' + str(width) + ' -i ' + str(ffmpegFileName) + ' -vcodec libx264 -b ' + str(bitrate) + 'k ' + str(fileOut))

    filename = 'process_video_define.txt'

    if os.path.exists(filename):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not

    highscore = open(filename, append_write)
    highscore.write(command + '\n')
    highscore.close()

    print(command, 'is added to ', filename)


def clahe(directory):

    most_recent_folder = directory

    print('Applying Clahe to all the videos found in directory: ', os.getcwd())
    filesFound = []

    ########### FIND MP4 FILES ###########
    for i in os.listdir(most_recent_folder):
        if i.__contains__(".mp4"):
            filesFound.append(i)
    os.chdir(most_recent_folder)
    print('Videos found: ', filesFound)
    for i in filesFound:
        currentVideo = i
        fileName, fileEnding = currentVideo.split('.', 2)
        saveName = str('CLAHE_') + str(fileName) + str('.avi')
        cap = cv2.VideoCapture(currentVideo)
        imageWidth = int(cap.get(3))
        imageHeight = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(saveName, fourcc, fps, (imageWidth, imageHeight), 0)
        while True:
            ret, image = cap.read()
            if ret == True:
                im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                claheFilter = cv2.createCLAHE(clipLimit=2, tileGridSize=(16, 16))
                claheCorrecttedFrame = claheFilter.apply(im)
                out.write(claheCorrecttedFrame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                print(str('Completed video ') + str(saveName))
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def cropvid_text(directory,filenames):
    global width,height

    #extract one frame
    currentDir = str(os.path.dirname(filenames))
    videoName = str(os.path.basename(filenames))
    os.chdir(currentDir)
    cap = cv2.VideoCapture(videoName)
    cap.set(1, 0)
    ret, frame = cap.read()
    fileName = str(0) + str('.bmp')
    filePath = os.path.join(currentDir, fileName)
    cv2.imwrite(filePath, frame)

    #find ROI

    img = cv2.imread(filePath)
    cv2.namedWindow('Select ROI', cv2.WINDOW_NORMAL)
    ROI = cv2.selectROI("Select ROI", img)
    width = abs(ROI[0] - (ROI[2] + ROI[0]))
    height = abs(ROI[2] - (ROI[3] + ROI[2]))
    topLeftX = ROI[0]
    topLeftY = ROI[1]
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #crop video with ffmpeg
    fileOut, fileType = videoName.split(".", 2)
    fileOutName = str(fileOut) + str('_cropped.mp4')
    command = str('ffmpeg -i ')+ str(directory) + '\\' + str(videoName) + str(' -vf ') + str('"crop=') + str(width) + ':' + str(
        height) + ':' + str(topLeftX) + ':' + str(topLeftY) + '" ' + str('-c:v libx264 -crf 0 -c:a copy ') + str(
        os.path.join(directory+'\\cropped_videos\\',fileOutName))

    filename = 'process_video_define.txt'

    if os.path.exists(filename):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not

    total = width+height+topLeftX +topLeftY

    if total != 0:

        highscore = open(filename, append_write)
        highscore.write(command + '\n')
        highscore.close()

        print(command,'is added to ',filename)

    elif total == 0:
        print('nothing added to the script as no coordinates was selected')


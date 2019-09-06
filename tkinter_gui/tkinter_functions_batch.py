import subprocess
import os
import cv2
from os import listdir
from os.path import isfile, join
import yaml
from PIL import Image
import glob


def downsamplevideo_batch(width,height,directory):
    filesFound = []
    ########### FIND FILES ###########
    for i in os.listdir(directory):
        if i.__contains__(".mp4"):
            filesFound.append(i)

    def execute(command):
        print(command)
        subprocess.call(command, shell=True, stdout = subprocess.PIPE)

    ########### DEFINE COMMAND ###########
    for i in filesFound:
        currentFile = i
        outFile = currentFile.replace('.mp4', '')
        outFile = str(outFile) + '_downsampled.mp4'
        output = os.path.basename(outFile)
        print('Downsampling video...')
        command = (str('ffmpeg -i ') + str(directory)+ '\\' + str(currentFile) + ' -vf scale='+str(width)+':'+ str(height) + ' ' + str(directory) + '\\' + outFile + ' -hide_banner')
        execute(command)
        print('Video Downsampled! ',output, ' is created')


def greyscale_batch(directory):
    filesFound = []

    def execute(command):
        print(command)
        subprocess.call(command, shell=True, stdout = subprocess.PIPE)

    ########### FIND FILES ###########
    for i in os.listdir(directory):
        if i.__contains__(".mp4"):

            filesFound.append(i)

    ########### DEFINE COMMAND ###########
    for i in filesFound:
        currentFile = i
        outFile = currentFile.replace('.mp4', '')
        outFile = str(outFile) + '_greyscale.mp4'
        output = os.path.basename(outFile)
        print('Converting into greyscale video...(this might take awhile)')
        command = (str('ffmpeg -i ') + str(directory)+'\\' +str(currentFile) + ' -vf format=gray '+ str(directory)+'\\' + outFile)
        execute(command)
        print('Video converted into greyscale! ', output, ' is created')

def superimposeframe_batch(directory):

    filesFound = []
    def execute(command):
        print(command)
        subprocess.call(command, shell=True, stdout=subprocess.PIPE)

    ########### FIND FILES ###########
    for i in os.listdir(directory):
        if i.__contains__(".mp4"):
            filesFound.append(i)
    ########### DEFINE COMMAND ###########
    for i in filesFound:
        currentFile = i
        outFile = currentFile.replace('.mp4', '')
        outFile = str(outFile) + '_frame_no.mp4'
        print('Adding frame numbers...')
        command = (str('ffmpeg -i ') + str(directory)+'\\' + str(currentFile) + ' -vf "drawtext=fontfile=Arial.ttf: text=\'%{frame_num}\': start_number=1: x=(w-tw)/2: y=h-(2*lh): fontcolor=black: fontsize=20: box=1: boxcolor=white: boxborderw=5" -c:a copy '+ str(directory)+'\\' + outFile)
        execute(command)
        print('Frame numbers added!')

def colorized_batch(directory):
    filesFound = []

    def execute(command):
        print(command)
        subprocess.call(command, shell=True, stdout=subprocess.PIPE)

    ########### FIND FILES ###########
    for i in os.listdir(directory):
        if i.__contains__(".mp4"):
            filesFound.append(i)
    ########### DEFINE COMMAND ###########
    for i in filesFound:
        currentFile = i
        outFile = currentFile.replace('.mp4', '')
        outFile = str(outFile) + '_colorized.mp4'
        command = (str('python bw2color_video3.py --prototxt colorization_deploy_v2.prototxt --model colorization_release_v2.caffemodel --points pts_in_hull.npy --input ' )+ str(directory)+'\\' + str(currentFile))
        execute(command)

def shortenvideos1_batch(directory,starttime,endtime):
    filesFound = []

    def execute(command):
        print(command)
        subprocess.call(command, shell=True, stdout=subprocess.PIPE)

    ########### FIND FILES ###########
    for i in os.listdir(directory):
        if i.__contains__(".mp4"):
            filesFound.append(i)
    ########### DEFINE COMMAND ###########
    for i in filesFound:
        currentFile = i
        outFile = currentFile.replace('.mp4', '')
        outFile = str(outFile) + '_shorten.mp4'
        output = os.path.basename(outFile)
        print('Cutting video....')
        command = (str('ffmpeg -i ') + str(directory)+'\\' + str(currentFile) + ' -ss ' + str(starttime) +' -to ' + str(endtime) + ' -c:v copy -c:a copy '+ str(directory)+'\\' + outFile)
        execute(command)
        print(output,' is generated!')

def shortenvideos2_batch(directory,time):
    filesFound = []

    def execute(command):
        print(command)
        subprocess.call(command, shell=True, stdout=subprocess.PIPE)

    ########### FIND FILES ###########
    for i in os.listdir(directory):
        if i.__contains__(".mp4"):
            filesFound.append(i)
    ########### DEFINE COMMAND ###########
    for i in filesFound:
        currentFile = i
        outFile = currentFile.replace('.mp4', '')
        outFile = str(outFile) + '_shorten.mp4'
        output = os.path.basename(outFile)
        print('Cutting video....')
        command = (str('ffmpeg -sseof ') + ' -' + str(time) + ' -i ' + str(directory)+'\\' + str(currentFile) + ' ' + str(directory)+'\\' + outFile)
        execute(command)
        print(output,' is generated!')

def convertavitomp4(filename):

    def execute(command):
        print(command)
        subprocess.call(command, shell=True, stdout = subprocess.PIPE)

    ########### DEFINE COMMAND ###########

    currentFile = filename
    outFile = currentFile.replace('.avi', '')
    outFile = str(outFile) + '_converted.mp4'
    output = os.path.basename(outFile)
    print('Converting avi to mp4...')
    command = (str('ffmpeg -i ') + str(currentFile) + ' ' + outFile)
    execute(command)
    print('Video converted! ',output, ' is generated!')
    return output

def convertpowerpoint_batch(directory):
    filesFound = []

    def execute(command):
        print(command)
        subprocess.call(command, shell=True, stdout=subprocess.PIPE)

    ########### FIND FILES ###########
    for i in os.listdir(directory):
        if i.__contains__(".mp4"):
            filesFound.append(i)
    ########### DEFINE COMMAND ###########
    for i in filesFound:
        currentFile = i
        outFile = currentFile.replace('.mp4', '')
        outFile = str(outFile) + '_powerpointready.mp4'
        output = os.path.basename(outFile)
        print('Making video into powerpoint compatible format... ')
        command = (str('ffmpeg -i ') + str(directory)+'\\' + str(currentFile) + ' -c:v libx264 -preset slow  -profile:v high -level:v 4.0 -pix_fmt yuv420p -crf 22 -codec:a aac ' + str(directory)+'\\' + outFile)
        execute(command)
        print('Done!')

def extract_allframescommand_batch(directory):
    filesFound = []

    def execute(command):
        print(command)
        subprocess.call(command, shell=True, stdout=subprocess.PIPE)

    ########### FIND FILES ###########
    for i in os.listdir(directory):
        if i.__contains__(".mp4"):
            filesFound.append(i)

    for i in filesFound:
        pathDir1 = str(i[:-4])
        pathDir = str(str(directory)+'\\' + pathDir1)
        print(pathDir)
        if not os.path.exists(pathDir):
            os.makedirs(pathDir)

        picFname = '%d.png'

        saveDirFilenames = os.path.join(pathDir, picFname)
        print(saveDirFilenames)

        fname = str(directory)+'\\' + str(i)
        print(fname)
        cap = cv2.VideoCapture(fname)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(fps)
        amount_of_frames = cap.get(7)
        print('The number of frames in this video = ',amount_of_frames)
        print('Extracting frames... (Might take awhile)')
        command = str('ffmpeg -i ' + str(fname) + ' ' + '-q:v 1' + ' ' + '-start_number 0' + ' ' + str(saveDirFilenames))
        print(command)
        subprocess.call(command, shell=True)
        print('All frames are extracted!')

def mergemovieffmpeg(directory,framespersec,vidformat,bit,imgformat):

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
    print(command)
    subprocess.call(command, shell=True)

def extractspecificframe(filename,startframe1,endframe1):

    cap = cv2.VideoCapture(filename)
    amount_of_frames = cap.get(7)
    pathDir = str(filename[:-4]+'\\frames')
    if not os.path.exists(pathDir):
        os.makedirs(pathDir)

    print(amount_of_frames)

    frames_OI = list(range(int(startframe1),int(endframe1)))
    #frames_OI.extend(range(7000,7200))
    #frames_OI.extend(range(9200,9350))

    for i in frames_OI:
        currentFrame = i
        cap.set(1, currentFrame)
        ret, frame = cap.read()
        fileName = str(currentFrame) + str('.png')
        filePath = os.path.join(pathDir, fileName)
        cv2.imwrite(filePath,frame)

def clahe_batch(directory):

    filesFound= []

    ########### FIND FILES ###########
    for i in os.listdir(directory):
        if i.__contains__(".mp4"):
            filesFound.append(i)

    os.chdir(directory)
    print('Applying CLAHE, this might take awhile...')

    for i in filesFound:
        currentVideo = i
        saveName = str('CLAHE_') + str(currentVideo[:-4]) + str('.avi')
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
    return saveName

def cropvid_batch(directory,filenames):
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
        os.path.join(directory,fileOutName))

    filename = str(directory)+'\\'+'process_video_define.txt'

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

def extract_frames_ini(directory):
    filesFound = []

    def execute(command):
        print(command)
        subprocess.call(command, shell=True, stdout=subprocess.PIPE)

    ########### FIND FILES ###########
    for i in os.listdir(directory):
        if i.__contains__(".mp4"):
            filesFound.append(i)


    for i in filesFound:
        pathDir1 = str(i[:-4])
        pathDir0 =str(str(os.path.dirname(directory)) +'\\frames\\input' )
        pathDir = str(str(pathDir0)+'\\' + pathDir1)

        if not os.path.exists(pathDir):
            os.makedirs(pathDir)
            picFname = '%d.png'

            saveDirFilenames = os.path.join(pathDir, picFname)

            fname = str(directory)+'\\' + str(i)
            cap = cv2.VideoCapture(fname)
            fps = cap.get(cv2.CAP_PROP_FPS)
            amount_of_frames = cap.get(7)
            print('The number of frames in this video = ',amount_of_frames)
            print('Extracting frames... (Might take awhile)')
            command = str('ffmpeg -i ' + str(fname) + ' ' + '-q:v 1' + ' ' + '-start_number 0' + ' ' + str(saveDirFilenames))
            print(command)
            subprocess.call(command, shell=True)
            print('Frames were extracted for',os.path.basename(pathDir))


        else:
            print(os.path.basename(pathDir),'existed, no action taken, frames should be in there')

    print('All frames were extracted.')
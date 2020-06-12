import os
import cv2


def downsamplevideo_auto(width,height,filesFound,outputdir):

    downsamplelist = []
    ########### DEFINE COMMAND ###########
    for i in filesFound:
        currentFile = i
        outFile = currentFile.replace('.mp4', '')
        outFile = str(outFile) + '_downsampled.mp4'
        output = os.path.basename(outFile)

        command = (str('ffmpeg -y -i ') + '"' + str(outputdir) + '\\' + os.path.basename(currentFile)+ '"' + ' -vf scale='+str(width)+':'+ str(height) + ' ' + '"'+ str(outputdir) + '\\' + output + '"'+ ' -hide_banner' + '\n'
                   'move \"' + str(outputdir) + '\\' + os.path.basename(currentFile) + '\" \"' + os.path.dirname(outputdir) + '\\' + 'tmp\"' + '\n'
                   'copy \"' + str(outputdir) + '\\' + output + '\" \"' +os.path.dirname(outputdir) +'\\' +'tmp\"' +'\n'
                   'rename \"' +os.path.join(str(outputdir),output) + '\" \"' + os.path.basename(currentFile)+'\"')

        downsamplelist.append(command)
    print('Downsample added into queue')
    return downsamplelist

def downsamplevideo_queue(width,height,filesFound,outputdir):

    currentFile = filesFound
    outFile = currentFile.replace('.mp4', '')
    outFile = str(outFile) + '_downsampled.mp4'
    output = os.path.basename(outFile)

    command = (str('ffmpeg -y -i ') + '"'+ str(outputdir) + '\\' + os.path.basename(currentFile) + '"'+ ' -vf scale='+str(width)+':'+ str(height) + ' ' + '"'+ str(outputdir) + '\\' + output+ '"' + ' -hide_banner' + '\n'
               'move \"' + str(outputdir) + '\\' + os.path.basename(currentFile) + '\" \"' + (outputdir) + '\\' + 'tmp\"' + '\n'
               'copy \"' + str(outputdir) + '\\' + output + '\" \"' + (outputdir) +'\\' +'tmp\"' +'\n'
               'rename \"' +os.path.join(str(outputdir),output) + '\" \"' + os.path.basename(currentFile)+'\"')

    print(filesFound,'added into the downsample queue')
    return command

def greyscale_auto(outputdir,filesFound):
    greyscale_list=[]

    ########### DEFINE COMMAND ###########
    for i in filesFound:
        currentFile = i
        outFile = currentFile.replace('.mp4', '')
        outFile = str(outFile) + '_grayscale.mp4'
        output = os.path.basename(outFile)
        command = (str('ffmpeg -y -i ') + '"'+ str(outputdir) + '\\' + os.path.basename(currentFile) + '"'+ ' -vf format=gray '+ '"'+ str(outputdir) + '\\' + output + '"'+ '\n'
                   'move \"' + str(outputdir) + '\\' + os.path.basename(currentFile) + '\" \"' + os.path.dirname(outputdir)+'\\'+'tmp\"' +'\n'                     
                   'copy \"' + str(outputdir) + '\\' + output + '\" \"' + os.path.dirname(outputdir)+'\\'+'tmp\"' +'\n'
                   'rename \"' + os.path.join(str(outputdir),output) + '\" \"' + os.path.basename(currentFile)+'\"')
        greyscale_list.append(command)
    print('Grayscale added into queue')
    return greyscale_list

def greyscale_queue(outputdir,filesFound):

    currentFile = filesFound
    outFile = currentFile.replace('.mp4', '')
    outFile = str(outFile) + '_grayscale.mp4'
    output = os.path.basename(outFile)
    command = (str('ffmpeg -y -i ') + '"'+ str(outputdir) + '\\' + os.path.basename(currentFile)+ '"' + ' -vf format=gray '+ '"'+ str(outputdir) + '\\' + output + '"'+ '\n'
               'move \"' + str(outputdir) + '\\' + os.path.basename(currentFile) + '\" \"' + (outputdir)+'\\'+'tmp\"' +'\n'                     
               'copy \"' + str(outputdir) + '\\' + output + '\" \"' + (outputdir)+'\\'+'tmp\"' +'\n'
               'rename \"' + os.path.join(str(outputdir),output) + '\" \"' + os.path.basename(currentFile)+'\"')

    print(filesFound,'added into the grayscale queue')
    return command

def superimposeframe_auto(outputdir,filesFound):
    superimposeframe_list =[]
    ########### DEFINE COMMAND ###########
    for i in filesFound:
        currentFile = i
        outFile = currentFile.replace('.mp4', '')
        outFile = str(outFile) + '_frame_no.mp4'
        output = os.path.basename(outFile)
        command = (str('ffmpeg -y -i ') + '"'+ str(outputdir)+'\\' + os.path.basename(currentFile)+ '"' + ' -vf "drawtext=fontfile=Arial.ttf: text=\'%{frame_num}\': start_number=0: x=(w-tw)/2: y=h-(2*lh): fontcolor=black: fontsize=20: box=1: boxcolor=white: boxborderw=5" -c:a copy ' + '"'+ str(outputdir) + '\\' + output + '"'+ '\n'
                   'move \"' + str(outputdir) + '\\' + os.path.basename(currentFile) + '\" \"' + os.path.dirname(outputdir)+'\\'+'tmp\"' +'\n'
                   'copy \"' + str(outputdir) + '\\' + output + '\" \"' + os.path.dirname(outputdir)+'\\'+'tmp\"' + '\n'
                   'rename \"' + os.path.join(str(outputdir),output) + '\" \"' + os.path.basename(currentFile)+'\"')
        superimposeframe_list.append(command)
    print('Superimpose frame added into queue.')
    return superimposeframe_list

def superimposeframe_queue(outputdir,filesFound):

    currentFile = filesFound
    outFile = currentFile.replace('.mp4', '')
    outFile = str(outFile) + '_frame_no.mp4'
    output = os.path.basename(outFile)
    command = (str('ffmpeg -y -i ') + '"'+ str(outputdir)+'\\' + os.path.basename(currentFile) + '"'+ ' -vf "drawtext=fontfile=Arial.ttf: text=\'%{frame_num}\': start_number=0: x=(w-tw)/2: y=h-(2*lh): fontcolor=black: fontsize=20: box=1: boxcolor=white: boxborderw=5" -c:a copy '+ '"'+ str(outputdir) + '\\' + output + '"'+ '\n'
               'move \"' + str(outputdir) + '\\' + os.path.basename(currentFile) + '\" \"' + (outputdir)+'\\'+'tmp\"' +'\n'
               'copy \"' + str(outputdir) + '\\' + output + '\" \"' + (outputdir)+'\\'+'tmp\"' + '\n'
               'rename \"' + os.path.join(str(outputdir),output) + '\" \"' + os.path.basename(currentFile)+'\"')

    print(filesFound,'added into the superimpose frame queue.')
    return command

def shortenvideos1_auto(outputdir,filesFound,starttime,endtime):
    shortenvideo_list = []

    ########### DEFINE COMMAND ###########
    for i in filesFound:
        currentFile = i
        outFile = currentFile.replace('.mp4', '')
        outFile = str(outFile) + '_shorten.mp4'
        output = os.path.basename(outFile)
        command = (str('ffmpeg -y -i ')+ '"' + str(outputdir)+'\\' + os.path.basename(currentFile)+ '"' + ' -ss ' + str(starttime) +' -to ' + str(endtime) + ' -async 1 '+ '"'+ str(outputdir)+'\\' + output + '"'+ '\n'
                        'move \"' + str(outputdir) + '\\' + os.path.basename(currentFile) + '\" \"' + os.path.dirname(outputdir)+'\\'+'tmp\"' +'\n'
                        'copy \"' + str(outputdir) + '\\' + output + '\" \"' + os.path.dirname(outputdir)+'\\'+'tmp\"' + '\n'
                        'rename \"' + os.path.join(str(outputdir),output) + '\" \"' + os.path.basename(currentFile)+'\"')
        shortenvideo_list.append(command)
    print('Shorten video added into queue')
    return shortenvideo_list

def shortenvideos1_queue(outputdir,filesFound,starttime,endtime):

    currentFile = filesFound
    outFile = currentFile.replace('.mp4', '')
    outFile = str(outFile) + '_shorten.mp4'
    output = os.path.basename(outFile)
    command = (str('ffmpeg -y -i ') + '"'+ str(outputdir)+'\\' + os.path.basename(currentFile) + '"'+ ' -ss ' + str(starttime) +' -to ' + str(endtime) + ' -async 1 '+ '"'+ str(outputdir)+'\\' + output + '"'+ '\n'
                    'move \"' + str(outputdir) + '\\' + os.path.basename(currentFile) + '\" \"' + (outputdir)+'\\'+'tmp\"' +'\n'
                    'copy \"' + str(outputdir) + '\\' + output + '\" \"' + (outputdir)+'\\'+'tmp\"' + '\n'
                    'rename \"' + os.path.join(str(outputdir),output) + '\" \"' + os.path.basename(currentFile)+'\"')

    print(filesFound,'added into the shorten video queue')
    return command

def clahe_auto(directory):

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

def cropvid_auto(filenames,outputdir):
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



    total = width+height+topLeftX +topLeftY

    if total != 0:
        command = (str('ffmpeg -y -i ') + str(outputdir) + '\\' + str(videoName) + str(' -vf ') + str('"crop=') + str(width) + ':' + str(height) + ':' + str(topLeftX) + ':' + str(topLeftY) + '" ' + str('-c:v libx264 -c:a copy ') + str(os.path.join(outputdir, fileOutName)) + '\n'
            'move \"' + str(outputdir) + '\\' + videoName + '\" \"' + (outputdir) + '\\' + 'tmp\"' + '\n'
             'copy \"' + str(outputdir) + '\\' + os.path.basename(fileOutName) + '\" \"' + (outputdir) + '\\' + 'tmp\"' + '\n'
            'rename \"' + os.path.join(str(outputdir), os.path.basename(fileOutName)) + '\" \"' + os.path.basename(videoName) + '\"')
        print(videoName,'added into the crop video queue.')
        os.remove(filePath)
    elif total == 0:
        command = []
        print('nothing added to the script as no coordinates was selected')

    if os.path.exists(filePath):
        os.remove(filePath)
    return command

def cropvid_queue(filenames,outputdir):
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

    total = width+height+topLeftX +topLeftY

    if total != 0:
        command = (str('ffmpeg -y -i ')+ '"' + str(outputdir) + '\\' + str(videoName)+ '"' + str(' -vf ') + str('"crop=') + str(width) + ':' + str(height) + ':' + str(topLeftX) + ':' + str(topLeftY) + '" ' + str('-c:v libx264 -c:a copy ') + '"'+ str(os.path.join(outputdir, fileOutName))+ '"' + '\n'
            'move \"' + str(outputdir) + '\\' + videoName + '\" \"' + (outputdir) + '\\' + 'tmp\"' + '\n'
             'copy \"' + str(outputdir) + '\\' + os.path.basename(fileOutName) + '\" \"' + (outputdir) + '\\' + 'tmp\"' + '\n'
            'rename \"' + os.path.join(str(outputdir), os.path.basename(fileOutName)) + '\" \"' + os.path.basename(videoName) + '\"')
        print(videoName, 'added into the crop video queue.')
        os.remove(filePath)
        return command
    else:
        print('nothing added to the script as no coordinates was selected')
        pass
    if os.path.exists(filePath):
        os.remove(filePath)


def clahe_batch(directory):

    filesFound= []

    ########### FIND FILES ###########
    for i in os.listdir(directory):
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

def clahe_queue(files):

    filesFound= [files]
    os.chdir(os.path.dirname(files))
    print('Applying CLAHE, this might take awhile...')

    for i in filesFound:
        currentVideo = os.path.basename(i)
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
import os
import shutil
import cv2
import subprocess


def extract_frames_ini(directory):
    filesFound = []

    def execute(command):
        print(command)
        subprocess.call(command, shell=True, stdout=subprocess.PIPE)

    ########### FIND FILES ###########
    for i in os.listdir(directory):
        # if i.__contains__(".mp4"):
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

def copy_frame_folders(source,inifile):
    source = str(source) + '\\'
    dest = str(os.path.dirname(inifile))
    dest1 = str((dest) + '\\frames\\input')
    files = []

    ########### FIND FILES ###########
    for i in os.listdir(source):
        files.append(i)

    for f in files:
        filetocopy = source + f
        if os.path.exists(dest1 + '\\' + f):
            print(f, 'already exist in', dest1)

        elif not os.path.exists(dest1 + '\\' + f):
            print('Copying frames of',f)
            shutil.copytree(filetocopy, dest1+'\\'+f)
            print(f, 'copied to', dest1)

    print('Finished copying frames!')


def copy_multivideo_ini(inifile,source,filetype):
    print('Copying videos...')
    source = str(source)+'\\'
    dest = str(os.path.dirname(inifile))
    dest1 = str((dest)+ '\\' + 'videos')
    files = []

    ########### FIND FILES ###########
    for i in os.listdir(source):
        if i.__contains__(str('.'+filetype)):
            files.append(i)

    for f in files:
        filetocopy=source +'\\'+f
        if os.path.exists(dest1+'\\'+f):
            print(f, 'already exist in', dest1)

        elif not os.path.exists(dest1+'\\'+f):
            shutil.copy(filetocopy, dest1)
            print(f, 'copied to', dest1)

    print('Finished copying videos!')

def copy_allcsv_ini(inifile,source):
    print('Copying csv files...')
    source = str(source)+'\\'
    dest = str(os.path.dirname(inifile))
    dest1 = str((dest)+ '\\' + 'csv'+ '\\'+ 'input_csv')
    files = []
    print(dest1)
    print(source)
    ########### FIND FILES ###########
    for i in os.listdir(source):
        if i.__contains__(".csv"):
            files.append(i)

    for f in files:
        filetocopy=source +'\\'+f
        if os.path.exists(dest1+'\\'+f):
            print(f, 'already exist in', dest1)

        elif not os.path.exists(dest1+'\\'+f):
            shutil.copy(filetocopy, dest1)
            print(f, 'copied to', dest1)

    print('Finished copying csv files!')

def copy_singlevideo_ini(inifile,source):
    print('Copying video...')
    dest = str(os.path.dirname(inifile))
    dest1 = str((dest) + '\\' + 'videos')

    if os.path.exists(dest1+'\\'+os.path.basename(source)):
        print(os.path.basename(source), 'already exist in', dest1)
    else:
        shutil.copy(source, dest1)
        print(os.path.basename(source),'copied to',dest1)

    print('Finished copying video!')

def copy_singlecsv_ini(inifile,source):
    print('Copying csv file...')
    dest = str(os.path.dirname(inifile))
    dest1 = str((dest) + '\\' + 'csv' + '\\' + 'input_csv')

    if os.path.exists(dest1+'\\'+os.path.basename(source)):
        print(os.path.basename(source), 'already exist in', dest1)
    else:
        shutil.copy(source, dest1)
        print(os.path.basename(source),'copied to',dest1)

    print('Finished copying csv file!')
import os
import shutil
import cv2
import subprocess
import sys
import glob

def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

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
            nametoprint = os.path.join('',*(splitall(dest1)[-4:]))
            print(f, 'copied to', nametoprint)

    print('Finished copying frames.')

def copy_singlevideo_DPKini(inifile,source):
    try:
        print('Copying video...')
        dest = str(os.path.dirname(inifile))
        dest1 = str((dest) + '\\videos\\input')

        if os.path.exists(dest1+'\\'+os.path.basename(source)):
            print(os.path.basename(source), 'already exist in', dest1)
        else:
            shutil.copy(source, dest1)
            nametoprint = os.path.join('', *(splitall(dest1)[-4:]))
            print(os.path.basename(source),'copied to',nametoprint)

        print('Finished copying video.')
    except:
        pass

def copy_singlevideo_ini(inifile,source):
    try:
        print('Copying video...')
        dest = str(os.path.dirname(inifile))
        dest1 = os.path.join(dest, 'videos')

        if os.path.exists(dest1+'\\'+os.path.basename(source)):
            print(os.path.basename(source), 'already exist in', dest1)
        else:
            shutil.copy(source, dest1)
            nametoprint = os.path.join('', *(splitall(dest1)[-4:]))
            print(os.path.basename(source),'copied to',nametoprint)

        print('Finished copying video.')
    except:
        pass

def copy_multivideo_DPKini(inifile,source,filetype):
    try:
        print('Copying videos...')
        source = str(source)+'\\'
        dest = str(os.path.dirname(inifile))
        dest1 = str((dest)+ '\\videos\\input')
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
                nametoprint = os.path.join('', *(splitall(dest1)[-4:]))
                print(f, 'copied to', nametoprint)

        print('Finished copying videos.')
    except:
        print('Please select a folder and enter in the file type')


def copy_multivideo_ini(inifile,source,filetype):
    try:
        print('Copying videos...')
        dest1 = os.path.join(os.path.dirname(inifile), 'videos')
        files = glob.glob(source + '/*.' + filetype)
        for file in files:
            filebasename = os.path.basename(file)
            if os.path.exists(os.path.join(dest1, filebasename)):
                print(filebasename, 'already exist in project')
            elif not os.path.exists(os.path.join(dest1, filebasename)):
                shutil.copy(file, dest1)
                print(filebasename, 'copied to project_folder/videos')
        print('Finished copying videos.')
    except:
        print('Please select a folder and enter in the file type')


def copy_allcsv_ini(inifile,source):
    print('Copying csv files...')
    dest = str(os.path.dirname(inifile))
    dest1 = os.path.join(dest, 'csv', 'input_csv', 'original_filename')
    if not os.path.exists(dest1):
        os.makedirs(dest1)
    dest2 = os.path.join(dest, 'csv', 'input_csv')
    files = glob.glob(source + '/*.csv')

    ###copy files to project_folder
    for file in files:
        filebasename = os.path.basename(file)
        if os.path.exists(os.path.join(dest1, filebasename)):
            print(file, 'already exist in', dest1)
        elif not os.path.exists(os.path.join(dest1, filebasename)):
            shutil.copy(file, dest1)
            shutil.copy(file, dest2)
            print(filebasename, 'copied into SimBA project')
            if (('.csv') and ('DeepCut') in filebasename) or (('.csv') and ('DLC_') in filebasename):
                if (('.csv') and ('DeepCut') in filebasename):
                    newFname = str(filebasename.split('DeepCut')[0]) + '.csv'
                if (('.csv') and ('DLC_') in filebasename):
                    newFname = str(filebasename.split('DLC_')[0]) + '.csv'
                newFname = os.path.join(dest2, newFname)
                try:
                    os.rename(os.path.join(dest2,filebasename),os.path.join(dest2,newFname))
                except FileExistsError:
                    os.remove(os.path.join(dest2,filebasename))
                    print(filebasename + ' already exist in project')
            else:
                pass
    print('Finished importing tracking data.')

def copy_singlecsv_ini(inifile,source):
    print('Copying csv file...')
    dest = str(os.path.dirname(inifile))
    dest1 = os.path.join(dest, 'csv', 'input_csv', 'original_filename')
    if not os.path.exists(dest1):
        os.makedirs(dest1)
    dest2 = os.path.join(dest, 'csv', 'input_csv')
    filebasename = os.path.basename(source)
    if os.path.exists(os.path.join(dest1, filebasename)):
        print(filebasename, 'already exist in project')
    else:
        shutil.copy(source, dest1)
        shutil.copy(source, dest2)
        if (('.csv') and ('DeepCut') in filebasename) or (('.csv') and ('DLC_') in filebasename):
            if (('.csv') and ('DeepCut') in filebasename):
                newFname = str(filebasename.split('DeepCut')[0]) + '.csv'
            if (('.csv') and ('DLC_') in filebasename):
                newFname = str(filebasename.split('DLC_')[0]) + '.csv'
            newFname = os.path.join(dest2, newFname)
            try:
                os.rename(os.path.join(dest2, filebasename), os.path.join(dest2, newFname))
            except FileExistsError:
                os.remove(os.path.join(dest2, filebasename))
                print(filebasename + ' already exist in project')
        else:
            pass
    print('Finished importing tracking data.')
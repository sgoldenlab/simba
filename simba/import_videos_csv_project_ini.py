import os, glob
import shutil
import cv2
import subprocess
import sys
import glob, os
from configparser import ConfigParser, MissingSectionHeaderError, NoOptionError
import pandas as pd
from simba.extract_frames_fast import *
from os import listdir
from os.path import isfile, join


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

def extract_frames_ini(directory, configinifile):
    filesFound = glob.glob(directory + '/*.avi') + glob.glob(directory + '/*.mp4')
    configFile = str(configinifile)
    config = ConfigParser()
    try:
        config.read(configFile)
    except MissingSectionHeaderError:
        print('ERROR:  Not a valid project_config file. Please check the project_config.ini path.')
    projectPath = config.get('General settings', 'project_path')
    for video in filesFound:
        fName = os.path.basename(video)
        fName = fName[:-4]
        outputPath = os.path.join(projectPath, 'frames', 'input', fName)
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
            video_to_frames(video, outputPath, overwrite=True, every=1, chunk_size=1000)
            print('Frames were extracted for',os.path.basename(outputPath))

        else:
            print(os.path.basename(outputPath),'already exist, no action taken.')

    print('All frames were extracted.')

def copy_frame_folders(source,inifile):
    dest = str(os.path.dirname(inifile))
    dest1 = os.path.join(dest, 'frames', 'input')
    files = []

    ########### FIND FILES ###########
    for i in os.listdir(source):
        files.append(i)

    for f in files:
        filetocopy = os.path.join(source, f)
        if os.path.exists(os.path.join(dest1, f)):
            print(f, 'already exist in', dest1)

        elif not os.path.exists(os.path.join(dest1, f)):
            print('Copying frames of',f)
            shutil.copytree(filetocopy, os.path.join(dest1,f))
            nametoprint = os.path.join('',*(splitall(dest1)[-4:]))
            print(f, 'copied to', nametoprint)

    print('Finished copying frames.')

def copy_singlevideo_DPKini(inifile,source):
    try:
        print('Copying video...')
        dest = str(os.path.dirname(inifile))
        dest1 = str(os.path.join(dest, 'videos', 'input'))

        if os.path.exists(os.path.join(dest1, os.path.basename(source))):
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
        filename, file_extension = os.path.basename(source), os.path.splitext(source)[1]
        filename = os.path.splitext(filename)[0]
        file_extension = file_extension.lower()
        newFileName = os.path.join(filename + file_extension)
        dest1 = os.path.join(dest, 'videos', newFileName)
        if os.path.isfile(dest1):
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
        dest = str(os.path.dirname(inifile))
        dest1 = os.path.join(dest, 'videos', 'input')
        files = []

        ########### FIND FILES ###########
        for i in os.listdir(source):
            if i.__contains__(str('.'+ filetype)):
                files.append(i)

        for f in files:
            filetocopy = os.path.join(source, f)
            if os.path.exists(os.path.join(dest1,f)):
                print(f, 'already exist in', dest1)

            elif not os.path.exists(os.path.join(dest1, f)):
                shutil.copy(filetocopy, dest1)
                nametoprint = os.path.join('', *(splitall(dest1)[-4:]))
                print(f, 'copied to', nametoprint)

        print('Finished copying videos.')
    except:
        print('Please select a folder and enter in the file type')


def copy_multivideo_ini(inifile,source,filetype):
    if filetype not in ('avi', 'mp4', 'MP4', 'AVI'):
        print('SimBA only works with .avi and .mp4 files. Please convert your videos to .mp4 or .avi to continue. ')
    else:
        print('Copying videos...')
        vidFolderPath = os.path.join(os.path.dirname(inifile), 'videos')
        files = glob.glob(source + '/*.' + filetype)
        for file in files:
            filebasename, file_extension = os.path.basename(file), os.path.splitext(file)[1]
            filebasename = os.path.splitext(filebasename)[0]
            file_extension = file_extension.lower()
            newFileName = os.path.join(filebasename + file_extension)
            dest1 = os.path.join(vidFolderPath, newFileName)
            if os.path.isfile(dest1):
                print(filebasename, 'already exist in project')
                print(dest1)
            elif not os.path.isfile(dest1):
                shutil.copy(file, dest1)
                print(filebasename, 'copied to project_folder/videos.')
        print('Finished copying videos.')


def copy_allcsv_ini(inifile,source):
    configFile = str(inifile)
    config = ConfigParser()
    try:
        config.read(configFile)
    except MissingSectionHeaderError:
        print('ERROR:  Not a valid project_config file. Please check the project_config.ini path.')
    wfileType = config.get('General settings', 'workflow_file_type')
    print('Copying tracking files to project using ' + str(wfileType) + ' file format.')
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
                    newFname = str(filebasename.split('DeepCut')[0]) + '.' + wfileType
                if (('.csv') and ('DLC_') in filebasename):
                    newFname = str(filebasename.split('DLC_')[0]) + '.' + wfileType
            else:
                newFname = str(filebasename.split('.')[0]) + '.' + wfileType
            if wfileType == 'parquet':
                currDf = pd.read_csv(os.path.join(dest2, filebasename))
                currDf = currDf.apply(pd.to_numeric, errors='coerce')
                currDf.to_parquet(os.path.join(dest2,newFname))
                os.remove(os.path.join(dest2, filebasename))
            if wfileType == 'csv':
                try:
                    os.rename(os.path.join(dest2,filebasename),os.path.join(dest2,newFname))
                except FileExistsError:
                    os.remove(os.path.join(dest2,filebasename))
                    print(filebasename + ' already exist in project')
            else:
                pass


def copy_singlecsv_ini(inifile,source):
    configFile = str(inifile)
    config = ConfigParser()
    try:
        config.read(configFile)
    except MissingSectionHeaderError:
        print('ERROR:  Not a valid project_config file. Please check the project_config.ini path.')
    wfileType = config.get('General settings', 'workflow_file_type')
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
                newFname = str(filebasename.split('DeepCut')[0]) + '.' + wfileType
            if (('.csv') and ('DLC_') in filebasename):
                newFname = str(filebasename.split('DLC_')[0]) + '.' + wfileType
        else:
            newFname = str(filebasename.split('.')[0]) + '.' + wfileType
        if wfileType == 'parquet':
            currDf = pd.read_csv(os.path.join(dest2, filebasename))
            currDf = currDf.apply(pd.to_numeric, errors='coerce')
            currDf.to_parquet(os.path.join(dest2, newFname))
            os.remove(os.path.join(dest2, filebasename))
        if wfileType == 'csv':
            try:
                os.rename(os.path.join(dest2, filebasename), os.path.join(dest2, newFname))
            except FileExistsError:
                os.remove(os.path.join(dest2, filebasename))
                print(filebasename + ' already exist in project')
        else:
            pass

    print('Csv imported.')

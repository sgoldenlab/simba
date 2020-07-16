import yaml
import cv2
import os
import shutil
import deeplabcut

def generatetempyaml(yamlfile,videolist):
    try:
        #copy yaml and rename
        tempyaml = os.path.dirname(yamlfile) +'\\temp.yaml'
        shutil.copy(yamlfile,tempyaml)

        #adding new videos to tempyaml
        deeplabcut.add_new_videos(tempyaml,[videolist],copy_videos=True)

        with open(tempyaml) as f:
            read_yaml = yaml.load(f, Loader=yaml.FullLoader)

        original_videosets = read_yaml['video_sets'].keys()

        keys=[]
        for i in original_videosets:
            keys.append(i)
        #remove the original video to get only newly added videos
        read_yaml['video_sets'].pop(keys[0],None)

        with open(tempyaml, 'w') as outfile:
            yaml.dump(read_yaml, outfile, default_flow_style=False)

        print('temp.yaml generated.')
    except FileNotFoundError:
        print('Please select a video file.')

def generatetempyaml_multi(yamlfile,videolist):

    #copy yaml and rename
    tempyaml = os.path.dirname(yamlfile) +'\\temp.yaml'
    shutil.copy(yamlfile,tempyaml)


    deeplabcut.add_new_videos(tempyaml,videolist,copy_videos=True)

    with open(tempyaml) as f:
        read_yaml = yaml.load(f, Loader=yaml.FullLoader)

    original_videosets = read_yaml['video_sets'].keys()

    keys=[]
    for i in original_videosets:
        keys.append(i)


    read_yaml['video_sets'].pop(keys[0],None)

    with open(tempyaml, 'w') as outfile:
        yaml.dump(read_yaml, outfile, default_flow_style=False)


def updateiteration(yamlfile,iteration):
    yamlPath = yamlfile
    with open(yamlPath) as f:
        read_yaml = yaml.load(f, Loader=yaml.FullLoader)

    read_yaml["iteration"] = int(iteration)

    with open(yamlPath, 'w') as outfile:
        yaml.dump(read_yaml, outfile, default_flow_style=False)
    print('Iteration set to',iteration)

def update_init_weight(yamlfile,initweights):
    yamlPath=yamlfile
    initweights,initw_filetype = os.path.splitext(initweights)

    with open(yamlPath) as f:
        read_yaml = yaml.load(f, Loader=yaml.FullLoader)

    iteration = read_yaml['iteration']

    yamlfiledirectory = os.path.dirname(yamlfile)
    iterationfolder = yamlfiledirectory +'\\dlc-models\\iteration-' +str(iteration)
    projectfolder = os.listdir(iterationfolder)
    projectfolder = projectfolder[0]


    posecfg = iterationfolder + '\\' + projectfolder +'\\train\\' + 'pose_cfg.yaml'

    with open(posecfg) as g:
        read_cfg = yaml.load(g, Loader=yaml.FullLoader)

    read_cfg['init_weights'] = str(initweights)

    with open(posecfg, 'w') as outfile:
        yaml.dump(read_cfg, outfile, default_flow_style=False)
    print(os.path.basename(initweights),'selected')

def select_numfram2pick(yamlfile,numframe):
    try:
        yamlPath = yamlfile
        with open(yamlPath) as f:
            read_yaml = yaml.load(f, Loader=yaml.FullLoader)

        read_yaml["numframes2pick"] = int(numframe)

        with open(yamlPath, 'w') as outfile:
            yaml.dump(read_yaml, outfile, default_flow_style=False)
    except:
        print('Please load .yaml file and enter the number of frames to pick to proceed')

def add_multi_video_yaml(yamlfile,directory):
    filesFound = []

    ########### FIND FILES ###########
    for i in os.listdir(directory):
        if ('.avi' in i) or ('.mp4' in i):
            a=os.path.join(directory,i)
            filesFound.append(a)


    videoFilePathsToAdd = filesFound
    yamlPath = yamlfile
    cropLineList = []

    with open(yamlPath) as f:
        read_yaml = yaml.load(f, Loader=yaml.FullLoader)

    ####################### GET DIMENSIONS #############################
    for i in videoFilePathsToAdd:
        cap = cv2.VideoCapture(i)
        width = int(cap.get(3))  # float
        height = int(cap.get(4))  # float
        cropLine = [0, width, 0, height]
        cropLineList.append(cropLine)


    for i in range(len(videoFilePathsToAdd)):
        currVidPath = videoFilePathsToAdd[i]
        currCropLinePath = str(cropLineList[i])
        currCropLinePath = currCropLinePath.strip("[]")
        currCropLinePath = currCropLinePath.replace("'","")
        read_yaml["video_sets"].update({currVidPath: {'crop': currCropLinePath}})

    with open(yamlPath, 'w') as outfile:
        yaml.dump(read_yaml, outfile, default_flow_style=False)

    print(len(filesFound),' videos added to config.yaml')

def add_single_video_yaml(yamlfile,videofile):
    yamlPath = yamlfile
    cap = cv2.VideoCapture(videofile)
    width = int(cap.get(3))  # float
    height = int(cap.get(4))  # float
    cropLine = [0, width, 0, height]
    cropLine = str(cropLine)
    currCropLinePath = cropLine.strip("[]")
    currCropLinePath = currCropLinePath.replace("'", "")
    with open(yamlPath) as f:
        read_yaml = yaml.load(f, Loader=yaml.FullLoader)

    read_yaml["video_sets"].update({videofile: {'crop': currCropLinePath}})

    with open(yamlPath, 'w') as outfile:
        yaml.dump(read_yaml, outfile, default_flow_style=False)


def copycsv(inifile,source):
    source = str(source)+'\\'
    dest = str(os.path.dirname(inifile))
    dest1 = str((dest)+ '\\' + 'csv' + '\\' + 'input_csv')
    files = []
    print(dest1)
    print(source)
    ########### FIND FILES ###########
    for i in os.listdir(source):
        if i.__contains__(".csv"):
            files.append(i)

    for f in files:
        try:
            shutil.copy(source+f,dest1)
            print(f,'copied to',dest1)
        except:
            print(f,'already exist in',dest1)


    print('DONE!',len(files),'copied to',dest1)
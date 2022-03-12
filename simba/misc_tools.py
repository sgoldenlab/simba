from configparser import NoSectionError
from pylab import *

def check_directionality_viable(animalBpDict):

    directionalitySetting = True
    NoseCoords = []
    EarLeftCoords = []
    EarRightCoords = []
    for animal in animalBpDict:
        for bp_cord in ['X_bps', 'Y_bps']:
            bp_list = animalBpDict[animal][bp_cord]
            for bp_name in bp_list:
                bp_name_components = bp_name.split('_')
                bp_name_components = [x.lower() for x in bp_name_components]
                if ('nose' in bp_name_components):
                    NoseCoords.append(bp_name)
                if ('ear' in bp_name_components) and ('left' in bp_name_components):
                    EarLeftCoords.append(bp_name)
                if ('ear' in bp_name_components) and ('right' in bp_name_components):
                    EarRightCoords.append(bp_name)

    for cord in [NoseCoords, EarLeftCoords, EarRightCoords]:
        if len(cord) != len(animalBpDict.keys()) * 2:
            directionalitySetting = False

    if directionalitySetting:
        NoseCoords = [NoseCoords[i * 2:(i + 1) * 2] for i in range((len(NoseCoords) + 2 - 1) // 2)]
        EarLeftCoords = [EarLeftCoords[i * 2:(i + 1) * 2] for i in range((len(EarLeftCoords) + 2 - 1) // 2)]
        EarRightCoords = [EarRightCoords[i * 2:(i + 1) * 2] for i in range((len(EarRightCoords) + 2 - 1) // 2)]

    return directionalitySetting, NoseCoords, EarLeftCoords, EarRightCoords


def check_multi_animal_status(config, noAnimals):
    try:
        multiAnimalIDList = config.get('Multi animal IDs', 'id_list')
        multiAnimalIDList = multiAnimalIDList.split(",")
        if multiAnimalIDList[0] != '':
            multiAnimalStatus = True
            print('Applying settings for multi-animal tracking...')
            return multiAnimalStatus, multiAnimalIDList

        else:
            multiAnimalStatus = False
            multiAnimalIDList = []
            for animal in range(noAnimals):
                multiAnimalIDList.append('Animal_' + str(animal + 1))
            print('Applying settings for classical tracking...')
            return multiAnimalStatus, multiAnimalIDList

    except NoSectionError:
        multiAnimalIDList = []
        for animal in range(noAnimals):
            multiAnimalIDList.append('Animal_' + str(animal + 1))
        multiAnimalStatus = False
        print('Applying settings for classical tracking...')
        return multiAnimalStatus, multiAnimalIDList

def line_length(p, q, n, M, coord):
    Px = np.abs(p[0] - M[0])
    Py = np.abs(p[1] - M[1])
    Qx = np.abs(q[0] - M[0])
    Qy = np.abs(q[1] - M[1])
    Nx = np.abs(n[0] - M[0])
    Ny = np.abs(n[1] - M[1])
    Ph = np.sqrt(Px*Px + Py*Py)
    Qh = np.sqrt(Qx*Qx + Qy*Qy)
    Nh = np.sqrt(Nx*Nx + Ny*Ny)
    if (Nh < Ph and Nh < Qh and Qh < Ph):
        coord.extend((q[0], q[1]))
        return True, coord
    elif (Nh < Ph and Nh < Qh and Ph < Qh):
        coord.extend((p[0], p[1]))
        return True, coord
    else:
        return False, coord


# inifile = r"Z:\DeepLabCut\DLC_extract\Troubleshooting\DLC_two_mice\project_folder\project_config.ini"
#
# config = ConfigParser()
# config.read(inifile)
#
# try:
#     noAnimals = config.getint('ROI settings', 'no_of_animals')
# except NoOptionError:
#     noAnimals = config.getint('General settings', 'animal_no')
#
# try:
#     wfileType = config.get('General settings', 'workflow_file_type')
# except NoOptionError:
#     wfileType = 'csv'
#
# projectPath = config.get('General settings', 'project_path')
# csv_dir_in = os.path.join(projectPath, 'csv', 'outlier_corrected_movement_location')
# logFolderPath = os.path.join(projectPath, 'logs')
# vidInfPath = os.path.join(projectPath, 'logs', 'video_info.csv')
# vidinfDf = pd.read_csv(vidInfPath)
# vidinfDf["Video"] = vidinfDf["Video"].astype(str)
#
# multiAnimalStatus, multiAnimalIDList = check_multi_animal_status(config, noAnimals)
# Xcols, Ycols, Pcols = getBpNames(inifile)
# cMapSize = int(len(Xcols) + 1)
# colorListofList = createColorListofList(noAnimals, cMapSize)
# animalBpDict = create_body_part_dictionary(multiAnimalStatus, multiAnimalIDList, noAnimals, Xcols, Ycols, [], colorListofList)
# noAnimals = len(animalBpDict.keys())
# check_directionality_viable(animalBpDict)



import pandas as pd
import fnmatch
import os
import re
from configparser import ConfigParser
pd.options.mode.chained_assignment = None

loop = 1
files = []
currentEndAttackList = []
configFile = r"Z:\DeepLabCut\DLC_extract\New_082119\project_folder\project_config.ini"
config = ConfigParser()
config.read(configFile)
model_nos = config.getint('SML settings', 'No_targets')

csv_dir = config.get('General settings', 'csv_path')
csv_dir_in = os.path.join(csv_dir, 'features_extracted')
csv_dir_out = os.path.join(csv_dir, 'targets_inserted')
if not os.path.exists(csv_dir_out):
    os.makedirs(csv_dir_out)

behaviourFile = config.get('SML settings', 'target_file')
target_names = []
target_listoflists = []

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


########### GET MODEL PATHS AND NAMES ###########
for i in range(model_nos):
    currentModelNames = 'target_name_' + str(loop)
    currentModelNames = config.get('SML settings', currentModelNames)
    target_names.append(currentModelNames)
    loop+=1
loop = 0

xls = pd.ExcelFile(behaviourFile)
sheetnames = xls.sheet_names
for file in os.listdir(csv_dir_in):
    for sheet in sheetnames:
        if fnmatch.fnmatch(file, sheet + '*.csv'):
            file = os.path.join(csv_dir_in, file)
            files.append(file)
files.sort(key=natural_keys)

for x in range(len(sheetnames)):
    target_listoflists = []
    currentEndAttackList = []
    currentList = xls.parse(x)
    for i in range(model_nos):
        target_listoflists.append(list(currentList[target_names[i]]))
    currentPandas = pd.read_csv(files[x])
    videoNumber = os.path.basename(files[x])
    videoNumber = re.sub('[^0-9]','', videoNumber)
    currentPandas = currentPandas.reset_index()
    currentPandas = currentPandas.drop('index', axis=1)
    for k in range(model_nos):
        currentCol = target_names[k]
        currentPandas[currentCol] = 0
    currentPandas['video_no'] = videoNumber
    currentPandas['frames'] = currentPandas.index

    for pp in range(model_nos):
        currList = target_listoflists[pp]
        currentColumn = target_names[pp]
        currentPandas[currentColumn][currentPandas['frames'].isin(currList)] = 1

    #currentPandas['scorer'] = currentPandas['frames']
    currentPandas.rename(columns={'Unnamed: 0': 'scorer'}, inplace=True)
    outFileName = str(sheetnames[x]) + str('.csv')
    outFileName = os.path.join(csv_dir_out, outFileName)
    currentPandas.to_csv(outFileName, index=False)
    print(outFileName)
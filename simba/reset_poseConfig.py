import os
import pandas as pd
import shutil


def reset_DiagramSettings(master):

    #get difference of list
    def Diff(li1, li2):
        return (list(set(li1) - set(li2)))

    ## get pose config folder
    currDir = os.path.dirname(__file__)
    poseconfigDir = os.path.join(currDir,'pose_configurations')

    #archive folder
    #make main dir if not exist
    if not os.path.exists(os.path.join(currDir,'pose_configurations_archive')):
        os.makedirs(os.path.join(currDir,'pose_configurations_archive'))

    #archive folder
    noArchiveFolder = len(os.listdir(os.path.join(currDir,'pose_configurations_archive')))
    shutil.copytree(poseconfigDir,os.path.join(currDir,'pose_configurations_archive','archive_'+str(noArchiveFolder+1)))

    #reset bp names
    bpcsv = os.path.join(poseconfigDir,'bp_names','bp_names.csv')
    bpDf = pd.read_csv(bpcsv,header=None)
    bpDf = bpDf.iloc[0:12]
    bpDf.to_csv(bpcsv,index=False,header=False)

    #reset configuration names
    pcncsv = os.path.join(poseconfigDir,'configuration_names','pose_config_names.csv')
    pcnDf = pd.read_csv(pcncsv,header=None)
    pcnDf = pcnDf.iloc[0:12]
    pcnDf.to_csv(pcncsv,index=False,header=False)

    #reset no animals
    noAcsv = os.path.join(poseconfigDir,'no_animals','no_animals.csv')
    noADf = pd.read_csv(noAcsv,header=None)
    noADf = noADf.iloc[0:12]
    noADf.to_csv(noAcsv,index=False,header=False)

    #remove pictures
    picpath = os.path.join(poseconfigDir,'schematics')
    picpathlist = os.listdir(os.path.join(poseconfigDir,'schematics'))
    oripiclist = ['Picture1.png',
                     'Picture2.png',
                     'Picture3.png',
                     'Picture4.png',
                     'Picture5.png',
                     'Picture6.png',
                     'Picture7.png',
                     'Picture8.png',
                    'Picture9.png','Picture10.png','Picture11.png','Picture12.png']

    diffpics_list = Diff(picpathlist,oripiclist)

    for i in diffpics_list:
        os.remove(os.path.join(picpath,i))

    print('Settings reset.')
    master.destroy()

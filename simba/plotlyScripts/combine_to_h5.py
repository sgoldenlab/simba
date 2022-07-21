import os, glob
import pandas as pd
import h5py
import numpy as np
from configparser import ConfigParser

def mergeH5file(projectconfigini, inputList):
    print('Generating Plotly data file from SimBA...')
    config = ConfigParser()
    config.read(projectconfigini)
    projectPath = config.get('General settings', 'project_path')
    machineresultsFolder = os.path.join(projectPath, 'csv', 'machine_results')
    logsFolder = os.path.join(projectPath, 'logs', 'machine_results')





    # machine_results_dir = r"Z:\Classifiers\Attack\project_folder\csv\machine_results"
    # filesFound = glob.glob(machine_results_dir + '/*.csv')
    # timeBinsFile = r"Z:\Classifiers\Attack\project_folder\logs\Time_bins_ML_results_20200615145809.csv"
    # timeBinsDf = pd.read_csv(timeBinsFile, index_col=0)
    # sklearnFile = r"Z:\Classifiers\Attack\project_folder\logs\sklearn_20200615100537.csv"
    # sklearnDf = pd.read_csv(sklearnFile, index_col=0)
    # severityFile = r"Z:\Classifiers\Attack\project_folder\logs\severity_20200615145538.csv"
    # severityDf = pd.read_csv(severityFile, index_col=0)
    #
    # ExampleStorage = pd.HDFStore('ExampleStore_4.h5', table=True, complib='blosc:zlib', complevel=9)
    # ExampleStorage['sklearn_results'] = sklearnDf
    # ExampleStorage['time_bins_results'] = timeBinsDf
    #
    # infoDf = pd.DataFrame(columns = ['Classifier_names', "Used_threshold", "fps", "Classifier_link"])
    # infoDf.loc[len(infoDf)] = ['Attack', '0.5', "30", 'https://osf.io/stvhr/' ]
    # infoDf.loc[len(infoDf)] = ['Sniffing', '0.5', "30", 'https://osf.io/sn72j/']
    #
    # ExampleStorage['Dataset_info'] = infoDf
    #
    # for csvfile in filesFound:
    #     currDf = pd.read_csv(csvfile, index_col=0)
    #     currDf = currDf[['Attack', 'Probability_Attack']]
    #     currDf['Sniffing'] = np.random.randint(0, 1, currDf.shape[0])
    #     currDf['Probability_Sniffing'] = np.random.uniform(low=0.00, high=1.00, size=(currDf.shape[0],))
    #     vidName = os.path.basename(csvfile).replace('.csv', '')
    #     identifier = 'VideoData/' + vidName
    #     ExampleStorage[identifier] = currDf
    #     print(ExampleStorage[identifier])
    # ExampleStorage.close()











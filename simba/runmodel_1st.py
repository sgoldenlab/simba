import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=DeprecationWarning)
import pickle
from configparser import ConfigParser
import os
import pandas as pd
from simba.drop_bp_cords import drop_bp_cords
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def validate_model_one_vid_1stStep(inifile,csvfile,savfile):
    configFile = str(inifile)
    config = ConfigParser()
    config.read(configFile)
    sample_feature_file = str(csvfile)
    sample_feature_file_Name = os.path.basename(sample_feature_file)
    sample_feature_file_Name = sample_feature_file_Name.split('.', 1)[0]
    classifier_path = savfile
    classifier_name = os.path.basename(classifier_path).replace('.sav','')
    inputFile = pd.read_csv(sample_feature_file)
    inputFile = inputFile.loc[:, ~inputFile.columns.str.contains('^Unnamed')]
    outputDf = inputFile
    inputFileOrganised = drop_bp_cords(inputFile, inifile)
    print(inputFileOrganised)
    print('Running model...')
    clf = pickle.load(open(classifier_path, 'rb'))
    ProbabilityColName = 'Probability_' + classifier_name
    predictions = clf.predict_proba(inputFileOrganised)
    outputDf[ProbabilityColName] = predictions[:, 1]

    # CREATE LIST OF GAPS BASED ON SHORTEST BOUT

    vidInfPath = config.get('General settings', 'project_path')
    vidInfPath = os.path.join(vidInfPath, 'logs', 'video_info.csv')
    vidinfDf = pd.read_csv(vidInfPath)
    fps = vidinfDf.loc[vidinfDf['Video'] == str(sample_feature_file_Name.replace('.csv', ''))]
    try:
        fps = int(fps['fps'])
    except TypeError:
        print('Error: make sure all the videos that are going to be analyzed are represented in the project_folder/logs/video_info.csv file')


    outFname = sample_feature_file_Name + '.csv'
    csv_dir_out_validation = config.get('General settings', 'csv_path')
    csv_dir_out_validation = os.path.join(csv_dir_out_validation,'validation')
    if not os.path.exists(csv_dir_out_validation):
        os.makedirs(csv_dir_out_validation)
    outFname = os.path.join(csv_dir_out_validation, outFname)
    outputDf.to_csv(outFname)
    print('Predictions generated.')












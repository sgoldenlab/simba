import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=DeprecationWarning)
import pickle
from configparser import ConfigParser, NoOptionError
import os
import pandas as pd
from simba.drop_bp_cords import drop_bp_cords
import warnings
from simba.rw_dfs import *
from simba.drop_bp_cords import get_workflow_file_format
from simba.drop_bp_cords import get_fn_ext
from copy import deepcopy

warnings.simplefilter(action='ignore', category=FutureWarning)

def validate_model_one_vid_1stStep(inifile,csvfile,classifier_path):
    config = ConfigParser()
    config.read(str(inifile))

    project_path = config.get('General settings', 'project_path')
    csv_path = config.get('General settings', 'csv_path')
    csv_dir_out_validation = os.path.join(csv_path, 'validation')
    if not os.path.exists(csv_dir_out_validation):
        os.makedirs(csv_dir_out_validation)

    wfileType = get_workflow_file_format(config)
    dir_name, sample_feature_file_name, ext = get_fn_ext(str(csvfile))
    _, classifier_name, _ = get_fn_ext(classifier_path)
    inputFile = read_df(str(csvfile), wfileType)
    inputFile = inputFile.loc[:, ~inputFile.columns.str.contains('^Unnamed')]
    inputFile = inputFile.drop(['scorer'], axis=1, errors='ignore')
    outputDf = deepcopy(inputFile)
    inputFileOrganised = drop_bp_cords(inputFile, inifile)

    print('Running model...')
    clf = pickle.load(open(classifier_path, 'rb'))
    ProbabilityColName = 'Probability_' + classifier_name

    try:
        predictions = clf.predict_proba(inputFileOrganised)
    except ValueError as e:
        print(e.args)
        print('Mismatch in the number of features in input file and what is expected from the model in file ' + str(sample_feature_file_name) + ' and model ' + str(classifier_name))

    try:
        outputDf[ProbabilityColName] = predictions[:, 1]
    except IndexError as e:
        print(e.args)
        print('IndexError: Your classifier has not been created properly. See The SimBA GitHub FAQ page for more information and suggested fixes.')

    outFname = sample_feature_file_name + '.' + wfileType
    outFname = os.path.join(csv_dir_out_validation, outFname)
    save_df(outputDf, wfileType, outFname)
    print(outFname)
    print('Predictions generated for {}'.format(outFname))
    print('Proceed to specify threshold and minimum bout length and click on "Validate"')


# validate_model_one_vid_1stStep(r"Z:\DeepLabCut\DLC_extract\Troubleshooting\DLC_two_mice\project_folder\project_config.ini",
#                                r"Z:\DeepLabCut\DLC_extract\Troubleshooting\DLC_2_black_060320\project_folder\csv\features_extracted\Together_1.csv",
#                                r"Z:\DeepLabCut\DLC_extract\Troubleshooting\DLC_2_black_060320\models\Approach.sav")














import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=DeprecationWarning)
from configparser import ConfigParser, MissingSectionHeaderError
from sklearn.feature_selection import RFE, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import os
import pandas as pd
from dtreeviz.trees import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from yellowbrick.classifier import ClassificationReport
from xgboost.sklearn import XGBClassifier
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.tree import export_graphviz
from subprocess import call
import pickle
import csv
import warnings
import numpy as np
import eli5
from eli5.sklearn import PermutationImportance
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import precision_recall_curve
import graphviz
import graphviz.backend
from dtreeviz.shadow import *
from sklearn import tree
from drop_bp_cords import drop_bp_cords, GenerateMetaDataFileHeaders
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, cross_validate
# import timeit
from matplotlib import pyplot


inifile = r"Z:\Classifiers\Attack\project_folder\project_config.ini"
# startTime = timeit.default_timer()
configFile = str(inifile)
config = ConfigParser()
try:
    config.read(configFile)
except MissingSectionHeaderError:
    print('ERROR:  Not a valid project_config file. Please check the project_config.ini path.')
modelDir = config.get('SML settings', 'model_dir')
modelDir_out = os.path.join(modelDir, 'generated_models')
if not os.path.exists(modelDir_out):
    os.makedirs(modelDir_out)
tree_evaluations_out = os.path.join(modelDir_out, 'model_evaluations')
if not os.path.exists(tree_evaluations_out):
    os.makedirs(tree_evaluations_out)
try:
    model_nos = config.getint('SML settings', 'No_targets')
    data_folder = config.get('create ensemble settings', 'data_folder')
    model_to_run = config.get('create ensemble settings', 'model_to_run')
    classifierName = config.get('create ensemble settings', 'classifier')
    train_test_size = config.getfloat('create ensemble settings', 'train_test_size')
    pose_estimation_body_parts = config.get('create ensemble settings', 'pose_estimation_body_parts')
except ValueError:
    print('ERROR: Project_config.ini contains errors in the [create ensemble settings] or [SML settings] sections. Please check the project_config.ini file.')
log_path = config.get('General settings', 'project_path')
log_path = os.path.join(log_path, 'logs')
features = pd.DataFrame()

# READ IN DATA FOLDER AND REMOVE ALL NON-FEATURE VARIABLES (POP DLC COORDINATE DATA AND TARGET DATA)
print('Reading in ' + str(len(os.listdir(data_folder))) + ' annotated files...')
for i in os.listdir(data_folder):
    if i.__contains__(".csv"):
        currentFn = os.path.join(data_folder, i)
        df = pd.read_csv(currentFn, index_col=0)
        features = features.append(df, ignore_index=True)
features = features.loc[:, ~features.columns.str.contains('^Unnamed')]
features = features.drop(["scorer"], axis=1, errors='ignore')
totalTargetframes = features[classifierName].sum()
try:
    targetFrame = features.pop(classifierName).values
except KeyError:
    print('Error: the dataframe does not contain any target annotations. Please check the csv files in the project_folder/csv/target_inserted folder')
features = features.fillna(0)
features = drop_bp_cords(features, inifile)
target_names = []
loop = 1
for i in range(model_nos):
    currentModelNames = 'target_name_' + str(loop)
    print(currentModelNames)
    currentModelNames = config.get('SML settings', currentModelNames)
    if currentModelNames != classifierName:
        target_names.append(currentModelNames)
    loop += 1
print('# of models to be created: 1')

for i in range(len(target_names)):
    currentModelName = target_names[i]
    print('xxxxx')
    features = features.pop(currentModelName).values
class_names = class_names = ['Not_' + classifierName, classifierName]
feature_list = list(features)
print('# of features in dataset: ' + str(len(feature_list)))



############################################################################################################################
## DROP HIGHLY CORRELATED FEATURES
saveCorrelationMatrix = 'yes'
featureCorrelationMatrix = features.corr().abs()
col_corr = set()
print(len(features.columns))
for i in range(len(featureCorrelationMatrix.columns)):
    for j in range(i):
        if (featureCorrelationMatrix.iloc[i, j] >= 0.95) and (featureCorrelationMatrix.columns[j] not in col_corr):
            colname = featureCorrelationMatrix.columns[i]  # getting the name of column
            col_corr.add(colname)
            if colname in features.columns:
                del features[colname]  # deleting the column from the dataset
if saveCorrelationMatrix == 'yes':
    print('Saving correlation matrix image...')
    sns.set(font_scale=0.5)
    pyplot.figure(figsize=(30, 30))
    snsFig = sns.heatmap(featureCorrelationMatrix,xticklabels='auto', yticklabels='auto')
    snsFig.figure.savefig('file.png')
    plt.close('all')

data_train, data_test, target_train, target_test = train_test_split(features, targetFrame,test_size=train_test_size)
## Recursive Feature Elimination ( RFE )
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
if model_to_run == 'RF':
    print('Training model ' + str(classifierName) + '...')
    RF_n_estimators = config.getint('create ensemble settings', 'RF_n_estimators')
    RF_max_features = config.get('create ensemble settings', 'RF_max_features')
    RF_criterion = config.get('create ensemble settings', 'RF_criterion')
    RF_min_sample_leaf = config.getint('create ensemble settings', 'RF_min_sample_leaf')
    clf = RandomForestClassifier(n_estimators=RF_n_estimators, max_features=RF_max_features, n_jobs=-1,
                                 criterion=RF_criterion, min_samples_leaf=RF_min_sample_leaf, bootstrap=True,
                                 verbose=1)
    rfecv = RFECV(estimator=clf,step=2, cv=StratifiedKFold(n_splits=5,shuffle=False,random_state=1001).split(data_train, target_train),scoring='f1',n_jobs=1, verbose=2)
    rfecv.fit(data_train, target_train)
    sel_features = [f for f, s in zip(features, rfecv.support_) if s]
    ranking = pd.DataFrame({'Features': features.columns})
    ranking['Rank'] = np.asarray(rfecv.ranking_)
    ranking.sort_values('Rank', inplace=True)








#sns.heatmap(featureCorrelationMatrix, annot=True, cmap=plt.cm.Reds)
#plt.savefig('corr.png', dpi=300)
#plt.close('all')





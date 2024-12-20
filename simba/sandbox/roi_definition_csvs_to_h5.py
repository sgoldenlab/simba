import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
import os
import ast
import numpy as np
from simba.roi_tools.ROI_multiply import create_emty_df
from simba.utils.checks import check_file_exist_and_readable, check_if_dir_exists

RECTANGLES_CSV_PATH = '/Users/simon/Desktop/test/rectangles_20240314085952.csv'
POLYGONS_CSV_PATH = '/Users/simon/Desktop/test/polygons_20240314085952.csv'
CIRCLES_CSV_PATH = None
SAVE_DIRECTORY = '/Users/simon/Desktop/test'


##########################################################################################################################
check_if_dir_exists(in_dir=SAVE_DIRECTORY)
save_path = os.path.join(SAVE_DIRECTORY, 'ROI_definitions.h5')
store = pd.HDFStore(save_path, mode="w")
if RECTANGLES_CSV_PATH is not None:
    check_file_exist_and_readable(file_path=RECTANGLES_CSV_PATH)
    r = pd.read_csv(RECTANGLES_CSV_PATH, index_col=0).to_dict(orient='records')
    for i in range(len(r)):
        r[i]['Color BGR'] = ast.literal_eval(r[i]['Color BGR'])
        r[i]['Tags'] = ast.literal_eval(r[i]['Tags'])
else:
    r = create_emty_df(shape_type='rectangles')
r = pd.DataFrame.from_dict(r)

if POLYGONS_CSV_PATH is not None:
    p = pd.read_csv(POLYGONS_CSV_PATH, index_col=0).to_dict(orient='records')
    for i in range(len(p)):
        p[i]['Color BGR'] = ast.literal_eval(p[i]['Color BGR'])
        p[i]['Tags'] = ast.literal_eval(p[i]['Tags'])
        p[i]['vertices'] = np.fromstring(p[i]['vertices'].replace('\n', '').replace('[', '').replace(']', '').replace('  ', ' '), sep=' ').reshape((-1, 2)).astype(np.int32)
else:
    p = create_emty_df(shape_type='polygons')
p = pd.DataFrame.from_dict(p)

if CIRCLES_CSV_PATH is not None:
    c = pd.read_csv(CIRCLES_CSV_PATH, index_col=0)
    for i in range(len(c)):
        c[i]['Color BGR'] = ast.literal_eval(c[i]['Color BGR'])
        c[i]['Tags'] = ast.literal_eval(c[i]['Tags'])
else:
    c = create_emty_df(shape_type='circlesDf')
c = pd.DataFrame.from_dict(c)

store["rectangles"] = r
store["circleDf"] = c
store["polygons"] = p
store.close()
print(f'ROI CSV definitions joined and saved at {save_path}')

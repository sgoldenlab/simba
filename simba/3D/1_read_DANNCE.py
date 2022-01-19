import scipy.io
import numpy as np
import pandas as pd
import ast

dannce_dict = scipy.io.loadmat('/Volumes/GoogleDrive/My Drive/GitHub/SimBA_DANNCE/data/predictions.mat')
dannce_pred = dannce_dict['predictions']
bodypart_lst = [x[0] for x in ast.literal_eval(str(dannce_pred.dtype))][:-1]
curr_prob = pd.DataFrame(dannce_pred[0][0][22]).to_csv('Test.csv')

out_df_list = []
for bp in range(0, len(bodypart_lst)):
    curr_pred = pd.DataFrame(dannce_pred[0][0][bp], columns=[bodypart_lst[bp] + '_x', bodypart_lst[bp] + '_y', bodypart_lst[bp] + '_z'])
    curr_pred[bodypart_lst[bp] + '_p'] = 1
    out_df_list.append(curr_pred)
sorted_df = pd.concat(out_df_list, axis=1)

multiindex_cols = []
for column in range(len(sorted_df.columns)):
    multiindex_cols.append(tuple(('DANNCE_3D_data', 'DANNCE_3D_data', sorted_df.columns[column])))
sorted_df.columns = pd.MultiIndex.from_tuples(multiindex_cols, names=['scorer', 'bodypart', 'coords'])
sorted_df.to_csv('DANNCE_import.csv')





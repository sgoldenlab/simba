import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

def pearson_filter(projectPath, featuresDf, del_corr_status, del_corr_threshold, del_corr_plot_status):
    print('Reducing features. Correlation threshold: ' + str(del_corr_threshold))
    col_corr = set()
    corr_matrix = featuresDf.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= del_corr_threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
                if colname in featuresDf.columns:
                    del featuresDf[colname]
    if del_corr_plot_status == 'yes':
        print('Creating feature correlation heatmap...')
        dateTime = datetime.now().strftime('%Y%m%d%H%M%S')
        plt.matshow(featuresDf.corr())
        plt.tight_layout()
        plt.savefig(os.path.join(projectPath, 'logs', 'Feature_correlations_' + dateTime + '.png'), dpi=300)
        plt.close('all')
        print('Feature correlation heatmap .png saved in project_folder/logs directory')

    return featuresDf




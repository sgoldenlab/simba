from __future__ import print_function
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import os, glob
import matplotlib.pyplot as plt
import seaborn as sns
from drop_bp_cords import drop_bp_cords

dataFolder = r"Z:\DeepLabCut\DLC_extract\SimBA_JJ_tab_SN_master_JJ\features_scripts\test_files"
classifierName = 'Attack'
features = pd.DataFrame()

for i in os.listdir(dataFolder):
    if i.__contains__(".csv"):
        currentFn = os.path.join(dataFolder, i)
        df = pd.read_csv(currentFn, index_col=0)
        features = features.append(df, ignore_index=True)
        print(features)
features = features.loc[:, ~features.columns.str.contains('^Unnamed')]
features = features.drop(["scorer", 'Probability_' + classifierName], axis=1, errors='ignore')
targetFrame = features.pop(classifierName).values
tsne = TSNE(n_components=2, learning_rate=200, random_state=0, verbose=1)
data_train, data_test, target_train, target_test = train_test_split(features, targetFrame, test_size=0.05)

tsne_obj= tsne.fit_transform(data_train)

tsne_df = pd.DataFrame({'X':tsne_obj[:,0],
                        'Y':tsne_obj[:,1],
                        'digit':target_train})


clusters = len(tsne_df.digit.unique())
# https://cmdlinetips.com/2019/07/dimensionality-reduction-with-tsne/

sns.scatterplot(x="X", y="Y",
              hue="digit",
              palette=['purple','green'],
              legend='full',
              data=tsne_df);
plt.show()

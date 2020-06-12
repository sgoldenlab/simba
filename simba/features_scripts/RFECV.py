import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
from sklearn.feature_selection import RFECV

def perf_RFCVE(projectPath, RFCVE_CVs, RFCVE_step_size, clf, data_train, target_train):
    selector = RFECV(estimator=clf, step=RFCVE_step_size, cv=RFCVE_CVs, scoring='f1', verbose=1)
    selector = selector.fit(data_train, target_train)

    print(selector.support_)















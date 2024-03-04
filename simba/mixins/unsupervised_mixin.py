__author__ = "Simon Nilsson"

import glob
import os
import pickle
from datetime import datetime
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

import simba

try:
    from typing import Literal
except:
    from typing_extensions import Literal


from simba.unsupervised.enums import Unsupervised
from simba.utils.enums import Paths
from simba.utils.printing import SimbaTimer


class UnsupervisedMixin(object):

    def __init__(self):
        self.datetime = datetime.now().strftime("%Y%m%d%H%M%S")
        self.timer = SimbaTimer(start=True)
        model_names_dir = os.path.join(
            os.path.dirname(simba.__file__), Paths.UNSUPERVISED_MODEL_NAMES.value
        )
        self.model_names = list(
            pd.read_parquet(model_names_dir)[Unsupervised.NAMES.value]
        )

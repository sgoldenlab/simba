from abc import ABC, abstractmethod

import pandas as pd
from typing import Union
import os

class AbstractFeatureExtraction(ABC):

    @abstractmethod
    def __init__(self,
                 config_path: Union[str, os.PathLike]):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def save(self,
             data: pd.DataFrame,
             save_path: Union[str, os.PathLike]):
        pass
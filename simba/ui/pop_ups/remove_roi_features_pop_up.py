import os
from simba.mixins.config_reader import ConfigReader
from typing import Union
from tkinter import messagebox
try:
    from typing import Literal
except:
    from typing_extensions import Literal

class RemoveROIFeaturesPopUp():
    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 dataset: Literal['features_extracted', 'targets_inserted']):

        answer = messagebox.askyesno(title='REMOVE ROI FEATURES', message=f"Are tou sure you want to remove ROI \nfeatures from data in the \nproject_folder/csv/{dataset} folder?")

        if answer:
            config = ConfigReader(config_path=config_path)
            config.read_roi_data()
            if dataset == 'targets_inserted':
                config.remove_roi_features(config.targets_folder)
            if dataset == 'features_extracted':
                config.remove_roi_features(config.features_dir)
        else:
            pass

# test = RemoveROIFeaturesPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Nastacia_unsupervised/project_folder/csv/features_extracted',
#                          dataset='features_extracted')
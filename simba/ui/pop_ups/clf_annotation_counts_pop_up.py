import os
from typing import Union

from simba.labelling.extract_labelling_meta import AnnotationMetaDataExtractor
from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.utils.errors import CountError

VIDEO_SPLIT = 'INCLUDE ANNOTATION COUNTS SPLIT BY VIDEO'
BOUT_SPLIT = 'INCLUDE ANNOTATED BOUTS INFORMATION'

class ClfAnnotationCountPopUp(PopUpMixin, ConfigReader):
    """
    :example:
    >>> ClfAnnotationCountPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
    """
    def __init__(self, config_path: Union[str, os.PathLike]):
        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        self.config_path = config_path
        PopUpMixin.__init__(self, title='COUNT NUMBER OF ANNOTATIONS IN SIMBA PROJECT', config_path=config_path)
        if len(self.clf_names) == 0:
            raise CountError(msg=f'No classifier names associated with SimBA project {config_path}', source=self.__class__.__name__)
        if len(self.target_file_paths) == 0:
            raise CountError(msg=f'No data files found inside the {self.targets_folder} directory. ' f'Cannot analyze annotation count without annotated data', source=self.__class__.__name__)
        self.settings_dict = self.create_cb_frame(cb_titles=[VIDEO_SPLIT, BOUT_SPLIT], main_frm=self.main_frm, frm_title='SETTINGS')
        self.create_run_frm(run_function=self.run, title='RUN')
        self.main_frm.mainloop()

    def run(self):
        split_by_video = self.settings_dict[VIDEO_SPLIT].get()
        annotated_bouts = self.settings_dict[BOUT_SPLIT].get()
        annotation_extractor = AnnotationMetaDataExtractor(config_path=self.config_path, split_by_video=split_by_video, annotated_bouts=annotated_bouts)
        annotation_extractor.run()
        annotation_extractor.save()
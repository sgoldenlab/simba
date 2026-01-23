import os

import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from typing import List, Optional, Tuple, Union

import numpy as np

from simba.data_processors.cuda.utils import _is_cuda_available

try:
    from ultralytics.models.sam import SAM2VideoPredictor
except:
    SAM2VideoPredictor = None

from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists, check_instance, check_int,
                                check_valid_array, check_valid_tuple)
from simba.utils.data import resample_geometry_vertices
from simba.utils.enums import Formats, Options
from simba.utils.errors import (InvalidInputError, SimBAGPUError,
                                SimBAPAckageVersionError)
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_fn_ext, get_video_meta_data, write_df)


class SamInference():

    """
    :example:
    >>> i = SamInference(video_path=r"MyVideo",
    >>>                 labels=[[1]],
    >>>                 prompts=[[166, 428]],
    >>>                 weights_path=r"D:\yolo_weights\sam2.1_b.pt",
    >>>                 save_dir=r'C:\troubleshooting\sam_results',
    >>>                 names=('Animal1',))
    >>> i.run()


    .. video:: _static/img/sam_example.webm
       :loop:

    """
    def __init__(self,
                 video_path: Union[str, os.PathLike],
                 weights_path: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike],
                 prompts: Union[np.ndarray, List[List[int]]],
                 labels: Union[np.ndarray, List[List[int]]],
                 names: Tuple[str, ...],
                 imgsz: Optional[int] = 1024,
                 confidence: Optional[float] = 0.25,
                 vertice_cnt: Optional[int] = 100):

        if not _is_cuda_available()[0]:
            raise SimBAGPUError(msg='No GPU detected.', source=self.__class__.__name__)
        if SAM2VideoPredictor is None:
            raise SimBAPAckageVersionError(msg='ultralytics.models.sam.SAM2VideoPredictor package not detected.', source=self.__class__.__name__)
        if os.path.isdir(video_path):
            self.video_paths = find_files_of_filetypes_in_directory(directory=video_path, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, raise_error=True)
        elif os.path.isfile(video_path):
            self.video_paths = [video_path]
        else:
            raise InvalidInputError(msg=f'{video_path} is not a valid file or directory path.', source=self.__class__.__name__)
        _ = [get_video_meta_data(video_path=x) for x in self.video_paths]
        check_instance(source=f'{self.__class__.__name__} prompts', instance=prompts, accepted_types=(list, np.ndarray,), raise_error=True, warning=False)
        check_instance(source=f'{self.__class__.__name__} labels', instance=labels, accepted_types=(list, np.ndarray,), raise_error=True, warning=False)
        prompts = np.array(prompts) if isinstance(prompts, (list,)) else prompts
        labels = np.array(labels) if isinstance(labels, (list,)) else labels
        check_valid_array(data=prompts, source=f'{self.__class__.__name__} prompts', accepted_ndims=(1, 2, 3,), accepted_dtypes=Formats.INTEGER_DTYPES.value)
        check_valid_array(data=labels, source=f'{self.__class__.__name__} labels', accepted_ndims=(1, 2,), accepted_dtypes=Formats.INTEGER_DTYPES.value)
        check_file_exist_and_readable(file_path=weights_path)
        check_if_dir_exists(in_dir=save_dir, raise_error=True)
        check_int(name=f'{self.__class__.__name__} imgsz', value=imgsz, min_value=1)
        check_int(name=f'{self.__class__.__name__} vertice_cnt', value=vertice_cnt, min_value=3)
        check_float(name=f'{self.__class__.__name__} confidence', value=confidence, min_value=10e-6, max_value=1.0)
        check_valid_tuple(x=names, source=f'{self.__class__.__name__} names', accepted_lengths=(len(labels[0]),))
        self.animal_name_dict = {v: x for v, x in enumerate(names)}
        self.prompts, self.lbls, self.vertice_cnt = prompts, labels, vertice_cnt
        self.save_dir, self.names = save_dir, names
        self.overrides = dict(conf=confidence, task="segment", mode="predict", imgsz=imgsz, model="sam2_b.pt")
        self.predictor = SAM2VideoPredictor(overrides=self.overrides)
        self.vertice_col_names = ['FRAME', 'NAME']
        for i in range(self.vertice_cnt):
            self.vertice_col_names.append(f"VERTICE_{i}_x"); self.vertice_col_names.append(f"VERTICE_{i}_y")

    def run(self):
        for video_cnt, video_path in enumerate(self.video_paths):
            video_labels, video_prompts = self.lbls[video_cnt], self.prompts[video_cnt]
            video_meta_data = get_video_meta_data(video_path=video_path)
            video_results = []
            _, video_name, _ = get_fn_ext(filepath=video_path)
            save_path = os.path.join(self.save_dir, f'{video_name}.csv')
            results = self.predictor(source=video_path, points=video_prompts, labels=video_labels, stream=True)
            for frm_cnt, video_predictions in enumerate(results):
                for animal_name_cnt, animal_name in enumerate(self.names):
                    if video_predictions.names is None or animal_name_cnt not in video_predictions.names.keys():
                        mask = np.full(shape=(int(self.vertice_cnt*2)), fill_value=-1, dtype=np.int32)
                        mask = np.insert(mask, 0, animal_name_cnt)
                        mask = np.insert(mask, 0, int(frm_cnt))
                        video_results.append(mask)
                    else:
                        mask = video_predictions.masks[animal_name_cnt].xy[0].astype(np.int64)
                        mask[:, 0] = np.clip(mask[:, 0], 0, video_meta_data['width'])
                        mask[:, 1] = np.clip(mask[:, 1], 0, video_meta_data['height'])
                        mask = resample_geometry_vertices(vertices=mask.reshape(1, len(mask), 2), vertice_cnt=self.vertice_cnt)[0].flatten().astype(np.int64)
                        mask = np.insert(mask, 0, animal_name_cnt)
                        mask = np.insert(mask, 0, int(frm_cnt))
                        video_results.append(mask)
            video_results = pd.DataFrame(video_results, columns=self.vertice_col_names)
            #video_results['NAME'] = video_results['NAME'].map(self.animal_name_dict)
            video_results.to_csv(path_or_buf=save_path)
            #write_df(df=video_results, file_type='csv', save_path=save_path, multi_idx_header=False)






# i = SamInference(video_path=r"D:\platea\platea_videos\videos\clipped\10B_Mouse_5-choice_MustTouchTrainingNEWFINAL_a7_clipped_3.mp4",
#                  labels=[[1]],
#                  prompts=[[166, 428]],
#                  weights_path=r"D:\yolo_weights\sam2.1_b.pt",
#                  save_dir=r'C:\troubleshooting\sam_results',
#                  names=('Animal1',))
# i.run()
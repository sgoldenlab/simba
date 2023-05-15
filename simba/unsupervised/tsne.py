from datetime import datetime
import os
from simba.utils.enums import Paths
import pandas as pd
from simba.misc_tools import (check_file_exist_and_readable,
                              SimbaTimer)
# from simba.unsupervised.misc import (check_that_directory_is_empty,
#                                      check_directory_exists,
#                                      read_pickle,
#                                      write_pickle,
#                                      define_scaler,
#                                      drop_low_variance_fields,
#                                      find_low_variance_fields,
#                                      scaler_transform)
from sklearn.manifold import TSNE
import random
import simba

class TSNEGridSearch(object):
    def __init__(self,
                 data_path: str,
                 save_dir: str
                 ):
        pass

    #     self.datetime = datetime.now().strftime('%Y%m%d%H%M%S')
    #     self.data_path = data_path
    #     self.save_dir = save_dir
    #     check_file_exist_and_readable(file_path=self.data_path)
    #     check_directory_exists(directory=self.save_dir)
    #     check_that_directory_is_empty(directory=self.save_dir)
    #     model_names_dir = os.path.join(os.path.dirname(simba.__file__), Paths.UNSUPERVISED_MODEL_NAMES.value)
    #     self.model_names = list(pd.read_parquet(model_names_dir)['NAMES'])
    #     self.timer = SimbaTimer()
    #     self.timer.start_timer()
    #
    # def fit(self,
    #         hyperparameters: dict):
    #     self.hyp, self.low_var_cols = hyperparameters, []
    #     self.data = read_pickle(data_path=self.data_path)
    #     self.original_feature_names = self.data['DATA'].columns
    #     if self.hyp['variance']:
    #         self.low_var_cols = find_low_variance_fields(data=self.data['DATA'], variance=self.hyp['variance'])
    #         self.data['DATA'] = drop_low_variance_fields(data=self.data['DATA'], fields=self.low_var_cols)
    #     self.out_feature_names = [x for x in self.original_feature_names if x not in self.low_var_cols]
    #     self.scaler = define_scaler(scaler_name=self.hyp['scaler'])
    #     self.scaler.fit(self.data['DATA'])
    #     self.scaled_data = scaler_transform(data=self.data['DATA'],scaler=self.scaler)
    #     self.__fit_tsne()
    #
    # def __fit_tsne(self):
    #     self.model_timer = SimbaTimer()
    #     self.model_timer.start_timer()
    #     self.results = {}
    #     self.results['SCALER'] = self.scaler
    #     self.results['LOW_VARIANCE_FIELDS'] = self.low_var_cols
    #     self.results['ORIGINAL_FEATURE_NAMES'] = self.original_feature_names
    #     self.results['OUT_FEATURE_NAMES'] = self.out_feature_names
    #     self.results['VIDEO NAMES'] = self.data['VIDEO_NAMES']
    #     self.results['START FRAME'] = self.data['START_FRAME']
    #     self.results['END FRAME'] = self.data['END_FRAME']
    #     self.results['POSE'] = self.data['POSE']
    #     self.results['DATA'] = self.scaler
    #     self.results['CLASSIFIER'] = self.data['CLF']
    #     self.results['CLASSIFIER PROBABILITY'] = self.data['CLF_PROBABILITY']
    #
    #     for h_cnt, perplexity in enumerate(self.hyp['perplexity']):
    #         self.h_cnt = h_cnt
    #         self.parameters = {'perplexity': perplexity}
    #         embedder = TSNE(n_components=2, perplexity=perplexity)
    #         embedder.fit(self.scaled_data.values)
    #         self.results['MODEL'] = embedder
    #         self.results['HASH'] = random.sample(self.model_names, 1)[0]
    #         self.results['TYPE'] = 'TSNE'
    #         write_pickle(data=self.results, save_path=os.path.join(self.save_dir, '{}.pickle'.format(self.results['HASH'])))
    #     self.timer.stop_timer()
    #     print('SIMBA COMPLETE: {} TSNE model(s) saved in {} (elapsed time: {}s)'.format(str(len(self.hyp['perplexity'])), self.save_dir, self.timer.elapsed_time_str))
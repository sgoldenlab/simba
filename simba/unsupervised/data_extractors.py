import os.path

from simba.unsupervised.misc import (check_directory_exists,
                                     find_embedding,
                                     read_pickle)
import pandas as pd
import numpy as np

class DataExtractorMultipleModels(object):
    def __init__(self,
                 embeddings_dir: str,
                 save_dir: str,
                 clusterer_dir: str or None,
                 settings: dict):

        check_directory_exists(directory=embeddings_dir)
        check_directory_exists(directory=save_dir)
        embeddings = read_pickle(data_path=embeddings_dir)

        self.save_dir = save_dir
        parameters = None

        if clusterer_dir:
            check_directory_exists(directory=clusterer_dir)
            clusterers = read_pickle(data_path=clusterer_dir)
            for k, v in clusterers.items():
                self.embedding_name, self.cluster_name = v['HASH'], v['NAME']
                embedding = find_embedding(embeddings=embeddings, hash=v['HASH'])
                data = embedding['models'].embedding_
                cluster_data = v['model'].labels_.reshape(-1, 1).astype(np.int8)
                data = pd.DataFrame(np.hstack((data, cluster_data)), columns=['X', 'Y', 'CLUSTER'])
                data = pd.concat([embedding['VIDEO_NAMES'],
                                  embedding['FRAME_IDS'],
                                  data,
                                  embedding['CLF'],
                                  embedding['CLF_PROBABILITY']], axis=1)
                if settings['include_features']:
                    feature_vals = self.__get_feature_values(embedding=embedding, normalized=settings['scaled_features'])
                    data = pd.concat([data, feature_vals], axis=1)
                if settings['parameter_log']:
                    parameters = pd.DataFrame({**embedding['parameters'], **v['parameters']}, index=[0]).T.rename(columns={0: 'PARAMETERS'})
                    self.__save(data=data, parameters=parameters)
        else:
            for k, v in embeddings.items():
                data = v['models'].embedding_
                self.embedding_name = v['HASH']
                data = pd.concat([v['VIDEO_NAMES'],
                                  v['FRAME_IDS'],
                                  data,
                                  v['CLF'],
                                  v['CLF_PROBABILITY']], axis=1)
                if settings['include_features']:
                    feature_vals = self.__get_feature_values(embedding=v, normalized=settings['scaled_features'])
                    data = pd.concat([data, feature_vals], axis=1)
                if settings['parameter_log']:
                    parameters = pd.DataFrame({**v['parameters']}, index=[0]).T.rename(columns={0: 'PARAMETERS'})
                self.__save(data=data, parameters=parameters)

    def __save(self, data: pd.DataFrame, parameters: pd.DataFrame or None):
        data_path = os.path.join(self.save_dir, f'{self.embedding_name}_{self.cluster_name}.csv')
        data.to_csv(data_path)
        if parameters is not None:
            parameters_path = os.path.join(self.save_dir, f'{self.embedding_name}_{self.cluster_name}_parameters.csv')
            parameters.to_csv(parameters_path)

    def __get_feature_values(self, embedding: object, normalized: bool):
        feature_vals = embedding['models']._raw_data
        if not normalized:
            feature_vals = embedding['scaler'].inverse_transform(feature_vals)
        return pd.DataFrame(feature_vals, columns=embedding['OUT_FEATURE_NAMES'])





# _ =DataExtractorMultipleModels(clusterer_dir='/Users/simon/Desktop/envs/troubleshooting/unsupervised/cluster_models/',
#                         embeddings_dir='/Users/simon/Desktop/envs/troubleshooting/unsupervised/dr_models/',
#                         save_dir='/Users/simon/Desktop/envs/troubleshooting/unsupervised/dr_models',
#                         settings = {'include_features': True, 'normalized_features': False, 'parameter_log': True})
#

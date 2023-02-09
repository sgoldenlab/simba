import pickle
import os, glob


def read_pickle(data_path: str):
    if os.path.isdir(data_path):
        data = {}
        files_found = glob.glob(data_path + '/*.pickle')
        for file_cnt, file_path in enumerate(files_found):
            with open(file_path, 'rb') as f:
                data[file_cnt] = pickle.load(f)
    if os.path.isfile(data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

    return data

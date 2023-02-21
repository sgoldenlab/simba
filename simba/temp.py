import pandas as pd
import numpy as np
from simba.misc_tools import detect_bouts


def bout_train_test_splitter(x_df: pd.DataFrame,
                             y_df: pd.Series,
                             test_size: float):

    def find_bouts(s: pd.Series):
        test_bouts_frames, train_bouts_frames = [], []
        bouts = detect_bouts(pd.DataFrame(s), target_lst=pd.DataFrame(s).columns, fps=-1)
        bouts = list(bouts.apply(lambda x: list(range(int(x['Start_frame']), int(x['End_frame']) + 1)), 1).values)
        test_bouts_idx = np.random.choice(np.arange(0, len(bouts)), int(len(bouts) * test_size))
        train_bouts_idx = np.array([x for x in list(range(len(bouts))) if x not in test_bouts_idx])
        for i in range(0, len(bouts)):
            if i in test_bouts_idx:
                test_bouts_frames.append(bouts[i])
            if i in train_bouts_idx:
                train_bouts_frames.append(bouts[i])
        return [i for s in test_bouts_frames for i in s], [i for s in train_bouts_frames for i in s]

    test_bouts_frames, train_bouts_frames = find_bouts(s=y_df)
    test_nonbouts_frames, train_nonbouts_frames = find_bouts(s=np.logical_xor(y_df, 1).astype(int))
    x_train = x_df[x_df.index.isin(train_bouts_frames + train_nonbouts_frames)].values
    x_test = x_df[x_df.index.isin(test_bouts_frames + test_nonbouts_frames)].values
    y_train = y_df[y_df.index.isin(train_bouts_frames + train_nonbouts_frames)].values
    y_test = y_df[y_df.index.isin(test_bouts_frames + test_nonbouts_frames)].values

    return x_train, x_test, y_train, y_test

    #
    #
    #
    # non_bouts = detect_bouts(pd.DataFrame(inverted_y_df), target_lst=pd.DataFrame(y_df).columns, fps=-1)
    # non_bouts = np.array(list(non_bouts.apply(lambda x: list(range(int(x['Start_frame']), int(x['End_frame']) + 1)), 1)))
    # test_nonbout_idx = np.random.choice(np.arange(0, bouts.shape[0]), int(bouts.shape[0] * test_size))


    #print(non_bouts)



    #train_absent_frames =




    #self.x_train, self.x_test, self.y_train, self.y_test



# files = glob.glob('/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/csv/targets_inserted' + '/*')
# df_list = []
# for file_p in files:
#     df_list.append(pd.read_csv(file_p, index_col=0))
# df = pd.concat(df_list, axis=0)
# target_df = df.pop('Attack')
# bout_train_test_splitter(x_df=df,
#                          y_df=target_df,
#                          test_size=0.20)
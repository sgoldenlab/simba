import os
import pandas as pd
import glob
import itertools
from configparser import ConfigParser, NoSectionError, NoOptionError
import copy
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

class FSTCC_perform(object):
    def __init__(self, config_path, time_window, behavior_list, create_graphs):
        self.time_delta, self.behavior_list, self.graph_status = int(time_window), behavior_list, create_graphs
        config = ConfigParser()
        config.read(config_path)
        project_path = config.get('General settings', 'project_path')
        video_info_path = os.path.join(project_path, 'logs', 'video_info.csv')
        data_folder = os.path.join(project_path, 'csv', 'machine_results')
        self.logs_folder = os.path.join(project_path, 'logs')
        self.date_time_string = datetime.now().strftime('%Y%m%d%H%M%S')
        try:
            self.wfileType = config.get('General settings', 'workflow_file_type')
        except NoOptionError:
            self.wfileType = 'csv'
        self.video_info_df = pd.read_csv(video_info_path)
        self.video_info_df["Video"] = self.video_info_df["Video"].astype(str)
        self.data_file_list = glob.glob(data_folder + "/*." + self.wfileType)
        self.processed_video_list = []
        self.session_length_list = []
        self.fps_list = []

    def calculate_sequence_data(self):
        self.out_df_list = []
        for file in self.data_file_list:
            video_out_df = pd.DataFrame( columns=['First behaviour', 'First behaviour start frame', 'First behavior end frame',
                         'Second behaviour', 'Second behaviour start frame',
                         'Difference: first behavior start to second behavior start',
                         'Time 2nd behaviour start to time window end'])
            video_name = os.path.basename(file)
            self.processed_video_list.append(video_name.replace('.' + self.wfileType, ''))
            print('Analyzing behavioral sequences: ' + str(video_name.replace('.' + self.wfileType, '')) + '...')
            video_settings = self.video_info_df.loc[self.video_info_df['Video'] == str(video_name.replace('.' + self.wfileType, ''))]
            try:
                self.fps = int(video_settings['fps'])
                self.fps_list.append(self.fps)
            except TypeError:
                print('Error: make sure all the videos that are going to be analyzed are represented in the project_folder/logs/video_info.csv file')
            self.frames_in_window = (self.fps / 1000) * self.time_delta
            if self.wfileType == 'csv': curr_df = pd.read_csv(file, index_col=0)
            if self.wfileType == 'parquet': curr_df = pd.read_parquet(file)
            self.session_length_list.append(len(curr_df))
            self.curr_df = curr_df[self.behavior_list]
            for currbehav in self.behavior_list:
                groupDf = pd.DataFrame()
                v = (self.curr_df[currbehav] != self.curr_df[currbehav].shift()).cumsum()
                u = self.curr_df.groupby(v)[currbehav].agg(['all', 'count'])
                m = u['all'] & u['count'].ge(1)
                groupDf['groups'] = self.curr_df.groupby(v).apply(lambda x: (x.index[0], x.index[-1]))[m]
                for first_behav_row in groupDf.itertuples(index=False):
                    first_behav_start, first_behav_end = first_behav_row[0][0], first_behav_row[0][1]
                    appendFlag = False
                    for second_behav_row in self.curr_df.loc[first_behav_start:first_behav_end + self.frames_in_window].itertuples():
                        behavior_list = second_behav_row[1:len(self.behavior_list)+1]
                        behavior_sum_list = sum(behavior_list)
                        if behavior_sum_list > 0:
                            second_behav_start, appendrow = second_behav_row[0], []
                            relevant_index = behavior_list.index(1)
                            first_behavior, second_behavior = str(currbehav), str(self.behavior_list[relevant_index])
                            timeBetweenBehaviors = (second_behav_start - first_behav_start)
                            SecondndBehavStartToTimeWindowEnd = ((first_behav_end + timeBetweenBehaviors) - second_behav_start)
                            appendrow = [first_behavior, first_behav_start, first_behav_end, second_behavior, second_behav_start, timeBetweenBehaviors, SecondndBehavStartToTimeWindowEnd]
                            if (first_behavior == second_behavior) and (second_behav_row[0] >= first_behav_start and second_behav_row[0] <= first_behav_end):
                                pass
                            else:
                                video_out_df.loc[len(video_out_df)] = appendrow
                                appendFlag = True
                    if appendFlag == False:
                        appendrow = [str(currbehav), first_behav_start, first_behav_end, 'None', second_behav_row[0], 'None', 'None']
                        video_out_df.loc[len(video_out_df)] = appendrow
            self.out_df_list.append(video_out_df)
        self.remove_duplicates()

    def remove_duplicates(self):
        for video_index, video_df in enumerate(self.out_df_list):
            video_df['Second behaviour start frame'] = video_df['Second behaviour start frame'].astype(int)
            duplicate = video_df[['First behaviour', 'First behaviour start frame', 'First behavior end frame']]
            duplicatedIndex = duplicate.duplicated()
            duplicatedIndexList = duplicatedIndex.index[duplicatedIndex == True].tolist()

            for indexVal in duplicatedIndexList:
                try:
                    currStartBehav, currStartFrame, currEndFrame = video_df.loc[indexVal, 'First behaviour'], int(video_df.loc[indexVal, 'First behaviour start frame']), int(video_df.loc[indexVal, 'First behavior end frame'])
                    duplicateRows = video_df.loc[(video_df['First behaviour'] == currStartBehav) & (video_df['First behaviour start frame'] == currStartFrame)]
                    index2Keep = int(duplicateRows[['Second behaviour start frame']].idxmin())
                    duplicateRowsList = list(duplicateRows.index)
                    duplicateRowsList.remove(index2Keep)
                    for value in duplicateRowsList: video_df.drop(value, inplace=True)
                except (KeyError, IndexError):
                    pass
            self.out_df_list[video_index] = video_df

    def calculate_FSTCC(self):
        behavior_combinations = list(itertools.combinations(self.behavior_list, 2))
        behavior_combinations.extend(([t[::-1] for t in behavior_combinations]))
        self.out_dict = {}
        for video_index, video_df in enumerate(self.out_df_list):
            video_name = self.processed_video_list[video_index]
            print('Analyzing FSTTCs: ' + str(video_name) + '...')
            session_length_ms = int((1000 / self.fps_list[video_index]) * self.session_length_list[video_index])
            video_df['Total_window_size'] = ((video_df['First behavior end frame'] - video_df['First behaviour start frame']) + self.time_delta)
            trigger_behaviors = video_df.groupby(['First behaviour', 'First behaviour start frame', 'First behavior end frame']).size().reset_index().rename(columns={0: 'count'})
            self.out_dict[video_index] = {'Video': video_name}
            for behavior_sequence in behavior_combinations:
                behavior_1, behavior_2 = behavior_sequence[0], behavior_sequence[1]
                out_dict_key = 'FSTTC' + '_' + behavior_1 + '_' + behavior_2
                no_behaviour_1 = len(trigger_behaviors[trigger_behaviors['First behaviour'] == behavior_1])
                no_behaviour_1_2 = len(video_df[(video_df['First behaviour'] == behavior_1) & (video_df['Second behaviour'] == behavior_2)])
                if (no_behaviour_1 > 0) & (no_behaviour_1_2 == 0):
                    self.out_dict[video_index][out_dict_key] = 0.0
                if (no_behaviour_1 == 0) & (no_behaviour_1_2 == 0):
                    self.out_dict[video_index][out_dict_key] = 'No events'
                if no_behaviour_1_2 > 0:
                    P = no_behaviour_1_2 / no_behaviour_1
                    behaviour_1_2_df = video_df[(video_df['First behaviour'] == behavior_1) & (video_df['Second behaviour'] == behavior_2)]
                    Ta = sum(behaviour_1_2_df['Total_window_size']) / session_length_ms
                    dfWithNonRemoved = behaviour_1_2_df[behaviour_1_2_df['Second behaviour'] != 'None']
                    dfWithNonRemoved['Time 2nd behaviour start to time window end'] = pd.to_numeric(dfWithNonRemoved['Time 2nd behaviour start to time window end'])
                    Tb = sum(dfWithNonRemoved['Time 2nd behaviour start to time window end']) / session_length_ms
                    FSTTC = 0.5 * ((P - Tb) / (1 - (P * Tb)) + ((P - Ta) / (1 - (P * Ta))))
                    self.out_dict[video_index][out_dict_key] = FSTTC

    def save_results(self):
        headers = []
        for key in self.out_dict[0]: headers.append(key)
        self.out_df = pd.DataFrame(columns=headers)
        for entry in self.out_dict:
            out_list = []
            for val in self.out_dict[entry]:
                out_list.append(self.out_dict[entry][val])
            self.out_df.loc[len(self.out_df)] = out_list
        file_name = 'FSTTC_' + self.date_time_string + '.csv'
        file_save_path = os.path.join(self.logs_folder, file_name)
        self.out_df.to_csv(file_save_path)
        print('FSTTC data saved at' + file_save_path)

    def plot_results(self):
        if self.graph_status:
            print('Plotting results...')
            melted_df = pd.melt(self.out_df, id_vars=['Video'], var_name='Transition', value_name='FSTTC')
            melted_df['Transition'] = melted_df['Transition'].replace({'FSTTC_': ''}, regex=True)
            melted_df = melted_df[melted_df['Transition'] != 'No events'].reset_index(drop=True)
            melted_df['FSTTC'] = pd.to_numeric(melted_df['FSTTC'], errors='coerce')
            figure_FSCTT = sns.violinplot(x="Transition", y="FSTTC", data=melted_df)
            figure_FSCTT.set_xticklabels(figure_FSCTT.get_xticklabels(), rotation=45, size=7)
            figure_FSCTT.figure.set_size_inches(13.7, 8.27)
            save_plot_name = 'FSTTC_' + self.date_time_string + '.png'
            save_plot_path = os.path.join(self.logs_folder, save_plot_name)
            figure_FSCTT.figure.savefig(save_plot_path, bbox_inches = "tight")
            print('FSTTC figure saved at' + save_plot_path)

# if __name__ == "__main__":
#     config_path = r"Z:\Classifiers\100620_all\project_folder\project_config.ini"
#     behavior_list = ['Lateral_threat', 'Attack', 'Escape', 'Defensive']
#     time_delta = 2000
#     create_graphs = True
#     FSTCC_performer = FSTCC_perform(config_path, time_delta, behavior_list, create_graphs)
#     FSTCC_performer.calculate_sequence_data()
#     FSTCC_performer.calculate_FSTCC()
#     FSTCC_performer.save_results()
#     FSTCC_performer.plot_results()






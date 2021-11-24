import pandas as pd
import os, glob
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib
import numpy as np
from configparser import ConfigParser, MissingSectionHeaderError, NoOptionError, NoSectionError
from simba.drop_bp_cords import *
from simba.rw_dfs import *
matplotlib.use('Agg')


def pup_retrieval_1(cofigini, prob_pup, prob_mother, dist_start_crit, carry_frames_seconds, smooth_factor, corenest_name, nest_name, dam_name, pup_name, smooth_function, max_time, carry_classifier_name, approach_classifier_name, dig_classifier_name):

    # cofigini = r"Z:\DeepLabCut\DLC_extract\Troubleshooting\Pup_retrieval_1\project_folder\project_config.ini"
    # prob_pup = 0.025
    # prob_mother = 0.5
    # dist_start_crit = 80
    # carry_frames_seconds = 3
    # smooth_factor = 5
    # corenest_name = 'corenest'
    # nest_name = 'nest'
    # dam_name = '1_mother'
    # pup_name = '2_pup'
    # smooth_function = 'gaussian'
    # max_time = 90
    # carry_classifier_name = 'carry'
    # approach_classifier_name = 'approach'
    # dig_classifier_name = 'doggy'

    config = ConfigParser()
    configFile = str(cofigini)
    try:
        config.read(configFile)
    except MissingSectionHeaderError:
        print('ERROR:  Not a valid project_config file. Please check the project_config.ini path.')

    project_path = config.get('General settings', 'project_path')
    no_classifiers = config.getint('SML settings', 'no_targets')
    animals_no = config.getint('General settings', 'animal_no')
    classifier_list = [carry_classifier_name, approach_classifier_name, dig_classifier_name]
    for i in range(len(classifier_list)):
        if classifier_list[i].lower() == 'none': classifier_list[i] = 'none'


    for clf in range(no_classifiers):
        classifier_list.append(config.get('SML settings', 'target_name_' + str(clf+1)))

    dist_pup_cornest_col = corenest_name + '_ ' + pup_name + '_distance'
    dist_mother_cornest_col = corenest_name + '_' + dam_name + '_distance'
    pup_in_corenest_col = corenest_name + '_ ' + pup_name + '_in_zone'
    pup_in_nest_col = nest_name + '_ ' + pup_name + '_in_zone'

    features_extracted_path = os.path.join(project_path, 'csv', 'features_extracted')
    machine_results_path = os.path.join(project_path, 'csv', 'machine_results')
    logs_dir_path = os.path.join(project_path, 'logs')
    features_files = glob.glob(features_extracted_path + '/*.csv')

    video_info_df = pd.read_csv(os.path.join(logs_dir_path, 'video_info.csv'))
    video_info_df["Video"] = video_info_df["Video"].astype(str)
    date_time_string = datetime.now().strftime('%Y%m%d%H%M%S')

    Xcols, Ycols, Pcols = getBpNames(cofigini)
    multiAnimalIDList = config.get('Multi animal IDs', 'id_list')

    if not multiAnimalIDList:
        multiAnimalIDList = []
        for animal in range(animals_no):
            multiAnimalIDList.append('Animal_' + str(animal + 1))
        multiAnimalStatus = False
        print('Applying settings for classical tracking...')

    else:
        multiAnimalIDList = multiAnimalIDList.split(",")
        multiAnimalStatus = True
        print('Applying settings for multi-animal tracking...')

    animalBpDict = create_body_part_dictionary(multiAnimalStatus, multiAnimalIDList, animals_no, Xcols, Ycols, [], [])

    body_parts_mother_x = animalBpDict[dam_name]['X_bps']
    body_parts_pup_x = animalBpDict[pup_name]['X_bps']
    body_parts_mother = [sub.replace('_x', '_p') for sub in body_parts_mother_x]
    body_parts_pup = [sub.replace('_x', '_p') for sub in body_parts_pup_x]

    colnames = ["Video", "Pup in nest (Frame)", "Pup in nest (s)", "Min_distance (pup to corenest)", "Reason (pup in nest)", "Pup in core-nest (frame)", "Pup in core-nest (s)", "Reason (pup in core-nest)"]
    for classifier in classifier_list:
        col_name_1, col_name_2 = classifier + ' (total time)', classifier + ' (before retrieval)'
        col_name_3, col_name_4 = classifier + ' (latency to first event)', classifier + ' (number of events)'
        col_name_5, col_name_6 = classifier + ' (Mean duration)', classifier + ' (mean interval)'
        colnames.extend((col_name_1, col_name_2, col_name_3, col_name_4, col_name_5, col_name_6))

    if (approach_classifier_name != 'none') and (dig_classifier_name != 'none'):
        colnames.extend(("Dig_time_after_approach_before_retrieval"," Dig_events_after_approach", "Mean_duration_dig_after_approach"))
    out_df = pd.DataFrame(columns=colnames)

    def generate_figure(dataframe, ycolname, xlabel, ylabel, title, date_time_string, hue, video_base_name):
        current_figure = sns.scatterplot(x=dataframe.index, y=dataframe[ycolname], hue=dataframe[hue], legend=False)
        current_figure.set(xlabel=xlabel, ylabel=ylabel, title=title)
        save_plot_name = title + '_' + video_base_name + '_' + date_time_string + '.png'
        save_plot_path = os.path.join(logs_dir_path, video_base_name)
        if not os.path.exists(save_plot_path): os.makedirs(save_plot_path)
        image_save_path = os.path.join(save_plot_path, save_plot_name)
        current_figure.figure.savefig(image_save_path, bbox_inches="tight")
        current_figure.clear()

    for counter, video_file in enumerate(features_files):
        video_base_name = os.path.basename(video_file)
        video_base_name_no_file_ending = video_base_name.replace('.csv', '')
        print('Processing video ' + video_base_name_no_file_ending + '. Video ' + str(counter+1) + ' / ' + str(len(features_files)))
        curr_feature_df = pd.read_csv(video_file, index_col=0)
        curr_machine_results_df = pd.read_csv(os.path.join(machine_results_path, video_base_name), usecols=classifier_list)
        curr_df = pd.concat([curr_feature_df,curr_machine_results_df], axis=1)
        del curr_feature_df; del curr_machine_results_df
        curr_df = curr_df.fillna(method='ffill')
        video_fps = video_info_df['fps'][video_info_df['Video'] == video_base_name.replace('.csv', '')].values[0]

        if max_time != 'none':
            max_frames = int(video_fps * max_time)
            if max_frames < len(curr_df):
                curr_df = curr_df.head(max_frames)

        for col_name in [dist_pup_cornest_col, dist_mother_cornest_col, pup_in_corenest_col, pup_in_nest_col]:
            if col_name not in curr_df:
                print('SimBA could not find a column named ' + str(col_name) + '. Make sure you have entered the correct ROI/pup/dam names')


        carry_frames = int(video_fps * carry_frames_seconds)

        curr_df['mean_p_mother'] = curr_df[body_parts_mother].mean(axis=1)
        curr_df['mean_p_pup'] = curr_df[body_parts_pup].mean(axis=1)
        curr_df['cumsum_nest_pup'] = curr_df[pup_in_nest_col].cumsum()

        generate_figure(curr_df, dist_mother_cornest_col, 'frame number', 'distance (mm)', 'distance between mother and corenest - before pre-processing', date_time_string, 'cumsum_nest_pup', video_base_name_no_file_ending)
        generate_figure(curr_df, dist_pup_cornest_col, 'frame number', 'distance (mm)', 'distance between pup and corenest - - before pre-processing', date_time_string, 'cumsum_nest_pup', video_base_name_no_file_ending)

        for classifier in classifier_list:
            curr_df.loc[curr_df['mean_p_mother'] < prob_mother, classifier] = 0

        first_row = curr_df[curr_df[dist_pup_cornest_col] > dist_start_crit].index[0]
        curr_df.loc[0:first_row, dist_pup_cornest_col] = curr_df.loc[first_row, dist_pup_cornest_col]
        curr_df.loc[0:first_row, pup_in_corenest_col] = 0
        curr_df.loc[0:first_row, pup_in_nest_col] = 0

        if 1 in curr_df[pup_in_nest_col].values:
            for nest_frame in curr_df[curr_df[pup_in_nest_col] == 1].index.tolist():
                sliced_df = curr_df[carry_classifier_name].loc[nest_frame - carry_frames+1:nest_frame].tolist()
                if sum(sliced_df) == 0:
                    curr_df.at[nest_frame, pup_in_nest_col] = 0
                else:
                    break

        rows_with_low_mean_pup_prob = curr_df[curr_df['mean_p_pup'] < prob_pup].index.tolist()
        curr_df.loc[rows_with_low_mean_pup_prob, pup_in_corenest_col] = 0
        curr_df.loc[rows_with_low_mean_pup_prob, pup_in_nest_col] = 0

        if smooth_function == 'gaussian':
            curr_df[dist_pup_cornest_col] = curr_df[dist_pup_cornest_col].rolling(window=video_fps, win_type='gaussian', center=True).mean(std=smooth_factor).fillna(curr_df[dist_pup_cornest_col])
            curr_df[dist_mother_cornest_col] = curr_df[dist_mother_cornest_col].rolling(window=video_fps, win_type='gaussian',center=True).mean(std=smooth_factor).fillna(curr_df[dist_mother_cornest_col])
        generate_figure(curr_df, dist_pup_cornest_col, 'frame number','distance (mm)', 'smoothened_distance_between_pup_and_core_nestt', date_time_string, dist_pup_cornest_col, video_base_name_no_file_ending)
        generate_figure(curr_df, dist_mother_cornest_col, 'frame number', 'distance (mm)', 'smoothened_distance_between_mother_and_core_nest', date_time_string, dist_mother_cornest_col, video_base_name_no_file_ending)

        closest_dist_between_pup_and_zone = round(curr_df[dist_pup_cornest_col].min(), 3)
        if 1 in curr_df[pup_in_nest_col].values:
            frame_when_pup_is_in_zone = curr_df[curr_df[pup_in_nest_col] == 1].index.min()
            time_seconds_until_zone = round(frame_when_pup_is_in_zone / video_fps, 3)
            reason_zone = "Pup in nest"
        else:
            frame_when_pup_is_in_zone = len(curr_df)
            time_seconds_until_zone = round(frame_when_pup_is_in_zone / video_fps, 3)
            reason_zone = "Pup not retrieved"

        if 1 in curr_df[pup_in_corenest_col].values:
            frame_when_pup_is_in_core_nest = curr_df[curr_df[pup_in_corenest_col] == 1].index.min()
            time_seconds_until_corenest = round(frame_when_pup_is_in_core_nest / video_fps, 3)
            reason_corenest = "Pup in core-nest"
        else:
            frame_when_pup_is_in_core_nest = len(curr_df)
            time_seconds_until_corenest = round(frame_when_pup_is_in_core_nest / video_fps, 3)
            reason_corenest = "Pup not in core-nest"

        latency_list, total_time_list, before_retrieval_list = [], [], []
        for classifier in classifier_list:
            total_time_list.append(round(curr_df[classifier].sum() / video_fps, 3))
            before_retrieval_list.append(round(curr_df.loc[0: frame_when_pup_is_in_zone, classifier].sum() / video_fps, 3))
            latency_list.append(round(curr_df[curr_df[classifier] == 1].index.min() / video_fps, 3))

        event_time_between_list, event_counter_list = [], []
        for classifier in classifier_list:
            groupDf = pd.DataFrame()
            v = (curr_df[classifier] != curr_df[classifier].shift()).cumsum()
            u = curr_df.groupby(v)[classifier].agg(['all', 'count'])
            m = u['all'] & u['count'].ge(1)
            groupDf['groups'] = curr_df.groupby(v).apply(lambda x: (x.index[0], x.index[-1]))[m]
            event_counter_list.append(len(groupDf))
            if len(groupDf) > 1:
                between_list = []
                iterator = groupDf.iterrows()
                for (i, row1), (j, row2) in zip(iterator, iterator):
                    between_list.append(round((row2[0][0] - row1[0][1]) / video_fps, 3))
            else:
                between_list = [np.nan]
            event_time_between_list.append(round(sum(between_list) / len(between_list), 3))

        before_enter_corenest_df = curr_df.loc[0: frame_when_pup_is_in_core_nest-1]
        mean_length_list = []
        for classifier in classifier_list:
            groupDf = pd.DataFrame()
            v = (before_enter_corenest_df[classifier] != before_enter_corenest_df[classifier].shift()).cumsum()
            u = before_enter_corenest_df.groupby(v)[classifier].agg(['all', 'count'])
            m = u['all'] & u['count'].ge(1)
            groupDf['groups'] = before_enter_corenest_df.groupby(v).apply(lambda x: (x.index[0], x.index[-1]))[m]
            bout_len_list = []
            for index, row in groupDf.iterrows():
                bout_len_list.append(round((row[0][1] - row[0][0]) / video_fps , 3))
            try:
                mean_length_list.append(round(sum(bout_len_list) / len(bout_len_list), 3))
            except ZeroDivisionError:
                mean_length_list.append(0)

        curr_df['cumsum_nest_pup'] = curr_df[pup_in_nest_col].expanding(1).sum()
        print(curr_df['cumsum_nest_pup'])
        retr_time_frame = curr_df[curr_df[pup_in_nest_col] == 1].index.min()
        if not retr_time_frame: retr_time_frame = len(curr_df)
        retr_time_s = round(retr_time_frame / video_fps, 3)

        # Dig time after first approach until retrieval
        if (approach_classifier_name != 'none') and (dig_classifier_name != 'none'):
            first_approach = curr_df[curr_df[approach_classifier_name] == 1].index.min()
            approach_to_retrieval_df = curr_df.loc[first_approach:retr_time_frame].reset_index(drop=True)
            dig_time_after_approach_to_ret = round(approach_to_retrieval_df[dig_classifier_name].sum() / video_fps, 3)
            groupDf = pd.DataFrame()
            v = (approach_to_retrieval_df[dig_classifier_name] != approach_to_retrieval_df[dig_classifier_name].shift()).cumsum()
            u = approach_to_retrieval_df.groupby(v)[dig_classifier_name].agg(['all', 'count'])
            m = u['all'] & u['count'].ge(1)

            try:
                groupDf['groups'] = approach_to_retrieval_df.groupby(v).apply(lambda x: (x.index[0], x.index[-1]))[m]
                dig_events_after_approach = len(groupDf)
                if dig_events_after_approach > 0:
                    duration_list = []
                    for index, row in groupDf.iterrows():
                        duration_list.append(round((row[0][1] - row[0][0]) / video_fps, 3))
                    mean_duration_dig_after_approach = (sum(duration_list) / len(duration_list))
                else:
                    mean_duration_dig_after_approach = np.nan
            except KeyError:
                mean_duration_dig_after_approach = np.nan


        generate_figure(curr_df, dist_pup_cornest_col, 'frame number', 'distance (mm)', 'cumulative time of pup in nest', date_time_string, 'cumsum_nest_pup', video_base_name_no_file_ending)
        generate_figure(curr_df, dist_mother_cornest_col, 'frame number', 'distance (mm)','distance corenest to mother', date_time_string, 'cumsum_nest_pup', video_base_name_no_file_ending)

        out_list = [video_base_name_no_file_ending, frame_when_pup_is_in_zone, time_seconds_until_zone, closest_dist_between_pup_and_zone, reason_zone, frame_when_pup_is_in_core_nest, time_seconds_until_corenest, reason_corenest]
        for clf in range(len(classifier_list)):
            total_time, before_ret, latency = total_time_list[clf], before_retrieval_list[clf], latency_list[clf]
            events, mean_dur, mean_interval = event_counter_list[clf], mean_length_list[clf], event_time_between_list[clf]
            out_list.extend((total_time, before_ret, latency, events, mean_dur, mean_interval))
        if (approach_classifier_name != 'none') and (dig_classifier_name != 'none'):
            out_list.extend((dig_time_after_approach_to_ret, dig_events_after_approach, mean_duration_dig_after_approach))
        out_df.loc[len(out_df)] = out_list

    pup_ret_file_path = os.path.join(logs_dir_path, 'Pup_retrieval_' + date_time_string + '.csv')
    out_df.to_csv(pup_ret_file_path)

    log_file_index = ['Date_time', 'Videos_#', 'Pub probability', 'Dam probability', 'Start distance criterion', 'Carry frames seconds', 'Smooth factor', 'Core-nest name', \
                      'Nest name', 'Dam name', 'Pup name', 'Smooth function', 'Carry classifier name', 'Approach classifier name', 'Dig classifiername']
    log_list = [date_time_string,  str(len(features_files)), prob_pup, prob_mother, dist_start_crit, carry_frames_seconds, smooth_factor, corenest_name, nest_name, dam_name, pup_name, smooth_function, carry_classifier_name, approach_classifier_name, dig_classifier_name]
    log_df = pd.DataFrame(log_list, index=log_file_index, columns=['Session values'])
    log_name = 'Log_pup_retrieval_' + str(date_time_string) + '.csv'
    log_save_path = os.path.join(logs_dir_path, log_name)
    log_df.to_csv(log_save_path)

    figure_df = out_df.copy()
    figure_df['Experiment'] = 1
    swarm_plot = sns.boxplot(x="Experiment", y="Pup in nest (s)", data=figure_df)
    swarm_plot = sns.swarmplot(x="Experiment", y="Pup in nest (s)", data=figure_df, color="white")
    swarm_plot.set(xlabel='', ylabel="Pup in nest (s)", title='Summary - pup retrieval time (s)')
    swarm_plot_name = 'Summary_pup_retrieval_times' + '_' + date_time_string + '.png'
    save_plot_path = os.path.join(logs_dir_path, swarm_plot_name)
    swarm_plot.figure.savefig(save_plot_path, bbox_inches="tight")
    swarm_plot.clear()
    print('All videos analysed, summary file saved @ ' + str(pup_ret_file_path))
    print('Summary plot saved @ ' + str(save_plot_path))
    print('Summary log file saved @ ' + str(log_save_path))
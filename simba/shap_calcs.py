import cv2
from collections import OrderedDict
import numpy as np
import itertools
from datetime import datetime
from configparser import ConfigParser, MissingSectionHeaderError, NoOptionError, NoSectionError
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")


def shap_summary_calculations(INIFILE, shap_df, classifier_name, BASELINE_VAL, save_path):
    dateTime = datetime.now().strftime('%Y%m%d%H%M%S')
    config = ConfigParser()
    configFile = str(INIFILE)
    try:
        config.read(configFile)
    except MissingSectionHeaderError:
        print('ERROR:  Not a valid project_config file. Please check the project_config.ini path.')

    pose_estimation_body_parts = config.get('create ensemble settings', 'pose_estimation_body_parts')
    if (pose_estimation_body_parts == '14') or (pose_estimation_body_parts == '16'):
        projectPath = config.get('General settings', 'project_path')
        shap_logs_path = os.path.join(projectPath, 'logs', 'shap')
        if not os.path.exists(shap_logs_path): os.makedirs(shap_logs_path)

        simba_cw = os.getcwd()
        simba_feat_cat_dir = os.path.join(simba_cw, 'simba', 'assets', 'shap', 'feature_categories')
        feat_cat_csv_path = os.path.join(simba_feat_cat_dir, 'shap_feature_categories.csv')
        simba_assets_path = os.path.join(simba_cw, 'simba', 'assets', 'shap')

        scale_dict = {'baseline_scale': os.path.join(simba_assets_path, 'baseline_scale.jpg'),
                      'small_arrow': os.path.join(simba_assets_path, 'down_arrow.jpg'),
                      'side_scale': os.path.join(simba_assets_path, 'side_scale.jpg'),
                      'color_bar': os.path.join(simba_assets_path, 'color_bar.jpg')}

        category_dict = {'Animal distances': {'icon': os.path.join(simba_assets_path, 'animal_distances.jpg')},
                         'Intruder movement': {'icon': os.path.join(simba_assets_path, 'intruder_movement.jpg')},
                         'Resident+intruder movement': {'icon': os.path.join(simba_assets_path, 'resident_intruder_movement.jpg')},
                         'Resident movement': {'icon': os.path.join(simba_assets_path, 'resident_movement.jpg')},
                         'Intruder shape': {'icon': os.path.join(simba_assets_path, 'intruder_shape.jpg')},
                         'Resident intruder shape': {'icon': os.path.join(simba_assets_path, 'resident_intruder_shape.jpg')},
                         'Resident shape': {'icon': os.path.join(simba_assets_path, 'resident_shape.jpg')}}

        pos_boxarrow_colors = [(253,141,60), (252,78,42), (227,26,28), (189,0,38), (128,0,38)]
        neg_boxarrow_colors = [(65,182,196), (29,145,192), (34,94,168), (37,52,148), (8,29,88)]
        ranges_lists = [list(range(0,20)), list(range(20,40)), list(range(40,60)), list(range(60,80)), list(range(80,101))]

        colCats = pd.read_csv(feat_cat_csv_path, header=[0, 1])
        firstIndices, secondIndices = list(colCats.columns.levels[0]), list(colCats.columns.levels[1])
        outputDfcols = secondIndices.copy()
        outputDfcols.extend(('Sum', 'Category'))

        for i in range(2):
            shap_for_cur_behavior_df = shap_df[shap_df[classifier_name] == i]
            outputDf = pd.DataFrame(columns=outputDfcols)
            for topIndex in firstIndices:
                meanList = []
                for botomIndex in secondIndices:
                    currDf = colCats.loc[:, list(itertools.product([topIndex], [botomIndex]))]
                    currCols = list(currDf. iloc[:, 0])
                    currCols = [x for x in currCols if str(x) != 'nan']
                    currShaps = shap_for_cur_behavior_df[currCols]
                    currShaps["Shap_sum"] = currShaps.sum(axis=1)
                    meanList.append(currShaps["Shap_sum"].mean())
                meanList.append(sum(meanList))
                meanList.append(topIndex)
                outputDf.loc[len(outputDf)] = meanList
            if i == 0:
                outputDf_path = os.path.join(save_path, 'SHAP_summary_' + classifier_name + '_ABSENT_' + str(dateTime) + '.csv')
            else:
                outputDf_path = os.path.join(save_path, 'SHAP_summary_' + classifier_name + '_PRSESENT_' + str(dateTime) + '.csv')
            outputDf = outputDf.set_index('Category')
            outputDf.to_csv(outputDf_path)

        shap_value_list = list(outputDf['Sum'] * 100)
        shap_value_list = [round(x, 0) for x in shap_value_list]

        for category, value in zip(category_dict, shap_value_list): category_dict[category]['value'] = value
        category_dict = dict(OrderedDict(sorted(category_dict.items(), key=lambda x: x[1]['value'], reverse=True)))

        img = 255 * np.ones([1680, 1680, 3], dtype=np.uint8)
        baseline_scale_img = cv2.imread(scale_dict['baseline_scale'])
        baseline_scale_top_left = (100, 800)
        baseline_scale_bottom_right = (baseline_scale_top_left[0] + baseline_scale_img.shape[0], baseline_scale_top_left[1] + baseline_scale_img.shape[1])
        baseline_scale_middle = ((int(700 + baseline_scale_img.shape[1] / 2)), 85)
        img[baseline_scale_top_left[0]:baseline_scale_bottom_right[0],
        baseline_scale_top_left[1]:baseline_scale_bottom_right[1]] = baseline_scale_img
        cv2.putText(img, 'baseline SHAP', baseline_scale_middle, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

        small_arrow_img = cv2.imread(scale_dict['small_arrow'])
        small_arrow_top_left = (baseline_scale_bottom_right[0], int(baseline_scale_top_left[1] + (baseline_scale_img.shape[1] / 100) * BASELINE_VAL))
        small_arrow_bottom_right = (small_arrow_top_left[0] + small_arrow_img.shape[0], small_arrow_top_left[1] + small_arrow_img.shape[1])
        img[small_arrow_top_left[0]:small_arrow_bottom_right[0], small_arrow_top_left[1]:small_arrow_bottom_right[1]] = small_arrow_img

        side_scale_img = cv2.imread(scale_dict['side_scale'])
        side_scale_top_left = (small_arrow_bottom_right[0] + 50, baseline_scale_top_left[1] - 50)
        side_scale_bottom_right = (side_scale_top_left[0] + side_scale_img.shape[0], side_scale_top_left[1] + side_scale_img.shape[1])
        # img[side_scale_top_left[0]:side_scale_bottom_right[0], side_scale_top_left[1]:side_scale_bottom_right[1]] = side_scale_img

        side_scale_y_tick_cords = [(side_scale_top_left[0], side_scale_top_left[1] - 75)]
        for i in range(1, 7): side_scale_y_tick_cords.append(
            (int(side_scale_top_left[0] + (side_scale_img.shape[0] / 4) * i), int(side_scale_top_left[1] - 75)))

        for value, tick in enumerate(side_scale_y_tick_cords):
            icon_name = list(category_dict.keys())[value]
            icon_img = cv2.imread(category_dict[icon_name]['icon'])
            icon_img = cv2.resize(icon_img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            icon_top_left = (tick[0] - int(icon_img.shape[0] / 2), tick[1] - 100)
            icon_bottom_right = (icon_top_left[0] + icon_img.shape[0], tick[1] + icon_img.shape[1] - 100)
            text_location = (int(icon_bottom_right[0] - (icon_bottom_right[0] - icon_top_left[0]) + 100),
                             int(icon_bottom_right[1] - (icon_bottom_right[1] - icon_top_left[1])) - 380)
            cv2.putText(img, str(icon_name), (text_location[1], text_location[0]), cv2.FONT_HERSHEY_COMPLEX, 0.75,
                        (0, 0, 0), 1)
            img[icon_top_left[0]:icon_bottom_right[0], icon_top_left[1]:icon_bottom_right[1]] = icon_img

        arrow_start = (int(small_arrow_top_left[1] + (small_arrow_img.shape[1] / 2)), side_scale_top_left[0])
        for value, shap_cat in enumerate(category_dict):
            arrow_width = int((baseline_scale_img.shape[1] / 100) * abs(category_dict[shap_cat]['value']))
            shap_value = category_dict[shap_cat]['value']
            if shap_value > 0:
                arrow_end = (arrow_start[0] + arrow_width, arrow_start[1])
                arrow_middle = int(((arrow_end[1] - arrow_start[1]) / 2) + arrow_start[1] - 7)
                for bracket_no, bracket in enumerate(ranges_lists):
                    if abs(shap_value) in bracket:
                        color = (pos_boxarrow_colors[bracket_no][2], pos_boxarrow_colors[bracket_no][1], pos_boxarrow_colors[bracket_no][0])
                cv2.arrowedLine(img, arrow_start, arrow_end, color, 5, tipLength=0.1)
                cv2.putText(img, '+' + str(abs(shap_value)) + '%', (arrow_end[0] - 7, arrow_middle - 15), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
                if value != (len(shap_value_list) - 1): arrow_start = (arrow_end[0], side_scale_y_tick_cords[value+1][0])

            if shap_value <= 0:
                arrow_end = (arrow_start[0] - arrow_width, arrow_start[1])
                arrow_middle = int(((arrow_start[1] - arrow_end[1]) / 2) + arrow_end[1] - 7)
                for bracket_no, bracket in enumerate(ranges_lists):
                    if abs(shap_value) in bracket:
                        color = (neg_boxarrow_colors[bracket_no][2], neg_boxarrow_colors[bracket_no][1], neg_boxarrow_colors[bracket_no][0])
                cv2.arrowedLine(img, arrow_start, arrow_end, color, 5, tipLength=0.1)
                cv2.putText(img, '-' + str(abs(shap_value)) + '%', (arrow_end[0] - 7, arrow_middle - 15), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
                if value != (len(shap_value_list) - 1): arrow_start = (arrow_end[0], side_scale_y_tick_cords[value + 1][0])

        small_arrow_top_left = (int(arrow_end[1]) + 20, int(arrow_end[0] - small_arrow_img.shape[1] / 2))
        small_arrow_bottom_right = (
            small_arrow_top_left[0] + small_arrow_img.shape[0], small_arrow_top_left[1] + small_arrow_img.shape[1])
        img[small_arrow_top_left[0]:small_arrow_bottom_right[0],
        small_arrow_top_left[1]:small_arrow_bottom_right[1]] = small_arrow_img

        color_bar_img = cv2.imread(scale_dict['color_bar'])
        color_bar_top_left = (arrow_end[1] + small_arrow_img.shape[0] + 25, baseline_scale_top_left[1])
        color_bar_bottom_right = (
            color_bar_top_left[0] + color_bar_img.shape[0], color_bar_top_left[1] + color_bar_img.shape[1])
        img[color_bar_top_left[0]:color_bar_bottom_right[0],
        color_bar_top_left[1]:color_bar_bottom_right[1]] = color_bar_img

        color_bar_middle = ((int(580 + baseline_scale_img.shape[1] / 2)), color_bar_bottom_right[0] + 50)
        cv2.putText(img, 'classification probability', color_bar_middle, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        image_name = 'SHAP_summary_line_graph_' + classifier_name + '_' + str(dateTime) + '.png'
        image_path = os.path.join(save_path, image_name)
        cv2.imwrite(image_path, img)
        print('SHAP summary graph saved at @' + image_path)
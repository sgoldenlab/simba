import dash
from dash.dependencies import  Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import dash_color_picker as dcp
import dash_daq as daq
import pandas as pd
import numpy as np
import h5py
import dash_colorscales as dcs
import random
import json
import math
import statistics
import os
from flask import request
import sys


filePath = ''
groupPath = ''

if not len(sys.argv) > 1:
    print("Expecting link argument.")
else:
    print("in p1.py link is " + sys.argv[1] + ' ' + sys.argv[2])
    filePath = sys.argv[1]
    groupPath = sys.argv[2]


external_stylesheets = ['dash_simba_base.css', '//use.fontawesome.com/releases/v5.0.7/css/all.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
hf = h5py.File(filePath, 'r')


# Input: string path of group
# Output: returns list of groups, dictionary of group and the videos in each group, and if groups exist
def get_groups(path):
    g_list = []
    g_dict = {}

    is_file = os.path.isfile(path)
    if is_file:
        df_groups = pd.read_csv(path)
        for group in df_groups.keys():
            g_list.append(group)
            g_dict[group] = df_groups[group].dropna().tolist()
    else:
        pass
    return g_list, g_dict, is_file


# Input: List of videos, behaviors, categories, hdf5 file path, and hdf5 file object
# Output: Returns complete dictionary of all behaviors, categories, and features
def get_feature_data(videos, behaviors, g_categories, path, h5file):
    dict_features = {}

    for b in behaviors:
        dict_features[b] = {}

        for category in g_categories:
            if category == 'VideoData':
                video = pd.read_hdf(path, key=category + '/' + videos[0])
                dict_features[b][category] = \
                    {column: get_probability_data(column, videos, path) for column in video.columns if '_'+b in column}
            else:
                for sub_c in h5file.get(category).keys():
                    video = pd.read_hdf(path, key=category + '/' + sub_c)

                    if b == 'Attack':
                        dict_features[b][sub_c] = {column: video[column] for column in video.columns}
                    else:
                        dict_features[b][sub_c] = {column: video[column] for column in video.columns if b in column}

    return dict_features


# Input: Current feature name, list of videos, and path of hdf5 file
# Output: Returns pandas dataframe for each video with probability features
def get_probability_data(feature, list_videos, path_name):
    feature_data = pd.DataFrame()

    for video in list_videos:
        read_video = pd.read_hdf(path_name, key='VideoData/' + video)
        feature_data[video] = read_video[feature]
        # dict_data[f][video].fillna(0)

    for g in groups_dict:
        feature_data[g+'_Mean'] = feature_data[groups_dict[g]].mean(axis=1)

    mean_data = feature_data.mean(axis=1)
    feature_data['Total_Mean'] = mean_data

    return feature_data


# Input: hdf5 file object
# Output: Returns list of categories and list of all hdf5 main keys
def get_categories(h5file):
    categories = []
    general_list = []
    for k in h5file.keys():
        if k != 'Classifier_names' and k != 'Video_info':
            if 'VideoData' in k:
                categories.append(k)
            else:
                hf_category = h5file.get(k)
                for key in hf_category.keys():
                    categories.append(key)
            general_list.append(k)
    return categories, general_list


# Input: Boolean if groups exist
# Output: Returns either visible or hidden style
def set_display(exist):
    if exist:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


# Return download path
def get_download_path():
    """Returns the default downloads path for linux or windows"""
    if os.name == 'nt':
        import winreg
        sub_key = r'SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders'
        downloads_guid = '{374DE290-123F-4565-9164-39C4925E467B}'
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, sub_key) as key:
            location = winreg.QueryValueEx(key, downloads_guid)[0]
        return location
    else:
        return os.path.join(os.path.expanduser('~'), 'downloads')


# # VARIABLES AND DATA
# # List of all videos in H5File
video_info = pd.read_hdf(filePath, key='Video_info')
H5list_videos = list(video_info.index)

# # List of Behaviors
classifiers = pd.read_hdf(filePath, key='Classifier_names')
list_behaviors = list(classifiers['Classifier_names'])

# # List of Categories
list_categories, general_categories = get_categories(hf)

# # List of Groups and which videos belong in which group
group_list, groups_dict, groups_exist = get_groups(groupPath)

# # Dictionary of fps for each video
fps_dict = dict(zip(H5list_videos, video_info['fps']))
for group in groups_dict:
    try:
        curr_fps = fps_dict[groups_dict[group][0]]
        fps_dict[group] = curr_fps
        for video in groups_dict[group]:
            if fps_dict[video] != curr_fps:
                print("Videos do not have the same fps")
    except IndexError:
        pass

# # Create and store all the data to be used in dash plotly in the form of a dictionary
all_data_behaviors = get_feature_data(H5list_videos, list_behaviors, general_categories, filePath, hf)


# # List of available and supported fonts
FONTS = ['Verdana', 'Helvetica', 'Calibri', 'Arial', 'Arial Narrow', 'Candara', 'Geneva', 'Courier New',  'Times New Roman']

# # Default graph configuration
config = {
  'toImageButtonOptions': {
    'format': 'svg',  # one of png, svg, jpeg, webp
    'filename': 'custom_image',
    'height': 500,
    'width': 700,
    'scale': 1  # Multiply title/legend/axis/canvas sizes by this factor
  }
}

hf.close()


# Returns color pickers based on number of groups
def create_color_pickers():
    output = []
    for g in group_list:
        new_picker = [html.Label(g), dcp.ColorPicker(id='ColorPicker'+str(g), color='#609BE0')]
        output += new_picker
    output += [html.Label('Mean'), dcp.ColorPicker(id='ColorPickerMean', color='#f22')]
    return output


app.layout = html.Div(children=[
    html.Div(id='header', className='row', children=[
        html.H1(children='SimBA Interactive Data Visualization Dashboard', className='nine columns'),

        html.Img(
            src=app.get_asset_url('TheGoldenLab.PNG'),
            className='three columns',
            style={
                'height': '15%',
                'width':'15%',
                'float': 'right',
                'position': 'relative',
            }
        )]
    ),


    html.Div(id='simba-control-tabs', className='three columns', children=[
        dcc.Tabs(id='simba-tabs', value='all_data', children=[

            dcc.Tab(
                label='Data',
                value='all_data',
                children=[
                    html.Div(id='plot_data', children=[
                        html.H6('Select Behavior to Plot'),
                        html.Label('Behaviors:'),
                        dcc.Dropdown(
                            # className='eight columns',
                            id='behaviors',
                            options=[
                                {'label': b,'value': b} for b in list_behaviors],
                            multi=False,
                            value=list_behaviors[0],
                            clearable=False),

                        html.Label('Category:'),
                        dcc.Dropdown(
                            id='category_selector',
                            # className='eight columns',
                            clearable=False),

                        html.Label('Feature:'),
                        dcc.Dropdown(
                            id='feature_selector',
                            clearable=False),


                        html.Div(
                            id='multi_group_data',
                            children=[
                                html.Hr(),
                                html.H6('Plotting Group Means'),
                                html.Label('Select Group(s):'),
                                dcc.Dropdown(
                                    id='multi_group_selector',
                                    # className='eight columns',
                                    options=[
                                        {'label': key,'value': key} for key in groups_dict.keys()],
                                    multi=True,
                                    clearable=False,
                                    value=group_list),

                                dcc.Checklist(
                                    id='group_mean',
                                    options=[{'label': 'Show Total Mean', 'value': 'showGroupMean', 'disabled': False}],
                                    value=['showGroupMean'],
                                    labelStyle={'display': 'inline-block'})],
                            style=set_display(groups_exist)),

                        html.Hr(),
                        html.H6('Plotting Individual Groups'),
                        html.Div(
                            id='show_group_opt',
                            children=[
                                html.Label('Select Single Group to Plot:',
                                           style=set_display(groups_exist)),
                                dcc.Dropdown(
                                    id='group_selector',
                                    options=[
                                        {'label': key, 'value': key} for key in groups_dict.keys()],
                                    multi=False,
                                    clearable=False)],
                            style=set_display(groups_exist)),

                        html.Label('Select Video(s):'),
                        dcc.Dropdown(
                            id='video_numbers',
                            multi=True,
                            clearable=False),
                        dcc.Checklist(
                            id='video_mean',
                            options=[{'label': 'Show Group Mean', 'value': 'showVideoMean', 'disabled': False}],
                            value=['showVideoMean'])]
                    )]
            ),

            dcc.Tab(
                label='Graph Settings',
                value='graph_settings',
                children=[
                    html.Div(className='twelve columns', id='graph_settings', children=[
                        html.Div(id='color_properties', className='twelve columns', children=[
                            html.H6('Color Properties'),

                            html.Div(id='color_pickers',
                                     children=create_color_pickers()),
                            html.Label('Colorscale Gradient for Individual Videos'),
                            dcs.DashColorscales(id='ind_group_colors', colorscale=['#fbb4ae', '#b3cde3', '#ccebc5', '#decbe4', '#fed9a6', '#ffffcc']),
                            html.Button('Update Colors', id='update_graph', n_clicks=0, type='submit'),
                            html.Hr()]),



                        html.Div(className='twelve columns',
                                 id='probability-graph-properties',
                                 children=[
                                     html.H6('Probability Graph Properties'),
                                     html.Div(
                                         id='multi_prob_prop',
                                         children=[
                                             html.Label('Set Group Viewing Frames:'),
                                             dcc.Input(id='multi_start_x', type='number', className='four columns'),
                                             dcc.Input(id='multi_end_x', type='number', className='four columns'),
                                             html.Button(id='submit_multi_range', className='three columns', n_clicks=1, children='Submit', type='submit'),
                                             html.Label('Multi Group (Seconds)'),
                                             dcc.Input(id='multi_start_sec', type='number', className='four columns', disabled=True),
                                             dcc.Input(id='multi_end_sec', type='number', className='four columns', disabled=True),
                                             html.Button(id='reset_multi_range', className='twelve columns', n_clicks=1, children='Reset Axes', type='reset')],
                                         style=set_display(groups_exist)),

                                     html.Label('Set Video Viewing Frames:', className='twelve columns'),
                                     dcc.Input(id='ind_start_x', type='number', className='four columns'),
                                     dcc.Input(id='ind_end_x', type='number', className='four columns'),
                                     html.Button(id='submit_ind_range', className='three columns', n_clicks=1, children='Submit', type='submit'),
                                     html.Label('Individual Group (Seconds)', className='twelve columns'),
                                     dcc.Input(id='ind_start_sec', type='number', className='four columns', disabled=True),
                                     dcc.Input(id='ind_end_sec', type='number', className='four columns', disabled=True),
                                     html.Button(id='reset_ind_range', className='twelve columns', n_clicks=1, children='Reset Axes', type='reset')],
                                 style={'display': 'block'}),


                        html.Div(className='twelve columns',
                                 id='bar-graph-properties',
                                 children=[
                                     html.H6('Bar Graph Properties'),
                                     html.Label('Error Type:'),
                                     dcc.RadioItems(
                                         id='error_type',
                                         options=[{'label': 'Above', 'value': 'above'},
                                                  {'label': 'Below', 'value': 'below'},
                                                  {'label': 'Both', 'value': 'both'}],
                                         value='both')]),


                        html.Div(className='twelve columns',
                                 id='additional-properties',
                                 children=[
                                     html.Hr(),
                                     html.H6('Additional Properties'),
                                     dcc.Checklist(
                                        id='graph_seconds',
                                        options=[{'label': 'Graph X Axis in Seconds', 'value': 'show_seconds'}],
                                        value=[],
                                        labelStyle={'display': 'inline-block'}),

                                     dcc.Checklist(
                                         id='show_grid',
                                         options=[{'label': "Show Grid lines", 'value': 'showGridLines'}],
                                         value=['showGridLines'],
                                         labelStyle={'display': 'inline-block'}),
                                     dcc.Checklist(
                                         id='show_bg',
                                         options=[{'label': "Show Background", 'value': 'showBg'}],
                                         value=['showBg'],
                                         labelStyle={'display': 'inline-block'}),

                                     html.Div(
                                         id='show_mean_graph_title',
                                         children=[
                                             html.Label('Group Means Title:'),
                                             dcc.Input(
                                                 id='mean_graph_title',
                                                 className='seven columns'
                                             ),
                                             html.Button(
                                                 children=[
                                                     html.I(n_clicks=0, className='fa fa-chart-area',
                                                            style={'font-size': '24px'}),
                                                     html.P('Set', style={'padding-left': '5px'})],
                                                 id='set_multi_title',
                                                 n_clicks=0,
                                                 className='four columns')],
                                         style=set_display(groups_exist)
                                     ),


                                     html.Label('Individual Videos Title:'),
                                     dcc.Input(
                                         id='ind_graph_title',
                                         className='seven columns'
                                     ),
                                     html.Button(
                                         children=[
                                             html.I(n_clicks=0, className='fa fa-chart-area', style={'font-size': '24px'}),
                                             html.P('Set', style={'padding-left': '5px'})],
                                         id='set_ind_title',
                                         n_clicks=0,
                                         className='four columns'),

                                     html.Label('Choose Font:'),
                                     dcc.Dropdown(
                                         id='fonts',
                                         options=[{'label': f, 'value': f} for f in FONTS],
                                         value=FONTS[0],
                                         multi=False,
                                         clearable=False),

                                     html.Label('Font Size:'),
                                     daq.NumericInput(
                                         id='font_size',
                                         value=12,
                                         min=8,
                                         max=48)])]
                             )]),
            dcc.Tab(
                label='Download Settings',
                value='download_settings',
                children=[
                    html.Div(
                        id='download-settings',
                        children=[
                            html.Div(id='csv_export', className='twelve columns', children=[
                                html.H6('CSV Export'),
                                html.Label('Enter csv file name:'),
                                dcc.Input(
                                    id='csv_file_name'
                                ),
                                html.P(children=['.csv']),
                                html.Button(
                                    children=[
                                        html.I(n_clicks=0, className='fa fa-download', style={'font-size': '24px'}),
                                        html.P('Group Means', style={'padding-left': '5px'})],
                                    id='download_group_data',
                                    n_clicks=0,
                                    style=set_display(groups_exist),
                                    className='five columns'),
                                html.Button(
                                    children=[
                                        html.I(n_clicks=0, className='fa fa-download', style={'font-size': '24px'}),
                                        html.P('Videos', style={'padding-left': '5px'})],
                                    id='download_data',
                                    n_clicks=0,
                                    className='five columns'),
                                html.Div(id='comments', className='twelve columns'),
                                html.Hr(className='twelve columns')]),
                            html.Div(id='img_export', className='twelve columns', children=[
                                html.H6('Image Export', className='twelve columns'),
                                html.Div(id='img-dim', className='twelve columns', children=[
                                    html.Label('Enter image dimensions (px):', className='twelve columns'),
                                    daq.NumericInput(
                                        id='img_height',
                                        className='three columns',
                                        value=500,
                                        min=0,
                                        max=1000
                                    ),
                                    html.P('x', className='one column'),
                                    daq.NumericInput(
                                        id='img_width',
                                        className='three columns',
                                        value=700,
                                        min=0,
                                        max=1000
                                    )]),
                                html.Label('Enter image name:', className='twelve columns'),
                                dcc.Input(
                                    id='img_file_name'
                                ),
                                html.P(id='img_type', children=['.svg']),
                                html.Label('File extension:'),
                                dcc.Dropdown(
                                    id='img_ext',
                                    options=[{'label': 'PNG', 'value': 'png'},
                                             {'label': 'JPEG', 'value': 'jpeg'},
                                             {'label': 'SVG', 'value': 'svg'}],
                                    value='svg',
                                ),
                                html.Button(
                                    children=[
                                        html.I(n_clicks=0, className='fa fa-save', style={'font-size': '24px'}),
                                        html.P('Save Download Settings', style={'padding-left': '5px'})],
                                    id='submit_download',
                                    n_clicks=0,
                                    type='submit')])
                    ])]
            )
        ])
    ]),

    html.Div(id='graphs', className='eight columns', children=[

        html.Div(
            id='multi_group_container',
            # style={'display': 'none'},
            children=[
                dcc.Graph(
                    id='plot_multi_groups',
                    style={'display': 'none'},
                    config=config
                )]),

        dcc.Graph(
            id='plot_ind_group',
            config=config,
            style={'display': 'block'}
        )
    ]),

    html.Div(id='credits', className='footer', children=[
        html.Div(className='six columns', children=[html.P('Created by '),
                                                    html.A('Sophia Hwang', href='https://github.com/sophihwang26', target='_blank'),
                                                    html.P(' and '),
                                                    html.A('Aasiya Islam', href='https://github.com/aasiya-islam', target='_blank')]),
        html.Div(className='six columns', children=[html.P('Built with Dash',
                                                           style={
                                                               'height': '15%',
                                                               'width': '15%',
                                                               'float': 'right',
                                                               'position': 'relative',
                                                           }
                                                           )])
    ])
])

# Sets config type text
@app.callback(
    Output('img_type', 'children'),
    [Input('img_ext', 'value')]
)
def update_ext_text(extension):
    return ['.' + extension]


# Sets config settings for graph
@app.callback(
    Output('graphs', 'children'),
    [Input('submit_download', 'n_clicks')],
    [State('img_height', 'value'),
     State('img_width', 'value'),
     State('img_file_name', 'value'),
     State('img_ext', 'value')]
)
def update_config(submit, height, width, filename, extension):
    if submit is not None:
        new_config = {
            'toImageButtonOptions': {
                'format': extension,  # one of png, svg, jpeg, webp
                'filename': filename,
                'height': height,
                'width': width,
                'scale': 1  # Multiply title/legend/axis/canvas sizes by this factor
            }
        }
        # print('config settings saved')
        return dcc.Graph(id='plot_multi_groups', config=new_config), \
               dcc.Graph(id='plot_ind_group', config=new_config)
    else:
        return dcc.Graph(id='plot_multi_groups', config=config), \
               dcc.Graph(id='plot_ind_group', config=config)


# Sets available options in dropdowns based on behavior selected
@app.callback(
    Output('feature_selector', 'options'),
    [Input('category_selector', 'value'),
     Input('behaviors', 'value')])
def set_features_options(selected_category, behavior):
    return [{'label': f, 'value': f} for f in list(all_data_behaviors[behavior][selected_category].keys())]


# Sets list of available videos to plot on graph based on group selected
@app.callback(
    Output('video_numbers', 'options'),
    [Input('group_selector', 'value'),
     Input('category_selector', 'value'),
     Input('behaviors', 'value'),
     Input('feature_selector', 'value')]
)
def set_group_options(selected_group, selected_category, selected_behavior, selected_feature):
    try:
        return [{'label': v, 'value': v} for v in groups_dict[selected_group]]
    except KeyError:
        print('No Groups')
        if 'VideoData' in selected_category:
            return [{'label': v, 'value': v} for v in H5list_videos]
        else:
            try:
                list_vid = list(all_data_behaviors[selected_behavior][selected_category][selected_feature].index)
                return [{'label': v, 'value': v} for v in list_vid]
            except KeyError:
                return []


# Sets default value for Feature
@app.callback(
    Output('feature_selector', 'value'),
    [Input('feature_selector', 'options')])
def set_feature_value(options):
    try:
        return options[0]['value']
    except IndexError:
        print('No features found.')
        return ''


# Sets options for category
@app.callback(
    Output('category_selector', 'options'),
    [Input('behaviors', 'value')])
def set_video_option(behavior):
    if behavior == 'Attack':
        return [{'label': c, 'value': c} for c in list_categories if 'Severity' not in c]
    else:
        return [{'label': c, 'value': c} for c in list_categories]

# Sets default value for Category
@app.callback(
    Output('category_selector', 'value'),
    [Input('category_selector', 'options')])
def set_category_value(options):
    try:
        return options[0]['value']
    except IndexError:
        return ''

# Sets default values for videos to plot
@app.callback(
    Output('video_numbers', 'value'),
    [Input('video_numbers', 'options')])
def set_video_option(options):
    try:
        # video_list = [(options[0]['value'])]
        # return video_list
        return [vid['value'] for vid in options]
    except IndexError:
        return ''



# Sets default value for single group to plot
@app.callback(
    Output('group_selector', 'value'),
    [Input('group_selector', 'options')])
def set_video_option(options):
    try:
        return options[0]['value']
    except IndexError:
        return ''


# Sets visibility of graphing options depending on bar or scatter plot
@app.callback(
    [Output('probability-graph-properties', 'style'),
     Output('bar-graph-properties', 'style'),
     Output('video_mean', 'style'),
     Output('group_mean', 'style'),
     Output('graph_seconds', 'style')],
    [Input('category_selector', 'value')])
def graph_options_visibility(category):
    if 'VideoData' in category:
        return {'display': 'block'}, {'display': 'none'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}
    else:
        return {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}


# Return ranges for numeric input given relayoutData and list of all frame
def determine_range(relayoutData, ranges, frames):
    index = 0
    # print(relayoutData)
    for word in json.dumps(relayoutData).split():
        word = word.replace(',', '')
        word = word.replace('}', '')
        # Determine closest frame number given relayoutData ranges
        try:
            number = float(word)
            absolute_different_function = lambda whole_x: abs(whole_x - number)
            closest_value = min(frames, key=absolute_different_function)
            ranges[index] = closest_value
            index += 1
        except ValueError:
            pass
    return ranges

# Sets value of start and end frames in Viewing Range numeric input box
@app.callback(
    [Output('ind_start_x', 'value'),
     Output('ind_end_x', 'value'),
     Output('ind_start_x', 'disabled'),
     Output('ind_end_x', 'disabled'),
     Output('multi_start_x', 'value'),
     Output('multi_end_x', 'value'),
     Output('multi_start_x', 'disabled'),
     Output('multi_end_x', 'disabled'),
     Output('ind_start_sec', 'value'),
     Output('ind_end_sec', 'value'),
     Output('multi_start_sec', 'value'),
     Output('multi_end_sec', 'value')],
    [Input('plot_ind_group', 'relayoutData'),
     Input('plot_multi_groups', 'relayoutData'),
     Input('graph_seconds', 'value'),
     Input('feature_selector', 'value'),
     Input('submit_multi_range', 'n_clicks'),
     Input('submit_ind_range', 'n_clicks'),
     Input('reset_multi_range', 'n_clicks'),
     Input('reset_ind_range', 'n_clicks')],
     [State('behaviors', 'value'),
     State('category_selector', 'value'),
     State('ind_start_x', 'value'),
     State('ind_end_x', 'value'),
     State('multi_start_x', 'value'),
     State('multi_end_x', 'value')]
)
def set_input_xrange(ind_relayout, multi_relayout, seconds, feature,
                     submit_multi, submit_ind, reset_multi, reset_ind,
                     behavior, category, prev_start_i, prev_end_i, prev_start_g, prev_end_g):
    if 'VideoData' not in category:
        return None, None, True, True, None, None, True, True, None, None, None, None
    else:
        x = all_data_behaviors[behavior][category][feature].index
        min_value = 0
        max_value = len(x) - 1
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        # print(changed_id)
        disabled = False

        # Initial Load
        if (prev_start_i or prev_end_i or prev_start_g or prev_end_g) is None:
            # print('Previously no Values in numeric input')
            current_indx = [min_value, max_value, 0, 1]
            current_multix = [min_value, max_value, 0, 1]
        elif 'reset_ind_range' in changed_id:
            current_indx = [min_value, max_value, 0, 1]
            current_multix = [prev_start_g, prev_end_g, 0, 1]
        elif 'reset_multi_range' in changed_id:
            current_indx = [prev_start_i, prev_end_i, 0, 1]
            current_multix = [min_value, max_value, 0, 1]
        # Else, assume the input already has preexisting numeric range values
        else:
            # print('Preexisting value in numeric input')
            current_indx = [prev_start_i, prev_end_i, 0, 1]
            current_multix = [prev_start_g, prev_end_g, 0, 1]

        # If User resized plotly graph, estimate numeric input
        if 'graph_seconds' in changed_id:
            # Disable if seconds are enabled
            if len(seconds) > 0:
                disabled = True
        elif 'plot_ind_group' in changed_id:
            # print('User resized on plotly')
            current_indx = determine_range(ind_relayout, current_indx, x)
        elif 'plot_multi_group' in changed_id:
            current_multix = determine_range(multi_relayout, current_multix, x)

        if len(seconds) > 0:
            disabled = True

        # Compute Seconds
        fps = int(fps_dict[H5list_videos[0]])
        start_i_sec, end_i_sec = round(current_indx[0]/fps, 3), round(current_indx[1]/fps, 3)
        start_g_sec, end_g_sec = round(current_multix[0] / fps, 3), round(current_multix[1] / fps, 3)

        return current_indx[0], current_indx[1], disabled, disabled, \
               current_multix[0], current_multix[1], disabled, disabled\
            , \
               start_i_sec, end_i_sec, start_g_sec, end_g_sec


# Find and return the total x axis in seconds and the title of x axis, Given list of total frames and behavior name
def show_seconds(start, end, name):
    fps = int(fps_dict[name])
    t = np.arange(start, end + 1/fps, 1/fps)
    # print('Seconds Range: ' + str(t[0]) + ' ' + str(t[-1]))
    return t, 'Seconds'


# Updates layout of the graph figure,
# given graph figure, plot title, x and y axis titles, boolean show grid, font type, font size
def update_layout(graph, plot_title, x_axis, y_axis, show_grid, font, size, show_bg):
    grid, bg = False, '#ffffff'

    if len(show_grid) > 0:
        grid = True
    if len(show_bg) > 0:
        bg = '#e5ecf6'

    graph['layout'] = go.Layout(title={'text': plot_title, 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
                                xaxis_title=x_axis, yaxis_title=y_axis, xaxis={'showgrid': grid},
                                yaxis={'showgrid': grid}, font=dict(family=font, size=size),
                                plot_bgcolor=bg)
    return graph


# Returns the start and end x axis range for graph figure,
# given start and end frame given by numeric input, list of all frame numbers
def set_graph_viewing_range(start, end, whole_x):
    try:
        low_number = start
        hi_number = end

        if start > end or start < 0 or end > len(whole_x)-1:
            pass

        absolute_low_different_function = lambda whole_x: abs(whole_x - low_number)
        closest_low_value = min(whole_x, key=absolute_low_different_function)

        absolute_hi_different_function = lambda whole_x: abs(whole_x - hi_number)
        closest_hi_value = min(whole_x, key=absolute_hi_different_function)

        start_ind = whole_x.index(closest_low_value)
        end_ind = whole_x.index(closest_hi_value)
    # Initial load, numeric input has no value
    except TypeError:
        start_ind = 0
        end_ind = len(whole_x)-1

    return start_ind, end_ind


@app.callback(
    Output('plot_ind_group', 'figure'),
    [Input('video_numbers', 'value'),
     Input('feature_selector', 'value'),
     Input('group_selector', 'value'),
     Input('video_mean', 'value'),
     Input('update_graph', 'n_clicks'),
     Input('ind_group_colors', 'colorscale'),
     Input('graph_seconds', 'value'),
     Input('show_grid', 'value'),
     Input('set_ind_title', 'n_clicks'),
     Input('fonts', 'value'),
     Input('font_size', 'value'),
     Input('show_bg', 'value'),
     Input('submit_ind_range','n_clicks'),
     Input('ind_start_sec', 'value'),
     Input('ind_end_sec', 'value')],
    [State('behaviors', 'value'),
     State('category_selector', 'value'),
     State('color_pickers', 'children'),
     State('ind_start_x', 'value'),
     State('ind_end_x', 'value'),
     State('ind_graph_title', 'value')])
def update_ind_graph(select_vids, feature_name, group_name, show_mean, update_colors,
                     colorset, seconds, showgrid, update_title, font, fontsize, showbg, update_ind_range, start_s, end_s,
                     behavior, category, colors, startx, endx, custom_title):
    fig = go.Figure()
    graph_title =  feature_name + ' ' + group_name
    color_name = {'Mean': colors[len(colors)-1]['props']['color']}
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    # Probability Line Graph
    if 'Probability' in feature_name:
        x_title = 'Frames'
        y_title = 'Probability'

        try:
            for v in select_vids:

                # one_line = all_data_behaviors[behavior][category][feature_name][v]
                # x_axis = list(range(all_data_behaviors[behavior][category][feature_name][v].count()))
                one_line = all_data_behaviors[behavior][category][feature_name].eval(v)
                x_axis = list(range(one_line.count()))

                start_x, end_x = set_graph_viewing_range(startx, endx, x_axis)

                # convert x axis to seconds if seconds box is checked
                if len(seconds) > 0:
                    x_axis, x_title = show_seconds(start_s, end_s, v)
                else:
                    # if range in numeric input has been submitted, update the viewing range in plotly'
                    x_axis = list(range(start_x, end_x+1))

                one_line = one_line[start_x: end_x+1]

                fig.add_trace(go.Scattergl(x=x_axis, y=one_line,
                                         mode='lines',
                                         name=v,
                                         line=dict(color=random.choice(colorset))))
            # Calculating Mean Line for Individual Group
            if len(show_mean) > 0:
                if groups_exist:
                    mean_group_name = group_name + '_Mean'
                else:
                    mean_group_name = 'Total_Mean'

                x_axis = list(range(all_data_behaviors[behavior][category][feature_name].eval(mean_group_name).count()))
                group_mean = all_data_behaviors[behavior][category][feature_name][select_vids].mean(axis=1)
                # print(len(group_mean))

                start_x, end_x = set_graph_viewing_range(startx, endx, x_axis)

                if len(seconds) > 0:
                    x_axis, x_title = show_seconds(start_s, end_s, behavior)
                else:
                    x_axis = list(range(start_x, end_x + 1))

                group_mean = group_mean[start_x: end_x+1]

                fig.add_trace(go.Scatter(x=x_axis,
                                         y=group_mean,
                                         mode='lines',
                                         name='Group Mean',
                                         line=dict(color=color_name['Mean'])))
        except KeyError:
            print('video not found')
            print(v)
            pass
    # Bar Graph
    else:
        x_title = 'Video / Group'
        if '# bout events' in feature_name:
            y_title = 'Mean # of Events'
        else:
            y_title = 'Mean Time (seconds)'
        try:
            for v in select_vids:
                try:
                    one_bar = [all_data_behaviors[behavior][category][feature_name].loc[v]]
                except KeyError:
                    one_bar = [0]
                x_axis = [v]

                fig.add_trace(go.Bar(x=x_axis, y=one_bar, text=one_bar, textposition='auto', name=v,
                                     marker_color=random.choice(colorset)))
        except KeyError:
            print('Video not found in H5 file.')
            pass

    if 'set_ind_title' in changed_id:
        if custom_title is not None:
            graph_title = custom_title

    fig = update_layout(fig, graph_title, x_title, y_title, showgrid, font, fontsize, showbg)
    return fig

# Input: Pandas dataframe of current group data
# Output: Returns list formatted mean and standard error value for a group, given list of values in a group
def calculate_group_statistics(group_data):
    mean_values = []
    std_error_values = []
    mean_values.append(round(statistics.mean(group_data), 3))
    count = len(group_data)
    std = statistics.stdev(group_data)
    std_error_values.append(round((std / math.sqrt(count)), 3))

    return mean_values, std_error_values


@app.callback(
    Output('plot_multi_groups', 'figure'),
    [Input('multi_group_selector', 'value'),
     Input('feature_selector', 'value'),
     Input('group_mean', 'value'),
     Input('update_graph', 'n_clicks'),
     Input('color_pickers', 'children'),
     Input('graph_seconds', 'value'),
     Input('show_grid', 'value'),
     Input('set_multi_title', 'n_clicks'),
     Input('fonts', 'value'),
     Input('font_size', 'value'),
     Input('show_bg', 'value'),
     Input('submit_multi_range', 'n_clicks'),
     Input('error_type', 'value'),
     Input('ind_start_sec', 'value'),
     Input('ind_end_sec', 'value')],
    [State('behaviors', 'value'),
     State('category_selector', 'value'),
     State('multi_start_x', 'value'),
     State('multi_end_x', 'value'),
     State('mean_graph_title', 'value')])
def update_multi_graph(list_groups, feature_name, show_group_mean, update_colors, colors, seconds,
                       showgrid, update_title, font, fontsize, showbg, update_multi_range, error, start_s, end_s,
                       behavior, category, startx, endx, custom_title):
    fig = go.Figure()
    graph_title = feature_name + ' - Group Means'
    color_name = {}
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    for group in list_groups:
        for c in colors:
            if c['type'] == 'ColorPicker' and group in c['props']['id']:
                color_name[group] = c['props']['color']
    color_name['Mean'] = colors[len(colors) - 1]['props']['color']

    if 'Probability' in feature_name:

        x_title = 'Frames'
        y_title = 'Probability'

        for group in list_groups:
            x_axis = list(range(all_data_behaviors[behavior][category][feature_name][group + '_Mean'].count()))
            one_mean_line = all_data_behaviors[behavior][category][feature_name][group+'_Mean']
            start_x, end_x = set_graph_viewing_range(startx, endx, x_axis)

            if len(seconds) > 0 and 'graph_seconds' in changed_id:
                x_axis, x_title = show_seconds(start_s, end_s, group)
            else:
                x_axis = list(range(start_x, end_x + 1))

            one_mean_line = one_mean_line[start_x: end_x + 1]

            fig.add_trace(go.Scattergl(x=x_axis, y=one_mean_line,
                                     mode='lines',
                                     name=group+' Mean',
                                     line=dict(color=color_name[group])))

        if len(show_group_mean) > 0:
            x_axis = list(range(all_data_behaviors[behavior][category][feature_name]['Total_Mean'].count()))
            total_mean_line = all_data_behaviors[behavior][category][feature_name]['Total_Mean']
            start_x, end_x = set_graph_viewing_range(startx, endx, x_axis)

            if len(seconds) > 0:
                x_axis, x_title = show_seconds(startx, endx, behavior)
            else:
                x_axis = list(range(start_x, end_x + 1))

            total_mean_line = total_mean_line[start_x: end_x + 1]

            fig.add_trace(
                go.Scatter(x=x_axis,
                           y=total_mean_line,
                           mode='lines',
                           name='Overall Mean',
                           line=dict(color=color_name['Mean'])))
    # Bar Graph
    else:
        x_title = 'Video'
        if '# bout events' in feature_name:
            y_title = 'Mean # of Events'
        else:
            y_title = 'Mean Time (seconds)'
        try:
            for group in list_groups:
                try:
                    group_values = all_data_behaviors[behavior][category][feature_name].loc[groups_dict[group]].tolist()
                    mean_values, std_error_values = calculate_group_statistics(group_values)
                except KeyError:
                    mean_values, std_error_values = [0], [0]
                x_axis = [group]

                # print(group_values)


                if 'both' in error:
                    symmetry = True
                    array = std_error_values
                    array_minus = []
                elif 'above' in error:
                    symmetry = False
                    array = std_error_values
                    array_minus = [0]
                else:
                    symmetry = False
                    array = [0]
                    array_minus = std_error_values

                fig.add_trace(go.Bar(x=x_axis, y=mean_values, text=mean_values, textposition='auto',
                                     error_y=dict(type='data', symmetric=symmetry, array=array, arrayminus=array_minus),
                                     hovertext=std_error_values, name=group, marker_color=color_name[group]))
        except KeyError:
            print('Group not found.')
            pass

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if 'set_multi_title' in changed_id:
        if custom_title is not None:
            graph_title = custom_title

    fig = update_layout(fig, graph_title, x_title, y_title, showgrid, font, fontsize, showbg)

    return fig

# Input: Graph figure, most recently changed property
# Output: Edited Pandas dataframe, title of output csv
def load_results(graph_fig, prop_id):
    df = pd.DataFrame()
    csv_title = graph_fig['layout']['title']['text']

    if 'Probability' in csv_title:
        for line in graph_fig['data']:
            # print(line)
            s = pd.Series(line['y'])
            # print(s)
            df = pd.concat([df, s.rename(line['name'])], axis=1)
        index_name = graph_fig['layout']['xaxis']['title']['text']

        df[index_name] = graph_fig['data'][0]['x']
        df = df.set_index(index_name)

    else:
        groups = []
        data = []
        error = []

        for bar in graph_fig['data']:
            groups += bar['x']
            data += bar['y']
            if 'download_group_data' in prop_id:
                error += bar['error_y']['array']

        df['Groups'] = groups
        df[csv_title] = data
        if 'download_group_data' in prop_id:
            df['Error'] = error

        df = df.set_index('Groups')
    return df, csv_title

# Downloads data .csv to project folder
@app.callback(
     Output('comments', 'children'),
     [Input('download_data', 'n_clicks'),
      Input('download_group_data', 'n_clicks')],
     [State('plot_ind_group', 'figure'),
      State('plot_multi_groups', 'figure'),
      State('csv_file_name', 'value')],
)
def download(ind_click, multi_click, ind_figure, multi_figure, file_name):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if 'download_data' in changed_id:
        figure = ind_figure

        click = ind_click
    else:
        figure = multi_figure
        click = multi_click

    if click > 0:
        download_df, str_name = load_results(figure, changed_id)
        # print(download_df)
        if file_name is not None:
            str_name = file_name

# Set csv save path to csv folder
        csvSavePath = get_download_path()
        csvSavePath = os.path.join(csvSavePath, str_name + '.csv')
        download_df.to_csv(csvSavePath)
        # print('Downloaded csv as ' + str_name + '.csv')
        return ['Saved ' + str_name + '.csv to Downloads']
    else:
        return []


if __name__ == '__main__':
    app.run_server(debug=False)

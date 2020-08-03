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
import plotly.io as pio



# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_stylesheets = ['dash_simba.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


# Extract video data from hdf5 file
# filePath = "/Volumes/Data/DeepLabCut/misc/plotly_dev/input_files/ExampleStore_2.h5"
# groupPath = '/Volumes/Data/DeepLabCut/misc/plotly_dev/input_files/Groups.csv'
filePath = r"Z:\DeepLabCut\misc\plotly_dev\input_files\ExampleStore_2.h5"
groupPath = r'Z:\DeepLabCut\misc\plotly_dev\input_files\Groups_copy.csv'

hf = h5py.File(filePath, 'r')
H5list_videos = hf.get('VideoData')

list_categories = [k for k in hf.keys() if k != 'Dataset_info']


def get_groups(path):
    g_list = []
    g_dict = {}
    df_groups = pd.read_csv(path)
    for group in df_groups.keys():
        g_list.append(group)
        g_dict[group] = df_groups[group].dropna().tolist()
    return g_list, g_dict


group_list, groups_dict = get_groups(groupPath)


def get_feature_data(videos, behaviors, categories, path):
    dict_features = {}

    for b in behaviors:
        dict_features[b] = {}

        for category in categories:
            if category == 'VideoData':
                video = pd.read_hdf(path, key=category + '/' + videos[0])
                dict_features[b][category] = {column: get_probability_data(column, videos, path) for column in video.columns if '_'+b in column}
            else:
                video = pd.read_hdf(path, key=category)
                dict_features[b][category] = {column: video[column] for column in video.columns if b in column}

    return dict_features


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


# Variables
H5list_videos = list(H5list_videos.keys())
list_behaviors = pd.read_hdf(filePath, key='Dataset_info')
fps_dict = dict(zip(list_behaviors['Classifier_names'], list_behaviors['fps']))
list_behaviors = list_behaviors['Classifier_names']
all_data_behaviors = get_feature_data(H5list_videos, list_behaviors, list_categories, filePath)
# dict_df_video_data = get_probability_data(['Probability_Attack', 'Probability_Sniffing'])
list_fonts = ['Verdana', 'Helvetica', 'Calibri', 'Arial', 'Arial Narrow', 'Candara', 'Geneva', 'Courier New',  'Times New Roman']

hf.close()

config = {
  'toImageButtonOptions': {
    'format': 'svg', # one of png, svg, jpeg, webp
    'filename': 'custom_image',
    'height': 500,
    'width': 700,
    'scale': 1 # Multiply title/legend/axis/canvas sizes by this factor
  }
}


def create_color_pickers():
    output=[]
    for g in group_list:
        new_picker = [html.Label(g), dcp.ColorPicker(id='ColorPicker'+str(g), color='#609BE0')]
        output += new_picker
    output += [html.Label('Mean'), dcp.ColorPicker(id='ColorPickerMean', color='#f22')]
    return output


app.layout = html.Div(children=[
    html.Div(id='header', className='row', children=[
        html.H1(children='SimBA Data Visualization', className='nine columns'),

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
                            clearable=False,
                            options=[{'label': c, 'value': c} for c in list_categories]),

                        html.Label('Feature:'),
                        dcc.Dropdown(
                            id='feature_selector',
                            clearable=False),

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
                            labelStyle={'display': 'inline-block'}),

                        html.Hr(),
                        html.H6('Plotting Individual Groups'),
                        html.Label('Select Single Group to Plot:'),
                        dcc.Dropdown(
                            id='group_selector',
                            options=[
                                {'label': key, 'value': key} for key in groups_dict.keys()],
                            multi=False,
                            clearable=False,
                            # fix later
                            value=group_list[0]),

                        html.Label('Select Video(s):'),
                        dcc.Dropdown(
                            id='video_numbers',
                            multi=True,
                            clearable=False),
                        dcc.Checklist(
                            id='video_mean',
                            options=[{'label': 'Show Group Mean', 'value': 'showVideoMean', 'disabled': False}],
                            value=['showVideoMean'],
                            labelStyle={'display': 'inline-block'})]
                    )]
            ),

            dcc.Tab(
                label='Graph Settings',
                value='graph_settings',
                children=[
                    html.Div(className='twelve columns', id='graph_settings', children=[
                        html.H6('Group Color Properties'),

                        html.Div(id='color_pickers',
                                 children=create_color_pickers()),
                        html.Label('Colorscale Gradient for Individual Group'),
                        dcs.DashColorscales(id='ind_group_colors', colorscale=['#fbb4ae', '#b3cde3', '#ccebc5', '#decbe4', '#fed9a6', '#ffffcc']),
                        html.Button('Update Colors', id='update_graph', n_clicks=0, type='submit'),

                        html.Hr(),

                        html.Div(className='twelve columns',
                                 id='probability-graph-properties',
                                 children=[
                                     html.H6('Probability Graph Properties'),
                                     html.Label('Set Multi Group Frames:'),
                                     dcc.Input(id='multi_start_x', type='number', className='four columns'),
                                     dcc.Input(id='multi_end_x', type='number', className='four columns'),
                                     html.Button(id='submit_multi_range', className='three columns', n_clicks=1, children='Submit', type='submit'),
                                     html.Label('Multi Group (Seconds)'),
                                     dcc.Input(id='multi_start_sec', type='number', className='four columns', disabled=True),
                                     dcc.Input(id='multi_end_sec', type='number', className='four columns', disabled=True),
                                     html.Button(id='reset_multi_range', className='twelve columns', n_clicks=1, children='Reset Axes', type='reset'),

                                     html.Label('Set Individual Group Frames:', className='twelve columns'),
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
                                         id= 'graph_seconds',
                                         options=[{'label': 'Graph X Axis in Seconds', 'value': 'show_seconds'}],
                                         value=[],
                                         labelStyle={'display': 'inline-block'}),
                                     dcc.Checklist(
                                         id='show_grid',
                                         options=[{'label': "Show Grid lines", 'value': 'showGridLines'}],
                                         value=['showGridLines'],
                                         labelStyle={'display': 'inline-block'}),

                                     html.Label('Choose Font:'),
                                     dcc.Dropdown(
                                         id='fonts',
                                         options=[{'label': f, 'value': f} for f in list_fonts],
                                         value=list_fonts[0],
                                         multi=False,
                                     clearable=False),

                                     html.Label('Font Size:'),
                                     daq.NumericInput(
                                         id='font_size',
                                         value=12,
                                         min=8,
                                         max=48),
                                     html.Button(
                                         id='download_data',
                                         n_clicks=0,
                                         children='Download Individual Group CSV'),
                                     html.Div(id='comments')])]
                             )]),
            dcc.Tab(
                label='Download Settings',
                value='download_settings',
                children=[
                    html.Div( id='download-settings', children=[
                        html.Label('Image Height:'),
                        daq.NumericInput(
                            id='img_height',
                            value=500,
                            min=0,
                            max=1000
                        ),
                        html.Label('Image Width:'),
                        daq.NumericInput(
                            id='img_width',
                            value=700,
                            min=0,
                            max=1000
                        ),
                        html.Label('Enter file name:'),
                        dcc.Input(
                            id='file_name'
                        ),
                        html.Button('Set Download Settings', id='submit_download', type='submit')
                    ])]
            )
        ])
    ]),

    html.Div(id='graphs', className='eight columns', children=[

        dcc.Graph(
            id='plot_multi_groups',
            config=config
        ),

        dcc.Graph(
            id='plot_ind_group',
            config=config
        )
    ])
])

# Sets config settings for graph
@app.callback(
    [Output('plot_ind_group', 'config'),
     Output('plot_multi_groups', 'config')],
    [Input('submit_download', 'n_clicks')],
    [State('img_height', 'value'),
     State('img_width', 'value'),
     State('file_name', 'value'),
     State('plot_ind_group', 'config')]
)
def update_config(submit, height, width, filename, ind_config):
    if submit is not None:
        print(ind_config)
        print(height)
        print(width)
        print(filename)
        new_config = {
            'toImageButtonOptions': {
                'format': 'png',  # one of png, svg, jpeg, webp
                'filename': filename,
                'height': height,
                'width': width,
                'scale': 1  # Multiply title/legend/axis/canvas sizes by this factor
            }
        }
        print('config settings saved')
        return new_config, new_config
    else:
        return config, config
# Sets available options in dropdowns based on behavior selected
@app.callback(
    Output('feature_selector', 'options'),
    [Input('category_selector', 'value'),
     Input('behaviors', 'value')])
def set_features_options(selected_category, behavior):
    return [{'label': f, 'value': f} for f in all_data_behaviors[behavior][selected_category]]


# Sets list of available videos to plot on graph based on group selected
@app.callback(
    Output('video_numbers', 'options'),
    [Input('group_selector', 'value')])
def set_group_options(selected_group):
    return [{'label': v, 'value': v} for v in groups_dict[selected_group]]


# Sets default value for Feature
@app.callback(
    Output('feature_selector', 'value'),
    [Input('feature_selector', 'options')])
def set_feature_value(options):
    try:
        return options[0]['value']
    except IndexError:
        return []

# Sets default value for Category
@app.callback(
    Output('category_selector', 'value'),
    [Input('category_selector', 'options')])
def set_category_value(options):
    try:
        return options[0]['value']
    except IndexError:
        return []

# Sets default values for videos to plot
@app.callback(
    Output('video_numbers', 'value'),
    [Input('video_numbers', 'options')])
def set_video_option(options):
    try:
        return [vid['value'] for vid in options]
    except KeyError:
        return []

# Sets visibility of graphing options
@app.callback(
    [Output('probability-graph-properties', 'style'),
     Output('bar-graph-properties', 'style')],
    [Input('category_selector', 'value')])
def graph_options_visibility(category):
    if 'VideoData' in category:
        return {'display': 'block'}, {'display': 'none'}
    else:
        return {'display': 'none'}, {'display': 'block'}


# Disables Mean checklist if bar graph
# @app.callback(
#     [Output('group_mean', 'options'),
#      Output('video_mean', 'options')],
#     [Input('category_selector', 'options')],
# )
# def disable_mean(category):
# 
#     if 'VideoData' not in category:
#         disabled = True
#     else:
#         disabled = False
#     group_options = [{'label': "Show Total Mean", 'value': 'showGroupMean', 'disabled': disabled}],
#     video_options = [{'label': 'Show Group Mean', 'value': 'showVideoMean', 'disabled': disabled}],
#
#     return group_options, video_options

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
        if ((prev_start_i or prev_end_i or prev_start_g or prev_end_g) is None) or 'reset_ind_range' in changed_id:
            # print('Previously no Values in numeric input')
            current_indx = [min_value, max_value, 0, 1]
            current_multix = [min_value, max_value, 0, 1]
        # Else, assume the input already has preexisting numeric range values
        else:
            print('Preexisting value in numeric input')
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
        fps = int(fps_dict[behavior])
        start_i_sec, end_i_sec = round(current_indx[0]/fps, 3), round(current_indx[1]/fps, 3)
        start_g_sec, end_g_sec = round(current_multix[0] / fps, 3), round(current_multix[1] / fps, 3)

        return current_indx[0], current_indx[1], disabled, disabled, \
               current_multix[0], current_multix[1], disabled, disabled\
            , \
               start_i_sec, end_i_sec, start_g_sec, end_g_sec


# Find and return the total x axis in seconds and the title of x axis, Given list of total frames and behavior name
def show_seconds(start, end, behavior_name):
    fps = int(fps_dict[behavior_name])
    t = np.arange(start, end + 1/fps, 1/fps)
    # print('Seconds Range: ' + str(t[0]) + ' ' + str(t[-1]))
    return t, 'Seconds'


# Updates layout of the graph figure,
# given graph figure, plot title, x and y axis titles, boolean show grid, font type, font size
def update_layout(graph, plot_title, x_axis, y_axis, value, font, size):
    grid = False
    if len(value) > 0:
        grid = True
    graph['layout'] = go.Layout(title={'text': plot_title, 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
                                xaxis_title=x_axis, yaxis_title=y_axis, xaxis={'showgrid': grid},
                                yaxis={'showgrid': grid}, font=dict(family=font, size=size))
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
    Output(component_id='plot_ind_group', component_property='figure'),
    [Input(component_id='video_numbers', component_property='value'),
     Input(component_id='feature_selector', component_property='value'),
     Input(component_id='group_selector', component_property='value'),
     Input(component_id='video_mean', component_property='value'),
     Input(component_id='update_graph', component_property='n_clicks'),
     Input(component_id='ind_group_colors', component_property='colorscale'),
     Input(component_id='graph_seconds', component_property='value'),
     Input(component_id='show_grid', component_property='value'),
     Input(component_id='fonts', component_property='value'),
     Input(component_id='font_size', component_property='value'),
     Input(component_id='submit_ind_range',component_property='n_clicks'),
     Input(component_id='ind_start_sec', component_property='value'),
     Input(component_id='ind_end_sec', component_property='value')],
    [State(component_id='behaviors', component_property='value'),
     State(component_id='category_selector', component_property='value'),
     State(component_id='color_pickers', component_property='children'),
     State(component_id='ind_start_x', component_property='value'),
     State(component_id='ind_end_x', component_property='value')])
def update_ind_graph(select_vids, feature_name, group_name, show_mean, update_colors,
                     colorset, seconds, showgrid, font, fontsize, update_ind_range, start_s, end_s,
                     behavior, category, colors, startx, endx):
    fig = go.Figure()
    # print(startx, endx)
    color_name = {'Mean': colors[len(colors)-1]['props']['color']}

    # Probability Line Graph
    if 'Probability' in feature_name:
        x_title = 'Frames'
        y_title = 'Probability'

        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

        for v in select_vids:
            one_line = all_data_behaviors[behavior][category][feature_name][v]
            x_axis = list(range(all_data_behaviors[behavior][category][feature_name][v].count()))
            start_x, end_x = set_graph_viewing_range(startx, endx, x_axis)

            # convert x axis to seconds if seconds box is checked
            if len(seconds) > 0:
                x_axis, x_title = show_seconds(start_s, end_s, behavior)
            else:
            # if range in numeric input has been submitted, update the viewing range in plotly'
                x_axis = list(range(start_x, end_x+1))

            one_line = one_line[start_x: end_x+1]

            fig.add_trace(go.Scatter(x=x_axis, y=one_line,
                                     mode='lines',
                                     name=v,
                                     line=dict(color=random.choice(colorset))))
        # Calculating Mean Line for Individual Group
        if len(show_mean) > 0:
            mean_group_name = group_name + '_Mean'
            group_mean = all_data_behaviors[behavior][category][feature_name][select_vids].mean(axis=1)
            x_axis = list(range(all_data_behaviors[behavior][category][feature_name][mean_group_name].count()))
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
    # Bar Graph
    else:
        x_title = 'Video'
        if 'Attack # bout events' in feature_name:
            y_title = 'Mean # of Events'
        else:
            y_title = 'Mean Time (seconds)'

        for v in select_vids:
            try:
                one_bar = [all_data_behaviors[behavior][category][feature_name+'.csv'].loc[v]]
            except KeyError:
                one_bar = [0]
            x_axis = [v]

            fig.add_trace(go.Bar(x=x_axis, y=one_bar, text=one_bar, textposition='auto', name=v,
                                 marker_color=random.choice(colorset)))

    fig = update_layout(fig, group_name + ' ' + feature_name, x_title, y_title, showgrid, font, fontsize)

    return fig

# Returns list formatted mean and standard error value for a group, given list of values in a group
def calculate_group_statistics(group_data):
    mean_values = []
    std_error_values = []
    mean_values.append(round(statistics.mean(group_data), 3))
    count = len(group_data)
    std = statistics.stdev(group_data)
    std_error_values.append(round((std / math.sqrt(count)), 3))

    return mean_values, std_error_values


@app.callback(
    Output(component_id='plot_multi_groups', component_property='figure'),
    [Input(component_id='multi_group_selector', component_property='value'),
     Input(component_id='feature_selector', component_property='value'),
     Input(component_id='group_mean', component_property='value'),
     Input(component_id='update_graph', component_property='n_clicks'),
     Input(component_id='color_pickers', component_property='children'),
     Input(component_id='graph_seconds', component_property='value'),
     Input(component_id='show_grid', component_property='value'),
     Input(component_id='fonts', component_property='value'),
     Input(component_id='font_size', component_property='value'),
     Input(component_id='submit_multi_range', component_property='n_clicks'),
     Input(component_id='error_type', component_property='value')],
    [State(component_id='behaviors', component_property='value'),
     State(component_id='category_selector', component_property='value'),
     State(component_id='multi_start_x', component_property='value'),
     State(component_id='multi_end_x', component_property='value')])
def update_multi_graph(list_groups, feature_name, show_group_mean, update_colors, colors, seconds, showgrid, font, fontsize, update_multi_range, error,
                       behavior, category, startx, endx):
    fig = go.Figure()
    # for each group selected in the checkbox, display the mean data of the current feature of the group selected

    color_name = {}

    for group in list_groups:
        for c in colors:
            if c['type'] == 'ColorPicker' and group in c['props']['id']:
                color_name[group] = c['props']['color']
    color_name['Mean'] = colors[len(colors) - 1]['props']['color']

    if 'Probability' in feature_name:

        x_title = 'Frames'
        y_title = 'Probability'

        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

        for group in list_groups:
            x_axis = list(range(all_data_behaviors[behavior][category][feature_name][group + '_Mean'].count()))
            one_mean_line = all_data_behaviors[behavior][category][feature_name][group+'_Mean']
            start_x, end_x = set_graph_viewing_range(startx, endx, x_axis)

            if len(seconds) > 0 and 'graph_seconds' in changed_id:
                x_axis, x_title = show_seconds(start_x, end_x, behavior)
            else:
                x_axis = list(range(start_x, end_x + 1))

            one_mean_line = one_mean_line[start_x: end_x + 1]

            fig.add_trace(go.Scatter(x=x_axis, y=one_mean_line,
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
        if 'Attack # bout events' in feature_name:
            y_title = 'Mean # of Events'
        else:
            y_title = 'Mean Time (seconds)'

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

    fig = update_layout(fig, feature_name + ' Group Means', x_title, y_title, showgrid, font, fontsize)

    return fig


def load_results(graph_fig):
    df = pd.DataFrame()
    csv_title = graph_fig['layout']['title']['text']

    if 'Probability' in csv_title:
        for line in graph_fig['data']:
            # print(line)
            df[line['name']] = line['y']
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
            error += bar['error_y']['array']
        df['Groups'] = groups
        df[csv_title] = data
        df['Error'] = error
        df = df.set_index('Groups')
    return df, csv_title

def download_image(fig):
    if not os.path.exists('images'):
        os.mkdir('images')
    fig.to_image(format='png', width='100', height='500', engine='kaleido')

@app.callback(
     Output('comments', 'children'),
     [Input('download_data', 'n_clicks')],
     [State('plot_ind_group', 'figure')],
)
def download(click, figure):
    if click > 0:
        download_df, str_name = load_results(figure)
        print(download_df)
        download_df.to_csv(str_name + '.csv')
        pio.write_image(figure, r'fig1.svg', format='svg')
        return ['Downloaded csv as ' + str_name + '.csv']
    else:
        return []
    # changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    # if 'download_data' in changed_id:
    #     download_df.to_csv(str_name+'.csv')

if __name__ == '__main__':
    app.run_server(debug=True)


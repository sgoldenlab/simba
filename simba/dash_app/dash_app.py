import json
import math
import os
import random
import statistics
import sys

import dash
import dash_color_picker as dcp
import dash_colorscales as dcs
import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State

#
# FILE_PATH = sys.argv[1]
# GROUP_PATH = sys.argv[2]
FILE_PATH = "/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/logs/SimBA_dash_storage_20230202105812.h5"
GROUP_PATH = None

external_stylesheets = [
    "dash_simba_base.css",
    "//use.fontawesome.com/releases/v5.0.7/css/all.css",
]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
FONTS = [
    "Verdana",
    "Helvetica",
    "Calibri",
    "Arial",
    "Arial Narrow",
    "Candara",
    "Geneva",
    "Courier New",
    "Times New Roman",
]

GRAPH_CONFIG = {
    "toImageButtonOptions": {
        "format": "svg",
        "filename": "custom_image",
        "height": 500,
        "width": 700,
        "scale": 1,
    }
}


class DashApp(object):
    def __init__(self):
        self.groups_exist = False
        self.h5_file_path, self.group_path = FILE_PATH, GROUP_PATH
        self.graph_config = GRAPH_CONFIG
        self.video_info_df = pd.read_hdf(self.h5_file_path, key="Video_info")
        self.video_names = list(self.video_info_df.index)
        self.clf_names = list(pd.read_hdf(self.h5_file_path, key="Classifier_names"))
        if os.path.isfile(self.group_path):
            self.get_groups()
        self.fps = dict(zip(self.video_names, self.video_info_df["fps"]))
        self.run()

    def get_groups(self):
        self.groups_exist = True
        self.groups = {}
        self.groups_lookup = {}
        df = pd.read_csv(self.group_path).dropna()
        for group_id in df["GROUP"].unique():
            self.groups[group_id] = df[df["GROUP"] == group_id].tolist()

    def set_display(self):
        if self.groups_exist:
            return {"display": "block"}
        else:
            return {"display": "none"}

    def run(self):
        app.layout = html.Div(
            children=[
                html.Div(
                    id="header",
                    className="row",
                    children=[
                        html.H1(
                            children="SimBA Interactive Data Visualization Dashboard",
                            className="nine columns",
                        ),
                        html.Img(
                            src=app.get_asset_url("TheGoldenLab.PNG"),
                            className="three columns",
                            style={
                                "height": "15%",
                                "width": "15%",
                                "float": "right",
                                "position": "relative",
                            },
                        ),
                    ],
                ),
                html.Div(
                    id="simba-control-tabs",
                    className="three columns",
                    children=[
                        dcc.Tabs(
                            id="simba-tabs",
                            value="all_data",
                            children=[
                                dcc.Tab(
                                    label="Data",
                                    value="all_data",
                                    children=[
                                        html.Div(
                                            id="plot_data",
                                            children=[
                                                html.H6("Select Behavior to Plot"),
                                                html.Label("Behaviors:"),
                                                dcc.Dropdown(
                                                    # className='eight columns',
                                                    id="behaviors",
                                                    options=[
                                                        {"label": b, "value": b}
                                                        for b in self.clf_names
                                                    ],
                                                    multi=False,
                                                    value=self.clf_names[0],
                                                    clearable=False,
                                                ),
                                                html.Label("Category:"),
                                                dcc.Dropdown(
                                                    id="category_selector",
                                                    # className='eight columns',
                                                    clearable=False,
                                                ),
                                                html.Label("Feature:"),
                                                dcc.Dropdown(
                                                    id="feature_selector",
                                                    clearable=False,
                                                ),
                                                html.Div(
                                                    id="multi_group_data",
                                                    children=[
                                                        html.Hr(),
                                                        html.H6("Plotting Group Means"),
                                                        html.Label("Select Group(s):"),
                                                        dcc.Dropdown(
                                                            id="multi_group_selector",
                                                            # className='eight columns',
                                                            options=[
                                                                {
                                                                    "label": key,
                                                                    "value": key,
                                                                }
                                                                for key in self.groups.keys()
                                                            ],
                                                            multi=True,
                                                            clearable=False,
                                                            value=self.groups,
                                                        ),
                                                        dcc.Checklist(
                                                            id="group_mean",
                                                            options=[
                                                                {
                                                                    "label": "Show Total Mean",
                                                                    "value": "showGroupMean",
                                                                    "disabled": False,
                                                                }
                                                            ],
                                                            value=["showGroupMean"],
                                                            labelStyle={
                                                                "display": "inline-block"
                                                            },
                                                        ),
                                                    ],
                                                    style=self.set_display(),
                                                ),
                                                html.Hr(),
                                                html.H6("Plotting Individual Groups"),
                                                html.Div(
                                                    id="show_group_opt",
                                                    children=[
                                                        html.Label(
                                                            "Select Single Group to Plot:",
                                                            style=self.set_display(),
                                                        ),
                                                        dcc.Dropdown(
                                                            id="group_selector",
                                                            options=[
                                                                {
                                                                    "label": key,
                                                                    "value": key,
                                                                }
                                                                for key in groups_dict.keys()
                                                            ],
                                                            multi=False,
                                                            clearable=False,
                                                        ),
                                                    ],
                                                    style=self.set_display(),
                                                ),
                                                html.Label("Select Video(s):"),
                                                dcc.Dropdown(
                                                    id="video_numbers",
                                                    multi=True,
                                                    clearable=False,
                                                ),
                                                dcc.Checklist(
                                                    id="video_mean",
                                                    options=[
                                                        {
                                                            "label": "Show Group Mean",
                                                            "value": "showVideoMean",
                                                            "disabled": False,
                                                        }
                                                    ],
                                                    value=["showVideoMean"],
                                                ),
                                            ],
                                        )
                                    ],
                                ),
                                dcc.Tab(
                                    label="Graph Settings",
                                    value="graph_settings",
                                    children=[
                                        html.Div(
                                            className="twelve columns",
                                            id="graph_settings",
                                            children=[
                                                html.Div(
                                                    id="color_properties",
                                                    className="twelve columns",
                                                    children=[
                                                        html.H6("Color Properties"),
                                                        html.Div(
                                                            id="color_pickers",
                                                            children=create_color_pickers(),
                                                        ),
                                                        html.Label(
                                                            "Colorscale Gradient for Individual Videos"
                                                        ),
                                                        dcs.DashColorscales(
                                                            id="ind_group_colors",
                                                            colorscale=[
                                                                "#fbb4ae",
                                                                "#b3cde3",
                                                                "#ccebc5",
                                                                "#decbe4",
                                                                "#fed9a6",
                                                                "#ffffcc",
                                                            ],
                                                        ),
                                                        html.Button(
                                                            "Update Colors",
                                                            id="update_graph",
                                                            n_clicks=0,
                                                            type="submit",
                                                        ),
                                                        html.Hr(),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="twelve columns",
                                                    id="probability-graph-properties",
                                                    children=[
                                                        html.H6(
                                                            "Probability Graph Properties"
                                                        ),
                                                        html.Div(
                                                            id="multi_prob_prop",
                                                            children=[
                                                                html.Label(
                                                                    "Set Group Viewing Frames:"
                                                                ),
                                                                dcc.Input(
                                                                    id="multi_start_x",
                                                                    type="number",
                                                                    className="four columns",
                                                                ),
                                                                dcc.Input(
                                                                    id="multi_end_x",
                                                                    type="number",
                                                                    className="four columns",
                                                                ),
                                                                html.Button(
                                                                    id="submit_multi_range",
                                                                    className="three columns",
                                                                    n_clicks=1,
                                                                    children="Submit",
                                                                    type="submit",
                                                                ),
                                                                html.Label(
                                                                    "Multi Group (Seconds)"
                                                                ),
                                                                dcc.Input(
                                                                    id="multi_start_sec",
                                                                    type="number",
                                                                    className="four columns",
                                                                    disabled=True,
                                                                ),
                                                                dcc.Input(
                                                                    id="multi_end_sec",
                                                                    type="number",
                                                                    className="four columns",
                                                                    disabled=True,
                                                                ),
                                                                html.Button(
                                                                    id="reset_multi_range",
                                                                    className="twelve columns",
                                                                    n_clicks=1,
                                                                    children="Reset Axes",
                                                                    type="reset",
                                                                ),
                                                            ],
                                                            style=self.set_display(),
                                                        ),
                                                        html.Label(
                                                            "Set Video Viewing Frames:",
                                                            className="twelve columns",
                                                        ),
                                                        dcc.Input(
                                                            id="ind_start_x",
                                                            type="number",
                                                            className="four columns",
                                                        ),
                                                        dcc.Input(
                                                            id="ind_end_x",
                                                            type="number",
                                                            className="four columns",
                                                        ),
                                                        html.Button(
                                                            id="submit_ind_range",
                                                            className="three columns",
                                                            n_clicks=1,
                                                            children="Submit",
                                                            type="submit",
                                                        ),
                                                        html.Label(
                                                            "Individual Group (Seconds)",
                                                            className="twelve columns",
                                                        ),
                                                        dcc.Input(
                                                            id="ind_start_sec",
                                                            type="number",
                                                            className="four columns",
                                                            disabled=True,
                                                        ),
                                                        dcc.Input(
                                                            id="ind_end_sec",
                                                            type="number",
                                                            className="four columns",
                                                            disabled=True,
                                                        ),
                                                        html.Button(
                                                            id="reset_ind_range",
                                                            className="twelve columns",
                                                            n_clicks=1,
                                                            children="Reset Axes",
                                                            type="reset",
                                                        ),
                                                    ],
                                                    style={"display": "block"},
                                                ),
                                                html.Div(
                                                    className="twelve columns",
                                                    id="bar-graph-properties",
                                                    children=[
                                                        html.H6("Bar Graph Properties"),
                                                        html.Label("Error Type:"),
                                                        dcc.RadioItems(
                                                            id="error_type",
                                                            options=[
                                                                {
                                                                    "label": "Above",
                                                                    "value": "above",
                                                                },
                                                                {
                                                                    "label": "Below",
                                                                    "value": "below",
                                                                },
                                                                {
                                                                    "label": "Both",
                                                                    "value": "both",
                                                                },
                                                            ],
                                                            value="both",
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="twelve columns",
                                                    id="additional-properties",
                                                    children=[
                                                        html.Hr(),
                                                        html.H6(
                                                            "Additional Properties"
                                                        ),
                                                        dcc.Checklist(
                                                            id="graph_seconds",
                                                            options=[
                                                                {
                                                                    "label": "Graph X Axis in Seconds",
                                                                    "value": "show_seconds",
                                                                }
                                                            ],
                                                            value=[],
                                                            labelStyle={
                                                                "display": "inline-block"
                                                            },
                                                        ),
                                                        dcc.Checklist(
                                                            id="show_grid",
                                                            options=[
                                                                {
                                                                    "label": "Show Grid lines",
                                                                    "value": "showGridLines",
                                                                }
                                                            ],
                                                            value=["showGridLines"],
                                                            labelStyle={
                                                                "display": "inline-block"
                                                            },
                                                        ),
                                                        dcc.Checklist(
                                                            id="show_bg",
                                                            options=[
                                                                {
                                                                    "label": "Show Background",
                                                                    "value": "showBg",
                                                                }
                                                            ],
                                                            value=["showBg"],
                                                            labelStyle={
                                                                "display": "inline-block"
                                                            },
                                                        ),
                                                        html.Div(
                                                            id="show_mean_graph_title",
                                                            children=[
                                                                html.Label(
                                                                    "Group Means Title:"
                                                                ),
                                                                dcc.Input(
                                                                    id="mean_graph_title",
                                                                    className="seven columns",
                                                                ),
                                                                html.Button(
                                                                    children=[
                                                                        html.I(
                                                                            n_clicks=0,
                                                                            className="fa fa-chart-area",
                                                                            style={
                                                                                "font-size": "24px"
                                                                            },
                                                                        ),
                                                                        html.P(
                                                                            "Set",
                                                                            style={
                                                                                "padding-left": "5px"
                                                                            },
                                                                        ),
                                                                    ],
                                                                    id="set_multi_title",
                                                                    n_clicks=0,
                                                                    className="four columns",
                                                                ),
                                                            ],
                                                            style=self.set_display(),
                                                        ),
                                                        html.Label(
                                                            "Individual Videos Title:"
                                                        ),
                                                        dcc.Input(
                                                            id="ind_graph_title",
                                                            className="seven columns",
                                                        ),
                                                        html.Button(
                                                            children=[
                                                                html.I(
                                                                    n_clicks=0,
                                                                    className="fa fa-chart-area",
                                                                    style={
                                                                        "font-size": "24px"
                                                                    },
                                                                ),
                                                                html.P(
                                                                    "Set",
                                                                    style={
                                                                        "padding-left": "5px"
                                                                    },
                                                                ),
                                                            ],
                                                            id="set_ind_title",
                                                            n_clicks=0,
                                                            className="four columns",
                                                        ),
                                                        html.Label("Choose Font:"),
                                                        dcc.Dropdown(
                                                            id="fonts",
                                                            options=[
                                                                {"label": f, "value": f}
                                                                for f in FONTS
                                                            ],
                                                            value=FONTS[0],
                                                            multi=False,
                                                            clearable=False,
                                                        ),
                                                        html.Label("Font Size:"),
                                                        daq.NumericInput(
                                                            id="font_size",
                                                            value=12,
                                                            min=8,
                                                            max=48,
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        )
                                    ],
                                ),
                                dcc.Tab(
                                    label="Download Settings",
                                    value="download_settings",
                                    children=[
                                        html.Div(
                                            id="download-settings",
                                            children=[
                                                html.Div(
                                                    id="csv_export",
                                                    className="twelve columns",
                                                    children=[
                                                        html.H6("CSV Export"),
                                                        html.Label(
                                                            "Enter csv file name:"
                                                        ),
                                                        dcc.Input(id="csv_file_name"),
                                                        html.P(children=[".csv"]),
                                                        html.Button(
                                                            children=[
                                                                html.I(
                                                                    n_clicks=0,
                                                                    className="fa fa-download",
                                                                    style={
                                                                        "font-size": "24px"
                                                                    },
                                                                ),
                                                                html.P(
                                                                    "Group Means",
                                                                    style={
                                                                        "padding-left": "5px"
                                                                    },
                                                                ),
                                                            ],
                                                            id="download_group_data",
                                                            n_clicks=0,
                                                            style=self.set_display(),
                                                            className="five columns",
                                                        ),
                                                        html.Button(
                                                            children=[
                                                                html.I(
                                                                    n_clicks=0,
                                                                    className="fa fa-download",
                                                                    style={
                                                                        "font-size": "24px"
                                                                    },
                                                                ),
                                                                html.P(
                                                                    "Videos",
                                                                    style={
                                                                        "padding-left": "5px"
                                                                    },
                                                                ),
                                                            ],
                                                            id="download_data",
                                                            n_clicks=0,
                                                            className="five columns",
                                                        ),
                                                        html.Div(
                                                            id="comments",
                                                            className="twelve columns",
                                                        ),
                                                        html.Hr(
                                                            className="twelve columns"
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    id="img_export",
                                                    className="twelve columns",
                                                    children=[
                                                        html.H6(
                                                            "Image Export",
                                                            className="twelve columns",
                                                        ),
                                                        html.Div(
                                                            id="img-dim",
                                                            className="twelve columns",
                                                            children=[
                                                                html.Label(
                                                                    "Enter image dimensions (px):",
                                                                    className="twelve columns",
                                                                ),
                                                                daq.NumericInput(
                                                                    id="img_height",
                                                                    className="three columns",
                                                                    value=500,
                                                                    min=0,
                                                                    max=1000,
                                                                ),
                                                                html.P(
                                                                    "x",
                                                                    className="one column",
                                                                ),
                                                                daq.NumericInput(
                                                                    id="img_width",
                                                                    className="three columns",
                                                                    value=700,
                                                                    min=0,
                                                                    max=1000,
                                                                ),
                                                            ],
                                                        ),
                                                        html.Label(
                                                            "Enter image name:",
                                                            className="twelve columns",
                                                        ),
                                                        dcc.Input(id="img_file_name"),
                                                        html.P(
                                                            id="img_type",
                                                            children=[".svg"],
                                                        ),
                                                        html.Label("File extension:"),
                                                        dcc.Dropdown(
                                                            id="img_ext",
                                                            options=[
                                                                {
                                                                    "label": "PNG",
                                                                    "value": "png",
                                                                },
                                                                {
                                                                    "label": "JPEG",
                                                                    "value": "jpeg",
                                                                },
                                                                {
                                                                    "label": "SVG",
                                                                    "value": "svg",
                                                                },
                                                            ],
                                                            value="svg",
                                                        ),
                                                        html.Button(
                                                            children=[
                                                                html.I(
                                                                    n_clicks=0,
                                                                    className="fa fa-save",
                                                                    style={
                                                                        "font-size": "24px"
                                                                    },
                                                                ),
                                                                html.P(
                                                                    "Save Download Settings",
                                                                    style={
                                                                        "padding-left": "5px"
                                                                    },
                                                                ),
                                                            ],
                                                            id="submit_download",
                                                            n_clicks=0,
                                                            type="submit",
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        )
                                    ],
                                ),
                            ],
                        )
                    ],
                ),
                html.Div(
                    id="graphs",
                    className="eight columns",
                    children=[
                        html.Div(
                            id="multi_group_container",
                            # style={'display': 'none'},
                            children=[
                                dcc.Graph(
                                    id="plot_multi_groups",
                                    style={"display": "none"},
                                    config=self.config,
                                )
                            ],
                        ),
                        dcc.Graph(
                            id="plot_ind_group",
                            config=self.config,
                            style={"display": "block"},
                        ),
                    ],
                ),
                html.Div(
                    id="credits",
                    className="footer",
                    children=[
                        html.Div(
                            className="six columns",
                            children=[
                                html.P("Created by "),
                                html.A(
                                    "Sophia Hwang",
                                    href="https://github.com/sophihwang26",
                                    target="_blank",
                                ),
                                html.P(" and "),
                                html.A(
                                    "Aasiya Islam",
                                    href="https://github.com/aasiya-islam",
                                    target="_blank",
                                ),
                            ],
                        ),
                        html.Div(
                            className="six columns",
                            children=[
                                html.P(
                                    "Built with Dash",
                                    style={
                                        "height": "15%",
                                        "width": "15%",
                                        "float": "right",
                                        "position": "relative",
                                    },
                                )
                            ],
                        ),
                    ],
                ),
            ]
        )


@app.callback(Output("img_type", "children"), [Input("img_ext", "value")])
def update_ext_text(extension):
    return [".{}".format(extension)]


if __name__ == "__main__":
    app.run_server(debug=False)

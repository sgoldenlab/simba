# Interactive Data Visualization in SimBA

## Overview

Once analyses have been performed in SimBA, users may need to visualize the results of the classifiers and have easy, interactive paths towards exporting the parts of the datasets of interest into third-party statistical and graphing applications and scripts. For this, SimBA has a built-in interactive graphical dashboard written in [Plotly](https://plotly.com/) and [Dash](https://github.com/plotly/dash) that allows users to inspect **huge** (or not so huge) data-sets, and create their own new datasets (through drag-and-drop, mouse-clicks, zoom-functions and more) without having to write any custom code.  In this tutorial, we outline and explain the different functions within the SimBA Plotly Dashboard, and how we can utilize the dashboard for analyzing larger datasets (using a data set containing classifications of 5 different behaviors, in 433 individual five-minute long videos) to third-party applications.  The SimBA dashboard was created by [Sophia Hwang](https://github.com/sophihwang26) and [Aasiya Islam](https://github.com/aasiya-islam).

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/Dash_logo.png" />
</p>


### PART 1: Generating a SimBA Dashboard file and opening the Dashboard

1. To open the SimBA Dashboard, we first need to create a single *collated* dashboard dataset file in SimBA. This single dashboard file is a highly compressed [HDF dataframe container](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) that contains all your data that we can play around with using our Plotly dashboard. 

>Note: This dataframe container also provides an efficient way of sharing data, and can be opened in SimBA at any location without requiring any other project files.

2. To generate this file, begin by loading your project in SimBA. In the main SimBA console window, click on `File` and `Load project`. In the **[Load Project]** window, click on `Browse File` and select the `project_config.ini` that belongs to your project, and then click `Load Project`. 
 
3. Navigate to the `Visualizations` tab, and you should see the following menu on the right hand side, titled `Plotly / Dash`. 

![](https://github.com/sgoldenlab/simba/blob/master/images/Dash_1.JPG "Plotly Graph Features")

4. The first part of this Plotly/Dash menu contains 5 tick-box menus. We will use these tick-box menus to specify *what* data we want to have contained within our dashboard file (e.g., what data we want to be able to be able to handle interactively). These boxes include:

* ```Sklearn results```: 

* ```Time bin analyses```:

* ```Probabilities```:

* ```Severity analyses```:

* ```Entire Machine Classification dataframes```:

In this tutorial we will go ahead and select `Sklearn results`, `Time bin analyses`, `Probabilities`.

Once we have selected the tick-boxes for the data we want to include, click on `Save SimBA/Plotly dataset`. This will generate a single, highly-compressed, `H5` dataframe container, which will be located in the `project_folder/logs` directory. The name of this file will be date-time stamped, and be named something like this: `SimBA_dash_file_20200829090616.h5`.

>Note: This is a highly compressed file. In this example tutorial, the 438  have been compressed into a very-much sharable 32MB that contain all the data indicated by the tick-boxes selected in the SimBA GUI 'Plotly/Dash' submenu:

![](https://github.com/sgoldenlab/simba/blob/master/images/Dash_2.JPG "Plotly Graph Features")


Depending on the number of videos that the user has within the project, this step may take some time. You can follow the progress in the main SimBA terminal window. 

### PART 2: Opening the SimBA Dashboard file

1. To open and inspect a `SimBA/Plotly dataset` h5 file, we will use the two menus circled in blue below:

![](https://github.com/sgoldenlab/simba/blob/master/images/Dash_4.JPG "SimBA Dashboard")

* ```SimBA Dashboard file (H5)```: In this menu, click on 'Browse File` and select the dataframe H5 dataframe container you wish to use within the Dashboard interface. 
>**Important**: The selected `SimBA Dashboard file.H5` file did not have to be generated within the currently opened project. The selected `SimBA Dashboard file.H5` could have been generated within the SimBA interface anywhere, within any project (regardless of pose-estimation tool, tracked body-parts, and the number of/specific classifier used). 

* ```SimBA Groups file (CSV)```: **(OPTIONAL)** If we want to plot and compare group-level metrics, then we need to indicate to SimBA and [Plotly](https://plotly.com/) which videos belong to which group. The most straightforward way of doing this is to create our own CSV file, where each column represents each group, and each row represents each video belonging to that group, and feed the information in this CSV file into the Dashboard. For the current tutorial example, with 438 videos, I have created a CSV file example that can be downloaded [HERE](https://github.com/sgoldenlab/simba/blob/master/misc/SimBA_Dash_tutorial_Group_information.csv). This file contains **two columns** representing the two groups (males v. females), with one row for each video in each group. For this example, we have 337 videos in the **male** group, and 101 videos in the **female** group. 


Once you have selected your files `SimBA Dashboard file` (.. and the optional SimBA Groups file), go ahead and click on the `Open SimBA / Plotly dataset` button. A little time will pass for the application to load, but eventually a window looking similar to this (on the left in the image below) should pop open. Alternatively, if you feel like this interface is finicky to work with, you can also navigate to the the IP `127.0.0.1:8050` address in your web browser (tested with Google Chrome) and you should see the native Dash interface (on the right in the image below; **click on the images below to enlarge**): 

<img src="https://github.com/sgoldenlab/simba/blob/master/images/Dash_6.JPG" width="425"/> <img src="https://github.com/sgoldenlab/simba/blob/master/images/Dash_7.JPG" width="425"/>

### PART 3: Graph overview

Here we feature the plots visualizing data for both a group mean comparison with groups selected on the `Data` menu on the Dashboard to the right-hand side, as well as an individual video comparison amongst a single group, also selected from the Dashboard.

### Part 1: Graph Types and Features


<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/probability_new.JPG" />
</p>

* Probability Graphs: These graphs plot the probability of the behavior as a continuous line graph per video frames as default. The x-axis can be changed [here](https://github.com/sgoldenlab/simba/blob/master/docs/plotly_dash.md#additional-properties) to visualize the probability vs. time in seconds for each video instead. By hovering over each of the line graphs, we can see the individual data points for each trace as a coordinate of (frames/seconds, probability) as shown below. You can 
also change this [here](https://github.com/sgoldenlab/simba/blob/master/docs/plotly_dash.md#part-3-plot-settings) to compare data points betwen traces while hovering. 


<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/bar_new.JPG" />
</p>

* Categorical Bar Graphs: These graphs plot specific features of the behavior as designated [here](https://github.com/sgoldenlab/simba/blob/master/docs/plotly_dash.md#select-behavior-to-plot), with the means of the feature and standard error of mean displayed for each group in the Group Means comparison, and the feature for each video in the Individual Groups comparison. The means and feature count can be displayed on the top of each of the bars respectively, and also displayed upon hovering over it with the group/video name and feature count, and for the group mean comparison, the standard error will be displayed as +/- after the mean value feature count, as shown below. The standard error bars can be configured to be displayed traditionally as both above and below the mean, or just one way in either direction [here](https://github.com/sgoldenlab/simba/blob/master/docs/plotly_dash.md#bar-graph-properties). 

* Legends: The legend primarily displays the color classification for each of the groups or individual videos displayed, which can be altered [here](https://github.com/sgoldenlab/simba/blob/master/docs/plotly_dash.md#color-properties)


### Part 2: Plot Settings

With each displayed plots, Plotly has many features and settings that allow users to better visualize the graphs and compare data as well as download the plots as images to your own computer. These settings can be accessed by hovering your mouse at the top right of any of the displayed graphs, like this:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/Dash_gif_01.gif" />
</p>

These settings are described in detailed in the [Plotly documentation], but are described in brief here below:


<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/image_download.gif" />
</p>


* ```Download Plot```: By clicking here, we can download each of our plots as images saved to our computers in our desired locations, and you can change the pixel dimensions as well as the file extension type of the image [here](https://github.com/sgoldenlab/simba/blob/master/docs/plotly_dash.md#image-export). The default will save as a 500 X 700 PNG. 


<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/zoom.gif" />
</p>


* ```Zoom```: By clicking here, we can zoom into a portion of our graphed data by clicking and dragging over the data, creating a box that highlights the data we would like to look at more closely. We can also do this by clicking and dragging the box over the data without clicking on the `Zoom` button, as this is the default feature of the cursor's click-and-drag with our graphs.
>Note: When plotting bar graphs, Zooming in using the click-and-drag method produces an error in which the graph immediately zooms back out. We are currently working on resolving the error.


<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/pan.gif" />
</p>


* ```Pan```: By clicking here, we can pan over a zoomed-in portion of the data and view more of the continuous frames, again by clicking and dragging the cursor over the graph either to the left or right of the data to view the surrounding data to the zoomed-in portion. If the data is not zoomed-in, the default graph will display all frames of the data together and you will not be able to pan the data to other relevant frames.


* ```Box Select```:

* ```Lasso Select```:


<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/zoom2.gif" />
</p>


* ```Zoom In```: By clicking here, we can zoom into a default portion of the data which will be the middle of the graph, and we can click multiple times to keep zooming in and looking closer at the default middle portion of the data. You can also zoom into a specified portion of the data by clicking and dragging over an area, as described above.

* ```Zoom Out```: Similar to above, by clicking here we can zoom out of the zoomed-in portion of the data, and it centers around the middle part of the graph, and continue to zoom out multiple times as needed.


<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/autoscale.gif" />
</p>


* ```Autoscale```: After zooming in or out of a portion of data, the axes should automatically autoscale to give the user the best view of the graphed data, but if it doesn't autoscale automatically, we can click here. Note that it may take extra time to autoscale if you are analyzing data from large datasets inclusive of several videos which slows down the program. 


<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/reset_axes.gif" />
</p>


* ```Reset Axes```:


<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/toggle_spikes.giff" />
</p>


* ```Toggle Spike Lines```:


<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/hover.gif" />
</p>


* ```Show closest data on hover```:

* ```Compare data on hover```:


### PART 4: Plot Dashboard Overview


The left-hand side of the Dashboard interface displays the Dashboard menus. These menus allow users to specify what data is displayed on the graphs shown on the right-hand side of the interface, how the data is plotted, how to save graphs, and export subsets of data into self-contained CSVs and/or [parquet](https://towardsdatascience.com/the-best-format-to-save-pandas-data-414dca023e0d) files for further processing. In this part of the tutorial, we will walk through all of the settings and their functions in the SimBA Dashboard.  

The Dashboard menus display three tabs: `Data`, `Graph Settings`, and `Download Settings`.

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/Dash_8.JPG" />
</p>

## Part 1: Data


With the first of these tabs being `Data`, users can select the classifier and types of data to plot, as well as select the types of groups to compare and which videos correspond to each respective group. 


<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/data_tab.PNG" />
</p>

### Select Behavior to Plot

In the following dropdown menus from the Data tab, the users can specify which classifier to plot, what category or type of data from that classifier, and the specific features of the data sets with our given scenario. 


<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/plot_behavior.JPG" />
</p>


* ```Behaviors:``` In this dropdown menu, we can select between any of the classified behaviors in the project, which in this case are `Attack`, `Mounting`, `Escape`, `Pursuit`, and `Defensive`. Selecting the classifier will update the graphs displayed to the right. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/Dash_gif_02.gif" />
</p>

* ```Category:``` In this dropdown menu, we can choose what type of data to display for the selected classifier, which will update the graphs to the right. Here we can select between `VideoData` which plots the probability of the behavior vs. frames, `SklearnData`, which plots different results of the features analyzed by Sklearn, and `TimeBins`, which plots the prevalence of the features for each time bin.

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/Dash_gif_03.gif" />
</p>

* ```Feature:``` In this dropdown menu, we can select the sub-type of data, aka the features or statistics, to plot for each respective category. VideoData plots probability in the form of a line graph. Sklearn plots seven statistics for each behavior with the mean and standard error represented in a bar graph. Here we can select from ` bout events`, `total events duration (s)`, `mean bout duration (s)`, `first occurrence (s)`, `mean interval (s)`, and `median interval (s)`. TimeBins plots the mean of the prevalence of these respective features for each time bin selected. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/behavior_feature.gif" />
</p>

### Plotting Group Means

With this feature, we can select different combinations of groups of videos to plot and compare the data of each to one another in the Group Means plot.

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/group_means.gif" />
</p>


* ```Select Group(s):``` In this dropdown menu, we can select the different groups to add to our plot, and to remove a group
from the selection we can click the `X` next to the group name.

For the VideoData continuous probability data, we can check the `Show Total Mean` box to add a trace to represent the total mean of the group data.


### Plotting Individual Groups

With this feature, we can select a single group of videos to compare the data of each video in the group to one another in the plot.


<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/individual_videos.gif" />
</p>


* ```Select Single Group to Plot:``` In this dropdown menu, we can select a single group from ort list of groups to plot.

* ```Select Video(s):``` In this dropdown menu, we can select which videos we would like to see represented and compared to one another. We can unselect videos to include in our individual video comparison by clicking the `X` next to the video name.

For the VideoData continuous probability data, we can check the `Show Group Mean` box to add a trace to represent the mean of the included videos
in the group data.


## Part 2: Graph Settings


With the second tab being `Graph Settings`, users can specify the different graph properties they would like to set such as the color scale, axes, background graph features, and miscellaneous properties. 


<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/graphsettings_new.JPG" />
</p>


### Color Properties


Here we can choose the color properties for both our groups and individual videos for the graphs. 


<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/group_colors.gif" />
</p>


For the group color selection, simply click on the colored circle underneath the name of the group which opens a color picker in which you can choose a specific color and shade for each group to be represented with. Once all the colors have been selected for the groups, you can click the `UPDATE COLORS` button to set the selection.


<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/colorscale.gif" />
</p>


* ```Colorscale Gradient for Individual Videos```: For the individual video color selection, you have the option of choosing from different pre-set or custom colorscales which can be found in the dropdown menu by clicking on the sample colorscale. Here you can choose the type of colorscale depending on what best suits your data visualization needs, as well as setting the number of swatches needed to potentially correspond to the number of videos being visualized. The colors will automatically update upon the selection of the colorscale gradient, and to return to the page, you can click the colorscale already displayed on the page, and to update the colors with a new combination of colors from the colorscale gradient, you can click on `UPDATE COLORS` to change them.


### Probability Graph Properties

Here we can select the properties for our different probability graphs, both the multi-group and individual group video graphs. Depending on whether you would like the graph visualized with video frames or seconds as the x-axis, as set [here](https://github.com/sgoldenlab/simba/blob/master/docs/plotly_dash.md#additional-properties), we can custom set the axes below.


<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/group_frames.gif" />
</p>


For visualizing the multi-group graphs in frames, you can set the minumum and maximum number of frames you would like visualized, respectively, and similarly if you're viewing your graph in seconds, you can set the numbers below. Once set, click `SUBMIT` to view the changes. To reset the axes, click `RESET AXES` to view the default minima and maxima values. The same steps can be applied for the individual group and video viewing frames/seconds below.


### Bar Graph Properties

For the bar graphs, as represented by the Sklearn and TimeBins data with our given scenario, we can specify what type of error bars we would like visualized, being either traditional error bars or just one way around the mean value. The standard error values can also be seen along with the mean if you hover over the bar itself on the graph.


<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/error_bars.gif" />
</p>


* ```Error Type:```: The default will display traditional error bars as `Both` but to display them only one way, select either the `Above` or `Below` options to
view the bars either above or below the bar, respectively. 


### Additional Properties

Here we can change different graph properties such as the background display, font, and axes if plotting probability data, custom setting the graph titles.


![](https://github.com/sgoldenlab/simba/blob/master/images/additionalproperties_3.JPG "Additional Properties Selection")


* ```Graph X Axis in Seconds```: For the probability graphs, you can choose to display the x-axis in units of seconds instead of frames by checking this box

* ```Show Grid Lines```: The white grid lines in the background of the plot will be displayed by default, but to remove them you can uncheck this box

* ```Show Background```: A light blue background will displayed behind the graph by default, but to remove this to display a white background you can check this box


<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/graph_titles.gif" />
</p>


* ``` Group Means Title``` & ```Individual Videos Title```: For the Group Means graph, the plot title will be the Feature + "Group Means" by default (e.g. "Probability_Attack Group Means"), and for the Individual Videos graph, the plot title will be the Feature + Group by default (e.g. Probability_Attack Group_1_test). To set your own plot titles, type in your new titles into the respective boxes and click `SET` to change them.

* ```Choose Font```: The default font displayed for the page is Verdana, but this can be changed using the dropdown menu as you can select from the following fonts: Verdana, Helvetica, Calibri, Arial, Arial Narrow, Candara, Geneva, Courier New, and Times New Roman. *Note that if you want to the new plot title to be updated with this font change, then you must click `SET` next to the title name once again to update both font and title name.

* ```Font Size```: The default font size displayed for the page is 12 pt, but this can also be increased or decreased by either clicking on the up or down arrows, respectively, or by highlighting and typing in a new font size in the display box. 


## Part 3: Download Settings


With the last tab being `Download Settings`, users can specify how they would like to download their datasets as CSVs and plots as images.

![](https://github.com/sgoldenlab/simba/blob/master/images/downloadsettings_2.JPG "Download Settings Tab")

### CSV Export 

You files end up `Your_PROJECT_NAME_\project_folder\logs` directory


You can download the data for each of the respective Group Means and Individual Videos plots as CSV files and save to your computer. 


<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/csv_export.JPG" />
</p>


* ```Enter csv file name```: To designate a CSV file name, type your desired file name into the box and click either the `MEANS.CSV` or the `VIDEOS.CSV` buttons to select which data you would like to download as a CSV. Once downloaded, it should display a message saying that you have "Downloaded csv as file_name.csv", as seen below. It will download into the same folder from which the _____.


<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/csv_download.JPG" />
</p>

For the Group Means data, the CSV will display the probability group data for each frame and the mean for the groups, or the feature means for each group with standard errors for the categorical data. For Individual Videos data, the CSV  will display the data in the same format with the data represented for each video instead of for each group. Examples can be seen below: 

![](https://github.com/sgoldenlab/simba/blob/master/images/probability_means_csv.JPG "CSV Probability Means Example")
![](https://github.com/sgoldenlab/simba/blob/master/images/probability_videos_csv.JPG "CSV Probability Videos Example")
![](https://github.com/sgoldenlab/simba/blob/master/images/sklearn_means_csv.JPG "CSV Sklearn Means Example")
![](https://github.com/sgoldenlab/simba/blob/master/images/sklearn_videos_csv.JPG "CSV Sklearn Videos Example")

### Image Export


You can also download the images for each of the plots and customize dimensions, name, and file extension. 


<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/image_download.gif" />
</p>


* ``` Enter image dimensions (px)```: The default image dimensions in pixels are 500 X 700, but you can change the height and width respectively by typing in a new value or by clicking the up and down arrows to increase or decrease the dimensions.

* ```Enter image name```: You can type your desired image download name by typing it in the box here.

* ``` File Extension```: The default file download extension is .SVG but you can also choose the file extension by selecting from the dropdown menu here and save the image as an .SVG, .PNG, or as a .JPEG.

Once you've fixed the download settings, click the `SET DOWNLOAD SETTINGS` button at the bottom, and to actually download the image, [go to the plot settings](https://github.com/sgoldenlab/simba/blob/master/docs/plotly_dash.md#part-3-plot-settings) and click "camera" button as `Download plot` which should download the image to the folder of your choice on your computer.

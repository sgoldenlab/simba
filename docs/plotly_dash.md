# SimBa Data Visualization: Plotly Dash Tutorial


# PART 1: Graphs and Plot Overview

![](https://github.com/sgoldenlab/simba/blob/master/images/cover%20photo.PNG "SimBA Plotly Overview")

### Part 1: Graph Overview

![](https://github.com/sgoldenlab/simba/blob/master/images/overall_bar.JPG "Graph Interface Overview")

### Part 2: Graph Types and Features

![](https://github.com/sgoldenlab/simba/blob/master/images/probability.JPG "Probability Graph Example")

![](https://github.com/sgoldenlab/simba/blob/master/images/bargraph.JPG "Bar Graph Example")

### Part 3: Plot Settings

![](https://github.com/sgoldenlab/simba/blob/master/images/probability_features.png "Plotly Graph Features")

# PART 2: Plot Dashboard

![](https://github.com/sgoldenlab/simba/blob/master/images/data_tab.JPG "Data Tab")
![](https://github.com/sgoldenlab/simba/blob/master/images/graphsettings_tab.JPG "Graph Settings Tab")
![](https://github.com/sgoldenlab/simba/blob/master/images/updated_downloadsettings.JPG "Download Settings Tab")

## Part 1: Data

### Select Behavior to Plot

In the following dropdown menus, we can select which type of behavior to visualize for our experiment along with the category of data
and the specific features of the data sets

![](https://github.com/sgoldenlab/simba/blob/master/images/plot_behavior.JPG "Plot Behavior Selection")

* ```Behaviors:``` In this dropdown menu, we can select between `Attack` and `Sniffing` behaviors

* ```Category:``` In this dropdown menu, we can select between `VideoData` which plots probability of behavior vs. frames,
`SklearnData` which plots different results of the features analyzed by Sklearn, and `TimeBins` which plots prevalence of the features
for each time bin

* ```Feature:``` In this dropdown menu, we can select the features or statistics to plot for each respective category. VideoData 
plots probability in the form of a line graph. Sklearn plots seven statistics for each behavior with the mean and standard error represented
in a bar graph. Here we can select from # bout events, total events duration (s), mean bout duration (s), first occurrence (s), mean interval
(s), and median interval (s). TimeBins plots the mean of the prevalence of these respective features for each time bin selected.

### Plotting Group Means

With this feature, we can select different combinations of groups of videos to plot and compare the data of each to one another in the Group Means plot

![](https://github.com/sgoldenlab/simba/blob/master/images/group_means.JPG "Group Selection")

* ```Select Group(s):``` In this dropdown menu, we can select the different groups to add to our plot, and to remove a group
from the selection we can click the `X` next to the group name

For the VideoData continuous probability data, we can check the `Show Total Mean` box to add a trace to represent the total mean of the group data

### Plotting Individual Groups

With this feature, we can select a single group of videos to compare the data of each video in the group to one another in the plot

![](https://github.com/sgoldenlab/simba/blob/master/images/individual_groups.JPG "Individual Group Selection")

* ```Select Single Group to Plot:``` In this dropdown menu, we can select a single group from out list of groups to plot

* ```Select Video(s):``` In this dropdown menu, we can select which videos we would like to see represented and compared to one another

For the VideoData continuous probability data, we can check the `Show Group Mean` box to add a trace to represent the mean of the included videos
in the group data 

## Part 2: Graph Settings

### Group Color Properties

Here we can choose the 

![](https://github.com/sgoldenlab/simba/blob/master/images/group_colors.png "Group Color Selection")

* ```Colorscale Gradient for Individual Group```

![](https://github.com/sgoldenlab/simba/blob/master/images/colorscales.png "Colorscale Selection")

`UPDATE COLORS`

### Probability Graph Properties

![](https://github.com/sgoldenlab/simba/blob/master/images/probability_propertiesJPG.JPGg "Probability Graph Properties Selection")

* ```Set Multi Group Frames:```
`SUBMIT`

* ```Multi Group (Seconds)```

`RESET AXES`

* ```Set Individual Group Frames:```
`SUBMIT`

* ```Individual Group (Seconds):```

`RESET AXES`

### Bar Graph Properties

![](https://github.com/sgoldenlab/simba/blob/master/images/bar_properties.JPG "Bar Graph Properties Selection")

* ```Error Type:```

`Above` `Below` `Both` 

### Additional Properties

![](https://github.com/sgoldenlab/simba/blob/master/images/additional_properties.JPG "Additional Properties Selection")

* ```Graph X Axis in Seconds```

* ```Show Grid Lines```

* ```Show Background```

* ```Choose Font:```

* ```Font Size```

`DOWNLOAD INDIVIDUAL GROUP CSV`

## Part 3: Download Settings

### CSV Export 

![](https://github.com/sgoldenlab/simba/blob/master/images/csv_export.JPG "CSV Export Settings")

* ```Enter csv file name:

`MEANS.CSV` `GROUP.CSV`

### Image Export
![](https://github.com/sgoldenlab/simba/blob/master/images/image_export.JPG "Image Export Settings")

* ```Image Height:```

* ```Image Width:```

* ```Enter file name:```

* ``` File Extension:```

`SET DOWNLOAD SETTINGS`

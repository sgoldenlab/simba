# <p align="center"> Feature sub-sets in SimBA </p>

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/feature_subsets_0.png" />
</p>

### INTRODUCTION

SimBA extracts features for builing and [running downstream machine learning models](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-5-extract-features). However, at times, we may just want to take advantage of some SimBA [feature calculators](https://github.com/sgoldenlab/simba/blob/master/simba/mixins/feature_extraction_mixin.py) and generate a subset of measurements for use in our own downstream applications. For some of the feature sub-sets available, see the image above. 


### INSTRUCTIONS

1) To create feature sub-sets, first import your pose-esimation data into SimBA and follow the instruction up-to and **including** outlier correction as documented in the [Scenario 1 tutorial](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md). Before calculating feature sub-sets in SimBA, ensure that the `project_folder/csv/outlier_corrected_movement_location` directory of your SimBA project is populated with files. 

2). Navigate to the [Extract features] tab, and click on the `CALCULATE FEATURE SUBSETS` button.

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/feature_subset_menu_1.png" />
</p>

3) In the **SETTINGS** frame, begin by selecting a directory where you want to store the output data. In the chosen directory, SimBA will store one CSV file per file in the `project_folder/csv/outlier_corrected_movement_location` directory. In these files, there will be one column per new feature and one row per video frame. It's a good idea to select an empty directory. 
   
5) We can choose to append these new feature subsets to our data in the `project_folder/csv/features_extracted` directory wit the aim of creating better classifiers. SimBA will then use these features when [generating machine learning classification predictions](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-8-run-machine-model), and the features will be included when [annotating](https://github.com/sgoldenlab/simba/blob/master/docs/label_behavior.md) the specific videos. To append the new features to the extracted features data inside the `project_folder/csv/features_extracted`, tick the `APPEND RESULTS TO FEATURES EXTRACTED FILES`.
   
6) If we have already annotated videos, then we may want to include these further features without having to re-annotate the data. To append the new features to the annotated data inside the `project_folder/csv/targets_inserted`, tick the `APPEND RESULTS TO TARGETS INSERTED FILES`.

> Note: If we decide to append the results to either or both the `features_extracted` and `targets_inserted` files, and do not want to store the new features in a separate directory, then tick the relevant checkboxes and leave the `SAVE DIRECTORY` as `No folder selected. 

7). When building classifiers, its important that all files representing each video has the same features and an equal number of features. Thus, before completing the appending of the new features, we may want to perform some integrity checks to confirm that each file has the same number of features and that the features have all the same names. To do this, check the `INCLUDE INTEGRITY CHECKS BEFORE APPENDING NEW DATA` CHECKBOX. 

> Note: If there are integrity issues, SimBA will print you an error message about the integrity issues, and **stop the appending of the new feature data**. Thus, if errors are discovered, SimBA will revert to the orginal `project_folder/csv/features_extracted` and/or `project_folder/csv/targets_inserted` file formats without appended data. 


8) Next, we need to select the features that we want to compute:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/feature_subset_menu_2.png" />
</p>


Tick the checkboxes for the features that we want to compute. Once complete, click the <kbd>RUN</kbd> button. 

9) Once complete, the `SAVE DIRECTORY` will be filled with one file for every video file represented in your `project_folder/csv/outlier_corrected_movement_location` directory (if you selected a save directory). In these files, every row represents a frame, and every column represents a feature in feature family. The number of columns (features) will depend on the number of body-parts and animals in your SimBA project. If you ticked the `APPEND RESULTS TO FEATURES EXTRACTED FILES` and/or `APPEND RESULTS TO TARGETS INSERTED FILES`, the files in these directories should also be populated with new features.

For smaller examples of expected output features, see:

* [Two-point body-part distances (mm).csv](https://github.com/sgoldenlab/simba/blob/master/misc/Two-point%20body-part%20distances%20(mm).csv)
* [Within-animal three-point body-part angles (degrees).csv](https://github.com/sgoldenlab/simba/blob/master/misc/Within-animal%20three-point%20body-part%20angles%20(degrees).csv)
* [Within-animal three-point convex hull (mm2).csv](https://github.com/sgoldenlab/simba/blob/master/misc/Within-animal%20three-point%20convex%20hull%20(mm2).csv)
* [Within-animal four-point convex hull (mm2).csv](https://github.com/sgoldenlab/simba/blob/master/misc/Within-animal%20four-point%20convex%20hull%20(mm2).csv)
* [Frame-by-frame body-part movement (mm).csv](https://github.com/sgoldenlab/simba/blob/master/misc/Frame-by-frame%20body-part%20movement%20(mm).csv)







# <p align="center"> Animal-anchored (ROIs) in SimBA </p>

The animal-anchored region-of-interest (ROI) interface allows users to define bounding boxes (or circles) around pose-estimated animal key-points. Once defined, we can calulate how often and when the different bounding boxes and key-points intersect which each other, in order to infer when and how animals interact with each other. 

# Before analyzing ROIs in SimBA

To analyze anchored-roi data in SimBA (for descriptive statistics, machine learning features, or both descriptive statistics and 
machine learning features) the tracking data **first** has to be processed the **up-to and including the 
*Outlier correction* step described in [Part 2 - Step 4 - Correcting outliers](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-4-outlier-correction)**. 
Thus, before proceeding to calculate anchored-ROI based measures, you should have one file for each of the videos in your 
project located within the `project_folder\csv\outlier_corrected_movement_location` sub-directory of your SimBA project.

Specifically, for working with anchored-ROI in SimBA, begin by 
(i) [Importing your videos to your project](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-2-import-videos-into-project-folder), 
(ii) [Import the tracking data and relevant videos to your project](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-3-import-dlc-tracking-data), 
(iii) [Set the video parameters](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-3-set-video-parameters), 
and lastly (iv) [Correct outliers](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-4-outlier-correction) (or click to indicate that you want to *Skip outlier correction* as detailed in the Correct outliers tutorial)



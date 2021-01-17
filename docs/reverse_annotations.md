# Reversing the 'directionality' of classifiers created in SimBA

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/reversering_1.PNG" />
</p>

### BACKGROUND

In this scenario, the user has already created a classifier in SimBA within a two-animal protocol. The classifier the user created has a "direction". This means that the created classifer takes into account which **specific** animal is doing the classified behavior, or alternative, the specific relationship between the two animals within a classified social interaction. Now the user wants a straighforward way of creating a second classifier with the reversed "direction", for example: 

* The original classifier detects when **animal 1** rears, but not when **animal 2** rears. The user wants a second classifier that detects when **animal 2** rears, but not **animal 1** rears. The user will then run the classifiers in tandem to get metrics for when either animals rears seperately. 

* The original classifier detects when **animal 1** pursues **animal 2**. The user wants a second classifier that detects when **animal 2** pursues **animal 1**. The user will then run the classifiers in tandem to get metrics for when either animal pursues the other seperatey.

* The original classifier detects when **animal 1** sniffs **animal 2**. The user wants a second classifier that detects when **animal 2** sniffs **animal 1**. The user will then run the classifiers in tandem to get metrics for when either animal sniffs the other seperatey. 

For example, [this video](https://www.youtube.com/watch?v=0OIFysQvUCI&list=PLi5Vwf0hhy1R6NDQJ3U28MOUJPfl2YWYl&index=17) contains 8 classifiers 
(approaching, sniffing, walking away, and rearing, for each of the two animals) that were created from annotations for only 4 classifiers (approaching, sniffing, walking away, and rearing, for animal 1 only). The directionality of each of these 4 classifiers were reversed in SimBA using the tool documented here, and then these new files were used to create 4 additional classifiers. 

>Note: This tool will only work if the behaviour of interest looks the same regardless of the directionality of the behavior. For example, if the sniffing/rearing/pursuit behavior of animal 1 isn't the same as when performed by animal 2, then a simple reversal as documented here won't work, and you need to go back and [create a new classifier from scratch](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md) with annotations for the reversed behavior. 


### Step 1 

Reversing a classifier requires there to be an original classifier, or muliple classifiers, within your project to reverse. Thus, you need to first have created a classifier as documented [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md).

Within you project folder for the classifier (or multiple classifiers) you want to reverse, make sure the `project_folder/csv` sub-directories are populated with files. Specifically, the (i) `project_folder/csv/outlier_corrected_movement_location`, (ii) `project_folder/csv/features_extracted`, and (iii)`project_folder/csv/targets_inserted` directories needs to contain data files, with one file in each directory representing the videos you want to reverse your annotations for. 

The files in the `project_folder/csv/outlier_corrected_movement_location` directory should have been created when you [corrected, or clicked to skip the outlier correction](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-4-outlier-correction), for your original classifier. 

The files in the `project_folder/csv/features_extracted` directory should have been created when you [clicked to peform feature extraction](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-5-extract-features), for your original classifier. 

The files in the `project_folder/csv/targets_inserted` directory should have been created when you [labelled behaviours using the SimBA behavioral annotation interface](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-6-label-behavior) or [appended third-party annotations](https://github.com/sgoldenlab/simba/blob/master/docs/third_party_annot.md), for your original classifier. 

### Step 2

[Load your SimBA project](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-1-load-project-config) and navigate to the tab 






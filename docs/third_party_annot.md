# APPENDING THIRD-PARTY ANNOTATIONS IN SIMBA

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/third_party_annot.png" />
</p>

SimBA has an [in-built behavior interface](https://github.com/sgoldenlab/simba/blob/master/docs/labelling_aggression_tutorial.md) that allow users to append experimenter-made labels to the [features extracted](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-5-extract-features) from the pose-estimation data. Accurate human labelling of images can be the most time-consuming part of creating reliable supervised machine learning models. Sometimes the experimenter already have accurate labels for the videos a set of videos, but the labels have been generated in a third-party annotation tool such as [BORIS]( https://www.boris.unito.it/), [Jwatcher](https://www.jwatcher.ucla.edu/), [Solomon coder](https://solomon.andraspeter.com/), or [MARS/BENTO](https://github.com/neuroethology/bentoMAT). SimBA allows the user to append labels stored in these formats, saving you from having to repeat the annotation process. 


For more detailed information on how to export [BORIS]( https://www.boris.unito.it/) annotations in a format compatible with SimBA - check out [THIS SHORT TUTORIAL](https://github.com/sgoldenlab/simba/blob/master/docs/append_boris.md). 

## How to import annotations into SimBA project

1. Once you have created a project, click on `File` --> `Load project` to load your project. For more information on creating a project, click [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#part-1-create-a-new-project-1).

2. Make sure you have extracted the features from your tracking data and there are files, one file representing each video in your project, located inside the `project_folder/csv/features_extracted` directory of your project. For more information on extracting features, click [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-5-extract-features).

3. After loading your project, click on the `Label behavior` tab, and you should see the following window:

<p align="center">
<img src=/images/lblbehavior.PNG />
</p>

4. In the **Import Third-Party behavior labels** sub-menu, click on the button that represents the thid-party application that the annotations were generated with.


5. A menu will appear that asks you yo select the folder that contains the annotation files in CSV file format. Browse for the folder and select it. 

6. Once you select the folder, new files will then be generated containing your annotations and features and they will be stored in the  `project_folder/csv/targets_inserted` directory of your project. You can follow the progress in the main SimBA window. 

## NOTES / TROUBLESHOOTING: MARS AND SOLOMON ANNOTATIONS

For SimBA to know which third-party annotation data should be appended to which video data, the files within SimBA and the Solomon/MARS annotation files need to have the same name. Thus, if you are processing two videos in SimbA named `Video_1` and `Video_2`, then the Solomon/MARS annotation files located in the folder defined in Step 5 above should be named `Video_1.csv` and `Video_2.csv `.  


## NOTES / TROUBLESHOOTING: BORIS ANNNOTATIONS

When you select a folder in Step 5 above, SimBA will: 

1. First thrawl through the user-defined directory and find all BORIS-styled CSV files. In the background, SimBA will then merge all the data in the detected BORIS-styled CSV files into a single dataframe and which is kept **in memory** only. 

2. Next, SimBA will open each file located in your `project_folder/csv/features_extracted` directory, one-by-one. If the first file in your `project_folder/csv/features_extracted` directory is called `Video_1`, then SimBA will search the **in-memory** dataframe for instances where the heading `Media file path` contain a filename that is `Video_1`. It is therefore important that the files you process in SimBA - and the files you annotated in BORIS - have the same name. For example, for me to successfully append BORIS annotations for a SimBA file called `Video_1`, the original BORIS annotation file for this video should look something like this:


<p align="center">
<img src=/images/BORIS_99.png />
</p>


3. There may be files within the `project_folder/csv/features_extracted` directory of your SimBA project that contains **no BORIS annotations** for either all or a sub-set of classifier behaviors. This would happen, for example, if there is a file in the `project_folder/csv/features_extracted` directory called `Video_1`, but there are no mentions of a `Video_1` in your BORIS `Media file path` headings. In these situations, SimBA will assume that these files contain **no** expressions of the behavior(s) of interest, and mark all frames in your video as **behavior absent**.  


#### Note 1: Keep in mind that the behaviors/classifiers has to be defined (with the same names as they appear in the third-party annotation files) in the SimBA project folder. You can add or remove classifier in `Further imports(data/video/frames)` --> `Add classifier` or `Remove existing classifier`

<p align="center">
<img src=/images/addorremoveC.PNG />
</p>

#### Note 2: If the third-party annotation files contain annotations for behaviors that you have **not** defined in your SimBA project, then those annotations will be discarded and not appended to your dataset.






# Appending third-party annotations in SimBA

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
A menu will appear that asks you yo select the folder that contains the annotation files in CSV file format. New files will then be generated containing your annotations and features and they will be stored in the  `project_folder/csv/targets_inserted` directory of your project.

>IMPORTANT: For SimBA to know which third-party annotation data should be appended to which video data, the two files need to have the same name. Thus, if you are processing two videos in SimbA named `Video_1` and `Video_2`, then the third-party annotation files should be named `Video_1` and `Video_2`.  

> Note 1: Keep in mind that the behaviors/classifiers has to be defined (with the same names as they appear in the third-party annotation files) in the SimBA project folder. You can add or remove classifier in `Further imports(data/video/frames)` --> `Add classifier` or `Remove existing classifier`

<p align="center">
<img src=/images/addorremoveC.PNG />
</p>

> Note 2: If the third-party annotation files contain annotations for behaviors that you have **not** defined in your SimBA project, then those annotations will be discarded and not appended. 

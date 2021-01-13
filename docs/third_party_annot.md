# Appending third-party annotations in SimBA

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/third_party_annot.png" />
</p>

SimBA has an [in-built behavior interface](https://github.com/sgoldenlab/simba/blob/master/docs/labelling_aggression_tutorial.md) that allow users to append experimenter-made labels to the [features extracted](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-5-extract-features) from the pose-estimation data. Accurate human labelling of images can be the most time-consuming part of creating reliable supervised machine learning models. Sometimes the experimenter already have accurate labels for the videos a set of videos, but the labels have been generated in a third-party annotation tool such as [BORIS]( https://www.boris.unito.it/), [Jwatcher](https://www.jwatcher.ucla.edu/), [Solomon coder](https://solomon.andraspeter.com/), or [MARS/BENTO](https://github.com/neuroethology/bentoMAT). SimBA allows the user to append labels stored in these formats, saving you from having to repeat the annotation process. 


## How to import annotations into SimBA project

1. Once you have created a project, click on `File` --> `Load project` to load your project.

2. Make sure you have extracted the features from your tracking data and there are *.csv* files in the project_folder/csv/features_extracted

3. Click on `Label behavior`, and you should see the following.

<p align="center">
<img src=/images/lblbehavior.PNG />
</p>

4. Under **Import Third-Party behavior labels**, click on the button and select the folder that contains **ONLY** the annotation files. It will then generate annotion .csvs in the project_folder/csv/targets_inserted folder.


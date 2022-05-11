# APPENDING THIRD-PARTY ANNOTATIONS (BEHAVIOR LABELS) IN SIMBA

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/third_party_annot.png" />
</p>

SimBA has an [in-built behavior annotation interface](https://github.com/sgoldenlab/simba/blob/master/docs/labelling_aggression_tutorial.md) that allow users to append experimenter-made labels to the [features extracted](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-5-extract-features) from  pose-estimation data. Accurate human labelling of images can be the most time-consuming part of creating reliable supervised machine learning models. Sometimes the experimenter  have accurate labels for a set of videos, but the labels have been generated in a third-party annotation tools. Some users may also prefer to use other dedicated behavior annotation tools rather than using the [annotation tool built into SimBA](https://github.com/sgoldenlab/simba/blob/master/docs/labelling_aggression_tutorial.md).

SimBA currently supports the import of annotations created in:

* [SOLOMON CODER](https://github.com/sgoldenlab/simba/blob/master/docs/third_party_annot.md#importing-bento-and-solomon-annotations) - https://solomon.andraspeter.com/
* [BORIS](https://github.com/sgoldenlab/simba/blob/master/docs/third_party_annot.md#importing-boris-annotations) - https://www.boris.unito.it/
* [BENTO](https://github.com/sgoldenlab/simba/blob/master/docs/third_party_annot.md#importing-bento-and-solomon-annotations) - https://github.com/neuroethology/bentoMAT
* [ETHOVISION](https://github.com/sgoldenlab/simba/blob/master/docs/third_party_annot.md#importing-ethovision-annotations) - https://www.noldus.com/ethovision-xt

If you have annotation created in any other propriatory or open-source tool and would like to append them to your data in SimBA, please reach out to us on [Gitter](https://gitter.im/SimBA-Resource/community) or [GitHub](https://github.com/sgoldenlab/simba) and we can work together to make the option available in the SimBA GUI.

## BEFORE IMPORTING THIRD-PARTY ANNOTATIONS (BEHAVIOR LABELS) INTO YOUR SIMBA PROJECT

In brief, before we can import the third-party annotations, we need to (i) create a SimBA project, (ii) import our pose-estimation data into this project, (iii) correct any outliers (or indicate to skip outlier correction), and (iv) extract features for our data set, thus: 

1. Once you have created a project, click on `File` --> `Load project` to load your project. For more information on creating a project, click [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#part-1-create-a-new-project-1).

2. Make sure you have extracted the features from your tracking data and there are files, one file representing each video in your project, located inside the `project_folder/csv/features_extracted` directory of your project. For more information on extracting features, click [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-5-extract-features).

3. After loading your project, click on the `Label behavior` tab, and you should see the following window, with the menu of intrest marked in red:

<p align="center">
<img src=/images/third_party_2.png />
</p>


### IMPORTING BENTO AND SOLOMON ANNOTATIONS

1. To import BENTO OR SOLOMON coder annotations, begin by clicking the button representing your annotation software. A file browser window will pop open, asking you to choose the folder that contains your annotation files. If you are importing **BENTO** annotations, SimBA expects a folder containing files with the file-ending `.annot`. If you are importing **Solomon** annotations, SimBA expects a folder containing files with the file-ending `.csv`. 

2. For an example layout of **BENTO** annotations that SimBA expects, click [HERE](https://github.com/sgoldenlab/simba/blob/master/misc/bento_example.annot). For an example layout of **Solomon** annotations that SimBA expects, click [HERE](https://github.com/sgoldenlab/simba/blob/master/misc/solomon_example.csv). If your files look (i) different from these files, and (ii) you are having trouble appending your BENTO or Solomon coder annotations in SimBA, please reach out to us on [Gitter](https://gitter.im/SimBA-Resource/community) or open a [GitHub issue](https://github.com/sgoldenlab/simba) report. 

3. Click `Select` for the folder that contains your annotation. You can follow the progress in the main SimBA terminal window. 

4. If the appending of your annotations where completed successfully, you should see files representing each of your videos inside the `project_folder/csv/targets_inserted` directory of your SimBA project. If you open these files, you should see one column (towards the very and of the file) representing each of your classifiers. These columns will be populated with `0` and `1`, representing the absence (`0`) and presence (`1`) of the behavior according to your annotations in the BENTO/SOLOMON annotation tools. 

> Note 1: For SimBA to know which third-party annotation files should be appended to which video data file, the files within SimBA and the Solomon/BENTO annotation files need to have the same name. Thus, if you are processing two videos in SimBA named `Video_1` and `Video_2`, then the Solomon/BENTO annotation files located in the folder defined in Step 5 above should be named `Video_1` and `Video_2` (excluding the file-endings). 

> Note 2: Keep in mind that the behaviors/classifiers has to be defined (with the same names as they appear in the third-party annotation files) in the SimBA project folder. For example, SimBA **will not** recognize a behaviour in the third-party annotation files called `sniffing` if the behavior in the SimBA project is defined as `Sniff` or `Sniffing`. To add / remove classifier(s), use the `Further imports(data/video/frames)` --> `Add classifier` or `Remove existing classifier` menus.

> Note 3: If the annotation files contain annotations for behaviors that you have **not** defined in your SimBA project, then those annotations will be discarded and not appended to your dataset.

## IMPORTING BORIS ANNOTATIONS

1. To import BORIS annotations, begin by clicking the appropriate button. A file browser window will pop open, asking you to choose the folder that contains your annotation files. If you are importing **BORIS** annotations, SimBA expects a folder containing files with the file-ending `.csv`. For an example layout of **BORIS** annotations that SimBA expects, click [HERE](https://github.com/sgoldenlab/simba/blob/master/misc/boris_example.csv). If your files look (i) different from these files, and (ii) you are having trouble appending BORIS annotations in SimBA, please reach out to us on [Gitter](https://gitter.im/SimBA-Resource/community) or open a [GitHub issue](https://github.com/sgoldenlab/simba) report. 

2. In the background, SimBA will thrawl through the user-defined directory and find all BORIS-styled CSV files. SimBA will then merge all the data in the detected BORIS-styled CSV files into a single dataframe and which is kept **in memory** only. Next, SimBA will open each file located in your `project_folder/csv/features_extracted` directory, one-by-one. If the first file in your `project_folder/csv/features_extracted` directory is called `Video_1`, then SimBA will search the **in-memory** dataframe for instances where the heading `Media file path` contain a filename that is `Video_1`. It is therefore important that the files you process in SimBA - and the files you annotated in BORIS - have the same name. For example, for me to successfully append BORIS annotations for a SimBA file called `Video_1`, the original BORIS annotation file for this video should look something like this:

<p align="center">
<img src=/images/BORIS_99.png />
</p>

3. There may be files within the `project_folder/csv/features_extracted` directory of your SimBA project that contains **no BORIS annotations** for either all or a sub-set of classifier behaviors. This would happen, for example, if there is a file in the `project_folder/csv/features_extracted` directory called `Video_1`, but there are no mentions of a `Video_1` in your BORIS `Media file path` headings. In these situations, SimBA will assume that these files contain **no** expressions of the behavior(s) of interest, and mark all frames in your video as **behavior absent**.  

4. If the appending of your annotations where completed successfully, you should see files representing each of your videos inside the `project_folder/csv/targets_inserted` directory of your SimBA project. If you open these files, you should see one column (towards the very and of the file) representing each of your classifiers. These columns will be populated with `0` and `1`, representing the absence (`0`) and presence (`1`) of the behavior according to your annotations in BORIS. 

> Note 1: Keep in mind that the behaviors/classifiers has to be defined (with the same names as they appear in the third-party annotation files) in the SimBA project folder. For example, SimBA **will not** recognize a behaviour in the third-party annotation files called `sniffing` if the behavior in the SimBA project is defined as `Sniff` or `Sniffing`. To add / remove classifier(s), use the `Further imports(data/video/frames)` --> `Add classifier` or `Remove existing classifier` menus.

> Note 2: If the annotation files contain annotations for behaviors that you have **not** defined in your SimBA project, then those annotations will be discarded and not appended to your dataset.

> Note 3: For a further documentation on importing BORIS annotations into your SimBA project, check out [THIS SHORT TUTORIAL](https://github.com/sgoldenlab/simba/blob/master/docs/append_boris.md). 

## IMPORTING ETHOVISION ANNOTATIONS

1. To import Ethovison annotations, begin by clicking the appropriate button. A file browser window will pop open, asking you to choose the folder that contains your annotation files. If you are importing **ETHOVISON** annotations, SimBA expects a folder containing files with the file-endings `.xlxs` and `.xls`. For an example layout of **ETHOVISON** annotations that SimBA expects, click [HERE](https://github.com/sgoldenlab/simba/blob/master/misc/ethovision_example.xlsx). If your files look (i) different from these files, and (ii) you are having trouble appending BORIS annotations in SimBA, please reach out to us on [Gitter](https://gitter.im/SimBA-Resource/community) or open a [GitHub issue](https://github.com/sgoldenlab/simba) report. 

2. Click `Select` for the folder that contains your annotation. You can follow the progress in the main SimBA terminal window. 

4. If the appending of your annotations where completed successfully, you should see files representing each of your videos inside the `project_folder/csv/targets_inserted` directory of your SimBA project. If you open these files, you should see one column (towards the very and of the file) representing each of your classifiers. These columns will be populated with `0` and `1`, representing the absence (`0`) and presence (`1`) of the behavior according to your annotations in the ETHOVISON annotation tool. 



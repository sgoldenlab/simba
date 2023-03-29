# THIRD-PARTY ANNOTATIONS (BEHAVIOR LABELS) IN SIMBA

SimBA has an [in-built behavior annotation interface](https://github.com/sgoldenlab/simba/blob/master/docs/labelling_aggression_tutorial.md) that allow users to append experimenter-made labels to the [features extracted](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-5-extract-features) from  pose-estimation data. Accurate human labelling of images can be the most time-consuming part of creating reliable supervised machine learning models. However, often experimenters have previously hand-labelled videos using manual software annotation tools and these labels can be appended to the pose-estimation datasets in SimBA. Some users may also prefer to use other dedicated behavior annotation tools rather than using the [annotation tool built into SimBA](https://github.com/sgoldenlab/simba/blob/master/docs/labelling_aggression_tutorial.md).

SimBA currently supports the import of annotations created in:

* [SOLOMON CODER](https://solomon.andraspeter.com/)
* [BORIS](https://www.boris.unito.it/)
* [BENTO](https://github.com/neuroethology/bentoMAT)
* [NOLDUS ETHOVISION](https://www.noldus.com/ethovision-xt)
* [DeepEthogram](https://github.com/jbohnslav/deepethogram)
* [NOLDUS OBSERVER](https://www.noldus.com/observer-xt)

If you have annotation created in any other software tool and would like to append them to your data in SimBA, please reach out to us on [Gitter](https://gitter.im/SimBA-Resource/community) or [GitHub](https://github.com/sgoldenlab/simba) and we can work together to make the option available in the SimBA GUI.

## BEFORE IMPORTING THIRD-PARTY ANNOTATIONS (BEHAVIOR LABELS) INTO YOUR SIMBA PROJECT

In brief, before we can import the third-party annotations, we need to (i) create a SimBA project, (ii) import our pose-estimation data into this project, (iii) correct any outliers (or indicate to skip outlier correction), and (iv) extract features for our data set, thus: 

1. Once you have created a project, click on `File` --> `Load project` to load your project. For more information on creating a project, click [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#part-1-create-a-new-project-1).

2. Make sure you have extracted the features from your tracking data and there are files, one file representing each video in your project, located inside the `project_folder/csv/features_extracted` directory of your project. For more information on extracting features, click [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-5-extract-features).

### TUTORIAL 

**1.** After loading your project, click on the `[LABEL BEHAVIOR]` tab, and the `Append third-party annotations` button inside the `LABELLING TOOLS` sub-frame and you should see the following pop-up:

<p align="center">
<img src=/images/third_party_label_new_1.png />
</p>

**2.** In the first drop-down menu named 'THIRD-PARTY APPLICATION`, select the application which your annotations were created in:

<p align="center">
<img src=/images/third_party_label_new_2.png />
</p>

**3.** Next, where it says `DATA DIRECTORY`, click `Browse Folder` and select the directory where your third-party annotations are stored. 

**4.** Next, we need to tell SimBA how to deal with inconsistancies in the annotation data and conflicts with the pose-estimation data (if they exist). While developing these tools in SimBA, we have been shared a lot of annotation files from a lot of users. We have noticed that the annotation files sometimes have oddities; e.g., behaviors that are annotated to end before they start or that start more times then they end etc etc.. We need to deal with these inconsistancies and conflicts when appeding the labels, and these settings gives the users some powers in how we do this. 

Each of the `WARNINGS AND ERRORS` dropdowns have two options: `WARNING`, and `ERROR`:

<p align="center">
<img src=/images/third_party_label_new_3.png />
</p>

If `WARNING` is selected, then SimBA will warn you with printed text that an inconsistancy has been found, where it was found, and try to remedy the issue. If `ERROR` is selected, then SimBA will shut down the appending process and print a text saying an inconsistancy has been found and where it was found, so that the user can look into and fix the issue. Below we will go through the potential inconsistancies and conflicts that SimBA will look for:

(i) `INVALID ANNOTATION DATA FILE FORMAT`: SimBA expects the annotation files to look a certain way depending on the third-party application. SimBA needs to make assumptions on - for example - where to find the time-stamps, the behaviors, and the video names etc. within your files, and when these assumptions are wrong, SimBA will show and *INVALID ANNOTATION DATA FILE FORMAT* error or warning. If this dropdown is set to **WARNING**, then SimBA will skip to read in the invalid file. If this dropdown is set to **ERROR**, then SimBA will halt the reading of your annotation files and show you en error. See below links for the file formats expected by SimBA from the differet annotation tools.

* [BENTO](https://github.com/sgoldenlab/simba/blob/master/misc/bento_example.annot)
* [BORIS](https://github.com/sgoldenlab/simba/blob/master/misc/boris_example.csv)
* [DEEPETHOGRAM](https://github.com/sgoldenlab/simba/blob/master/misc/deep_ethogram_labels.csv)
* [ETHOVISION](https://github.com/sgoldenlab/simba/blob/master/misc/ethovision_example.xlsx)
* [OBSERVER](https://github.com/sgoldenlab/simba/blob/master/misc/Observer_example_1.xlsx)
* [OBSERVER](https://github.com/sgoldenlab/simba/blob/master/misc/Observer_example_2.xlsx)
* [SOLOMON](https://github.com/sgoldenlab/simba/blob/master/misc/solomon_example.csv)


(ii) `ADDITIONAL THIRD PARTY BEHAVIOR DETECTED`: At times, the annotation files contain annotations for behaviors that you have **not** defined in your SimBA project. 

Example 1: You have defined two classifiers/behaviors in your SimBA project called: `Attack` and `Sniffing`. In your annotation files, SimBA finds annotations for `Attack`, `Sniffing`, and `Grooming`. As `Grooming` is not defined in your SimBA project we are not sure what to do with those annotations.

Example 2: You have defined two classifiers/behaviors in your SimBA project called: `Attack` and `Sniffing`. In your annotation files, SimBA finds annotations for `attack`, `sniffing`. As `attack` and `sniffing` is not defined in your SimBA project we are not sure what to do with those annotations. 

If this dropdown is set to **WARNING**, then SimBA will show you a warning and **discard** the behaviors not defined in your SimBA project. If this dropdown is set to **ERROR**, then SimBA will stop appending the annotations and show you an error about the additional classifiers/annotations found and where SImBA found them.

(ii) `ANNOTATION OVERLAP CONFLICT`: Many third-party annotation softwares give you a `BEHAVIOR` and `EVENT` columns (e.g., Noldus tools). For example, a specific log in a file can be a behavior of `Grooming` and the event is `START`, and the next row log the behavior is `Grooming` and the event is `STOP`. But what happens when a behavior is recorded as `STOP` before any record of an associated `START`? This would happen if you have sequantial logs of `Grooming` as `START`->`STOP`->`STOP`->`START` or `STOP`->`START`->`START`->`STOP` etc. 

If this dropdown is set to **WARNING**, then SimBA will show you a warning and try to find the innaccurate start->stop annotations and discard them. If this dropdown is set to **ERROR**, then SimBA will stop appending the annotations and show you an error about which behavior and video the overlap was found.


(iii) `ZERO THIRD-PARTY VIDEO BEHAVIOR ANNOTATIONS FOUND`: At times, a specific video contains zero third-party annotations for a specific behavior. 

If this dropdown is set to **WARNING**, then SimBA will show you a warning and set all frames in your data to annotated as **behavior absent**. If this dropdown is set to **ERROR**, then SimBA will stop appending the annotations and show you an error about which behavior and video contains zero annotations as behavior present. 

(iv) `ANNOTATION AND POSE FRAME COUNT FONFLICT`: It happens that the video annotated in your annotation tool and the pose-estimation data imported to SimBA has different number of frames. For example, the annotation data for a specific video may contains annotations for 2k frames, **but** the imported pose-estimation data for the same video is 1.5k frames long. More, if the annotation file indicates that your behavior is present between frame 1750 and frame 1800, then SimBA does not know what to do with those annotations as the pose-estimation data only has 1500 entries.  

If this dropdown is set to **WARNING**, then SimBA will show you a warning and **discard** any annotations made after the last frame in the pose-estimation data. If this dropdown is set to **ERROR**, then SimBA will stop appending your labels when this conflict occurs, and give you information about where the conflict was found (which video, which behavior, which frames, and how many frames). 

(v) `ANNOTATION EVENT COUNT CONFLICT`: Many third-party annotation softwares give you a `BEHAVIOR` and `EVENT` columns (e.g., Noldus tools). For example, a specific log in a file can be a behavior of `Grooming` and the event is `START`, and the next row log the behavior is `Grooming` and the event is `STOP`. But what happens when a behavior is recorded as have started N times and stopped N-1 or N+1 times? This would happen if you have sequantial logs of `Grooming` as `START`->`STOP`->`START`->`STOP`->'START' or `START`->`STOP`->`STOP` etc. Then we have an extra event with no counterevent which SimBA doesn't know how to deal with. 

If this dropdown is set to **WARNING**, then SimBA will show you a warning and try to find the innaccurate start/stop annotations and discard them. If this dropdown is set to **ERROR**, then SimBA will stop appending the annotations and show you an error about which behavior and video the count conflict was found in.

(vi) `ANNOTATION DATA FILE NOT FOUND`: Say we have two video data files representing our pose-estimation in our SimBA project - `Video_1` and `Video_2` -  that we want to append third-party annotations to. We import our third-party annotations for two videos `Video_1` and `Video_3`. When SimBA gets to `Video_2` and looks for the third-party annotations, it doesn't find any. 

If this dropdown is set to **WARNING**, then SimBA will show you a warning and skip appending annotations to the video that lack annotation data. If this dropdown is set to **ERROR**, then SimBA will stop appending the annotations and show you an error about which video is lacking annotation data.






then SimBA will **discard the annotations for the last 500 frames**. SimBA will print a warning in the main terminal window if this happens. Please make sure that the FPS of the imported video [registered in the video_info.csv file](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-3-set-video-parameters) and the video annotated in the third-party tool are identical - otherwise SimBA will get the frame numbers jumbled up. 
















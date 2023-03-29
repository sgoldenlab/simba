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

If `NONE` is selected, SimBA will not warn you, or try to remedy, if an inconsistancy/conflict is found. *Use this setting only when you know your annotation data contains **no** errors and you 

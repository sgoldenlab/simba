# APPENDING THIRD-PARTY ANNOTATIONS (BEHAVIOR LABELS) IN SIMBA

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/third_party_annot.png" />
</p>


SimBA has an [in-built behavior annotation interface](https://github.com/sgoldenlab/simba/blob/master/docs/labelling_aggression_tutorial.md) that allow users to append experimenter-made labels to the [features extracted](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-5-extract-features) from  pose-estimation data. Accurate human labelling of images can be the most time-consuming part of creating reliable supervised machine learning models. However, often experimenters have previously hand-labelled videos using manual software annotation tools and these labels can be appended to the pose-estimation datasets in SimBA. Some users may also prefer to use other dedicated behavior annotation tools rather than using the [annotation tool built into SimBA](https://github.com/sgoldenlab/simba/blob/master/docs/labelling_aggression_tutorial.md).


SimBA currently supports the import of annotations created in:

* [SOLOMON CODER](https://solomon.andraspeter.com/)
* [BORIS](https://www.boris.unito.it/)
* [BENTO](https://github.com/neuroethology/bentoMAT)
* [ETHOVISION](https://www.noldus.com/ethovision-xt)
* [DeepEthogram](https://github.com/jbohnslav/deepethogram)
* [Noldus Observer](https://www.noldus.com/observer-xt)

If you have annotation created in any other software tool and would like to append them to your data in SimBA, please reach out to us on [Gitter](https://gitter.im/SimBA-Resource/community) or [GitHub](https://github.com/sgoldenlab/simba) and we can work together to make the option available in the SimBA GUI.

## BEFORE IMPORTING THIRD-PARTY ANNOTATIONS (BEHAVIOR LABELS) INTO YOUR SIMBA PROJECT

In brief, before we can import the third-party annotations, we need to (i) create a SimBA project, (ii) import our pose-estimation data into this project, (iii) correct any outliers (or indicate to skip outlier correction), and (iv) extract features for our data set, thus: 

1. Once you have created a project, click on `File` --> `Load project` to load your project. For more information on creating a project, click [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#part-1-create-a-new-project-1).

2. Make sure you have extracted the features from your tracking data and there are files, one file representing each video in your project, located inside the `project_folder/csv/features_extracted` directory of your project. For more information on extracting features, click [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-5-extract-features).

3. After loading your project, click on the `Label behavior` tab, and you should see the following window, with the menu of interest marked in red:

<p align="center">
<img src=/images/third_party_2.png />
</p>

### TUTORIAL 

1.

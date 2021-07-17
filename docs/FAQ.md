# Friendly Asked Questions (FAQ)

# Please search the [SimBA issues tracker](https://github.com/sgoldenlab/simba/issues), or search the [Gitter chat channel](https://gitter.im/SimBA-Resource/community) for more questions and answers. 

##  1. I get 'TypeError: cannot convert the series to <class 'int'>' or 'TypeError: cannot convert the series to <class 'float'>' when trying to extract features, generate movies/frames, or when extracting outliers

>This error comes up when SimBA can't find the resolution and/or pixels per millimeter of your video. This data is stored in the `project_folder/logs/video_info.csv` file. If you open this CSV file, make sure that the left-most column named `Video` contains the names of your video files, do not contain any duplicate rows (where video names appear in more than one row), and that the resolution columns and pixels/mm column contains values. 

## 2. When I click on a video to set its parameters (i.e., "pixel per millimeter"), or try to open a video to dray ROI regions I get an OpenCV, I get an error report with something like "cv2.Error. The function is not implemented. Rebuild the library with Windows..."

To fix this, make sure your python environment has the correct version of OpenCV installed. Within your environment, try to type:

(1) `pip install opencv-python==3.4.5.20`

or

`conda install opencv-python==3.4.5.20`.

Then launch SimBA by typing `SimBA`, and see if that fixes the issue. 

(2) Alternatively, if this does not fix it, try:

`pip uninstall opencv-python`

followed by:

`pip install opencv-contrib-python` or `conda install opencv-contrib-python`

Then launch SimBA by typing `SimBA`, and see if that fixes the issue. 

## 3. I get a `QHull` (e..g., QH6154 or 6013) error when extracting the features

This error typically happens when a video tracked with DLC/DPK/SLEAP does not contain an animal. Did can happen because one or all animal is missing from the video frame, or you have filtered the data prior to importing it into SimBA to remove body-parts with low detection probabilities. Because no animal is present in the video, DeepLabCut and other pose-estimation tools places all body-parts at the same co-ordinate with a low probability (no-where, or frequently at coordinate (0,0), which is the upper-left most pixel in the image). SimBA tries to use these co-ordinates to calculate metrics from the hull of the animal, but bacause the coordinates are in 1D rather than 2D, it produces the `QHull` error. To fix it, try to use the video pre-processing tools in SimBA to trim the videos and discard the portions where no animals are present:

[Tutorial: Batch pre-process videos in SimBA](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial_process_videos.md)

[Tutorial: Video pre-processing tools in SimBA](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md)

A second possible reason for this error is that the body-part locations where filtered in the pose-estimation package. For example, in DLC, you may have indicated to drop body-part location predictions with probabilities below a certain threshold (i.e., you set a high DLC *p-cutoff*). SimBA will try to make behavioral predictions for all frames and this error can happen if there are no body-part locations to predict from. To fix it, we recommend to go back and analyze your videos without filtering the data and generate body-part locatuon predictions for **all** frames.

However, in some use-cases, it is not possible to keep the animal(s) in the frame for the entire video and/or removing segments of the video where the animal is absent. For these use-cases, SimBA includes an **interpolation tool** that can be used when importing data which handles missing pose-estimation body-part coordinates. For information on how to use the SimBA interpolation tool, click [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-4-extract-frames-into-project-folder)


## 4. The frames folder is empty after clicking to extract frames, or my videos have not been generated appropriately

>Make sure that: 

1. You have installed FFmpeg: [INSTALLATION INSTRUCTIONS](https://m.wikihow.com/Install-FFmpeg-on-Windows)

2. You are running Python 3.6.0 (or python 3.6.10 in conda). 

## 5. **SimBA won't launch - there's an error, with some complaint about Shapely**

Check out these issue threads for potential fixes:

https://github.com/sgoldenlab/simba/issues/12

https://github.com/sgoldenlab/simba/issues/11#issuecomment-596805732

https://github.com/sgoldenlab/simba/issues/70#issuecomment-703180294

## 6. SimBA won't start, and there is GPU related errors such as "ImportError: Could not find 'cudart64_100.dll'.

If you are running SimBA on a computer fitted with a RTX 2080ti GPU, make sure;

1. CUDA 10 is installed - https://developer.nvidia.com/cuda-10.0-download-archive
2. cuDNN 7.4.2 for CUDA 10 is installed - https://developer.nvidia.com/rdp/cudnn-archive
3. Tensorflow-gpu 1.14.0 is installed, run: pip install tensorflow-gpu==1.14.0
4. Tensorflow (without GPU support) is *not* installed: run pip uninstall tensorflow
5. Protobuf 3.6.0 is installed: pip install protobuf==3.6.0

If you are running SimBA on a computer fitted with a never RTX 3080ti GPU and struggling to get it running, reach out to us and we will send on instructions.

## 7. I get an error when launching the ROI interface - it is complaining about `ValueError: cannot set WRITEABLE flag to True of this array`. It may also have seen `Missing optional dependency 'tables`

Make sure you are running a later version of pytables(>= version 3.51). Also make sure you have numpy 1.18.1 and pandas 0.25.3 installed. To be sure of this, run:

`pip install tables --upgrade` or `pip install tables==3.5.1`. Pytables 3.5.1 may not be available in conda so do a `pip install tables==3.6.1`. 

`pip install pandas==0.25.3`

`pip install numpy==1.18.1`

## 8. My videos are very long and can be a pain to annotate in the SimBA annotation GUI, can I skip annotating some frames and still build an accurate classification model based on annotated/not-annotated frames?

When you first open the SimBA labelling GUI for a new, not previously annotated video (e.g., a video that has **not** gone through [SimBA "pseudo-labelling](https://github.com/sgoldenlab/simba/blob/master/docs/pseudoLabel.md)), SimBA automatically treats all frames in that video as **NOT** containing any of your behaviours of interest. 

If you decide, for example, to only annotate half of the video, then SimBA and any ensuing machine learning training steps will automatically treat the second half of that video as **examples of the absence of your behaviour(s)**. This is all well if you know, for certain, that the second part of the video does not contain any examples of your behavior of interest. 

However, if the second part of your video **does** contain examples of your behaviors of interest, the machine learning algorithm will suffer a lot(!). Because what you are doing in this scenario is giving the machine examples of your behaviors of interest, while at the same time telling the algorithm that it **isn't** your behaviour of interest: finding the relationships between your features and your behavior will be so much more difficult (and maybe impossible) if you give it the computer the wrong information.

## 9. When I try to execute some process in SimBA (e.g., feature extraction, or generate frames or videos etc), I get a TypeError that may look somthing like this:
```
TypeError("cannot convert the series to " "{0}".format(str(converter)))
TypeError: cannot convert the series to <class 'float'>
```
When you execute your process (e.g., Feature extraction), SimBA looks in the folder containing the output of the previous process (e.g., `project_filder/csv/outlier_corrected_movement_location`) and will aim to analyze all of the CSV files that this folder contains. To analyze it appropriatly (across rolling time windows etc.), SimBA also needs to know which **fps**, and **pixels per millimiter** the video file associated with this CSV file has. This fps and pixel per millimeter information is stored in your `project_folder/logs/video_info.csv` file, and SimBA will attempt to grab it for you. To do this, SimBA will take the filename of the CSV located in the `project_filder/csv/outlier_corrected_movement_location` folder, strip it of its file ending, and look in the first column of your `project_folder/logs/video_info.csv` file for a matching name. So if your first CSV file is called *Video1.csv*, SimBA will look in the first column of your `project_folder/logs/video_info.csv` for *Video1*. Here are the most common reasons for it going wrong and you see this error:

1. There is no *Video1* in your `project_folder/logs/video_info.csv` file. You may have renamed your files somewhere along the process or introduced a typo (e.g., there is a `Video1 ` or `Video 1` or possibly `video1`, but there is **no** `Video1` which is what SimBA is looking for. 

2. There are several `Video1` rows in your `project_folder/logs/video_info.csv` file. SimBA happens to find them all, can't decide which one is the correct one, and breaks. Make sure you only have one row representing each video in your project in your `project_folder/logs/video_info.csv` file. 

3. Another CSV file, has somehow nestled into your `project_filder/csv/outlier_corrected_movement_location` folder along the way (and this file is neither part of the project or has been processed in the [`Video Parameters`](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-3-set-video-parameters) tool in SimBA. SimBA sees it as it is present in the `project_filder/csv/outlier_corrected_movement_location` folder, but when it looks in the `project_folder/logs/video_info.csv` file - the **fps**, and **pixels per millimiter** is missing and you get thrown this error. 

## 10. When I install or update SimBA, I see a bunch or messages in the console, in red text, telling me some error has happened, similar or the same as this:
```diff
- ERROR: imbalanced-learn 0.7.0 has requirement scikit-learn>=0.23, but you'll have scikit-learn 0.22.2 which is incompatible.
- ERROR: deeplabcut 2.0.9 has requirement numpy~=1.14.5, but you'll have numpy 1.18.1 which is incompatible.
- ERROR: deeplabcut 2.0.9 has requirement python-dateutil~=2.7.3, but you'll have python-dateutil 2.8.1 which is incompatible.
```
These are warnings, and are not fatal. You should be able to ignore them - go ahead with launching SimBA by typing `simba` in the console. 

## 11. When I install or update SimBA, I see a bunch or messages in the console, telling there me about some `dependency conflicts`. The messages may look a little like this:

![](https://github.com/sgoldenlab/simba/blob/master/images/Dependencies.png)

These errors are related to an update in the pypi package manager version 20.3.3, [where they introduced stricter version control](https://pip.pypa.io/en/stable/news/). I suggest trying either:

* If you are installing SimBA via git - try typing `pip3 install -r simba/SimBA/requirements.txt --no-dependencies` rather than `pip3 install -r simba/SimBA/requirements.txt`

* Try downgrading pip before installing SimBA:
  - Run `pip install pip==20.1.1`
  - Next, run `pip install simba-uw-tf`, `pip install simba-uw-no-tf` or `pip install simba-uw-tf-dev`, depending [on which version of SimBA you want to run](https://github.com/sgoldenlab/simba/blob/master/docs/installation.md#installing-simba-option-1-recommended).  

* Install SimBA using pip and the `--no-dependencies` argument:
  - Type `pip install simba-uw-tf --no-dependencies`, `pip install simba-uw-no-tf--no-dependencies` or `pip install simba-uw-tf-dev --no-dependencies`, depending [on which version of SimBA you want to run](https://github.com/sgoldenlab/simba/blob/master/docs/installation.md#installing-simba-option-1-recommended).
  - Next, install any missing package dependencies manually. A list of dependencies can be found H
here: https://github.com/sgoldenlab/simba/blob/master/docs/installation.md#python-dependencies

## 12. When run my [classifier on new videos](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-3-run-the-classifier-on-new-data), or trying to to [validate my classifier on a single video](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#optional-step-before-running-machine-model-on-new-data) my predictions seem to be generated, but then I get either an IndexError: `Index 1 is out of bounds for axis 1 of size 1`, or an error msg telling me that my classifier hasn't been generated properly:

This means there is something odd going on with this classifier, and the classifier most likely was not  created properly. Your classifier is expected to return **two** values: the probability for absence of behaviour, and probability for presence of the behaviour. In this case your classifier only returns one value. 

This could happen if you did not provide the classifier with examples for **both** the presence of the behavior, and absence of the behavior, during the [behavioral annotation step](https://github.com/sgoldenlab/simba/blob/master/docs/labelling_aggression_tutorial.md). It could also happen if you only [imported annotations](https://github.com/sgoldenlab/simba/blob/master/docs/third_party_annot.md) for the presence *or* absence of the behavior, not both. This means SimBA can't generate probability scores for the presence/absence of the behavior, as you have not provided the classifier examples of both behavioral states. [Go back and re-create your classifier](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-6-label-behavior) using annotations for both the presence AND absence of the behaviour. 

## 13. When I try to correct outliers, I get an error about the wrong number of columns, for example: `Value error: Length mismatch: Expected axis has 16 elements, new values have 24`. 

One possibility here is that you are importing the wrong files from SLEAP / DLC / DeepPoseKit into SimBA. For example, if you are seeing the error message as exactly stated above, you are tracking 8 body-parts, each which should have 3 columns in your CSV file (8 * 3 = 24) but SimBA can only find 16 (8 * 2 = 16) and thus one column per body-part is missing. This could happen if you are importing your hand-annotated, human, body-part annotation files (which only has 2 columns per body-part - there is no 'probability' associated with your human hand labels) rather than the actual tracking files. 

If the files you are importing has a name akin to `CollectedData_MyName.csv` you are importing the **wrong** files and go back to your pose-estimation tools to generate actual machine predictions for your videos. If you are importing files with names akin to `MyVideoName_DeepCut_resnet50_Project1Nov11shuffle1_500000.csv` you are importing the right pose-estimation tracking files. 

## 14. When I try to to run my classifier on new videos, I get an error message telling me about a *mismatch* error - something along the lines of: **Mismatch in the number of features in input file and what is expected from the model in file...**

This means that the model was created with a different number of features than the number of features in the files you are now trying to analyze. 

For example, when you created the model, SimBA grabbed the files inside your `project_folder/features_extracted` directory, and each of those files happened to contain 450 columns (you might have more or less columns than this, I'm just making this up for this example). You now have new files inside your `project_folder/features_extracted` directory, which you are analyzing with the model you already have created. These files contain 455 columns, and that's why you see this error message. Among other things, the mismatch in columns could have been produced by:

* You created a standard classifier without [adding ROI features](https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial.md#part-3-generating-features-from-roi-data). For the new files you are trying to analyze however, you **did** [calculate ROI features](https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial.md#part-3-generating-features-from-roi-data). This will produce a mismatch in the number of columns between your model and your current files. 

* Before creating your model, you [calculated ROI features](https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial.md#part-3-generating-features-from-roi-data). These features where added to your files inside your `project_folder/features_extracted` directory. For the new files you are trying to analyze however, you **did not** [calculate ROI features](https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial.md#part-3-generating-features-from-roi-data). This would also produce a mismatch in the number of columns between your model and your current files. 


## 15. After a Microsoft Windows 10 update, I get a GPU CUDA/cudnn error - it was working before the update and now it is complaining about about `.cv2 DLL`

In Windows, try an go to Settings > Apps > Manage Optional Features > Add a Feature. And then check the boxes for both Windows Media Player and for the Media Feature Pack. Restart the computer, and try launching SimBA again. 





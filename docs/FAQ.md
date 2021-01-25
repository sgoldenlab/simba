# Friendly Asked Questions (FAQ)

# Please search the [SimBA issues tracker](https://github.com/sgoldenlab/simba/issues), or search the [Gitter chat channel](https://gitter.im/SimBA-Resource/community) for more questions and answers. 

##  I get 'TypeError: cannot convert the series to <class 'int'>' or 'TypeError: cannot convert the series to <class 'float'>' when trying to extract features, generate movies/frames, or when extracting outliers

>This error comes up when SimBA can't find the resolution and/or pixels per millimeter of your video. This data is stored in the `project_folder/logs/video_info.csv` file. If you open this CSV file, make sure that the left-most column named `Video` contains the names of your video files, do not contain any duplicate rows (where video names appear in more than one row), and that the resolution columns and pixels/mm column contains values. 


## When I click on a video to set its parameters (i.e., "pixel per millimeter"), or try to open a video to dray ROI regions I get an OpenCV, I get an error report with something like `cv2.Error. The function is not implemented. Rebuild the library with Windows...`

To fix this, make sure your python environment has the correct version of OpenCV installed. Within your environment, try to type:

`pip install opencv-python==3.4.5.20`

or

`conda install opencv-python==3.4.5.20`

Then launch SimBA by typing `SimBA`, and see if that fixes the issue. 




## I get a `QHull` (e..g., QH6154 or 6013) error when extracting the features

>This error typically happens when a video tracked with DLC/DPK does not contain an animal (or one animal is missing from the video when you are tracking two animals). Because no animal is present in the video, DeepLabCut places all body-parts at the same co-ordinate with a low probability (no-where, or frequently at coordinate (0,0), which is the upper-left most pixel in the image). SimBA tries to use these co-ordinates to calculate metrics from the hull of the animal, but bacause the coordinates are in 1D rather than 2D, it produces the `QHull` error. To fix it, use the video pre-processing tools in SimBA to trim the videos and discard the portions where no animals are present:

[Tutorial: Batch pre-process videos in SimBA](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial_process_videos.md)

[Tutorial: Video pre-processing tools in SimBA](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md)

## The frames folder is empty after clicking to extract frames, or my videos have not been generated appropriately

>Make sure that: 

1. You have installed FFmpeg: [INSTALLATION INSTRUCTIONS](https://m.wikihow.com/Install-FFmpeg-on-Windows)

2. You are running Python 3.6.0 (or python 3.6.10 in conda). 

## **SimBA won't launch - there's an error, with some complaint about Shapely**

Check out these issue threads for potential fixes:

https://github.com/sgoldenlab/simba/issues/12

https://github.com/sgoldenlab/simba/issues/11#issuecomment-596805732

https://github.com/sgoldenlab/simba/issues/70#issuecomment-703180294

## SimBA won't start, and there is GPU related errors such as "ImportError: Could not find 'cudart64_100.dll'.

Make sure;

1. CUDA 10 is installed - https://developer.nvidia.com/cuda-10.0-download-archive
2. cuDNN 7.4.2 for CUDA 10 is installed - https://developer.nvidia.com/rdp/cudnn-archive
3. Tensorflow-gpu 1.14.0 is installed, run: pip install tensorflow-gpu==1.14.0
4. Tensorflow (without GPU support) is *not* installed: run pip uninstall tensorflow
5. Protobuf 3.6.0 is installed: pip install protobuf==3.6.0

## I get an error when launching the ROI interface - it is complaining about `ValueError: cannot set WRITEABLE flag to True of this array`. It may also have seen `Missing optional dependency 'tables`

Make sure you are running a later version of pytables(>= version 3.51). Also make sure you have numpy 1.18.1 and pandas 0.25.3 installed. To be sure of this, run:

`pip install tables --upgrade` or `pip install tables==3.5.1`. Pytables 3.5.1 may not be available in conda so do a `pip install tables==3.6.1`. 

`pip install pandas==0.25.3`

`pip install numpy==1.18.1`

## My videos are very long and can be a pain to annotate in the SimBA annotation GUI, can I skip annotating some frames and still build an accurate classification model based on annotated/not-annotated frames?

When you first open the SimBA labelling GUI for a new, not previously annotated video (e.g., a video that has **not** gone through [SimBA "pseudo-labelling](https://github.com/sgoldenlab/simba/blob/master/docs/pseudoLabel.md)), SimBA automatically treats all frames in that video as **NOT** containing any of your behaviours of interest. 

If you decide, for example, to only annotate half of the video, then SimBA and any ensuing machine learning training steps will automatically treat the second half of that video as **examples of the absence of your behaviour(s)**. This is all well if you know, for certain, that the second part of the video does not contain any examples of your behavior of interest. 

However, if the second part of your video **does** contain examples of your behaviors of interest, the machine learning algorithm will suffer a lot(!). Because what you are doing in this scenario is giving the machine examples of your behaviors of interest, while at the same time telling the algorithm that it **isn't** your behaviour of interest: finding the relationships between your features and your behavior will be so much more difficult (and maybe impossible) if you give it the computer the wrong information.

## When I try to execute some process in SimBA (e.g., feature extraction, or generate frames or videos etc), I get a TypeError that may look somthing like this:
```
TypeError("cannot convert the series to " "{0}".format(str(converter)))
TypeError: cannot convert the series to <class 'float'>
```
When you execute your process (e.g., Feature extraction), SimBA looks in the folder containing the output of the previous process (e.g., `project_filder/csv/outlier_corrected_movement_location`) and will aim to analyze all of the CSV files that this folder contains. To analyze it appropriatly (across rolling time windows etc.), SimBA also needs to know which **fps**, and **pixels per millimiter** the video file associated with this CSV file has. This fps and pixel per millimeter information is stored in your `project_folder/logs/video_info.csv` file, and SimBA will attempt to grab it for you. To do this, SimBA will take the filename of the CSV located in the `project_filder/csv/outlier_corrected_movement_location` folder, strip it of its file ending, and look in the first column of your `project_folder/logs/video_info.csv` file for a matching name. So if your first CSV file is called *Video1.csv*, SimBA will look in the first column of your `project_folder/logs/video_info.csv` for *Video1*. Here are the most common reasons for it going wrong and you see this error:

1. There is no *Video1* in your `project_folder/logs/video_info.csv` file. You may have renamed your files somewhere along the process or introduced a typo (e.g., there is a `Video1 ` or `Video 1` or possibly `video1`, but there is **no** `Video1` which is what SimBA is looking for. 

2. There are several `Video1` rows in your `project_folder/logs/video_info.csv` file. SimBA happens to find them all, can't decide which one is the correct one, and breaks. Make sure you only have one row representing each video in your project in your `project_folder/logs/video_info.csv` file. 

3. Another CSV file, has somehow nestled into your `project_filder/csv/outlier_corrected_movement_location` folder along the way (and this file is neither part of the project or has been processed in the [`Video Parameters`](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-3-set-video-parameters) tool in SimBA. SimBA sees it as it is present in the `project_filder/csv/outlier_corrected_movement_location` folder, but when it looks in the `project_folder/logs/video_info.csv` file - the **fps**, and **pixels per millimiter** is missing and you get thrown this error. 

## When I install or update SimBA, I see a bunch or messages in the console, in red text, telling me some error has happened, similar or the same as this:
```diff
- ERROR: imbalanced-learn 0.7.0 has requirement scikit-learn>=0.23, but you'll have scikit-learn 0.22.2 which is incompatible.
- ERROR: deeplabcut 2.0.9 has requirement numpy~=1.14.5, but you'll have numpy 1.18.1 which is incompatible.
- ERROR: deeplabcut 2.0.9 has requirement python-dateutil~=2.7.3, but you'll have python-dateutil 2.8.1 which is incompatible.
```
These are warnings, and are not fatal. You should be able to ignore them - go ahead with launching SimBA by typing `simba` in the console. 

## When I install or update SimBA, I see a bunch or messages in the console, telling there me about some `dependency conflicts`. The messages may look a little like this:

![](https://github.com/sgoldenlab/simba/blob/master/images/Dependencies.png)

These errors are related to an update in the pypi package manager version 20.3.3, [where they introduced stricter version control](https://pip.pypa.io/en/stable/news/). I suggest trying either:

* If you are installing SimBA via git - try typing `pip3 install -r simba/SimBA/requirements.txt --no-dependencies` rather than `pip3 install -r simba/SimBA/requirements.txt`

* Try downgrading pip before installing SimBA:
  - Run `pip install pip==20.1.1`
  - Next, run `pip install simba-uw-tf`, `pip install simba-uw-no-tf` or `pip install simba-uw-tf-dev`, depending [on which version of SimBA you want to run](https://github.com/sgoldenlab/simba/blob/master/docs/installation.md#installing-simba-option-1-recommended).  

* Install SimBA using pip and the `--no-dependencies` argument:
  - Type `pip install simba-uw-tf --no-dependencies`, `pip install simba-uw-no-tf--no-dependencies` or `pip install simba-uw-tf-dev --no-dependencies`, depending [on which version of SimBA you want to run](https://github.com/sgoldenlab/simba/blob/master/docs/installation.md#installing-simba-option-1-recommended).



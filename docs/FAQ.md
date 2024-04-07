# Friendly Asked Questions (FAQ)

## Please search the [SimBA issues tracker](https://github.com/sgoldenlab/simba/issues), or search the [Gitter chat channel](https://gitter.im/SimBA-Resource/community) for more questions and answers. If that does not help, reach out to us by posting a new issue or write to us on Gitter.  


###  1. I get 'TypeError: cannot convert the series to <class 'int'>' or 'TypeError: cannot convert the series to <class 'float'>' when trying to extract features, generate movies/frames, or when extracting outliers
<details>
  <summary>Show solutions!</summary>
<br/><br/>
This error typilcally comes up when SimBA can't find the resolution and/or pixels per millimeter of your video. This data is stored in the `project_folder/logs/video_info.csv` file. If you open this CSV file, make sure that the left-most column named `Video` contains the names of your video files, do not contain any duplicate rows (where video names appear in more than one row), and that the resolution columns and pixels/mm column contains values. 
  
</details>

### 2. When I click on a video to set its parameters (i.e., "pixel per millimeter"), or try to open a video to dray ROI regions I get an OpenCV, I get an error report with something like "cv2.Error. The function is not implemented. Rebuild the library with Windows..."

<details>
  <summary>Show solutions!</summary>
<br/><br/>
To fix this, make sure your python environment has the correct version of OpenCV installed. Within your environment, try to type:

(1) `pip install opencv-python==3.4.5.20` or `conda install opencv-python==3.4.5.20`.

Then launch SimBA by typing `SimBA`, and see if that fixes the issue. 

(2) Alternatively, if this does not fix it, try typing:

`pip uninstall opencv-python` followed by either  `pip install opencv-contrib-python` or `conda install opencv-contrib-python`

Then launch SimBA by typing `SimBA`, and see if that fixes the issue. 
  
</details>

### 3. I get a `QHull` (e..g., QH6154 or 6013) error when extracting the features

<details>
  <summary>Show solutions!</summary>
<br/><br/>
This error typically happens when a video tracked with DLC/DPK/SLEAP does not contain an animal. This can happen because one or all animal is missing from the video frame, or you have filtered the data prior to importing it into SimBA to remove body-parts with low detection probabilities. Because no animal is present in the video, DeepLabCut and other pose-estimation tools places all body-parts at the same co-ordinate with a low probability (no-where, or frequently at coordinate (0,0), which is the upper-left most pixel in the image). SimBA tries to use these co-ordinates to calculate metrics from the hull of the animal (e.g., the animal volume), but bacause the coordinates are in 1D rather than 2D, it produces the `QHull` error. To fix it, try to use the video pre-processing tools in SimBA to trim the videos and discard the portions where no animals are present:

[Tutorial: Batch pre-process videos in SimBA](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial_process_videos.md)

[Tutorial: Video pre-processing tools in SimBA](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md)

A second possible reason for this error is that the body-part locations where filtered in the pose-estimation package. For example, in DLC, you may have indicated to drop body-part location predictions with probabilities below a certain threshold (i.e., you set a high DLC *p-cutoff*). SimBA will try to make behavioral predictions for all frames and this error can happen if there are no body-part locations to predict from. To fix it, we recommend to go back and analyze your videos without filtering the data and generate body-part location predictions for **all** frames.

However, in some use-cases, it is not possible to keep the animal(s) in the frame for the entire video and/or removing segments of the video where the animal is absent. For these use-cases, SimBA includes an **interpolation tool** that can be used when importing data which handles missing pose-estimation body-part coordinates. For information on how to use the SimBA interpolation tool, click [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-4-extract-frames-into-project-folder)

</details>

### 4. The frames folder is empty after clicking to extract frames, or my videos have not been generated appropriately

<details>
  <summary>Show solutions!</summary>
<br/><br/>

1. You have installed FFmpeg: [INSTALLATION INSTRUCTIONS](https://m.wikihow.com/Install-FFmpeg-on-Windows)

2. You are running Python 3.6.0 (or python 3.6.10 in conda).
  
</details>

### 5. SimBA won't launch - there's an error with some complaint about Shapely

<details>
  <summary>Show solutions!</summary>
<br/><br/>

Check out these issue threads for potential fixes:

https://github.com/sgoldenlab/simba/issues/12

https://github.com/sgoldenlab/simba/issues/11#issuecomment-596805732

https://github.com/sgoldenlab/simba/issues/70#issuecomment-703180294
  
</details>


### 6. SimBA won't start, and there is GPU related errors, such as "ImportError: Could not find 'cudart64_100.dll'.

<details>
  <summary>Show solutions!</summary>
<br/><br/>


If you are running SimBA on a computer fitted with a RTX 2080ti GPU, make sure;

1. CUDA 10 is installed - https://developer.nvidia.com/cuda-10.0-download-archive
2. cuDNN 7.4.2 for CUDA 10 is installed - https://developer.nvidia.com/rdp/cudnn-archive
3. Tensorflow-gpu 1.14.0 is installed, run: pip install tensorflow-gpu==1.14.0
4. Tensorflow (without GPU support) is *not* installed: run pip uninstall tensorflow
5. Protobuf 3.6.0 is installed: pip install protobuf==3.6.0

If you are running SimBA on a computer fitted with a never RTX 3080TI/ 3070 GPU, and to get the packages to recognize your GPU, then have a look at [THESE INTSTRUCTIONS](https://github.com/sgoldenlab/simba/blob/master/docs/pose_on_rtx30x.md).

</details>


### 7. I get an error when launching the ROI interface - it is complaining about `ValueError: cannot set WRITEABLE flag to True of this array`. It may also have seen `Missing optional dependency 'tables`. 

<details>
  <summary>Show solutions!</summary>
<br/><br/>
  
Make sure you are running a later version of pytables(>= version 3.51). Also make sure you have numpy 1.18.1 and pandas 0.25.3 installed. To be sure of this, run:

`pip install tables --upgrade` or `pip install tables==3.5.1`. Pytables 3.5.1 may not be available in conda so do a `pip install tables==3.6.1`. 

`pip install pandas==0.25.3`

`pip install numpy==1.18.1`
  
</details>

### 8. My videos are very long and can be a pain to annotate in the SimBA annotation GUI, can I skip annotating some frames and still build an accurate classification model based on annotated/not-annotated frames?

<details>
  <summary>Show solutions!</summary>
<br/><br/>

When you first open the SimBA labelling GUI for a new, not previously annotated video (e.g., a video that has **not** gone through [SimBA "pseudo-labelling](https://github.com/sgoldenlab/simba/blob/master/docs/pseudoLabel.md)), SimBA automatically treats all frames in that video as **NOT** containing any of your behaviours of interest. 

If you decide, for example, to only annotate half of the video, then SimBA and any ensuing machine learning training steps will automatically treat the second half of that video as **examples of the absence of your behaviour(s)**. This is all well if you know, for certain, that the second part of the video does not contain any examples of your behavior of interest. 

However, if the second part of your video **does** contain examples of your behaviors of interest, the machine learning algorithm will suffer a lot(!). Because what you are doing in this scenario is giving the machine examples of your behaviors of interest, while at the same time telling the algorithm that it **isn't** your behaviour of interest: finding the relationships between your features and your behavior will be so much more difficult (and maybe impossible) if you give it the computer the wrong information.

</details>
  
### 9. When I try to execute some process in SimBA (e.g., feature extraction, or generate frames or videos etc), I get a TypeError that may look somthing like this:
```
TypeError("cannot convert the series to " "{0}".format(str(converter)))
TypeError: cannot convert the series to <class 'float'>
```
<details>
  <summary>Show solutions!</summary>
<br/><br/>

When you execute your process (e.g., Feature extraction), SimBA looks in the folder containing the output of the previous process (e.g., `project_filder/csv/outlier_corrected_movement_location`) and will aim to analyze all of the CSV files that this folder contains. To analyze it appropriatly (across rolling time windows etc.), SimBA also needs to know which **fps**, and **pixels per millimiter** the video file associated with this CSV file has. This fps and pixel per millimeter information is stored in your `project_folder/logs/video_info.csv` file, and SimBA will attempt to grab it for you. To do this, SimBA will take the filename of the CSV located in the `project_filder/csv/outlier_corrected_movement_location` folder, strip it of its file ending, and look in the first column of your `project_folder/logs/video_info.csv` file for a matching name. So if your first CSV file is called *Video1.csv*, SimBA will look in the first column of your `project_folder/logs/video_info.csv` for *Video1*. Here are the most common reasons for it going wrong and you see this error:

1. There is no *Video1* in your `project_folder/logs/video_info.csv` file. You may have renamed your files somewhere along the process or introduced a typo (e.g., there is a `Video1 ` or `Video 1` or possibly `video1`, but there is **no** `Video1` which is what SimBA is looking for. 

2. There are several `Video1` rows in your `project_folder/logs/video_info.csv` file. SimBA happens to find them all, can't decide which one is the correct one, and breaks. Make sure you only have one row representing each video in your project in your `project_folder/logs/video_info.csv` file. 

3. Another CSV file, has somehow nestled into your `project_filder/csv/outlier_corrected_movement_location` folder along the way (and this file is neither part of the project or has been processed in the [`Video Parameters`](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-3-set-video-parameters) tool in SimBA. SimBA sees it as it is present in the `project_filder/csv/outlier_corrected_movement_location` folder, but when it looks in the `project_folder/logs/video_info.csv` file - the **fps**, and **pixels per millimiter** is missing and you get thrown this error. 

</details>
  
  
### 10. When I install or update SimBA, I see a bunch or messages in the console, in red text, telling me some error has happened, similar or the same as this:
```diff
- ERROR: imbalanced-learn 0.7.0 has requirement scikit-learn>=0.23, but you'll have scikit-learn 0.22.2 which is incompatible.
- ERROR: deeplabcut 2.0.9 has requirement numpy~=1.14.5, but you'll have numpy 1.18.1 which is incompatible.
- ERROR: deeplabcut 2.0.9 has requirement python-dateutil~=2.7.3, but you'll have python-dateutil 2.8.1 which is incompatible.
```
<details>
  <summary>Show solutions!</summary>
<br/><br/>

These are warnings, and are not fatal. You should be able to ignore them - go ahead with launching SimBA by typing `simba` in the console. 
</details>


### 11. When I install or update SimBA, I see a bunch or messages in the console, telling there me about some `dependency conflicts`. The messages may look a little like this:

<details>
  <summary>Show solutions!</summary>
<br/><br/>


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
  
</details>

### 12. When run my [classifier on new videos](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-3-run-the-classifier-on-new-data), or trying to to [validate my classifier on a single video](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#optional-step-before-running-machine-model-on-new-data) my predictions seem to be generated, but then I get either an IndexError: `Index 1 is out of bounds for axis 1 of size 1`, or an error msg telling me that my classifier hasn't been generated properly:

<details>
  <summary>Show solutions!</summary>
<br/><br/>

This means there is something odd going on with this classifier, and the classifier most likely was not  created properly. Your classifier is expected to return **two** values: the probability for absence of behaviour, and probability for presence of the behaviour. In this case your classifier only returns one value. 

This could happen if you did not provide the classifier with examples for **both** the presence of the behavior, and absence of the behavior, during the [behavioral annotation step](https://github.com/sgoldenlab/simba/blob/master/docs/labelling_aggression_tutorial.md). It could also happen if you only [imported annotations](https://github.com/sgoldenlab/simba/blob/master/docs/third_party_annot.md) for the presence *or* absence of the behavior, not both. This means SimBA can't generate probability scores for the presence/absence of the behavior, as you have not provided the classifier examples of both behavioral states. [Go back and re-create your classifier](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-6-label-behavior) using annotations for both the presence AND absence of the behaviour. 

</details>

### 13. When I try to correct outliers, I get an error about the wrong number of columns, for example: `Value error: Length mismatch: Expected axis has 16 elements, new values have 24`. 

<details>
  <summary>Show solutions!</summary>
<br/><br/>
  
One possibility here is that you are importing the wrong files from SLEAP / DLC / DeepPoseKit into SimBA. For example, if you are seeing the error message as exactly stated above, you are tracking 8 body-parts, each which should have 3 columns in your CSV file (8 * 3 = 24) but SimBA can only find 16 (8 * 2 = 16) and thus one column per body-part is missing. This could happen if you are importing your hand-annotated, human, body-part annotation files (which only has 2 columns per body-part - there is no 'probability' associated with your human hand labels) rather than the actual tracking files. 

If the files you are importing has a name akin to `CollectedData_MyName.csv` you are importing the **wrong** files and go back to your pose-estimation tools to generate actual machine predictions for your videos. If you are importing files with names akin to `MyVideoName_DeepCut_resnet50_Project1Nov11shuffle1_500000.csv` you are importing the right pose-estimation tracking files. 
  
</details>

### 14. When I try to to run my classifier on new videos, I get an error message telling me about a *mismatch* error - something along the lines of: `Mismatch in the number of features in input file and what is expected from the model in file...`

<details>
  <summary>Show solutions!</summary>
<br/><br/>

This means that the model was created with a different number of features than the number of features in the files you are now trying to analyze. 

For example, when you created the model, SimBA grabbed the files inside your `project_folder/features_extracted` directory, and each of those files happened to contain 450 columns (you might have more or less columns than this, I'm just making this up for this example). You now have new files inside your `project_folder/features_extracted` directory, which you are analyzing with the model you already have created. These files contain 455 columns, and that's why you see this error message. Among other things, the mismatch in columns could have been produced by:

* You created a standard classifier without [adding ROI features](https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial.md#part-3-generating-features-from-roi-data). For the new files you are trying to analyze however, you **did** [calculate ROI features](https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial.md#part-3-generating-features-from-roi-data). This will produce a mismatch in the number of columns between your model and your current files. 

* Before creating your model, you [calculated ROI features](https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial.md#part-3-generating-features-from-roi-data). These features where added to your files inside your `project_folder/features_extracted` directory. For the new files you are trying to analyze however, you **did not** [calculate ROI features](https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial.md#part-3-generating-features-from-roi-data). This would also produce a mismatch in the number of columns between your model and your current files. 

</details>
  
### 15. After a Microsoft Windows 10 update, I get a GPU CUDA/cudnn error - it was working before the update and now it is complaining about about `.cv2 DLL`

<details>
  <summary>Show solutions!</summary>
<br/><br/>

In Windows, try an go to Settings > Apps > Manage Optional Features > Add a Feature. And then check the boxes for both Windows Media Player and for the Media Feature Pack. Restart the computer, and try launching SimBA again. 
  
</details>

### 16. I get en error message, complaining that something like `Could not find a version that satisfies the requirement tensorflow-gpu==1.14.0. No Matching distribution found for tensorflow==1.14.0.`

<details>
  <summary>Show solutions!</summary>
<br/><br/>


You may be running the wrong version of python. You want to be running Python 64-bit, but may have installed the Python 32-bit version. 
  
</details>

### 17. I get an error when I try to import multi-animal data (H5 files) from DeepLabCut into SimBA - the error says that SimBA cannot locate my video files. 

<details>
  <summary>Show solutions!</summary>
<br/><br/>

This error may read (depending on what version of SimBA you are using):

* `Cannot locate video MyVideo in mp4 or avi format`, or
* `ERROR: SimBA searched your project_folder/videos directory for a video file representing the MyVideo and could not find a match. Above is a list of possible video filenames that SimBA searched for within your projects video directory without success.`

Here, you have provided SimBA with a directory containing DeepLabCut tracking data, and now we need to open the video so that we can indicate which animal is which to keep downstream processing consistant across videos. For SimBA to find the video, SimBA makes three assumptions behind the hood:

1. The video file is located inside the `project_folder/videos` directory. 
2. The video file is in `.mp4` or `.avi` format. 
3. If we split the DeepLabCut H5 filename into 2 parts at a given place, then the first part of the H5 filename will contain the video file-name. Specifically, DeepLabCut output files can be very long and contain a lot of information, examples could be (with the video filename highlighted in bold:

* **My_video_007**DLC_dlcrnetms5_main_projectJul1shuffle1_100000_el.h5, or
* **White_mice_together_no_tail_ends**DLC_resnet50_two_white_mice_052820May28shuffle1_200000_bx.h5

To find the correct video name, SimBA tries to split each filename into two at a bunch of hard-coded split-points (e.g., 'DLC_resnet50', 'DLC_resnet_50', 'DLC_dlcrnetms5') and keeps the only the text prior to these split-points. It then appends the potential file extension ('.mp4', '.MP4', '.avi', '.AVI') and checks if those files are present in your `project_folder/videos` directory, one by one. If it can't find a match for any possible combination and permutation, you get this error. Likely, you may have manually renamed your pose-estimation file-names so the assumptions SimBA makes are no longer valid. 

</details>

### 18. I'm trying to lunch SimBA on a Mac, and I get an `ImportError` reading `ImportError: Python is not installed as a framework. The Mac OS X backend will not be able to function correctly if Python is not installed as a framework...`. Or, while trying to launch SimBA on my Mac, I get an error reading `_tkinter.TclError: expected boolean value but got ""`. 

<details>
  <summary>Show solutions!</summary>
<br/><br/>

This error is caused by running the wrong version of Python. You want to make sur that the tkinter version inside your environment is no later than `8.6.10`. If you are using conda, we suggest using Python `3.6.13`. For more information, see [THIS ISSUE](https://github.com/sgoldenlab/simba/issues/143).
  
</details>

### 19. I have installed SimBA on MacOS. When I try to launch SimBA by typing `simba`, I get an error saying:  ```ImportError: Python is not installed as a framework. The Mac OS X backend will not be able to function correctly if Python is not installed as a framework. See the Python documentation for more information on installing Python as a framework on Mac OS X. Please either reinstall Python as a framework, or try one of the other backends. If you are using (Ana)Conda please install python.app and replace the use of 'python' with 'pythonw'. See 'Working with Matplotlib on OSX' in the Matplotlib FAQ for more information.```


<details>
  <summary>Show solutions!</summary>
<br/><br/>

In your conda environment: 

(i) uninstall `matplotlib` by typing `pip uninstall matplotlib`
  
(ii) install `matplotlib` using conda, by typing `conda install matplotlib`
  
(iii) try to start simba by typing `simba`
  
</details>


### 20. I have installed SimBA on MacOS. When I try to launch SimBA by typing `simba`, I get a long error message, which ends with: ```libc++abi.dylib: terminating with uncaught exception of type NSException Abort trap: 6```


<details>
  <summary>Show solutions!</summary>
<br/><br/>

(i) use anaconda to import this conda environment file [environment.yml.zip](https://github.com/sgoldenlab/simba/files/9097623/environment.yml.zip)

(ii) There is a directory in your root called ~/.matplotlib. Create a file ~/.matplotlib/matplotlibrc there and add the following code: backend: TkAgg
https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python

(iii) Do "conda install shapely" in this new anaconda environment

Type simba to check if it works!
  
For more information, see this [GitHub issue](https://github.com/sgoldenlab/simba/issues/196)


</details>

### 21. I am trying to install SimBA on a machine with Linux, and I get an error ```OSError: Could not find library geos_c or load any of its variants ['libgeos_c.so.1', 'libgeos_c.so’]```. I may also get an additional error complaining about `opencv-python` 

<details>
  <summary>Show solutions!</summary>
<br/><br/>

This error is produced when SimBA is trying to install the packages `shapeley` and `opencv` on your machine. Try to install SimBA with

(i)Install SimBA with `pip install simba-uw-tf-dev --no-deps`

(ii) install the requirements manually, and then install wxpython seperately using the suggestions e.g. here: wxWidgets/Phoenix#465

</details>

### 22. I am trying to launch SimBA by typing `simba`, and I get a long error message that ends with `ImportError: cannot import name 'available_if'`
<details>
  <summary>Show solutions!</summary>
<br/><br/>

This error is produced when SimBA is installed with the wrong version of `pip`. To fix the error:

(i) Upgrade `pip` by typing `pip install pip==21.2.2`

(ii) Install SimBA by typing `pip install simba-uw-tf-dev`
  
(iii) Launch SimBA by typing `simba`. `
  
</details>


### 23. When installing shapely with "conda install -c conda-forge shapely" I see the error msg "CondaSSLError: OpenSSL appears to be unavailable on this machine. OpenSSL is required to download and install packages." 

<details>
  <summary>Show solutions!</summary>
<br/><br/>

Try to install `shapely` with "pip install shapely` or `conda install shapely` rather than "conda install -c conda-forge shapely". Then try to re-launch SimBA with `simba`. Also see [THIS GITHUB ISSUE](https://github.com/sgoldenlab/simba/issues/214)
  
</details>

### 24. When building and evaluating a behavior classification model, I had no issues! However, now, when I am analyzing new data I hit an error complaining about feature number mismatch: e.g., `FEATURE NUMBER ERROR: Mismatch in the number of features in input file and what is expected from the model in file MyVideo and model 0 ('Number of features of the model must match the input. Model n_features is 750 and input n_features is 621 ',)

<details>
  <summary>Show solutions!</summary>
<br/><br/>

When you create the features, SimBA will try to count how many frames represents different time windows, ranging from half-a-second at the largest window to 1/15th of a second (66ms) being the smallest time window. However, say that the video in your project with the lowest  frames-per-second is 2. In this situation it is not possible to get the number of frames that represent 66ms, because a single frame represents 500ms. SimBA will then discard this time window calculation, and do not calculate any features in the 66ms time window. 

Say you created your classifier where the lowest FPS was 15 or higher. SimBA will then create all features in all time windows. Next, when you create the features for the new data, the lowest FPS is 5. SimBA will then omit calculating the 66ms features, and you end up with fewer features then what you used to create the model.

If your classifiers performs well, and you don't want to update them, I would stick to analyzing videos with FPS 15 or higher and omit lower FPS videos from analyses. However, if you need your classifier to be able to handle lower FPS videos (e.g., 7), then you have to update  the classifier with annotated videos containing the lowest FPS you want the classifier to be able to handle.
  
</details>

### 25. My pose-estimation tracking looks good when I visualize it in the pose-estimation tool. However, after I import it into SimBA, it doesn’t look good anymore, why??

<details>
  <summary>Show solutions!</summary>
<br/><br/> 
  
Some situations when the tracking can look good in pose-estimation package, but then become messed up when importing it into SimBA:

i) The copy of the video analyzed pose-estimation has different frame rate or resolution then the copy of video imported into SimBA. E.g., say you analyzed a video in DLC that was 15fps, and 640x400 resolution. You imported the pose-estimation CSV file, and then you cropped it and/or changed the frame rate to 10fps and imported it to SimBA. The body-parts are no longer in the pixel location in relation to video in which pose-estimation was performed.

ii) Say you have a fair bit of intermittent missing body-parts in pose-estimation. Missing body-parts are placed at (0,0) - top left corner of the image. You then apply a linear smoothing function in SimBA (Gaussian, or Savitzky-Golay) for missing body-parts. It is possible that you see the body-parts flying about as SimBA trying to interpolate the body-parts all over the place between (0,0) and the actual location of the body-parts. You can try another smoothing function (e.g., nearest).

iii) Say you perform outlier correction, but you apply a criterion that is too stringent. SimBA will then remove body-part movements that are true body-part movement, and the animal body-part location predictions can appear to be “stuck” in the video while the animal is actually moving.

(iv) When visualizing the pose-estimated data in the pose-estimation tool (e.g., DLC), the default body-part circle colors can be a single color. For example, if both left and right ear of a mouse are visualized using red circles, and they switch positions, it is **very** difficult spot. 

SimBA however, assign different colors to the different body-parts, and these body-part switches/jumps/shifts therefore become more noticeable. To check if this is the case, go back to your pose-estimation package and use their tools to visualize the data using different colors to see if the switches are noticeable.  

Some pose-estimation tools, like DeepLabCut also has a filtering method, which can remove body-part predictions where the confidence is low. This will make the visualizations in DeepLabCut much better. However, these low-confidence predictions become apparent when imported into SimBA. 

</details>

### 26. When I try to install SimBA I get an error about scikit learn missing '_OneToOneFeatureMixin' e.g., end of error message may read: ``from sklearn.base import _OneToOneFeatureMixin as OneToOneFeatureMixin ImportError: cannot import name '_OneToOneFeatureMixin'``

<details>
  <summary>Show solutions!</summary>
<br/><br/>

Check out this potential solution on [stackoverflow](https://stackoverflow.com/questions/56549270/importerror-cannot-import-name-multioutputmixin-from-sklearn-base). 

Specifically, try:

i) `pip uninstall imblearn` followed by;
ii) ``conda install -c conda-forge imbalanced-learn``

Then try to relaunch simba by typing `simba`.

</details>

### 27. I'm on Linux, and when I install SimBA I get a complaint error while installint the SHAP library, complaining about ``gcc`` and exit code 1

<details>
  <summary>Show solutions!</summary>
<br/><br/>

Check out this solution on [stackoverflow](https://github.com/watson-developer-cloud/python-sdk/issues/418#issuecomment-674038596) 

Specifically, try:

i) `sudo apt-get install gcc` followed by `sudo apt-get install g++`

Then try to relaunch simba by typing `simba`.

</details>





Author [Simon N](https://github.com/sronilsson)

# Use user-defined 'feature extraction' script in SimBA

SimBA extracts ['features' from pose-estimation data](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-5-extract-features), and use these features together with [behavior annotations](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-6-label-behavior) to [build predictive classifiers](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-7-train-machine-model) of behavior. In SimBA (when using pre-defined pose-estimation bosy-part configurations), these features compose of curated metrics such as the distances between particular animal body-parts, and particular movements, which is relevant for social behavior. The exact features, and the number of features, is hard-coded in SimBA and is determined by the number of body-part key-points tracked in pose-estimation. For examples of the features calculated by SimBA when using pose-estimation from 2 animals and 16-body-parts, see [this file](https://github.com/sgoldenlab/simba/blob/master/misc/Feature_description.csv). When using SimBA with [user-defined body-part configurations]

In some scenarios, however, employing SimBAs' built-in feature-extraction scripts is not suitable, and the user may want to extract other features that are of particuar relevance to their behavior of interest. The rationale for the "user-defined 'feature extraction'" function in SimBA is 3-fold. Through the user-defined 'feature extraction' function in SimBA, users can:

1. Advanced users can write their own, brand-new, feature extraction scripts and use them within the SimBA GUI environment. These can be shared independetly of the SimBA program, and could improve classifier accuracy and computational time in experimental settings for which SimBA was not originally designed (i.e., non-social behavior). 

2. As SimBA is developing - the default hard-coded feature extraction files could, at times, be updated (i) with further, additional, features that we have found powerful for classifying particular social behaviors, or (ii) originally calculated features that we have found to lack predictive power in social settings may be removed to speed up computational time. The ability to deploy user-defined 'feature extraction' scrips make new versions of SimBA back-compatible, and users can specify to use feature extraction scripts that came with any prior version of SimBA within the newest latest-and-the-greatest SimBA GUI environment.      

3. By default, when using user-defined pose-estimation body-part configurations in SimBA, a generic feature battery of features is calculated in SimBA which encompass the distance between all body-parts and their velocities in rolling windows.  This may  not be optimal - as this generic feature set could include many features that are not relevant for the behavior of interest, while also missing some key features that could increase predictive accuracy if they were included. If the user is tracking a large number of body-parts using a user-defined pose-estimation configuration, the feature set can also turn very large. The ability to to use user-generated feature extraction scripts in SimBA overcomes all these hurdles.       

### Step 1: Use a user-defined feature extraction script in SimBA

1. Before using a user-defined feture extraction script in SimBA, import the pose-estimation tracking files and correct outliers. Instructions for how to import pose-estimation tracking files and correcting outliers can be found in the walk-through tutorial for [Scenario 1](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md) and [Part I](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-1-generate-project-config%5D) or the generic SimBA tutorial.  

2. Navigate to the 

User can now use their own extract features script to run. Click [here](https://osf.io/cmkub/) to download the sample script.

## Pre-requisite

![](/images/effolder.PNG)

1. The user should have an empty `__init__.py` file in the script path.

2. The name of the extract features script can be any name but it cannot contain any spaces. Eg: *arbitraryscriptname.py* is good, while *arbitrary script name .py* is bad.

3. The name of the folder that contains the script should not contain any spaces too.

4. In the python script, the main function should be name **extract_features_userdef** that allows the config ini file to pass in as an argument. Hence,
it should look something like this `def extract_features_userdef(inifile):`


<p align="center">
  <img width="288" height="35" src="/images/defextractf.PNG">
</p>


## How to run your own script

![](/images/extractfusrdef.PNG)

1. First, load your project and navigate to the `[Extract Features]` tab.

2. Under **Extract Features**, check the checkbox `Apply user defined feature extraction script`.

3. Select your script by clicking on `Browse File`.

4. Click on `Extract Features` button to run your script through SimBA.


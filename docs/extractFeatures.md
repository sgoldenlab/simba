# Use user-defined 'feature extraction' script in SimBA

SimBA extracts ['features' from pose-estimation data](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-5-extract-features), and use these features together with [behavior annotations](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-6-label-behavior) to [build predictive classifiers](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-7-train-machine-model) of behavior. In SimBA (when using pre-defined pose-estimation bosy-part configurations), these features compose of curated metrics such as the distances between particular animal body-parts, and certain movements that may be particluarly relevant for studying social behaviours. The exact features, and the number of features, is hard-coded in SimBA and is determined by the number of body-part key-points tracked during pose-estimation. For examples of the features calculated by SimBA when using pose-estimation from 2 animals and 16-body-parts, see [this file](https://github.com/sgoldenlab/simba/blob/master/misc/Feature_description.csv). When using SimBA with [user-defined body-part configurations]

In some scenarios, however, employing SimBAs' built-in feature-extraction scripts is not suitable, and the user may want to extract other features that are of particuar relevance to their behavior of interest and experimental protocol. We address this with the `user-defined feature extraction` function in SimBA.  The `user-defined 'feature extraction` function in SimBA gives users significant flexibility:

1. Advanced users can write their own, brand-new, feature extraction scripts and use them within the SimBA GUI environment. These can be shared independetly of the SimBA program, and may improve classifier accuracy and computational time in experimental settings for which SimBA was not originally targeted (i.e., non-social behavior). 

2. As SimBA is developing - the default hard-coded feature extraction files could, at times, be updated (i) with further, additional, features that we have found powerful for classifying particular social behaviors, or (ii) originaly calculated features, that we have found to lack predictive power in social settings, may be removed to speed up computational time. The ability to deploy user-defined 'feature extraction' scrips make newer versions of SimBA back-compatible, and users can specify to use feature extraction scripts that came with any prior version of SimBA within the latest SimBA GUI environment.      

3. By default, when using user-defined pose-estimation body-part configurations in SimBA, a generic feature battery of features is calculated in SimBA which encompass the distance between all body-parts and their velocities in rolling windows.  This may  not be optimal - as this generic feature set could include many features that are not relevant for the behavior of interest, while also missing some key features that could increase predictive accuracy if they were included. If the user is tracking a large number of body-parts using a user-defined pose-estimation configuration, the generaically defined feature set can also turn very large. The ability to deplay user-generated feature extraction scripts in SimBA overcomes these hurdles.

### Step 1: Use a user-defined feature extraction script in SimBA

1. Before using a user-defined a feature extraction script in SimBA, load your project, import the pose-estimation tracking files and correct outliers. For instructions on how to load your project, importing pose-estimation tracking files, and correcting outliers, read the walk-through tutorial for [Scenario 1](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md) and/or [Part I](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-1-generate-project-config%5D) or the generic SimBA tutorial.  

2. Navigate to the `Extract features` tab in SimBA. and you should see the following window with the "User-defined feature extraction" menu circled in red in the image. 

![alt-text-1](/images/feat_1.JPG "Feat_1")

3. Tick the `Apply user defined feature extraction script` box, and click the `Browse File` button next to the `Script Path` entry-box. Go ahead and select the path to the folder containing your feature extraction script. There are some critical rules that the your new `feature extraction script` needs to follow in order to work within the SimBA GUI environment and within your project:

* The feature extraction script should be located in a folder which contains **two** files in total: (i) the feature extraction script itself and, (ii) a file called `__init__.py`. The `__init__.py` file should be completely empty. The folder **should not** have any spaces in it's name. For example, you may have a folder called `My_feature_extraction_script` on your Desktop (**not `My feature extraction script`**), which content looks like this:

![alt-text-1](/images/feat_3.JPG "Feat_3")

* The feature extraction script itself (e.g., `My_feature_extraction_script.py` in the image above) can be named anything. However, it also cannot contain any spaces in the filename. For example, *My_feature_extraction_script.py* is good, while *My feature extraction script.py* is bad.

* The inside of the *My_feature_extraction_script.py* script should contain a function accepeting a single argument. This main function should be named **extract_features_userdef**, and takes the main project config file as the argument. Hence, the function and its argument should look like this: 

<p align="center">
  <img width="288" height="35" src="/images/defextractf.PNG">
</p>

* For example outlines of SimBA feature extraction files, for either designing your own feature extraction script within SimBA, or modifying existing feature extraction scripts, please see the SimBA [OSF repository](https://osf.io/emxyw/)

4. After selecting the feature extraction script you would like to use in you project, click on `Extract Features` in the `Extract Features` sub-menu. 



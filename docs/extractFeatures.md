# User-defined 'feature extraction' script in SimBA

SimBA extracts ['features' from pose-estimation data](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-5-extract-features), and use these features together with [behavior annotations](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-6-label-behavior) to [build predictive classifiers](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-7-train-machine-model) of the behavior. In SimBA (when using pre-defined pose-estimation body-part configurations), these features compose of curated metrics such as computes explainable feature representations of movements, angles, paths, velocities, distances, and sizes within individual frames and as rolling time-window aggregates. The exact features, and the number of features, is hard-coded in SimBA and is determined by the number of body-part key-points tracked during pose-estimation. For examples of the features calculated by SimBA when using pose-estimation from 2 animals and 16-body-parts, see [this file](https://github.com/sgoldenlab/simba/blob/master/misc/Feature_description.csv).

In some scenarios, however, employing these built-in feature-extraction scripts is not suitable, and the user may want to extract other features that are of particuar relevance to their behavior of interest and experimental protocol. We address this with the `user-defined feature extraction` function in SimBA.  The `user-defined 'feature extraction` function in SimBA gives users significant flexibility, e.g.:

*1. Advanced users can write their own, brand-new, feature extraction scripts and use them within the SimBA GUI environment. These can be shared independently of the SimBA program, and may improve classifier accuracy and computational time in experimental settings for which SimBA was not originally targeted (i.e., non-social behavior).* 

*2. As SimBA is developing - the default hard-coded feature extraction files could, at times, be updated (i) with further, additional, features that we have found powerful for classifying particular social behaviors, or (ii) originaly calculated features, that we have found to lack predictive power in social settings, may be removed to speed up computational time. The ability to deploy user-defined 'feature extraction' scrips make newer versions of SimBA back-compatible, and users can specify to use feature extraction scripts that came with any prior version of SimBA within the latest SimBA GUI environment.*

*3. By default, when using user-defined pose-estimation body-part configurations in SimBA, a generic feature battery of features is calculated in SimBA which encompass the distance between all body-parts and their velocities in rolling windows.  This may  not be optimal - as this generic feature set could include many features that are not relevant for the behavior of interest, while also missing some key features that could increase predictive accuracy if they were included. If the user is tracking a large number of body-parts using a user-defined pose-estimation configuration, the generaically defined feature set can also turn very large. The ability to deplay user-generated feature extraction scripts in SimBA overcomes these hurdles.*

*4. Many features that SimBA calculates (e.g., distances between individual body-parts within animals, or the size of the animal represented as a convex hull) are only really relevant for classifying behaviors if the animals show variability in these features across sequential frames. Shape-shifting animals like, like rodents, do show variability in these features, while non-shape-shifting animals, like fish, do **not** show variability in these features. If we are working with non-shape shifting animals like fish we may therefore want to calculate alternative features; like angular features, dispersion time-series decomposition, rotation etc to build more accurate classifiers instead.* 

*5. SimBA has a large battery of feature caluclators only accessable through the API (as of 12/23). These feature calculators tap into [frequentist](https://simba-uw-tf-dev.readthedocs.io/en/latest/simba.mixins.html#module-simba.mixins.statistics_mixin), [circular](https://simba-uw-tf-dev.readthedocs.io/en/latest/simba.mixins.html#module-simba.mixins.circular_statistics), and [time-series](https://simba-uw-tf-dev.readthedocs.io/en/latest/simba.mixins.html#module-simba.mixins.timeseries_features_mixin) statistics. They also compute [geometric](https://simba-uw-tf-dev.readthedocs.io/en/latest/simba.mixins.html#module-simba.mixins.geometry_mixin) manipulations and time-dependent [network (graph)](https://simba-uw-tf-dev.readthedocs.io/en/latest/simba.mixins.html#module-simba.mixins.network_mixin) based measures and other ML related distribution measures. To take advantage of these, users will currently have to write standalone classes calling these methods.*


### Use a user-defined feature extraction script in SimBA

1. Before using a user-defined a feature extraction script in SimBA, load your project, import the pose-estimation tracking files and correct outliers. For instructions on how to load your project, importing pose-estimation tracking files, and correcting outliers, read the walk-through tutorial for [Scenario 1](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md) and/or [Part I](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-1-generate-project-config%5D) or the generic SimBA tutorial.  

2. Navigate to the `Extract features` tab in SimBA. and you should see the following window with the "User-defined feature extraction" menu marked in red in the image. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/feature_extraction_user_defined_2023.png" />
</p>

3. Tick the `Apply user defined feature extraction script` box, and click the `Browse File` button next to the `Script path` entry-box. Go ahead and select the path to your feature extraction script.

> Note: Your feature extraction script needs to be a `.py` file with a **single** python class. If there are more then one class in your feature extraction script, then SimBA will use the first class. 

4. After selecting the feature extraction script you would like to use in you project, click on `Extract Features` in the `Extract Features` sub-menu. You can follow the progress in the SimBA GUI main window and/or the operating system terminal which you used to boot up the SimBA GUI.

### Writing a custom feature extraction script and running it in SimBA: Design, expected layout and gotcha's. 

>Note: For example outlines of SimBA feature extraction files, for either designing your own feature extraction script within SimBA, or modifying existing feature extraction scripts, please see the [hard-coded feature extraction scripts in SimBA](https://github.com/sgoldenlab/simba/tree/master/simba/feature_extractors) or the SimBA [OSF repository](https://osf.io/emxyw/). For an example user-defined feature extraction script used to score behavior in zebrafish, see [THIS FILE](https://github.com/sgoldenlab/simba/blob/master/simba/feature_extractors/misc/fish_feature_extractor_2023_version_5.py). For an example classifyin pup/dam behavior, see [THIS FILE](https://github.com/lapphe/AMBER-pipeline/blob/main/SimBA_AMBER_project/AMBER_2_0__feature_extraction/amber_feature_extraction_20230815.py). For an example feature extraction script that calculates geometric features and bounding boxes from pose-estimation data, see [THIS FILE](https://github.com/sgoldenlab/simba/blob/master/misc/geometry_feature_extraction.py).


When reading in your custom feature extraction file, SimBA will check its layout an execute it accordingly, with some expectations:

(1) SimBA expects you to use a file with a `.py` file extension. 

(2) SimBA expects the file to contain a single python class. If several classes are found in the file, then the first class will be used. 

(3) If the feature extraction class relies on argparse AND inherits ``simba.mixins.abstract_classes.AbstractFeatureExtraction``
(see [THIS FILE](https://github.com/sgoldenlab/simba/blob/master/misc/geometry_feature_extraction.py) for an example custom feature extraction class that relies on argparse AND inherits ``simba.mixins.abstract_classes.AbstractFeatureExtraction``), then the feature extraction class will be executed though the subprocess module. This ensures that the GUI cannot interfere with the feature extraction process, and reliably exexution of multicore processes if present in the feature-extraction class.

(4) If the feature extraction class does not rely on argparse, then the class will be loaded and executed in python through ``sys``. It
might still be quick and reliable (it often is). However, I have noted that some function, in particular functions that rely on ``multiprocessing.imap`` are disrupted by the GUI with unacceptable effects on runtime when executed thriugh ``sys``.

(5) If you are intrested in the code that SimBA uses to parse and exexute your custom feature extraction code, you can find it [HERE](https://github.com/sgoldenlab/simba/blob/master/simba/utils/custom_feature_extractor.py). If you have suggested improvements, bug reports, or ideas, please consider reaching out to us on [Gitter](https://app.gitter.im/#/room/#SimBA-Resource_community:gitter.im) or by opening an [issue](https://github.com/sgoldenlab/simba/issues) here on GitHub. 

#
Author [Simon N](https://github.com/sronilsson))

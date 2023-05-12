# <p align="center"> Feature sub-sets in SimBA </p>

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/feature_subsets_0.png" />
</p>

### INTRODUCTION

SimBA extracts features for builing and [running downstream machine learning models](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-5-extract-features). However, at times, we may just want to take advantage of some SimBA [feature calculators](https://github.com/sgoldenlab/simba/blob/master/simba/mixins/feature_extraction_mixin.py) and generate a subset of measurements for use in our own downstream applications. For some of the feature sub-sets available, see the image above. 


### INSTRUCTIONS

1) To create feature sub-sets, first import your pose-esimation data into SimBA and follow the instruction up-to and **including** outlier correction as documented in the [Scenario 1 tutorial](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md). Before calculating feature sub-sets in SimBA, ensure that the `project_folder/csv/outlier_corrected_movement_location` directory of your SimBA project is populated with files. 

2). Navigate to the [Extract features] tab, and click on the `CALCULATE FEATURE SUBSETS` button.

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/feature_subsets_2.png" />
</p>

3) In the `FEATURE FAMILY` drop-down, select the type of measurements you want to calculate. 

> Note I: If you find a set of features is missing from the drop-down, let us know by opening a [GitHub issue](https://github.com/sgoldenlab/simba/issues) or reach out to us on [Gitter](https://gitter.im/SimBA-Resource/community) and we will work to get it in.  

> Note II: Calculating convex hull sizes is expensive. Thus, if you are calculating 3- or 4-point convex hull areas and have a lot of animals/body-parts, chances are it will take a while. SimBA currently (03/2023) rely on [joblib.Parallel](https://joblib.readthedocs.io/en/latest/parallel.html) and [scipy.spatial.ConvexHull](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html) to compute convex hulls [HERE](https://github.com/sgoldenlab/simba/blob/cfd8806a75e507dac8df2cfaef6934d51703ba70/simba/mixins/feature_extraction_mixin.py#L69). If you know of faster methods, please consider contributing! 



<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/feature_subsets_3.png" />
</p>

4). In the `SAVE DIRECTORY` selection box, select a directory where you want to save your feature data. It's a good idea to select an empty directory. 


5). Once filled in, hit the `RUN` button. You can follow the progress in the main SimBA window. 


6). Once complete, the `SAVE DIRECTORY` will be filled with one file for every video file represented in your `project_folder/csv/outlier_corrected_movement_location` directory. In these files, every row represents a frame, and every column represents a feature in feature family. The number of columns (features) will depend on the number of body-parts and animals in your SimBA project.

For smaller examples of expected output, see:

* [Two-point body-part distances (mm).csv](https://github.com/sgoldenlab/simba/blob/master/misc/Two-point%20body-part%20distances%20(mm).csv)
* [Within-animal three-point body-part angles (degrees).csv](https://github.com/sgoldenlab/simba/blob/master/misc/Within-animal%20three-point%20body-part%20angles%20(degrees).csv)
* [Within-animal three-point convex hull (mm2).csv](https://github.com/sgoldenlab/simba/blob/master/misc/Within-animal%20three-point%20convex%20hull%20(mm2).csv)
* [Within-animal four-point convex hull (mm2).csv](https://github.com/sgoldenlab/simba/blob/master/misc/Within-animal%20four-point%20convex%20hull%20(mm2).csv)
* [Frame-by-frame body-part movement (mm).csv](https://github.com/sgoldenlab/simba/blob/master/misc/Frame-by-frame%20body-part%20movement%20(mm).csv)







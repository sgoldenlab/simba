# Steps to run grooming, rearing, and Straub tail classifiers. 

## 1) PRE-PROCESSING: Egocentric alignment and background subtraction for Straub tail classification. 

For accurately classyifing Straub tail, we need features constructed from **images** of the animals tail (not, as done by default, features constructed from the **pose-estimated locations** of the animal). To get standardized images of the animals 
tails, we need to create copies of the original videos and original pose-estimation data where the background have been subtracted and the videos have been egocentrically aligned. 

For these steps, I have jupyter notebook examples below, but please let me know if your prefer to go through these steps by clicking in a graphical user interface and I can make it available. 

#### Background subtraction

As we are constructing features from images of the animal and its tail, we don't want the background (i.e., the arena floor) to be visable. If it was, then pixel values around and under the tail would influence the feature values. To remove the background,
we compute the average frame pixel values, and remove the mean from each image as below.

[docs__static_img_bg_removed_ex_2_clipped.webm](https://github.com/user-attachments/assets/261c36e0-c59d-4f57-9422-430277d3b78b)

To create copies of videos where the background is removed, you can use [THIS](https://simba-uw-tf-dev.readthedocs.io/en/latest/nb/bg_remove.html) notebook.

#### Egocentric alignment

After peforming the background subtraction, we need to [egocentrically aligned](https://simba-uw-tf-dev.readthedocs.io/en/latest/simba.data_processors.html#simba.data_processors.egocentric_aligner.EgocentricalAligner) copies of the background subtracted videos. 
This means that we spin the videos around so that the animal is always faacing in the same direction (e.g., east), as in the example below. We do this to remove any variability associated with animal direction: i.e., the animal direction in itself should can not change how the tail looks like.

[EgocentricalAligner.webm](https://github.com/user-attachments/assets/7caf920b-0e86-49c2-bfde-2b606de6d6d8)

As we know the location of the animal snout and center through the pose-estimated locations from SLEAP/DLC to, we can rotate the images so that the center body-part is always located in near the center of the video, 
and the nose is always pointing 0 degrees axis-aligned of the center. Note, as we are rotating the images, we should also create copies of the original pose-estimation data where the same rotation factors have been applied. This ensures that we still know where the body-parts are 
located in the rotated videos,  and we can perform any calculations from the body-parts in the rotated data the same way as we could in the original, un-rotated, data. 

To perform pose-estimation based video and data rotation, use [THIS](https://simba-uw-tf-dev.readthedocs.io/en/latest/nb/egocentric_align.html) notebook. 

> [!NOTE] 
> Again I can insert GUI options for doing these operations (background subtraction and egocentric alignment) if you prefer!

## 2) DATA IMPORT: Interpolation and smoothing.

Next, import the raw, original pose-estimation data into SimBA, perform `Body-part: Nearest` interpolation. This ensures that for any frame a body-part is missing in, the body-part will take the location of when the body-part was last visibale. 

Also perform Savitzky-Golay smoothing using a 400ms time-window to remove pose-estimation jitter. 

Do not perform any outlier correction. 

## 3) CUSTOM FEATURE EXTRACTION.

We will compute a bunch of features that measures the animals hull, and parts of the hull (e.g., lower body, upper body, head) and how their shapes and sizes vary in sliding time-windows. To do this, we will use the 
custom feature extraction script in SimBA as documented [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/extractFeatures.md). The feature extraction script that we will be using, is called `MitraFeatureExtractor`,
and is located [HERE](https://github.com/sgoldenlab/simba/blob/master/simba/sandbox/MitraFeatureExtractor.py). 

## 4) RUN REARING AND GROOMING CLASSIFIERS

After feature extraction, your good to go and run the `grooming` and `rearing` classifier. Run the classifiers
in SimBA as documented [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-3-run-the-classifier-on-new-data).

GROOMING THRESHOLD:             0.35                                            
GROOMING MINIMUM BOUT LENGTH:   200ms
REARING THRESHOLD:              0.30
REARING MINIMUM BOUT LENGTH:    200ms

## COMPUTE ADDITIONAL STRAUB TAIL FEATURES

Now, in order to run the Straub tail classifier, we need to compute some additional features, and combine them with the 
features computed in the prior step, to create the final feature set for the Straub tail classifier.  

**Step i**: To compute these features, use [THIS](https://github.com/sgoldenlab/simba/blob/master/simba/sandbox/mitra_tail_analyzer.py)
script. This produces a set of files, one for each of the videos, inside your passed ``save_dir``

**Step ii**: In step 2, we need to combine these features created in (i) with some features created in step `"3) CUSTOM FEATURE EXTRACTION"` to create the final
feature set for the tail classifier. To do this, run [THIS](https://github.com/sgoldenlab/simba/blob/master/simba/sandbox/mitra_appand_additional.py) scrip.
This produces a final set of files inside the passed ``SAVE_DIR``.

**Step ii**: Move the files ouputted in step (ii) to the ``project_folder/csv/features_extracted`` directory of your SimBA project (move the files you have used for the grooming/rearing classifications first). 
Alternatively, place thse files inside the ``project_folder/csv/features_extracted`` directory of a new SimBA project only designed to classify Straub tail. 

## 4) RUN STRAUB TAIL CLASSIFIER

Run the straub tail classifier in SimBA as documented [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-3-run-the-classifier-on-new-data).

STRAUB TAIL THRESHOLD:             0.30
STRAUB TAIL MINIMUM BOUT LENGTH:   200ms






















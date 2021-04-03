







## HYPERPARAMETERS

`Trackin propability (pup)` - the minimum mean **pup** body-part probability to consider in classifications. If the mean pup body-part probability does not reach this threshold for a given frame, then the behavioral classifications for that frame will be set to behavior absent. DEFAULT: 0.025.  

`Trackin propability (dam)` - the minimum mean **dam** body-part probability to consider in classifications. If the mean dam body-part probability does not reach this threshold for a given frame, then the behavioral classifications for that frame will be set to behavior absent. DEFAULT: 0.5.

`Start distance criterion (mm)` - the minimum start distance between the pup and the core-nest in millimeters. If pose-estimation places the pup at a shorter distance to the core-nest in the beginning of the video, then the the pup-to-corenest distance in these frames will be corrected. The pup-to-corenest distance in such frames will be corrected to be the distance in the first frame when the pup is located more or equal to the start distance criterion from the core-nest. DEFAULT: 80.

`Carry frames (s)` - for the pup to be in the nest, the dam needs to carry the pup before the pup enters the nesy. If the pup is detected in the core-nest, but the dam has not carried the pup in N seconds prior to the pup enters the nest, the pup will be removed from the nest. DEFULT: 3.  

`Core-nest name` - the name of the core-nest as defined in the [SimBA ROI definitions table](https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial.md#part-1-defining-rois-in-simba)

`Nest-name` - the name of the core-nest as defined in the [SimBA ROI definitions table](https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial.md#part-1-defining-rois-in-simba

`Dam name` - the name of the dam (mother) as set in the [SimBA project definitions table](https://github.com/sgoldenlab/simba/blob/master/docs/Multi_animal_pose.md#step-4-import-your-tracking-data) *Note*: The animal names can also be accessed in the *project_folder/project_config.ini* under the [Multi animal IDs] heading. 

`Pup name` - the name of the pup as set in the [SimBA project definitions table](https://github.com/sgoldenlab/simba/blob/master/docs/Multi_animal_pose.md#step-4-import-your-tracking-data) *Note*: The animal names can also be accessed in the *project_folder/project_config.ini* under the [Multi animal IDs] heading. 

`Smooth function` - function to smooth the animal movement trajectories. DEFULT: gaussian. 

`Smooth factor` - the standard deviation of the gaussian smoothing function. DEFULT: 5.

`Carry classifier name` - The name of the *carry* classifier as defined in the [SimBA project creation menu](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#part-1-create-a-new-project-1).

`Approach classifier name` - The name of the *approach* classifier as defined in the [SimBA project creation menu](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#part-1-create-a-new-project-1).

`Dig classifier name` - The name of the *dig* classifier as defined in the [SimBA project creation menu](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#part-1-create-a-new-project-1).

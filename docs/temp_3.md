Analyze machine predictions: BY SEVERITY: This type of analysis is only relevant if your behavior can be graded on a scale ranging from mild (the behavior occurs in the presence of very little body part movements) to severe (the behavior occurs in the presence of a lot of body part movements). For instance, attacks could be graded this way, with 'mild' or 'moderate' attacks happening when the animals aren't moving as much as they are in other parts of the video, while 'severe' attacks occur when both animals are tussling at full force. This button and code calculates the ‘severity’ of each frame (or bout) classified as containing the behavior based on a user-defined scale. Clicking the severity button brings up the following menu. We go through the meaning of each setting below:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/severity_analysis_pop_up.png" />
</p>


**CLASSIFIER:** This drop-down shows the classifiers in your SimBA project. Select the classifier which you want to score the severity for. 
**BRACKETS:** Select the size of the severity scale. E.g., select 10 if you want to score your classifications on a 10-point scale.
**ANIMALS:** Select which animals body-parts you want to use to calculate the movement. E.g., select ALL ANIMALS to calculate the movement based on all animals and their body-parts.
**BRACKET TYPE:** If `QUANTIZE`, then SimBA  creates **N equally sized bins** (with N defined in the BRACKETS dropdown). If `QUANTILE`, 
SimBA forces an equal number of frames into each bin and creates **N unequally sized bins**. For more detailed, see the differences between [pandas.qcut](https://pandas.pydata.org/docs/reference/api/pandas.qcut.html) and [pandas.cut](https://pandas.pydata.org/docs/reference/api/pandas.cut.html).
**DATA TYPE:** When binning the severity, we can either obtain severity scores for each (i) individual classifified frame, or (ii) for each classified bout. Select `BOUTS` to get a severity score for each classified bout and `FRAMES` to get a severity score per classified frame. 
**MOVEMENT NORMALIZATION TYPE:** When creating the severity bins, we can either (i) use all the movement data represented by all files with classifications (all data within the project_folder/csv/machine_results directory). This selection will results in a single bin reference scale that are applied equally to all videos, or (ii) we can create the scales by referencing the movement only the video itself. This selection will results in different severity scale bins for the different videos. 
**SAVE BRACKET DEFINITIONS:** If ticked, SimBA will save a CSV log file containing the bin definitions for each analysed video in the project (saved inside the project_folder/logs directory).

**VISUALIZE**: If ticked, SimBA will generate visualization example clips.
**SHOW POSE-ESTIMATED LOCATIONS**: If ticked, SimBA will include pose-estimated body-part locations (as circles) in the video clips.
**VIDEO SPEED**: The FPS of the example clips relative to the original video. E.g., if `1`, the clips will be saved at origginal speed. If `0.5`, the clips will be saved at hald the original speed.
**CLIP COUNT**: 







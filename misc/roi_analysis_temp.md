# Part 2. Analyzing ROI data.

Once we have [drawn the ROIs](https://github.com/sgoldenlab/simba/blob/master/docs/roi_tutorial_new_2025.md), we can compute descriptive statistics based on movement of the animal(s) in relation to the ROIs 
(for example, how much time the animals spends in teh ROIs, or how many times the animals enter the ROIs. To compute ROI statistics, click in the <kbd>ANALYZE ROI DATA: AGGREGATES</kbd> button in the `[ ROI ]` tab and you should see the 
following pop up:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/roi_analyze_tutorial_1.webp" />
</p>


1) In the `# OF ANIMALS` dropdown, select the number of animals you want to compute ROI statistics for.
   
2) In the `PROBABILITY THRESHOLD` entry-box, select the minimum pose-estimation probability score (between 0 and 1) that should be considered when performing ROI analysis. Any frame body-part probability score above the entered value will be filtered out.  
> [!CAUTION]
> If possible, we recommend having reliable pose-estimation data in every frame. This includes pre-process all videos, and remove any segments of the videos where animals are not present in the video as documented [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial_process_videos.md),
> and performing pose-estimation interpolation of missing data at data import as documented [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#to-import-multiple-dlc-csv-file).

3). In the `SELECT BODY-PART(S)` frame, use the dropdown menus to select the body-parts that you wish to use to infer the locations of the animals. 

4). In the `DATA OPTIONS` frame, select the data that you wish to compute:

 * `TOTAL ROI TIME (S)`: The total time, in seconds, that each animal spends in each defined ROI in each video.
 * `ROI ENTRIES (COUNT): The total number of times, that each animal enters each defined ROI in each video.
 * `FIRST ROI ENTRY TIME (S)`: The video time, in seconds, when each animal first enters each defined ROI in each video (NOTE: will be `None` if an animal never enters a defined ROI).   
 * `LAST ROI ENTRY TIME (S)`: The video time, in seconds, when each animal last enters each defined ROI in each video (NOTE: will be `None` if an animal never enters a defined ROI).
 * `MEAN ROI BOUT TIME (S)`: The mean length, in seconds, of each sequence the animal spends in each defined ROI in each video (NOTE: will be `None` if an animal never enters a defined ROI).
 * `DETAILED ROI BOUT DATA (SEQUENCES)`: If checked, the SimBA ROI analysis generates a CSV file within the `project_folder/logs` directory named something like *Detailed_ROI_bout_data_20231031141848.csv*. This file contains the exact frame numbers, time stamps, and duration of each seqence when animals enter and exits each user-drawn ROIs. (NOTE: no file will be created if no animal in any video never enters an ROI)
 * `ROI MOVEMENT (VELOCITY AND DISTANCES)`: The total distance moved, and the average velocity, of each animal in each defined ROI in each video.

5). In the `FORMAT OPTION` frame, select how the output data should be formatted, and any addtional video meta data that should be included in the output which could be helpful for sanity checks. 

* `TRANSPOSE OUTPUT TABLE`: If checked, one row in the output data will represent a video. If unchecked, one row in the output data will represent a data measurment.
* `INCLUDE FPS DATA`: If checked, the FPS used to compute the metrics of each video will be included in the output table. 
* `INCLUDE VIDEO LENGTH DATA`: If checked, the length of each video in seconds will be included in the output table.
* `INCLUDE INCLUDE PIXEL PER MILLIMETER DATA`: If checked, the pixel per millimeter conversion factor (used to compute distances and velocity) of each video will be included in the output table.

6. Once the above is filled in, click the <kbd>RUN</kbd> button. You can foillow the progress in the main SimBA terminal.

Once complete, a file will be stored in the logs folder of your SimBA project named something like `ROI_descriptive statistics_20250306162014.csv`. If you did **not** check the
`TRANSPOSE OUTPUT TABLE` checkbox, the file will look something like [THIS](https://github.com/sgoldenlab/simba/blob/master/misc/ROI_descriptive%20statistics_non_transposed.csv), where each row represent a specific video, ROI, animal, and measurement. If you **did** check the  `TRANSPOSE OUTPUT TABLE` checkbox, the file will look something like [THIS](https://github.com/sgoldenlab/simba/blob/master/misc/ROI_descriptive%20statistics_transposed.csv), where each row represent a specific video
and each column represents a specific measurment.

If you did check the `DETAILED ROI BOUT DATA (SEQUENCES)` checkbox, an additional file will be crated in the SimBA project logs folder named something like `Detailed_ROI_data_20250306162014.csv` that can be expected to look something like [THIS](https://github.com/sgoldenlab/simba/blob/master/misc/Detailed_ROI_data_20250307101923.csv). This file contains a row of information (entry and exit time, frame and entry duration) for every ROI, for every animal, and every video. It also contains columns for the animal name and body-part name for reference. 

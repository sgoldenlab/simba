# YOLO Pose Estimation Visualization in SimBA

SimBA has a graphical user interface (GUI) for visualizing YOLO-based pose estimation results on videos. This GUI allows users to overlay detected keypoints, bounding boxes, and tracking information onto videos, creating annotated output videos that show the pose estimation results.

<img width="704" height="203" alt="image" src="https://github.com/user-attachments/assets/b8d732dc-cdfc-47d8-a959-09b1811b37a6" />

## Table of Contents

- [Prerequisites](#prerequisites)
- [Accessing the YOLO Pose Visualizer Popup](#accessing-the-yolo-pose-visualizer-popup)
- [Interface Overview](#interface-overview)
- [Usage Instructions](#usage-instructions)
- [Output](#output)
- [Important Notes](#important-notes)
- [Troubleshooting](#troubleshooting)
- [Related Documentation](#related-documentation)


<a id="prerequisites"></a>
## Prerequisites

Before using the YOLO Pose Visualizer popup, ensure that:

1. **YOLO Pose Estimation Results**: You need CSV files containing YOLO pose estimation results (output from YOLO pose inference). These files should contain keypoint coordinates and confidence scores for each frame.
2. **Source Videos**: You need the original video files that correspond to the pose estimation results. The video filenames should match the CSV filenames (excluding extensions).
3. **Sufficient Disk Space**: Ensure you have enough disk space to save the annotated output videos, which will be the same size or larger than the input videos.

<a id="accessing-the-yolo-pose-visualizer-popup"></a>
## Accessing the YOLO Pose Visualizer Popup

The YOLO visualization menu can be accessed in the SimBA main interface. The popup window is titled **"PLOT YOLO POSE ESTIMATION RESULTS"** and provides an interface for configuring and creating annotated videos with pose estimation overlays.

<a id="interface-overview"></a>
## Interface Overview

<img width="922" height="493" alt="image" src="https://github.com/user-attachments/assets/8f5535fe-18c3-4b43-a6c1-17dc2738e86f" />

The popup window is organized into several sections:

### SETTINGS

This section contains visualization parameters that control how the pose estimation results are displayed:

- **SAVE DIRECTORY**: Directory where the annotated output videos will be saved.
  - **Effect**: All generated videos are saved to this directory. Make sure you have sufficient disk space, as output videos can be large. The output videos will have the same names as the input videos with pose estimation overlays.

- **PLOT TRACKS**: Whether to visualize tracking information (track IDs and paths). Options: TRUE or FALSE (default).
  - **Effect**: When TRUE, uses `YOLOPoseTrackVisualizer` which shows track IDs and maintains consistent colors for each tracked animal across frames. This is useful for multi-animal scenarios where you want to see which detections belong to the same track. When FALSE (default), uses `YOLOPoseVisualizer` which shows keypoints without track information. **Use case**: Enable tracks if you used tracking during inference and want to visualize animal identities over time. Disable for single-animal scenarios or when track information isn't needed.

- **CPU CORE COUNT**: Number of CPU cores to use for parallel video processing. Options range from 1 to the maximum number of CPU cores available. Default is approximately one-third of the maximum cores.
  - **Effect**: More CPU cores can significantly speed up video processing, especially for long videos or when processing multiple videos. However, using all cores may slow down other applications. **Trade-off**: More cores = faster processing but higher CPU usage. Default (one-third of cores) is usually a good balance. Increase if you want faster processing and have cores available. Decrease if you need to reserve CPU for other tasks.

- **SHOW BOUNDING BOXES**: Whether to draw bounding boxes around detected animals. Options: TRUE (default) or FALSE.
  - **Effect**: When TRUE, draws rectangular bounding boxes around each detected animal, making it easier to see detection boundaries. When FALSE, only keypoints and skeleton connections are shown. **Recommendation**: Keep TRUE (default) to clearly see detection boundaries, especially useful for debugging or presentations. Set FALSE for cleaner visualizations focusing only on keypoints.

- **VERBOSE**: Whether to output detailed progress information during processing. Options: TRUE (default) or FALSE.
  - **Effect**: When TRUE, you'll see detailed progress updates, frame counts, processing times, and any warnings in the terminal/console. This is helpful for monitoring progress, especially for long videos. When FALSE, only essential messages are shown. **Recommendation**: Keep TRUE (default) to monitor progress, especially for long-running visualization tasks.

- **THRESHOLD**: Confidence threshold for rendering detections. Only detections with confidence scores above this threshold are drawn. Options range from 0.0 to 1.0 in increments of 0.1. Default is 0.0.
  - **Effect**: Lower thresholds (0.0-0.3) show all detections, including low-confidence ones, which can help visualize all detected keypoints but may show noisy or incorrect detections. Higher thresholds (0.5-1.0) only show high-confidence detections, creating cleaner visualizations but potentially hiding some valid detections. **Trade-off**: Lower threshold = more complete visualization (including potential false positives) vs. higher threshold = cleaner but potentially incomplete visualization. Start with 0.0 to see all detections, increase to 0.3-0.5 to filter out low-confidence detections.

- **LINE THICKNESS**: Thickness of skeleton lines connecting keypoints. Options: AUTO (default), or 1-20 pixels.
  - **Effect**: When AUTO (default), line thickness is automatically calculated based on video frame dimensions for optimal visibility. Manual values (1-20) set a fixed pixel thickness. Thicker lines (10-20) are more visible but may obscure details. Thinner lines (1-5) are more subtle but may be hard to see in high-resolution videos. **Recommendation**: Use AUTO (default) for optimal results. Adjust manually only if you need specific line thickness for presentation or publication purposes.

- **CIRCLE SIZE**: Radius of keypoint circles in pixels. Options: AUTO (default), or 1-20 pixels.
  - **Effect**: When AUTO (default), circle size is automatically calculated based on video frame dimensions for optimal visibility. Manual values (1-20) set a fixed pixel radius. Larger circles (10-20) are more visible but may obscure fine details. Smaller circles (1-5) are more precise but may be hard to see. **Recommendation**: Use AUTO (default) for optimal results. Adjust manually only if you need specific circle sizes for presentation or publication purposes.

### PLOT SINGLE VIDEO

This section allows you to visualize pose estimation results for a single video:

- **DATA PATH (CSV)**: Select the path to the CSV file containing YOLO pose estimation results for the video.
- **VIDEO PATH**: Select the path to the original source video file.
- **CREATE SINGLE VIDEO**: Button to start visualization for the single selected video.

### PLOT MULTIPLE VIDEOS

This section allows you to visualize pose estimation results for multiple videos in batch:

- **DATA DIRECTORY**: Select the directory containing CSV files with YOLO pose estimation results. Each CSV file should correspond to a video file.
- **VIDEO DIRECTORY**: Select the directory containing the original source video files. Video filenames should match CSV filenames (excluding extensions).
- **CREATE MULTIPLE VIDEOS**: Button to start batch visualization for all matching video/CSV pairs.

> **Note**: For multiple videos, the tool automatically matches CSV files to video files based on filenames (excluding extensions). Videos without matching CSV files, or CSV files without matching videos, will generate warnings but won't stop the process.

<a id="usage-instructions"></a>
## Usage Instructions

### Step 1: Configure Visualization Settings

1. Click on **SAVE DIRECTORY** and select the folder where annotated output videos should be saved.

2. Adjust visualization parameters in the **SETTINGS** section:
   - Set **PLOT TRACKS** to TRUE if you want to visualize tracking information (track IDs and consistent colors per animal)
   - Set **SHOW BOUNDING BOXES** to TRUE (default) to display bounding boxes around detections
   - Adjust **THRESHOLD** to filter low-confidence detections (0.0 shows all, higher values filter more)
   - Set **LINE THICKNESS** and **CIRCLE SIZE** to AUTO (default) or specific values
   - Set **CPU CORE COUNT** based on available CPU resources
   - Keep **VERBOSE** as TRUE (default) to monitor progress

### Step 2: Select Video(s) and Data

**For a single video:**

1. Click on **DATA PATH (CSV)** and select the CSV file containing pose estimation results.
2. Click on **VIDEO PATH** and select the corresponding source video file.
3. Click the **CREATE SINGLE VIDEO** button to start visualization.

**For multiple videos:**

1. Click on **DATA DIRECTORY** and select the folder containing CSV result files.
2. Click on **VIDEO DIRECTORY** and select the folder containing source video files.
3. Ensure that CSV filenames match video filenames (excluding extensions). For example:
   - `video1.csv` should match `video1.mp4`
   - `experiment_001.csv` should match `experiment_001.avi`
4. Click the **CREATE MULTIPLE VIDEOS** button to start batch visualization.

### Step 3: Monitor Progress

If **VERBOSE** is set to TRUE, you will see progress information in the terminal/console window, including:
- Current video being processed
- Frame counts and processing times
- Any warnings about missing files or mismatches

The visualization process will:
1. Load the pose estimation data from CSV files
2. Load corresponding video frames
3. Overlay keypoints, skeleton connections, and optional bounding boxes
4. Save annotated videos to the specified save directory

<a id="output"></a>
## Output

The visualization process creates annotated output videos saved in the specified **SAVE DIRECTORY**. The output videos:

- Have the same names as the input videos (with pose estimation overlays)
- Contain color-coded keypoints for each detected animal/class
- Show skeleton connections between keypoints (if applicable)
- Display bounding boxes around detections (if enabled)
- Show track IDs and consistent colors per track (if track visualization is enabled)
- Maintain the same frame rate and resolution as the input videos

The output format matches the input video format (MP4, AVI, etc.).

<a id="important-notes"></a>
## Important Notes

1. **File Matching**: For multiple videos, ensure CSV filenames match video filenames (excluding extensions). The tool matches files based on base names, so `data.csv` will match `data.mp4`, `data.avi`, etc.

2. **Performance**: Visualization can be time-consuming for long videos or high-resolution videos. Processing time depends on:
   - Video length and resolution
   - Number of detections per frame
   - Number of CPU cores used
   - Whether tracking visualization is enabled

3. **Memory Considerations**: Processing large videos or many videos simultaneously may require significant RAM. If you encounter memory issues:
   - Process videos in smaller batches
   - Reduce the number of CPU cores used
   - Close other applications

4. **Track Visualization**: Track visualization (PLOT TRACKS = TRUE) requires that your CSV files contain track ID information. If your pose estimation results don't include tracking data, track visualization may not work correctly. Use regular visualization (PLOT TRACKS = FALSE) for non-tracked results.

5. **Threshold Filtering**: The threshold setting filters detections based on confidence scores. Setting a higher threshold creates cleaner visualizations but may hide valid low-confidence detections. Use threshold 0.0 to see all detections, then adjust based on your needs.

6. **Visualization Quality**: The AUTO settings for line thickness and circle size are optimized for most use cases. Manual settings may be needed for specific presentation or publication requirements.

7. **Color Coding**: Different animals/classes are automatically assigned different colors. When track visualization is enabled, each track maintains a consistent color across frames.

<a id="troubleshooting"></a>
## Troubleshooting

- **"No matching files found"**: Ensure that CSV filenames match video filenames (excluding extensions). Check that files are in the correct directories and have matching base names.

- **"File not found errors"**: Verify that:
  - CSV files exist and are readable
  - Video files exist and are readable
  - File paths are correct
  - Files are in the specified directories

- **"Missing video files" or "Missing data files" warnings**: These warnings indicate that some CSV files don't have matching videos, or vice versa. The tool will skip these files and continue processing matching pairs. Check your file naming to ensure consistency.

- **Slow processing**: 
  - Increase **CPU CORE COUNT** if you have available cores
  - Reduce video resolution if possible
  - Process videos in smaller batches
  - Disable track visualization if not needed (PLOT TRACKS = FALSE)

- **Out of memory errors**: 
  - Reduce **CPU CORE COUNT** to use less memory
  - Process videos one at a time instead of in batch
  - Close other applications using memory
  - Process shorter video segments

- **Poor visualization quality**: 
  - Adjust **THRESHOLD** to filter out low-confidence detections
  - Increase **LINE THICKNESS** and **CIRCLE SIZE** for better visibility
  - Enable **SHOW BOUNDING BOXES** to see detection boundaries
  - Check that your pose estimation results are accurate

- **Tracks not showing correctly**: 
  - Ensure your CSV files contain track ID columns
  - Verify that tracking was enabled during pose estimation inference
  - Check that **PLOT TRACKS** is set to TRUE
  - If tracks still don't work, use regular visualization (PLOT TRACKS = FALSE)

- **Colors not consistent across frames**: 
  - Enable **PLOT TRACKS** to maintain consistent colors per track
  - Verify that track IDs are present in your CSV data
  - Check that the same track IDs are used consistently across frames

<a id="related-documentation"></a>
## Related Documentation

- For running YOLO pose estimation inference to generate the CSV results, see the [YOLO pose inference documentation](https://github.com/sgoldenlab/simba/blob/master/docs/yolo_pose_inference_popup.md).
- For training YOLO pose estimation models, see the [YOLO pose training documentation](https://github.com/sgoldenlab/simba/blob/master/docs/yolo_pose_train_popup.md).
- For importing YOLO pose estimation results into SimBA, see pose estimation import tools.
- For programmatic access to visualization functions, see:
  - `simba.plotting.yolo_pose_visualizer.YOLOPoseVisualizer` - Regular pose visualization
  - `simba.plotting.yolo_pose_track_visualizer.YOLOPoseTrackVisualizer` - Track-based pose visualization

##
Author [Simon N](https://github.com/sronilsson)

